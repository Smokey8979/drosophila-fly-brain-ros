"""
Main Fly Brain ROS 2 Control Node.

Connects the Drosophila whole-brain connectome LIF neural simulation to a 4-wheel
differential/skid-steer ground robot in ROS 2 / Gazebo:
  - Subscribes to /scan (sensor_msgs/msg/LaserScan) using sensor data QoS
  - Computes obstacle distances across left, right, and front visual sectors
  - Encodes distances into Poisson spike trains for LC4 and P9 neural populations
  - Advances the 138k-neuron PyTorch LIF brain engine asynchronously
  - Decodes descending motor neuron spikes into linear and angular velocity commands
  - Validates commands through SafetySupervisor (watchdog, hard stop, acceleration limit)
  - Publishes geometry_msgs/msg/Twist to /cmd_vel at a documented controller rate
  - Publishes zero-velocity failsafe commands on startup, shutdown, and fault conditions
"""

import json
import math
import os
import queue
import sys
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from time import perf_counter as time

# Try importing genuine rclpy; provide lightweight compatibility fallback if not installed
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    HAVE_RCLPY = True
except ImportError:
    HAVE_RCLPY = False

    # Standalone mock types for test execution without ROS 2 system installation
    class Node:
        def __init__(self, node_name: str, **kwargs):
            self._node_name = node_name
            self._params = {}
            self._subscriptions = []
            self._publishers = []
            self._timers = []

        def declare_parameter(self, name, value=None):
            self._params[name] = value

        def get_parameter(self, name):
            class _Param:
                def __init__(self, val):
                    self.value = val
            return _Param(self._params.get(name))

        def create_subscription(self, msg_type, topic, callback, qos_profile):
            sub = {"type": msg_type, "topic": topic, "cb": callback, "qos": qos_profile}
            self._subscriptions.append(sub)
            return sub

        def create_publisher(self, msg_type, topic, qos_profile):
            class _Pub:
                def __init__(self, t):
                    self.topic = t
                    self.published = []
                def publish(self, msg):
                    self.published.append(msg)
            pub = _Pub(topic)
            self._publishers.append(pub)
            return pub

        def create_timer(self, timer_period_sec, callback):
            timer = {"period": timer_period_sec, "cb": callback, "active": True}
            self._timers.append(timer)
            return timer

        def destroy_node(self):
            pass

        def get_logger(self):
            name = self._node_name
            class _Logger:
                def info(self, msg): print(f"[{name}] [INFO] {msg}")
                def warn(self, msg): print(f"[{name}] [WARN] {msg}")
                def error(self, msg): print(f"[{name}] [ERROR] {msg}")
            return _Logger()

    class LaserScan:
        def __init__(self):
            self.ranges = []
            self.angle_min = -2.35619
            self.angle_max = 2.35619
            self.angle_increment = 0.02618
            self.range_min = 0.10
            self.range_max = 15.0

    class Twist:
        def __init__(self):
            class _Vector3:
                def __init__(self):
                    self.x = 0.0
                    self.y = 0.0
                    self.z = 0.0
            self.linear = _Vector3()
            self.angular = _Vector3()

    class QoSProfile:
        def __init__(self, reliability=None, history=None, depth=10):
            self.reliability = reliability
            self.history = history
            self.depth = depth

    class ReliabilityPolicy:
        BEST_EFFORT = 1
        RELIABLE = 2

    class HistoryPolicy:
        KEEP_LAST = 1

    class String:
        def __init__(self):
            self.data = ""

import collections
import hashlib
import shutil

# Include package modules
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.brain_engine import FlyBrainEngine
from fly_brain_ros.sensory_encoder import SensoryEncoder
from fly_brain_ros.motor_decoder import MotorDecoder
from fly_brain_ros.safety_supervisor import SafetySupervisor
from fly_brain_ros.learning_subsystem import ConnectomeLearner, EpisodicMemory
from fly_brain_ros.internal_state import InternalStateManager
from fly_brain_ros.neuron_metadata import get_neuron_info, query_neuron_connections, BIOLOGICAL_METADATA
from fly_brain_ros.checkpoint_config import (
    get_canonical_checkpoint_path,
    validate_checkpoint_file,
    DEFAULT_CANONICAL_PATH,
)


def resolve_connectome_path(configured_path: str, default_filename: str) -> str:
    """Intelligently resolves connectome data file paths, avoiding rigid machine-specific paths."""
    if configured_path and os.path.exists(configured_path):
        return os.path.abspath(configured_path)

    # Search candidates
    search_dirs = [
        Path(__file__).resolve().parents[2] / 'fly-brain' / 'data',
        Path(__file__).resolve().parents[1] / 'data',
        Path.cwd() / 'fly-brain' / 'data',
        Path('/home/om/Desktop/robo_fly/fly-brain/data'),
    ]

    for candidate_dir in search_dirs:
        p = candidate_dir / default_filename
        if p.exists():
            return str(p)

    return configured_path or default_filename


class FlyBrainControllerPipeline:
    """Thread-safe, decoupled control pipeline connecting sensors to the fly brain."""

    def __init__(self,
                 comp_path=None,
                 conn_path=None,
                 cache_dir=None,
                 dt_ms=0.1,
                 steps_per_cycle=10,
                 device="cpu",
                 safe_dist=1.2,
                 crit_dist=0.35,
                 max_loom_rate=200.0,
                 nominal_fwd_rate=100.0,
                 kv=0.02,
                 kw=0.03,
                 krev=0.02,
                 max_v=0.4,
                 max_w=1.0,
                 wheel_separation=0.36,
                 wheel_radius=0.085,
                 cmd_timeout_sec=0.5,
                 obstacle_hard_stop_dist=0.25):

        self.comp_path = resolve_connectome_path(comp_path, '2025_Completeness_783.csv')
        self.conn_path = resolve_connectome_path(conn_path, '2025_Connectivity_783.parquet')
        self.cache_dir = cache_dir or os.path.dirname(self.conn_path)

        # Validate that connectome completeness and connectivity files exist
        if not os.path.exists(self.comp_path):
            raise FileNotFoundError(f"Connectome completeness file not found: {self.comp_path}")
        if not os.path.exists(self.conn_path) and not os.path.exists(os.path.join(self.cache_dir, 'weight_csr.pkl')):
            raise FileNotFoundError(f"Connectome connectivity file not found: {self.conn_path}")

        self.steps_per_cycle = steps_per_cycle

        self.engine = FlyBrainEngine(
            comp_path=self.comp_path,
            conn_path=self.conn_path,
            cache_dir=self.cache_dir,
            dt_ms=dt_ms,
            device=device
        )

        self.encoder = SensoryEncoder(
            safe_dist=safe_dist,
            crit_dist=crit_dist,
            max_loom_rate=max_loom_rate,
            nominal_fwd_rate=nominal_fwd_rate
        )

        self.decoder = MotorDecoder(
            kv=kv,
            kw=kw,
            krev=krev,
            max_v=max_v,
            max_w=max_w,
            wheel_separation=wheel_separation,
            wheel_radius=wheel_radius
        )

        self.safety = SafetySupervisor(
            max_linear_vel=max_v,
            max_angular_vel=max_w,
            timeout_sec=cmd_timeout_sec,
            obstacle_hard_stop_dist=obstacle_hard_stop_dist
        )

        # Threading & Asynchronous queues
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.worker_thread = None
        self.running = False

        self.last_cmd = {'linear_x': 0.0, 'angular_z': 0.0, 'v_left_wheels': 0.0, 'v_right_wheels': 0.0}
        self.last_status = "INITIALIZING"
        self.last_sector_distances = (10.0, 10.0, 10.0)

        # Smooth isolated neural motor spikes between worker updates.
        self.command_smoothing_alpha = 0.25
        # Exponentially decaying motor-event trace for command decoding.
        self.motor_trace_tau_sec = 0.4
        self.motor_spike_trace = {
            nid: 0.0 for nid in self.decoder.monitored_ids
        }
        self.last_motor_trace_time = time()
        self.filtered_vx = 0.0
        self.filtered_wz = 0.0

        # Learning & Episodic Memory Subsystem
        self.learner = ConnectomeLearner()
        # Simulated Internal State & Self-Monitoring Manager
        self.internal_state = InternalStateManager(dt=0.10, safe_distance=safe_dist)
        # Bounded Time-Correlated Sensory-to-Motor Timeline
        self.timeline = collections.deque(maxlen=40)
        self.last_modulated_vx = 0.0
        self.last_modulated_wz = 0.0
        self.default_checkpoint_path = get_canonical_checkpoint_path()
        self.is_asleep = False
        self.lifecycle_state = "AWAKE"


    def start(self):
        """Initializes the connectome and starts the persistent brain worker thread."""
        init_time = self.engine.initialize(exc_flywire_ids=self.encoder.encode_distances(1.0, 1.0).keys())
        self.safety.set_ready(True)
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return init_time

    def stop(self):
        """Clean shutdown of brain worker thread."""
        self.running = False
        self.safety.set_ready(False)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

    def is_healthy(self) -> bool:
        """Returns True if the background neural worker thread is active and running."""
        return self.running and (self.worker_thread is not None) and self.worker_thread.is_alive()

    def handle_sensor_ranges(self, d_left, d_right, d_front=None):
        """Accepts sensor updates and pushes encoded rates into the brain queue."""
        self.safety.notify_sensor_heartbeat()
        self.last_sector_distances = (d_left, d_right, d_front if d_front is not None else min(d_left, d_right))
        rates_dict = self.encoder.encode_distances(d_left, d_right, d_front)

        # Non-blocking replace of latest sensory rate
        try:
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put_nowait((
                rates_dict,
                float(d_left),
                float(d_right),
                float(d_front if d_front is not None else min(d_left, d_right)),
            ))
        except queue.Full:
            pass

    def get_latest_command(self):
        """Retrieves latest safe motor command."""
        try:
            while not self.output_queue.empty():
                self.last_cmd, self.last_status = self.output_queue.get_nowait()
        except queue.Empty:
            pass

        return self.last_cmd, self.last_status

    def get_telemetry_snapshot(self):
        """Compiles real-time telemetry from the 138k connectome pipeline."""
        cmd, status = self.get_latest_command()
        active_subnetwork = self.engine.get_active_subnetwork_summary(max_active=20) if self.engine else {}
        consolidation_report = self.learner.episodic_memory.consolidate()
        return {
            'status': status,
            'linear_x': cmd.get('linear_x', 0.0),
            'angular_z': cmd.get('angular_z', 0.0),
            'v_left_wheels': cmd.get('v_left_wheels', 0.0),
            'v_right_wheels': cmd.get('v_right_wheels', 0.0),
            'raw_linear_x': getattr(self, 'last_raw_vx', 0.0),
            'raw_angular_z': getattr(self, 'last_raw_wz', 0.0),
            'modulated_linear_x': getattr(self, 'last_modulated_vx', 0.0),
            'modulated_angular_z': getattr(self, 'last_modulated_wz', 0.0),
            'filtered_linear_x': self.filtered_vx,
            'filtered_angular_z': self.filtered_wz,
            'avoidance_direction': getattr(self.safety, 'avoidance_direction', 0),
            'clear_consecutive_cycles': getattr(self.safety, 'clear_consecutive_cycles', 0),
            'sector_distances': {
                'left': round(self.last_sector_distances[0], 2),
                'right': round(self.last_sector_distances[1], 2),
                'front': round(self.last_sector_distances[2], 2),
                'min': round(min(self.last_sector_distances), 2),
            },
            'motor_rates': cmd.get('motor_rates', {}),
            'network_activity': getattr(self.engine, 'last_step_stats', {}),
            'step_duration_sec': cmd.get('step_duration_sec', 0.0),
            'simulated_step_sec': cmd.get('simulated_step_sec', 0.0),
            'is_healthy': self.is_healthy(),
            'is_asleep': getattr(self, 'is_asleep', False),
            'lifecycle_state': getattr(self, 'lifecycle_state', 'AWAKE'),
            'total_neurons': self.engine.num_neurons if self.engine else 138639,
            'internal_state': self.internal_state.get_state_dict(),
            'learning': self.learner.to_dict(),
            'episodic_memory_recent': self.learner.episodic_memory.episodes[-10:],
            'consolidation': consolidation_report,
            'subnetwork': active_subnetwork,
            'timeline': list(self.timeline)[-20:],
        }

    def create_checkpoint(self, filepath: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Atomically saves versioned checkpoint of neural, learning, memory, and internal state.

        Guarantees safe atomic replacement: writes to temporary file, syncs to disk (fsync),
        verifies cryptographic SHA-256 and schema completeness on the written file, backs up
        previous checkpoint, and only replaces atomically upon 100% verification.
        """
        save_path = get_canonical_checkpoint_path(filepath)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        tmp_path = save_path + f".tmp.{os.getpid()}_{int(time() * 1000)}"
        bak_path = save_path + ".bak"

        payload = {
            "manifest": {
                "schema_version": "2.0.0",
                "created_at": time(),
                "session_id": session_id or f"session_{int(time())}",
                "brain_model": "Drosophila_138k_LIF_v783",
                "total_neurons": self.engine.num_neurons if self.engine else 138639,
            },
            "neural_state": self.engine.save_dynamic_state() if self.engine else {},
            "learning_state": self.learner.to_dict(),
            "internal_affect_state": self.internal_state.to_dict(),
            "episodic_memory": self.learner.episodic_memory.to_dict(),
            "lifecycle_state": {
                "status": self.last_status,
                "is_asleep": getattr(self, 'is_asleep', False),
                "lifecycle_state": getattr(self, 'lifecycle_state', 'AWAKE'),
                "timestamp": time(),
                "last_sector_distances": list(self.last_sector_distances),
            }
        }

        # Canonical JSON roundtrip before computing SHA-256 integrity hash
        canonical_obj = json.loads(json.dumps(payload))
        raw_bytes = json.dumps(canonical_obj, sort_keys=True).encode("utf-8")
        canonical_obj["manifest"]["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        payload = canonical_obj

        # Write to temporary file with flush and fsync
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Pre-replacement integrity validation
        is_valid, err_msg, summary = validate_checkpoint_file(tmp_path)
        if not is_valid:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Checkpoint atomic write verification failed: {err_msg}")

        # Preserve previous valid checkpoint as backup
        if os.path.exists(save_path):
            try:
                shutil.copy2(save_path, bak_path)
            except Exception:
                pass

        # Atomic replacement of target
        os.replace(tmp_path, save_path)
        return payload["manifest"]

    def restore_checkpoint(self, filepath: Optional[str] = None, is_wake: bool = False) -> bool:
        """Validates and restores checkpoint, falling back to backup if corrupted.
        
        Args:
            filepath: Path to JSON checkpoint.
            is_wake: If True, executes sleep restorative reset on computational fatigue
                     and arousal while strictly preserving learned policy parameters,
                     episodic memory records, and neural state.
        """
        load_path = get_canonical_checkpoint_path(filepath)
        candidates = [load_path, load_path + ".bak"]

        for target in candidates:
            if not os.path.exists(target):
                continue

            is_valid, err_msg, summary = validate_checkpoint_file(target)
            if not is_valid:
                continue

            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if self.engine and "neural_state" in data:
                    self.engine.restore_dynamic_state(data["neural_state"])

                if "learning_state" in data:
                    self.learner.load_dict(data["learning_state"])

                if "internal_affect_state" in data:
                    self.internal_state.load_dict(data["internal_affect_state"])
                    if is_wake:
                        # Biological sleep restoration: reset fatigue and normalize arousal
                        self.internal_state.sleep_reset()

                if "episodic_memory" in data:
                    self.learner.episodic_memory.load_dict(data["episodic_memory"])

                # Post-restoration safety: discard stale transient commands
                self.filtered_vx = 0.0
                self.filtered_wz = 0.0
                self.last_modulated_vx = 0.0
                self.last_modulated_wz = 0.0
                self.last_raw_vx = 0.0
                self.last_raw_wz = 0.0
                self.last_cmd = {'linear_x': 0.0, 'angular_z': 0.0, 'v_left_wheels': 0.0, 'v_right_wheels': 0.0}
                self.last_status = "NOMINAL"
                # Flush queues
                while not self.input_queue.empty():
                    try: self.input_queue.get_nowait()
                    except Exception: break
                while not self.output_queue.empty():
                    try: self.output_queue.get_nowait()
                    except Exception: break

                return True
            except Exception:
                continue

        return False


    def _worker_loop(self):
        """Asynchronous execution loop advancing brain simulation without blocking ROS."""
        current_rates = {}
        current_left_dist = 10.0
        current_right_dist = 10.0
        current_front_dist = 10.0
        current_min_dist = 10.0

        while self.running:
            # 1. Fetch newest sensory input rates if available
            try:
                (
                    current_rates,
                    current_left_dist,
                    current_right_dist,
                    current_front_dist,
                ) = self.input_queue.get(timeout=0.05)

                current_min_dist = min(
                    current_left_dist,
                    current_right_dist,
                    current_front_dist,
                )
                self.engine.set_rates_by_id(current_rates)
            except queue.Empty:
                pass

            if not self.running:
                break

            # Sleep Mode Guard: While asleep, suspend motor output generation and hold zero velocity
            if getattr(self, "is_asleep", False):
                self.last_raw_vx = 0.0
                self.last_raw_wz = 0.0
                self.last_modulated_vx = 0.0
                self.last_modulated_wz = 0.0
                self.filtered_vx = 0.0
                self.filtered_wz = 0.0
                for nid in self.decoder.monitored_ids:
                    self.motor_spike_trace[nid] = 0.0
                zero_cmd = {
                    'linear_x': 0.0,
                    'angular_z': 0.0,
                    'v_left_wheels': 0.0,
                    'v_right_wheels': 0.0,
                    'motor_rates': {},
                    'step_duration_sec': 0.0,
                    'simulated_step_sec': 0.0,
                }
                while not self.output_queue.empty():
                    try: self.output_queue.get_nowait()
                    except Exception: break
                try:
                    self.output_queue.put_nowait((zero_cmd, "ASLEEP"))
                except Exception:
                    pass
                import time as _t
                _t.sleep(0.05)
                continue

            # 2. Advance the brain in one batch. The engine's step()
            # already accumulates spike counts over every simulation step.
            t_step_start = time()
            spikes = self.engine.step(num_steps=self.steps_per_cycle)
            step_duration = time() - t_step_start
            self.safety.notify_brain_heartbeat()

            # 3. Read accumulated counts for the monitored motor neurons.
            motor_spike_counts = {}
            for nid in self.decoder.monitored_ids:
                idx = self.engine.flyid2i.get(nid)
                if idx is not None:
                    motor_spike_counts[nid] = float(spikes[idx])

            # 4. Decode a real-time trace of motor spike events.
            # This is a control-rate estimate, not a biological firing-rate
            # measurement. Preserve simulated time separately for diagnostics.
            simulated_window_sec = (
                self.steps_per_cycle * self.engine.dt_ms
            ) / 1000.0

            trace_now = time()
            trace_elapsed = max(
                trace_now - self.last_motor_trace_time, 1e-3
            )
            self.last_motor_trace_time = trace_now

            trace_decay = math.exp(
                -trace_elapsed / self.motor_trace_tau_sec
            )

            for nid in self.decoder.monitored_ids:
                self.motor_spike_trace[nid] = (
                    self.motor_spike_trace[nid] * trace_decay
                    + motor_spike_counts.get(nid, 0.0)
                )

            decoded = self.decoder.decode_spikes(
                self.motor_spike_trace,
                window_sec=self.motor_trace_tau_sec,
            )
            self.last_raw_vx = float(decoded['linear_x'])
            self.last_raw_wz = float(decoded['angular_z'])

            # 5. Apply learned policy modulation (clutter speed & directional torque)
            mod_vx, mod_wz = self.learner.modulate_command(
                self.last_raw_vx,
                self.last_raw_wz,
                d_front=current_front_dist,
                d_left=current_left_dist,
                d_right=current_right_dist,
            )
            self.last_modulated_vx = mod_vx
            self.last_modulated_wz = mod_wz

            alpha = self.command_smoothing_alpha
            self.filtered_vx += alpha * (mod_vx - self.filtered_vx)
            self.filtered_wz += alpha * (mod_wz - self.filtered_wz)

            learned_steer_start = self.learner.get_effective_steer_threshold(self.safety.obstacle_steer_start_dist)
            safe_vx, safe_wz, status = self.safety.filter_command(
                self.filtered_vx,
                self.filtered_wz,
                min_obstacle_dist=current_min_dist,
                left_obstacle_dist=current_left_dist,
                right_obstacle_dist=current_right_dist,
                front_obstacle_dist=current_front_dist,
                steer_start_override=learned_steer_start,
            )

            # Update learning subsystem and internal state
            active_dir = getattr(self.safety, 'avoidance_direction', 0)
            learn_out = self.learner.step(
                current_front_dist, current_left_dist, current_right_dist,
                status, active_dir
            )

            tot_spikes = sum(motor_spike_counts.values())
            affect_out = self.internal_state.step(
                current_front_dist, current_left_dist, current_right_dist,
                int(tot_spikes), status, self.learner.episodic_memory
            )

            # Append to time-correlated sensory-brain-motor timeline
            self.timeline.append({
                "t": round(trace_now, 3),
                "d_f": round(current_front_dist, 2),
                "d_l": round(current_left_dist, 2),
                "d_r": round(current_right_dist, 2),
                "raw_v": round(self.last_raw_vx, 3),
                "raw_w": round(self.last_raw_wz, 3),
                "mod_v": round(mod_vx, 3),
                "mod_w": round(mod_wz, 3),
                "safe_v": round(safe_vx, 3),
                "safe_w": round(safe_wz, 3),
                "status": status,
                "dir": active_dir,
                "surprise": affect_out["sensory_surprise"],
                "vigilance": affect_out["environmental_vigilance"],
                "valence": affect_out["valence_affect"],
            })

            # Only clear filtered commands and motor traces on critical hardware faults or emergency stops.
            # During normal safety slowdown and avoidance maneuvers, maintain continuity.
            if status in ("FAULT_STALE_SENSOR", "FAULT_BRAIN_TIMEOUT", "EMERGENCY_STOP", "FAULT_INVALID_VALUES"):
                self.filtered_vx = 0.0
                self.filtered_wz = 0.0
                for nid in self.motor_spike_trace:
                    self.motor_spike_trace[nid] = 0.0
            elif status in ("SAFETY_OVERRIDE_OBSTACLE", "SAFETY_BLOCKED"):
                self.filtered_vx = 0.0
                self.filtered_wz = 0.0

            # TEMPORARY DIAGNOSTICS: remove after identifying the fault.
            if not hasattr(self, "_debug_cycles"):
                self._debug_cycles = 0
            self._debug_cycles += 1

            if self._debug_cycles % 10 == 0:
                print(
                    "[FLY_BRAIN_DEBUG] "
                    f"cycle={self._debug_cycles} "
                    f"step_duration={step_duration:.4f}s "
                    f"simulated_step={simulated_window_sec:.6f}s "
                    f"network_activity={self.engine.last_step_stats} "
                    f"obstacle_distances=(L={current_left_dist:.3f}, R={current_right_dist:.3f}, F={current_front_dist:.3f})m "
                    f"min_obstacle_dist={current_min_dist:.3f}m "
                    f"motor_spikes={motor_spike_counts} "
                    f"motor_rates={decoded['rates']} "
                    f"raw_vx={decoded['linear_x']:.5f} "
                    f"raw_wz={decoded['angular_z']:.5f} "
                    f"filtered_vx={self.filtered_vx:.5f} "
                    f"filtered_wz={self.filtered_wz:.5f} "
                    f"safe_vx={safe_vx:.5f} "
                    f"safe_wz={safe_wz:.5f} "
                    f"safety_status={status}",
                    flush=True,
                )

            # 6. Distribute to 4-wheel left and right pairs
            b = self.decoder.b
            v_l = safe_vx - (safe_wz * b / 2.0)
            v_r = safe_vx + (safe_wz * b / 2.0)

            command_payload = {
                'linear_x': safe_vx,
                'angular_z': safe_wz,
                'v_left_wheels': v_l,
                'v_right_wheels': v_r,
                'motor_rates': decoded['rates'],
                'step_duration_sec': step_duration,
                'simulated_step_sec': simulated_window_sec,
                'min_dist': current_min_dist,
            }

            try:
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put_nowait((command_payload, status))
            except queue.Full:
                pass


class FlyBrainNode(Node):
    """Real ROS 2 node wrapping the Drosophila connectome control pipeline."""

    def __init__(self, node_name="fly_brain_node"):
        super().__init__(node_name)

        # -------------------------------------------------------------
        # 1. Parameter Declarations (Matching fly_brain_params.yaml)
        # -------------------------------------------------------------
        # Model Configuration
        self.declare_parameter("connectome_comp_path", "")
        self.declare_parameter("connectome_conn_path", "")
        self.declare_parameter("cache_dir", "")
        self.declare_parameter("dt_ms", 0.1)
        self.declare_parameter("steps_per_cycle", 10)
        self.declare_parameter("device", "cpu")

        # Sensory Encoding Parameters
        self.declare_parameter("safe_distance", 1.2)
        self.declare_parameter("critical_distance", 0.35)
        self.declare_parameter("max_loom_rate", 200.0)
        self.declare_parameter("nominal_forward_rate", 100.0)

        # Motor Decoding Parameters
        self.declare_parameter("linear_gain_kv", 0.02)
        self.declare_parameter("angular_gain_kw", 0.03)
        self.declare_parameter("reverse_gain_krev", 0.02)
        self.declare_parameter("max_linear_speed", 0.4)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("wheel_separation", 0.36)
        self.declare_parameter("wheel_radius", 0.085)

        # Safety Supervisor Parameters
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("obstacle_hard_stop_dist", 0.25)
        self.declare_parameter("checkpoint_file", DEFAULT_CANONICAL_PATH)
        self.declare_parameter("start_asleep", False)
        self.declare_parameter("startup_grace_period_sec", 0.0)

        # ROS Communication Configuration
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("controller_rate_hz", 10.0)

        # -------------------------------------------------------------
        # 2. Retrieve & Validate Parameters
        # -------------------------------------------------------------
        comp_path = self.get_parameter("connectome_comp_path").value
        conn_path = self.get_parameter("connectome_conn_path").value
        cache_dir = self.get_parameter("cache_dir").value
        dt_ms = float(self.get_parameter("dt_ms").value)
        steps_per_cycle = int(self.get_parameter("steps_per_cycle").value)
        device = str(self.get_parameter("device").value)

        safe_dist = float(self.get_parameter("safe_distance").value)
        crit_dist = float(self.get_parameter("critical_distance").value)
        max_loom_rate = float(self.get_parameter("max_loom_rate").value)
        nominal_fwd_rate = float(self.get_parameter("nominal_forward_rate").value)

        kv = float(self.get_parameter("linear_gain_kv").value)
        kw = float(self.get_parameter("angular_gain_kw").value)
        krev = float(self.get_parameter("reverse_gain_krev").value)
        max_v = float(self.get_parameter("max_linear_speed").value)
        max_w = float(self.get_parameter("max_angular_speed").value)
        b = float(self.get_parameter("wheel_separation").value)
        r = float(self.get_parameter("wheel_radius").value)

        cmd_timeout = float(self.get_parameter("cmd_timeout_sec").value)
        hard_stop_dist = float(self.get_parameter("obstacle_hard_stop_dist").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.rate_hz = max(1.0, float(self.get_parameter("controller_rate_hz").value))
        self.cmd_timeout_sec = cmd_timeout
        self.hard_stop_dist = hard_stop_dist

        # -------------------------------------------------------------
        # 3. Instantiate Persistent Connectome Pipeline
        # -------------------------------------------------------------
        self.get_logger().info(f"Initializing Drosophila connectome pipeline on {device}...")
        self.pipeline = FlyBrainControllerPipeline(
            comp_path=comp_path,
            conn_path=conn_path,
            cache_dir=cache_dir,
            dt_ms=dt_ms,
            steps_per_cycle=steps_per_cycle,
            device=device,
            safe_dist=safe_dist,
            crit_dist=crit_dist,
            max_loom_rate=max_loom_rate,
            nominal_fwd_rate=nominal_fwd_rate,
            kv=kv,
            kw=kw,
            krev=krev,
            max_v=max_v,
            max_w=max_w,
            wheel_separation=b,
            wheel_radius=r,
            cmd_timeout_sec=cmd_timeout,
            obstacle_hard_stop_dist=hard_stop_dist
        )

        init_duration = self.pipeline.start()
        self.get_logger().info(
            f"Connectome initialized in {init_duration:.3f}s: "
            f"{self.pipeline.engine.num_neurons} neurons active."
        )

        # Persistent Checkpoint Restoration & Deliberate Startup Policy
        self.checkpoint_file = get_canonical_checkpoint_path(str(self.get_parameter("checkpoint_file").value))
        self.start_asleep_param = bool(self.get_parameter("start_asleep").value)

        # Deliberate startup policy: default to AWAKE regardless of how the previous session ended,
        # separating durable model state (weights, memories, affect) from operational process state.
        if self.start_asleep_param:
            self.is_asleep = True
            self.lifecycle_state = "ASLEEP"
            self.pipeline.is_asleep = True
            self.pipeline.lifecycle_state = "ASLEEP"
            self.last_motion_state = "ASLEEP"
        else:
            self.is_asleep = False
            self.lifecycle_state = "AWAKE"
            self.pipeline.is_asleep = False
            self.pipeline.lifecycle_state = "AWAKE"
            self.last_motion_state = "WAITING_FOR_SENSORS"

        if self.checkpoint_file and os.path.exists(self.checkpoint_file):
            restored = self.pipeline.restore_checkpoint(self.checkpoint_file, is_wake=False)
            if restored:
                self.get_logger().info(
                    f"Successfully restored persistent brain checkpoint from {self.checkpoint_file}. "
                    f"Startup lifecycle initialized to deliberate operational state: {self.lifecycle_state}."
                )
            else:
                self.get_logger().warn(f"Failed to restore checkpoint from {self.checkpoint_file}")

        self.node_startup_time = time()
        self.startup_grace_period_sec = float(self.get_parameter("startup_grace_period_sec").value)


        # Watchdog Tracking
        self.last_scan_received_time = time()
        self.scan_received_count = 0
        self.published_cmd_count = 0

        # Authoritative Control Multiplexing State
        self.control_mode = "NEURAL"  # "NEURAL" or "TELEOP"
        self.teleop_raw_vx = 0.0
        self.teleop_raw_wz = 0.0
        self.teleop_filtered_vx = 0.0
        self.teleop_filtered_wz = 0.0
        self.last_teleop_time = 0.0
        self.teleop_timeout_sec = 0.5

        # -------------------------------------------------------------
        # 4. ROS Publishers and Subscribers
        # -------------------------------------------------------------
        # Sensor-data QoS for LiDAR (best effort, keep last 10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            sensor_qos
        )

        # Sole authoritative velocity publisher to Gazebo bridge
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        # Teleoperation command and control mode subscriptions
        self.teleop_sub = self.create_subscription(
            Twist,
            '/fly_brain/teleop_cmd',
            self.teleop_cmd_callback,
            10
        )

        self.mode_sub = self.create_subscription(
            String,
            '/fly_brain/control_mode',
            self.control_mode_callback,
            10
        )

        self.telemetry_pub = self.create_publisher(
            String,
            '/fly_brain/telemetry',
            10
        )

        # Lifecycle control subscription (SLEEP, WAKE, CONSOLIDATE, RESET_LEARNING)
        self.lifecycle_sub = self.create_subscription(
            String,
            '/fly_brain/lifecycle_control',
            self.lifecycle_callback,
            10
        )

        # Reinforcement learning feedback subscription (Reward, Punish)
        self.learning_feedback_sub = self.create_subscription(
            String,
            '/fly_brain/learning_feedback',
            self.learning_feedback_callback,
            10
        )

        # Interactive neuron inspector query & response
        self.neuron_query_sub = self.create_subscription(
            String,
            '/fly_brain/query_neuron',
            self.query_neuron_callback,
            10
        )
        self.neuron_details_pub = self.create_publisher(
            String,
            '/fly_brain/neuron_details',
            10
        )

        # Control cycle timer
        timer_period = 1.0 / self.rate_hz
        self.control_timer = self.create_timer(timer_period, self.control_cycle)

        self.get_logger().info(
            f"FlyBrainNode started at {self.rate_hz} Hz. "
            f"Subscribed to {self.scan_topic}, /fly_brain/teleop_cmd, /fly_brain/control_mode. "
            f"Authoritative publisher to {self.cmd_vel_topic} and /fly_brain/telemetry."
        )

    def teleop_cmd_callback(self, msg: Twist):
        """Receives non-authoritative teleoperation inputs for safety supervisor validation."""
        self.last_teleop_time = time()
        self.teleop_raw_vx = float(msg.linear.x)
        self.teleop_raw_wz = float(msg.angular.z)

    def control_mode_callback(self, msg: String):
        """Switches authoritative control mode between NEURAL connectome and TELEOP teacher."""
        mode_str = str(msg.data).strip().upper()
        if mode_str in ("TELEOP", "TEACHER", "MANUAL"):
            if self.control_mode != "TELEOP":
                self.get_logger().info("Control source switched to TELEOP (Dashboard Teacher Override with Safety).")
                self.control_mode = "TELEOP"
                # Match filtered velocity to avoid any motion discontinuity
                self.teleop_filtered_vx = self.pipeline.filtered_vx
                self.teleop_filtered_wz = self.pipeline.filtered_wz
                self.teleop_raw_vx = 0.0
                self.teleop_raw_wz = 0.0
                self.last_teleop_time = time()
        elif mode_str in ("NEURAL", "AUTO", "AUTONOMOUS"):
            if self.control_mode != "NEURAL":
                self.get_logger().info("Control source switched to NEURAL (Autonomous Fly Brain Connectome).")
                self.control_mode = "NEURAL"
                self.teleop_raw_vx = 0.0
                self.teleop_raw_wz = 0.0
                self.teleop_filtered_vx = 0.0
                self.teleop_filtered_wz = 0.0

    def lifecycle_callback(self, msg: String):
        """Processes lifecycle transitions (SLEEP, WAKE, CONSOLIDATE, RESET_LEARNING)."""
        cmd = msg.data.strip().upper()
        now = time()
        if cmd == "SLEEP":
            # Protect deliberate AWAKE startup from stale / late-joining discovery messages
            if 0.0 <= (now - self.node_startup_time) < self.startup_grace_period_sec and not self.start_asleep_param:
                self.get_logger().warn(
                    f"Ignored stale/external SLEEP command received during startup window ({now - self.node_startup_time:.3f}s < {self.startup_grace_period_sec}s). "
                    f"Robot remains in deliberate startup state: {self.lifecycle_state}."
                )
                return

            self.get_logger().info("Received SLEEP request: gating motor commands, consolidating memory, and saving atomic checkpoint...")
            self.lifecycle_state = "ENTERING_SLEEP"
            self.is_asleep = True
            self.pipeline.is_asleep = True
            self.pipeline.lifecycle_state = "ASLEEP"

            # Immediately clamp output velocity to zero
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_twist)

            # Consolidate memory during sleep
            try:
                self.pipeline.learner.episodic_memory.consolidate()
            except Exception:
                pass

            target = get_canonical_checkpoint_path(self.checkpoint_file)
            manifest = self.pipeline.create_checkpoint(target)
            self.lifecycle_state = "ASLEEP"
            self.get_logger().info(f"Sleep checkpoint committed: {manifest.get('sha256', '')[:12]} at {target}. Robot is ASLEEP.")

        elif cmd == "WAKE":
            self.lifecycle_state = "WAKING"
            target = get_canonical_checkpoint_path(self.checkpoint_file)
            self.get_logger().info(f"Received WAKE request: restoring from {target}...")
            self.pipeline.restore_checkpoint(target, is_wake=True)
            self.teleop_raw_vx = 0.0
            self.teleop_raw_wz = 0.0
            self.last_teleop_time = 0.0
            self.teleop_filtered_vx = 0.0
            self.teleop_filtered_wz = 0.0
            # Force receipt of fresh sensor scan after wake before motion
            self.last_scan_received_time = 0.0
            self.pipeline.safety.notify_sensor_heartbeat()
            self.pipeline.safety.notify_brain_heartbeat()
            self.pipeline.is_asleep = False
            self.pipeline.lifecycle_state = "AWAKE"
            self.is_asleep = False
            self.lifecycle_state = "AWAKE"
            self.get_logger().info("Robot is AWAKE. Autonomous brain and safety supervisor control restored.")

        elif cmd == "CONSOLIDATE":
            report = self.pipeline.learner.episodic_memory.consolidate()
            self.get_logger().info(f"Memory consolidated: {report.get('summary')}")

        elif cmd == "RESET_LEARNING":
            self.pipeline.learner.reset_learning()
            self.get_logger().info("Learned navigation parameters reset to baseline.")

    def learning_feedback_callback(self, msg: String):
        """Processes real-time reinforcement feedback (reward / punish) from web dashboard."""
        try:
            data = json.loads(msg.data)
            reward = float(data.get("reward", 0.0))
            note = str(data.get("note", ""))
            res = self.pipeline.learner.apply_feedback(reward, note)
            self.pipeline.internal_state.process_feedback(reward)
            self.get_logger().info(
                f"Reinforcement feedback applied: reward={reward:+.2f}, "
                f"cumulative={res.get('cumulative_reward')}, updates={res.get('updates')}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to process learning feedback: {e}")

    def query_neuron_callback(self, msg: String):
        """Returns detailed physiological inspection for the requested neuron."""
        query_id = msg.data.strip()
        details = self.pipeline.engine.inspect_neuron(query_id)
        resp = String()
        resp.data = json.dumps(details)
        self.neuron_details_pub.publish(resp)

    def scan_callback(self, msg: LaserScan):
        """Processes real LaserScan measurements into sector distances for the brain."""
        self.last_scan_received_time = time()
        self.scan_received_count += 1

        ranges = msg.ranges
        if not ranges:
            self.pipeline.handle_sensor_ranges(0.0, 0.0, 0.0)
            return

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        r_min = getattr(msg, 'range_min', 0.10)
        r_max = getattr(msg, 'range_max', 15.0)

        left_ranges = []
        right_ranges = []
        front_ranges = []
        valid_count = 0

        for i, r in enumerate(ranges):
            # Guard against NaN, infinite, or negative values
            if math.isnan(r) or math.isinf(r) or r < 0.0:
                continue

            # Readings inside dead-zone indicate critical contact threat
            eff_r = max(0.01, r) if r < r_min else min(float(r), r_max)
            valid_count += 1

            angle = angle_min + i * angle_inc

            # Front sector: [-15 deg, +15 deg] = [-0.2618 rad, +0.2618 rad]
            if -0.2618 <= angle <= 0.2618:
                front_ranges.append(eff_r)

            # Left sector: [+15 deg, +90 deg] = [0.2618 rad, +1.5708 rad]
            if 0.2618 < angle <= 1.5708:
                left_ranges.append(eff_r)

            # Right sector: [-90 deg, -15 deg] = [-1.5708 rad, -0.2618 rad]
            if -1.5708 <= angle < -0.2618:
                right_ranges.append(eff_r)

        if valid_count == 0:
            # Completely invalid or blinded sensor: fail-safe zero distance
            self.pipeline.handle_sensor_ranges(0.0, 0.0, 0.0)
            return

        d_l = min(left_ranges) if left_ranges else r_max
        d_r = min(right_ranges) if right_ranges else r_max
        d_f = min(front_ranges) if front_ranges else min(d_l, d_r)

        # Dispatch real sensor sector ranges into the neural pipeline
        self.pipeline.handle_sensor_ranges(d_l, d_r, d_f)

    def control_cycle(self):
        """Timer callback publishing safe Twist commands to /cmd_vel."""
        now = time()
        twist = Twist()
        active_source = "NONE"
        status = "NOMINAL"
        motion_state = "MOVING"

        # Rule 0: Hard Sleep / Inactive Lifecycle Gate
        # When asleep or entering sleep, velocity is strictly clamped to 0.0 at the authoritative gate.
        if self.is_asleep or self.lifecycle_state in ("ASLEEP", "ENTERING_SLEEP"):
            motion_state = "ASLEEP"
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.published_cmd_count += 1

            if motion_state != getattr(self, 'last_motion_state', None):
                self.get_logger().info(f"[MOTION_STATE] {getattr(self, 'last_motion_state', 'INIT')} -> {motion_state} (asleep=True, gate clamped to 0.0)")
                self.last_motion_state = motion_state

            try:
                snapshot = self.pipeline.get_telemetry_snapshot()
                snapshot['control_mode'] = self.control_mode
                snapshot['active_source'] = "SLEEP_GATE"
                snapshot['status'] = "ASLEEP"
                snapshot['motion_state'] = motion_state
                snapshot['lifecycle_state'] = self.lifecycle_state
                snapshot['is_asleep'] = True
                snapshot['safe_vx'] = 0.0
                snapshot['safe_wz'] = 0.0
                snapshot['raw_linear_x'] = 0.0
                snapshot['raw_angular_z'] = 0.0
                snapshot['published_cmd_count'] = self.published_cmd_count
                snapshot['scan_received_count'] = self.scan_received_count
                telemetry_msg = String()
                telemetry_msg.data = json.dumps(snapshot)
                self.telemetry_pub.publish(telemetry_msg)
            except Exception:
                pass
            return

        # Rule 1: Stale Sensor Watchdog Check (Requires fresh sensor scan before motion)
        if self.scan_received_count == 0 or (now - self.last_scan_received_time) > self.cmd_timeout_sec:
            motion_state = "WAITING_FOR_SENSORS"
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.published_cmd_count += 1

            if motion_state != getattr(self, 'last_motion_state', None):
                dt_scan = now - self.last_scan_received_time if self.scan_received_count > 0 else 999.0
                self.get_logger().info(
                    f"[MOTION_STATE] {getattr(self, 'last_motion_state', 'INIT')} -> {motion_state} "
                    f"(scans_received={self.scan_received_count}, dt_scan={dt_scan:.2f}s)"
                )
                self.last_motion_state = motion_state

            try:
                snapshot = self.pipeline.get_telemetry_snapshot()
                snapshot['control_mode'] = self.control_mode
                snapshot['active_source'] = "WATCHDOG"
                snapshot['status'] = "WAITING_FOR_SENSORS"
                snapshot['motion_state'] = motion_state
                snapshot['lifecycle_state'] = self.lifecycle_state
                snapshot['is_asleep'] = False
                snapshot['safe_vx'] = 0.0
                snapshot['safe_wz'] = 0.0
                snapshot['raw_linear_x'] = 0.0
                snapshot['raw_angular_z'] = 0.0
                snapshot['published_cmd_count'] = self.published_cmd_count
                snapshot['scan_received_count'] = self.scan_received_count
                telemetry_msg = String()
                telemetry_msg.data = json.dumps(snapshot)
                self.telemetry_pub.publish(telemetry_msg)
            except Exception:
                pass
            return

        # Rule 2: Neural Worker Health Check
        if not self.pipeline.is_healthy():
            motion_state = "SAFETY_STOP"
            if motion_state != getattr(self, 'last_motion_state', None):
                self.get_logger().error(f"[MOTION_STATE] {getattr(self, 'last_motion_state', 'INIT')} -> {motion_state} (neural worker thread unhealthy)")
                self.last_motion_state = motion_state
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.published_cmd_count += 1
            return

        # Rule 3: Single Authoritative Command Selection Architecture
        d_l, d_r, d_f = self.pipeline.last_sector_distances
        d_min = min(d_l, d_r, d_f)
        hard_stop_threshold = getattr(self, 'hard_stop_dist', getattr(self.pipeline.safety, 'hard_stop_dist', 0.25))

        if self.control_mode == "TELEOP":
            # Check teleoperation watchdog
            if (now - self.last_teleop_time) > self.teleop_timeout_sec:
                target_vx = 0.0
                target_wz = 0.0
            else:
                target_vx = self.teleop_raw_vx
                target_wz = self.teleop_raw_wz

            alpha = 0.35
            self.teleop_filtered_vx += alpha * (target_vx - self.teleop_filtered_vx)
            self.teleop_filtered_wz += alpha * (target_wz - self.teleop_filtered_wz)

            # Pass teleoperation through Authoritative Safety Supervisor
            safe_vx, safe_wz, status = self.pipeline.safety.filter_command(
                self.teleop_filtered_vx,
                self.teleop_filtered_wz,
                min_obstacle_dist=d_min,
                left_obstacle_dist=d_l,
                right_obstacle_dist=d_r,
                front_obstacle_dist=d_f,
            )
            active_source = "TELEOP"
            raw_cmd_vx = target_vx
            raw_cmd_wz = target_wz
        else:
            # Autonomous Connectome Control
            cmd, status = self.pipeline.get_latest_command()
            safe_vx = float(cmd.get('linear_x', 0.0))
            safe_wz = float(cmd.get('angular_z', 0.0))
            active_source = "NEURAL"
            raw_cmd_vx = getattr(self.pipeline, 'last_raw_vx', safe_vx)
            raw_cmd_wz = getattr(self.pipeline, 'last_raw_wz', safe_wz)
            if d_min <= hard_stop_threshold:
                safe_vx, safe_wz, status = 0.0, 0.0, "SAFETY_OVERRIDE_OBSTACLE"

        # Rule 4: Critical Fault & Emergency Stops
        if (
            d_min <= hard_stop_threshold
            or status in ("FAULT_STALE_SENSOR", "FAULT_BRAIN_TIMEOUT", "EMERGENCY_STOP", "FAULT_INVALID_VALUES", "SAFETY_BLOCKED", "SAFETY_OVERRIDE_OBSTACLE")
            or (isinstance(status, str) and (status.startswith("FAULT") or status.startswith("SAFETY_BLOCKED") or status.startswith("SAFETY_OVERRIDE")))
        ):
            motion_state = "SAFETY_STOP"
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif abs(safe_vx) < 1e-4 and abs(safe_wz) < 1e-4:
            motion_state = "AWAKE_NO_MOTOR_OUTPUT"
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            motion_state = "MOVING"
            twist.linear.x = float(safe_vx)
            twist.angular.z = float(safe_wz)

        # Sole authoritative velocity command published to /cmd_vel
        self.cmd_vel_pub.publish(twist)
        self.published_cmd_count += 1

        if motion_state != getattr(self, 'last_motion_state', None):
            self.get_logger().info(
                f"[MOTION_STATE] {getattr(self, 'last_motion_state', 'INIT')} -> {motion_state} "
                f"(vx={twist.linear.x:.3f}, wz={twist.angular.z:.3f}, source={active_source}, status={status})"
            )
            self.last_motion_state = motion_state

        # Rule 5: Publish Comprehensive Telemetry
        try:
            snapshot = self.pipeline.get_telemetry_snapshot()
            snapshot['control_mode'] = self.control_mode
            snapshot['active_source'] = active_source
            snapshot['status'] = status
            snapshot['motion_state'] = motion_state
            snapshot['safe_vx'] = float(safe_vx)
            snapshot['safe_wz'] = float(safe_wz)
            snapshot['raw_linear_x'] = float(raw_cmd_vx)
            snapshot['raw_angular_z'] = float(raw_cmd_wz)
            snapshot['published_cmd_count'] = self.published_cmd_count
            snapshot['scan_received_count'] = self.scan_received_count
            snapshot['teleop_watchdog_active'] = (now - self.last_teleop_time <= self.teleop_timeout_sec)
            telemetry_msg = String()
            telemetry_msg.data = json.dumps(snapshot)
            self.telemetry_pub.publish(telemetry_msg)
        except Exception:
            pass

    def destroy_node(self):
        """Clean shutdown publishing safe zero velocity and stopping worker thread."""
        try:
            self.get_logger().info("Shutting down FlyBrainNode. Publishing zero-velocity stop...")
        except Exception:
            pass
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        try:
            self.cmd_vel_pub.publish(stop_twist)
        except Exception:
            pass
        self.pipeline.stop()
        try:
            super().destroy_node()
        except Exception:
            pass


def main(args=None):
    """Main ROS 2 node entry point."""
    if HAVE_RCLPY:
        rclpy.init(args=args)
        node = FlyBrainNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass
        except Exception:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
    else:
        print("[FlyBrainNode] Warning: rclpy is not installed in the current environment.")
        print("[FlyBrainNode] To run standalone demonstration, use: ros2 run fly_brain_ros standalone_demo")
        node = FlyBrainNode()
        try:
            import time as tm
            tm.sleep(1.0)
        finally:
            node.destroy_node()


if __name__ == '__main__':
    main()
