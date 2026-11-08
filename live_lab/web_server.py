"""Companion Web Server & Live Neuro-Robotics Laboratory Dashboard.

Provides an advanced browser-based interface accessible at http://127.0.0.1:5000:
  - Window 1: Live Robot Arena (LiDAR rays, dynamic Gazebo obstacles, trajectory trail, kinematics)
  - Window 2: Scientific 3D Connectome Viewer & Neuron Inspector (138,639-neuron WebGL point cloud,
              anatomical neuropils, real-time spike highlights, exact 64-bit FlyWire inspection)
  - Window 3: Synchronized Sensory -> Brain -> Motor Timeline (rolling real-time strip chart)
  - Window 4: Adaptive Learning & Reinforcement Console (manual Reward/Punish, steer adaptation,
              directional bias, clutter speed factor, episodic memory history)
  - Window 5: Simulated Internal State, Affect & Self-Monitoring (Surprise, Vigilance, Confidence,
              Uncertainty, Fatigue, Valence index)
  - Window 6: Sleep / Wake Lifecycle Console (safe zero-motion halt, atomic checkpointing,
              cryptographic manifest verification)
"""

import argparse
import collections
import json
import math
import os
import signal
import sys
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

# Configure system paths for repository and package
current_file = Path(__file__).resolve()
if current_file.parents[1].name == "fly_brain_ros":
    workspace_root = current_file.parents[2]
    pkg_root = current_file.parents[1]
else:
    workspace_root = current_file.parents[1]
    pkg_root = workspace_root / "fly_brain_ros"

sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(pkg_root))
sys.path.insert(0, str(pkg_root / "fly_brain_ros"))

from flask import Flask, Response, jsonify, request, render_template_string
from virtual_world.arena import FlyArena
from virtual_world.closed_loop_agent import EmbodiedFlyAgent
from live_lab.interactive_experiments import InteractiveExperimentRunner

# Import ROS 2 & Lifecycle Supervisor
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False

try:
    from fly_brain_ros.lifecycle_supervisor import SimulationLifecycleSupervisor
    from fly_brain_ros.neuron_metadata import (
        get_neuron_info,
        BIOLOGICAL_METADATA,
        resolve_neuron_name_or_alias,
    )
    from fly_brain_ros.checkpoint_config import (
        get_canonical_checkpoint_path,
        validate_checkpoint_file,
        DEFAULT_CANONICAL_PATH,
    )
    from fly_brain_ros.connectome_spatial_layout import (
        generate_spatial_atlas,
        NEUROPILS,
    )
    from fly_brain_ros.learning_subsystem import ConnectomeLearner, EpisodicMemory
    from fly_brain_ros.internal_state import InternalStateManager
except ImportError:
    try:
        from lifecycle_supervisor import SimulationLifecycleSupervisor
        from neuron_metadata import (
            get_neuron_info,
            BIOLOGICAL_METADATA,
            resolve_neuron_name_or_alias,
        )
        from checkpoint_config import (
            get_canonical_checkpoint_path,
            validate_checkpoint_file,
            DEFAULT_CANONICAL_PATH,
        )
        from connectome_spatial_layout import generate_spatial_atlas, NEUROPILS
        from learning_subsystem import ConnectomeLearner, EpisodicMemory
        from internal_state import InternalStateManager
    except ImportError:
        SimulationLifecycleSupervisor = None
        def get_neuron_info(fid, idx=None):
            return {
                "flywire_id": str(fid),
                "name": f"Neuron_{fid}",
                "functional_group": "Connectome Interneuron",
                "role": "LIF integration",
            }
        def resolve_neuron_name_or_alias(name):
            return None
        BIOLOGICAL_METADATA = {}
        DEFAULT_CANONICAL_PATH = os.environ.get(
            "FLY_BRAIN_CHECKPOINT_PATH",
            str(Path(__file__).resolve().parents[1] / "checkpoints" / "fly_brain_checkpoint.json")
        )
        def get_canonical_checkpoint_path(p=None):
            return p or DEFAULT_CANONICAL_PATH
        def validate_checkpoint_file(p):
            return os.path.exists(p), "OK" if os.path.exists(p) else "Not found", {}
        def generate_spatial_atlas():
            return np.zeros((138639, 4), dtype=np.float32), []
        NEUROPILS = {}
        ConnectomeLearner = None
        InternalStateManager = None

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app = Flask(__name__, static_folder=static_dir, static_url_path="/static")

# Cached 3D spatial geometry
cached_spatial_atlas: Optional[np.ndarray] = None
cached_key_neurons: Optional[List[Dict[str, Any]]] = None

# ---------------------------------------------------------------------------
# ROS 2 Live Simulation Bridge Node
# ---------------------------------------------------------------------------
if HAVE_ROS2:
    class LiveROSDashboardBridge(Node):
        """ROS 2 Node bridging live Gazebo Fortress and Fly Brain Node to web dashboard."""

        def __init__(self):
            super().__init__('live_lab_dashboard_bridge')
            self.lock = threading.Lock()
            self.last_msg_time = 0.0

            # Kinematic and ground truth poses
            self.robot_x = 0.0
            self.robot_y = -6.0
            self.robot_theta = 1.5708
            self.robot_v = 0.0
            self.robot_omega = 0.0
            self.v_left_wheels = 0.0
            self.v_right_wheels = 0.0
            self.trail: List[List[float]] = []

            # Dynamic LiDAR rays and obstacle points
            self.rays: List[Dict[str, Any]] = []
            self.dynamic_obstacle_points: List[List[float]] = []
            self.sector_distances = {'left': 10.0, 'right': 10.0, 'front': 10.0, 'min': 10.0}
            self.last_scan_time = 0.0
            self.scan_metadata: Dict[str, Any] = {}

            # Enriched Telemetry
            self.telemetry: Dict[str, Any] = {
                'status': 'INITIALIZING',
                'raw_linear_x': 0.0,
                'raw_angular_z': 0.0,
                'modulated_linear_x': 0.0,
                'modulated_angular_z': 0.0,
                'filtered_linear_x': 0.0,
                'filtered_angular_z': 0.0,
                'linear_x': 0.0,
                'angular_z': 0.0,
                'avoidance_direction': 0,
                'clear_consecutive_cycles': 0,
                'motor_rates': {},
                'network_activity': {},
                'total_neurons': 138639,
                'is_healthy': True,
                'is_asleep': False,
                'lifecycle_state': 'AWAKE',
                'step_duration_sec': 0.0,
                'simulated_step_sec': 0.0,
                'published_cmd_count': 0,
                'scan_received_count': 0,
                'internal_state': {},
                'learning': {},
                'episodic_memory_recent': [],
                'consolidation': {},
                'subnetwork': {},
                'timeline': [],
            }

            # Neuron Query Cache
            self.neuron_cache: Dict[str, Any] = {}

            # QoS Configuration
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

            # Subscriptions
            self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)
            self.create_subscription(Odometry, '/odom', self._on_odom, 10)
            self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
            self.create_subscription(String, '/fly_brain/telemetry', self._on_telemetry, 10)
            self.create_subscription(String, '/fly_brain/neuron_details', self._on_neuron_details, 10)

            # Publishers
            self.teleop_cmd_pub = self.create_publisher(Twist, '/fly_brain/teleop_cmd', 10)
            self.control_mode_pub = self.create_publisher(String, '/fly_brain/control_mode', 10)
            self.lifecycle_pub = self.create_publisher(String, '/fly_brain/lifecycle_control', 10)
            self.query_neuron_pub = self.create_publisher(String, '/fly_brain/query_neuron', 10)
            self.learning_feedback_pub = self.create_publisher(String, '/fly_brain/learning_feedback', 10)

        def _on_odom(self, msg: Odometry):
            with self.lock:
                self.last_msg_time = time.time()
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                self.robot_x = float(p.x)
                self.robot_y = float(p.y)

                siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
                cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
                self.robot_theta = math.atan2(siny_cosp, cosy_cosp)
                self.robot_v = float(msg.twist.twist.linear.x)
                self.robot_omega = float(msg.twist.twist.angular.z)

                if not self.trail or (math.hypot(p.x - self.trail[-1][0], p.y - self.trail[-1][1]) > 0.05):
                    self.trail.append([round(p.x, 3), round(p.y, 3)])
                    if len(self.trail) > 300:
                        self.trail.pop(0)

        def _on_scan(self, msg: LaserScan):
            with self.lock:
                now = time.time()
                self.last_msg_time = now
                self.last_scan_time = now
                ranges = msg.ranges
                if not ranges:
                    return

                rays: List[Dict[str, Any]] = []
                dynamic_points: List[List[float]] = []

                angle_min = float(msg.angle_min)
                angle_inc = float(msg.angle_increment)
                r_max = float(getattr(msg, 'range_max', 15.0))
                r_min = float(getattr(msg, 'range_min', 0.10))

                # LiDAR sensor offset on 4-wheel robot: +0.10m forward of base_link along X (per model.sdf)
                sensor_offset_x = 0.10
                sensor_x = self.robot_x + sensor_offset_x * math.cos(self.robot_theta)
                sensor_y = self.robot_y + sensor_offset_x * math.sin(self.robot_theta)

                # Process all scan measurements
                subsample = max(1, len(ranges) // 180)
                for i in range(0, len(ranges), subsample):
                    r = ranges[i]
                    if math.isnan(r) or math.isinf(r) or r < r_min:
                        continue

                    is_hit = bool(r < r_max * 0.98)
                    hit_dist = min(r, r_max)
                    beam_angle = angle_min + i * angle_inc
                    world_angle = self.robot_theta + beam_angle
                    rx = sensor_x + hit_dist * math.cos(world_angle)
                    ry = sensor_y + hit_dist * math.sin(world_angle)

                    rays.append({
                        "angle": round(world_angle, 3),
                        "beam_angle": round(beam_angle, 3),
                        "dist": round(hit_dist, 2),
                        "rx": round(rx, 2),
                        "ry": round(ry, 2),
                        "sx": round(sensor_x, 2),
                        "sy": round(sensor_y, 2),
                        "hit": is_hit,
                    })

                    if is_hit:
                        dynamic_points.append([round(rx, 2), round(ry, 2)])

                self.rays = rays
                self.dynamic_obstacle_points = dynamic_points
                self.scan_metadata = {
                    "timestamp": now,
                    "total_samples": len(ranges),
                    "valid_hits_count": len(dynamic_points),
                    "range_min": r_min,
                    "range_max": r_max,
                    "angle_min": angle_min,
                    "angle_max": float(msg.angle_max),
                    "angle_increment": angle_inc,
                    "sensor_offset_x": sensor_offset_x,
                }

        def _on_cmd_vel(self, msg: Twist):
            with self.lock:
                self.last_msg_time = time.time()
                vx = float(msg.linear.x)
                wz = float(msg.angular.z)
                b = 0.36
                self.v_left_wheels = vx - (wz * b / 2.0)
                self.v_right_wheels = vx + (wz * b / 2.0)

        def _on_telemetry(self, msg: String):
            with self.lock:
                self.last_msg_time = time.time()
                try:
                    data = json.loads(msg.data)
                    self.telemetry = data
                    if 'sector_distances' in data:
                        self.sector_distances = data['sector_distances']

                    # Update supervisor state based on verified node lifecycle state
                    if lifecycle_supervisor is not None:
                        node_lc = data.get("lifecycle_state", "AWAKE")
                        if node_lc == "AWAKE" and lifecycle_supervisor.current_state in (
                            lifecycle_supervisor.STATE_SLEEPING,
                            lifecycle_supervisor.STATE_RESTORING,
                        ):
                            lifecycle_supervisor._record_transition(
                                lifecycle_supervisor.STATE_AWAKE,
                                "Active connectome telemetry confirmed AWAKE from fly_brain_node."
                            )
                        elif node_lc == "ASLEEP" and lifecycle_supervisor.current_state != lifecycle_supervisor.STATE_SLEEPING:
                            lifecycle_supervisor._record_transition(
                                lifecycle_supervisor.STATE_SLEEPING,
                                "Node confirmed ASLEEP state."
                            )
                except Exception:
                    pass

        def _on_neuron_details(self, msg: String):
            with self.lock:
                try:
                    data = json.loads(msg.data)
                    fid = str(data.get("flywire_id", "")).strip()
                    name = str(data.get("name", "")).strip().lower()
                    idx = str(data.get("local_index", "")).strip()
                    query = str(data.get("query", "")).strip()
                    if fid:
                        data["flywire_id"] = fid
                        self.neuron_cache[fid] = data
                    if name:
                        self.neuron_cache[name] = data
                    if idx:
                        self.neuron_cache[idx] = data
                    if query:
                        self.neuron_cache[query] = data
                except Exception:
                    pass

        def publish_teleop_cmd(self, linear_x: float, angular_z: float):
            try:
                msg = Twist()
                msg.linear.x = float(linear_x)
                msg.angular.z = float(angular_z)
                self.teleop_cmd_pub.publish(msg)
            except Exception:
                pass

        def set_control_mode(self, mode_str: str):
            try:
                msg = String()
                msg.data = mode_str
                self.control_mode_pub.publish(msg)
            except Exception:
                pass

        def send_lifecycle_command(self, cmd_str: str):
            try:
                msg = String()
                msg.data = cmd_str
                self.lifecycle_pub.publish(msg)
            except Exception:
                pass

        def query_neuron(self, identifier: str):
            try:
                msg = String()
                msg.data = str(identifier)
                self.query_neuron_pub.publish(msg)
            except Exception:
                pass

        def publish_learning_feedback(self, reward: float, note: str = ""):
            try:
                msg = String()
                msg.data = json.dumps({"reward": float(reward), "note": str(note)})
                self.learning_feedback_pub.publish(msg)
            except Exception:
                pass


# Initialize ROS 2 spin worker thread
ros_bridge: Optional[Any] = None
_ros_thread = None
_stop_ros_spin = threading.Event()

if HAVE_ROS2:
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
        ros_bridge = LiveROSDashboardBridge()

        def _ros_spin_worker():
            while not _stop_ros_spin.is_set():
                try:
                    if rclpy.ok():
                        rclpy.spin_once(ros_bridge, timeout_sec=0.1)
                    else:
                        break
                except Exception:
                    break

        _ros_thread = threading.Thread(target=_ros_spin_worker, daemon=True)
        _ros_thread.start()
    except Exception as e:
        print(f"[web_server] Notice: ROS 2 bridge fallback: {e}")

def _cleanup_ros():
    _stop_ros_spin.set()
    global _ros_thread
    if _ros_thread is not None and _ros_thread.is_alive():
        try:
            _ros_thread.join(timeout=1.0)
        except Exception:
            pass
    if HAVE_ROS2:
        try:
            if ros_bridge is not None:
                ros_bridge.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

import atexit
atexit.register(_cleanup_ros)

# Lifecycle supervisor instance
lifecycle_supervisor: Optional[SimulationLifecycleSupervisor] = None
if SimulationLifecycleSupervisor is not None:
    canonical_cp = get_canonical_checkpoint_path()
    lifecycle_supervisor = SimulationLifecycleSupervisor(
        checkpoint_path=canonical_cp,
        ros_bridge_node=ros_bridge,
        initial_state=SimulationLifecycleSupervisor.STATE_AWAKE,
    )


# Authoritative 20m x 20m Learning Lab Arena Geometry (matching learning_lab_arena.world)
ARENA_BOUNDS = {
    "width": 20.0,
    "height": 20.0,
    "min_x": -10.0,
    "max_x": 10.0,
    "min_y": -10.0,
    "max_y": 10.0,
    "walls": [
        {"name": "north", "x1": -10.0, "y1": 10.0, "x2": 10.0, "y2": 10.0, "thickness": 0.2},
        {"name": "south", "x1": -10.0, "y1": -10.0, "x2": 10.0, "y2": -10.0, "thickness": 0.2},
        {"name": "east",  "x1": 10.0, "y1": -10.0, "x2": 10.0, "y2": 10.0, "thickness": 0.2},
        {"name": "west",  "x1": -10.0, "y1": -10.0, "x2": -10.0, "y2": 10.0, "thickness": 0.2},
    ],
    "fixed_obstacles": [
        {"name": "center_pillar", "type": "cylinder", "x": 0.0, "y": 0.0, "radius": 1.2},
        {"name": "left_pillar", "type": "cylinder", "x": -3.5, "y": -2.0, "radius": 0.8},
        {"name": "right_pillar", "type": "cylinder", "x": 3.5, "y": -2.0, "radius": 0.8},
    ],
    "zones": [
        {"name": "sucrose_reward", "type": "circle", "x": 5.5, "y": 5.5, "radius": 1.8, "color": "rgba(0, 230, 118, 0.22)", "border": "#00e676"},
        {"name": "shock_hazard", "type": "circle", "x": -5.5, "y": 5.5, "radius": 1.8, "color": "rgba(255, 51, 75, 0.22)", "border": "#ff334b"},
    ]
}


# ---------------------------------------------------------------------------
# Fallback Standalone Simulation Engine (When ROS is not running)
# ---------------------------------------------------------------------------
class LiveSimulationBackend:
    def __init__(self, seed: int = 42):
        self.lock = threading.Lock()
        self.arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=seed)
        self.agent = EmbodiedFlyAgent(arena=self.arena, seed=seed)
        self.agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0)

        self.learner = ConnectomeLearner() if ConnectomeLearner else None
        self.internal_state = InternalStateManager() if InternalStateManager else None

        self.is_teaching = False
        self.is_autonomous = True
        self.teacher_v = 0.0
        self.teacher_omega = 0.0
        self.fps = 10.0
        self.step_count = 0
        self.is_asleep = False

    def get_state(self) -> Dict[str, Any]:
        """Compile complete telemetry state for client visualization."""
        global ros_bridge, lifecycle_supervisor
        is_ros = HAVE_ROS2 and (ros_bridge is not None) and (time.time() - ros_bridge.last_msg_time < 3.0)

        # 1. Prefer Live ROS 2 / Gazebo Telemetry if active
        if is_ros:
            with ros_bridge.lock:
                tel = ros_bridge.telemetry
                status = tel.get('status', 'NOMINAL')
                raw_vx = tel.get('raw_linear_x', 0.0)
                raw_wz = tel.get('raw_angular_z', 0.0)
                safe_vx = tel.get('linear_x', ros_bridge.robot_v)
                safe_wz = tel.get('angular_z', ros_bridge.robot_omega)
                is_asleep = bool(tel.get('is_asleep', False) or tel.get('lifecycle_state') == "ASLEEP")

                scan_age = round(time.time() - ros_bridge.last_scan_time, 2) if ros_bridge.last_scan_time > 0 else 999.0
                lidar_fresh = bool(scan_age < 1.5 and ros_bridge.last_scan_time > 0)
                lidar_payload = {
                    "timestamp": ros_bridge.last_scan_time,
                    "fresh": lidar_fresh,
                    "age_sec": scan_age,
                    "total_rays": len(ros_bridge.rays),
                    "valid_hits": len(ros_bridge.dynamic_obstacle_points),
                    "sensor_offset_x": 0.10,
                    "range_min": ros_bridge.scan_metadata.get('range_min', 0.10),
                    "range_max": ros_bridge.scan_metadata.get('range_max', 15.0),
                    "angle_min": ros_bridge.scan_metadata.get('angle_min', -2.356),
                    "angle_max": ros_bridge.scan_metadata.get('angle_max', 2.356),
                }

                return {
                    "is_ros": True,
                    "step": tel.get('published_cmd_count', 0),
                    "fps": 10.0,
                    "mode": "ROS 2 Gazebo Live Connectome",
                    "status": "ASLEEP" if is_asleep else status,
                    "motion_state": tel.get('motion_state', 'ASLEEP' if is_asleep else ('MOVING' if abs(safe_vx) > 0.001 else 'AWAKE_NO_MOTOR_OUTPUT')),
                    "is_asleep": is_asleep,
                    "lifecycle_state": tel.get('lifecycle_state', 'AWAKE'),
                    "x": round(ros_bridge.robot_x, 2),
                    "y": round(ros_bridge.robot_y, 2),
                    "theta": round(ros_bridge.robot_theta, 3),
                    "v": 0.0 if is_asleep else round(ros_bridge.robot_v, 2),
                    "omega": 0.0 if is_asleep else round(ros_bridge.robot_omega, 3),
                    "wheel_left": 0.0 if is_asleep else round(ros_bridge.v_left_wheels, 2),
                    "wheel_right": 0.0 if is_asleep else round(ros_bridge.v_right_wheels, 2),
                    "raw_vx": 0.0 if is_asleep else round(raw_vx, 3),
                    "raw_wz": 0.0 if is_asleep else round(raw_wz, 3),
                    "modulated_vx": 0.0 if is_asleep else round(tel.get('modulated_linear_x', raw_vx), 3),
                    "modulated_wz": 0.0 if is_asleep else round(tel.get('modulated_angular_z', raw_wz), 3),
                    "filtered_vx": 0.0 if is_asleep else round(tel.get('filtered_linear_x', 0.0), 3),
                    "filtered_wz": 0.0 if is_asleep else round(tel.get('filtered_angular_z', 0.0), 3),
                    "safe_vx": 0.0 if is_asleep else round(safe_vx, 3),
                    "safe_wz": 0.0 if is_asleep else round(safe_wz, 3),
                    "avoidance_direction": tel.get('avoidance_direction', 0),
                    "clear_consecutive_cycles": tel.get('clear_consecutive_cycles', 0),
                    "sector_distances": ros_bridge.sector_distances,
                    "motor_rates": tel.get('motor_rates', {}),
                    "network_activity": tel.get('network_activity', {}),
                    "step_duration_sec": tel.get('step_duration_sec', 0.0),
                    "simulated_step_sec": tel.get('simulated_step_sec', 0.0),
                    "total_neurons": tel.get('total_neurons', 138639),
                    "is_healthy": tel.get('is_healthy', True),
                    "rays": ros_bridge.rays,
                    "obstacle_points": ros_bridge.dynamic_obstacle_points,
                    "trail": ros_bridge.trail,
                    "control_mode": tel.get('control_mode', 'NEURAL'),
                    "active_source": tel.get('active_source', 'NEURAL'),
                    "is_teaching": (tel.get('control_mode') == 'TELEOP'),
                    "is_autonomous": (tel.get('control_mode') != 'TELEOP'),
                    "internal_state": tel.get('internal_state') if tel.get('internal_state') else (self.internal_state.get_state_dict() if self.internal_state else {
                        "sensory_surprise": 0.0, "environmental_vigilance": 0.0, "confidence": 1.0,
                        "uncertainty": 0.0, "computational_fatigue": 0.0, "valence_affect": 0.50,
                        "valence": 0.50, "vigilance": 0.0, "energy": 1.0, "arousal": 0.0,
                        "memory_lesson": "Nominal terrain."
                    }),
                    "learning": tel.get('learning') if tel.get('learning') else (self.learner.to_dict() if self.learner else {
                        "speed_factor": 1.0, "steer_dist": 1.5, "delta_steer_dist": 0.0, "weights": {}
                    }),
                    "episodic_memory_recent": tel.get('episodic_memory_recent', []),
                    "consolidation": tel.get('consolidation', {}),
                    "subnetwork": tel.get('subnetwork', {}),
                    "timeline": tel.get('timeline', []),
                    "lifecycle": lifecycle_supervisor.get_status() if lifecycle_supervisor else {},
                    "lidar": lidar_payload,
                    "arena_bounds": ARENA_BOUNDS,
                }

        # 2. Standalone / Sleeping Mode State
        lc_status = lifecycle_supervisor.get_status() if lifecycle_supervisor else {"lifecycle_state": "STANDALONE"}
        is_asleep = (lc_status.get('lifecycle_state') == "SLEEPING" or self.is_asleep)
        learning_dict = self.learner.to_dict() if self.learner else {}
        internal_dict = self.internal_state.get_state_dict() if self.internal_state else {
            "sensory_surprise": 0.0,
            "environmental_vigilance": 0.0,
            "confidence": 1.0,
            "uncertainty": 0.0,
            "computational_fatigue": 0.0,
            "valence_affect": 0.50,
            "valence": 0.50,
            "vigilance": 0.0,
            "energy": 1.0,
            "arousal": 0.0,
            "memory_lesson": "Drosophila in persistent sleep state. Awaiting wake request.",
        }

        return {
            "is_ros": False,
            "step": self.step_count,
            "fps": 10.0,
            "mode": f"Lifecycle: {lc_status.get('lifecycle_state', 'STANDALONE')}",
            "status": "ASLEEP" if is_asleep else "STANDALONE",
            "motion_state": "ASLEEP" if is_asleep else ("MOVING" if abs(getattr(self, 'v', 0.0)) > 0.001 else "AWAKE_NO_MOTOR_OUTPUT"),
            "is_asleep": is_asleep,
            "lifecycle_state": "ASLEEP" if is_asleep else "AWAKE",
            "x": 0.0,
            "y": -6.0,
            "theta": 1.571,
            "v": 0.0,
            "omega": 0.0,
            "wheel_left": 0.0,
            "wheel_right": 0.0,
            "raw_vx": 0.0,
            "raw_wz": 0.0,
            "modulated_vx": 0.0,
            "modulated_wz": 0.0,
            "filtered_vx": 0.0,
            "filtered_wz": 0.0,
            "safe_vx": 0.0,
            "safe_wz": 0.0,
            "avoidance_direction": 0,
            "clear_consecutive_cycles": 0,
            "sector_distances": {'left': 10.0, 'right': 10.0, 'front': 10.0, 'min': 10.0},
            "motor_rates": {},
            "network_activity": {},
            "total_neurons": 138639,
            "is_healthy": True,
            "rays": [],
            "obstacle_points": [],
            "trail": [],
            "control_mode": "SLEEP" if is_asleep else "NEURAL",
            "active_source": "SLEEP_GATE" if is_asleep else "NONE",
            "is_teaching": False,
            "is_autonomous": True,
            "internal_state": internal_dict,
            "learning": learning_dict,
            "episodic_memory_recent": self.learner.episodic_memory.episodes[-10:] if self.learner else [],
            "consolidation": {},
            "subnetwork": {},
            "timeline": [],
            "lifecycle": lc_status,
            "lidar": {
                "timestamp": 0.0,
                "fresh": False,
                "age_sec": 999.0,
                "total_rays": 0,
                "valid_hits": 0,
                "sensor_offset_x": 0.10,
                "range_min": 0.10,
                "range_max": 15.0,
                "angle_min": -2.356,
                "angle_max": 2.356,
            },
            "arena_bounds": ARENA_BOUNDS,
        }


sim = LiveSimulationBackend()

# ---------------------------------------------------------------------------
# Scientific HTML5 Dashboard Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Drosophila Connectome Neuro-Robotics Laboratory</title>
  <style>
    :root {
      --bg-base: #090b10;
      --bg-panel: #11141c;
      --bg-header: #0c0e14;
      --border-panel: #1e2433;
      --border-subtle: #161b26;
      --text-main: #f0f3f8;
      --text-muted: #838ca0;
      --accent-cyan: #00d4e6;
      --accent-green: #00e676;
      --accent-amber: #ffb300;
      --accent-purple: #9c64ff;
      --accent-red: #ff334b;
      --mono-font: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 0;
      background: var(--bg-base);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow-x: hidden;
      font-size: 13px;
    }
    header {
      background: var(--bg-header);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-panel);
      position: sticky; top: 0; z-index: 1000;
    }
    .header-left { display: flex; align-items: center; gap: 14px; }
    h1 {
      margin: 0; font-size: 1.05rem; font-weight: 700;
      color: #fff; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px;
    }
    h1 span.tag {
      font-size: 0.7rem; font-weight: 600; padding: 2px 7px; border-radius: 3px;
      background: rgba(0,212,230,0.12); color: var(--accent-cyan); border: 1px solid rgba(0,212,230,0.3);
    }
    .badge {
      padding: 3px 8px; border-radius: 4px; font-size: 0.72rem;
      font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
      display: inline-flex; align-items: center; gap: 5px;
    }
    .badge-awake { background: rgba(0,230,118,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.4); }
    .badge-sleep { background: rgba(156,100,255,0.15); color: var(--accent-purple); border: 1px solid rgba(156,100,255,0.4); }
    .badge-warn { background: rgba(255,179,0,0.15); color: var(--accent-amber); border: 1px solid rgba(255,179,0,0.4); }
    .badge-danger { background: rgba(255,51,75,0.15); color: var(--accent-red); border: 1px solid rgba(255,51,75,0.4); }

    .header-controls { display: flex; align-items: center; gap: 8px; }
    .btn {
      padding: 5px 12px; border: 1px solid transparent; border-radius: 4px;
      font-size: 0.78rem; font-weight: 600; cursor: pointer;
      transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 5px;
      background: #1b202e; color: #cfd7e6;
    }
    .btn:hover { filter: brightness(1.2); }
    .btn:active { transform: translateY(1px); }
    .btn-sleep { background: #3b1b6e; color: #e1ccff; border-color: #6332aa; }
    .btn-wake { background: #004d66; color: #80e5ff; border-color: #0088b3; }
    .btn-mode { background: #161b26; border-color: #2b3447; color: var(--text-muted); }
    .btn-mode.active { background: #006080; border-color: var(--accent-cyan); color: #fff; }
    .btn-reward { background: rgba(0,230,118,0.15); color: var(--accent-green); border-color: rgba(0,230,118,0.4); }
    .btn-punish { background: rgba(255,51,75,0.15); color: var(--accent-red); border-color: rgba(255,51,75,0.4); }
    .btn-action { background: #1a2233; color: var(--accent-cyan); border-color: #293855; }

    .grid-container {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
      padding: 12px 18px;
    }
    .card {
      background: var(--bg-panel);
      border: 1px solid var(--border-panel);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .card-title {
      margin: 0 0 10px 0;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.4px;
      color: #9cb2cc;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 6px;
      text-transform: uppercase;
    }

    /* 3D Connectome Viewer Window */
    .col-viewer { grid-column: span 7; display: flex; flex-direction: column; }
    #viewer3dContainer {
      width: 100%; height: 380px;
      background: #050608; border-radius: 4px;
      position: relative; overflow: hidden;
      border: 1px solid #141822;
    }
    .viewer-toolbar {
      display: flex; justify-content: space-between; align-items: center;
      gap: 8px; margin-top: 8px; flex-wrap: wrap;
    }
    .layer-toggles { display: flex; gap: 4px; }
    .btn-layer {
      padding: 3px 8px; font-size: 0.7rem; border-radius: 3px;
      background: #151a24; border: 1px solid #232b3c; color: var(--text-muted); cursor: pointer;
    }
    .btn-layer.active {
      background: #1c2638; border-color: var(--accent-cyan); color: var(--accent-cyan); font-weight: 600;
    }
    .viewer-hud {
      position: absolute; bottom: 8px; left: 8px;
      font-size: 0.68rem; color: #6a748a; pointer-events: none;
      background: rgba(8,10,14,0.7); padding: 4px 8px; border-radius: 3px;
    }
    .viewer-legend {
      position: absolute; top: 8px; right: 8px;
      font-size: 0.68rem; color: #8d98b0;
      background: rgba(8,10,14,0.85); padding: 6px 10px; border-radius: 4px;
      border: 1px solid #1e2636; line-height: 1.4;
    }

    /* Neuron Inspector */
    .inspector-panel {
      margin-top: 10px;
      background: #0b0e14;
      border: 1px solid #1a202d;
      border-radius: 5px;
      padding: 10px 12px;
      font-size: 0.78rem;
    }
    .search-row { display: flex; gap: 6px; margin-bottom: 8px; }
    .search-row input {
      flex: 1; background: #121620; border: 1px solid #242c3d;
      color: #fff; padding: 5px 9px; border-radius: 4px; font-size: 0.78rem;
      font-family: var(--mono-font);
    }
    .inspector-grid {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 8px;
    }
    .inspect-metric {
      background: #11151f; padding: 6px 8px; border-radius: 4px; border: 1px solid #181f2d;
    }
    .inspect-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; }
    .inspect-val { font-size: 0.85rem; font-weight: 700; color: #fff; font-family: var(--mono-font); margin-top: 2px; }
    .synapse-tables { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
    .syn-col { background: #0f121a; border-radius: 4px; padding: 6px 8px; max-height: 100px; overflow-y: auto; font-size: 0.72rem; }
    .syn-row { display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-family: var(--mono-font); }

    /* Right Column: Robot Arena, Learning, Affect */
    .col-right { grid-column: span 5; display: flex; flex-direction: column; gap: 12px; }
    
    /* Arena Viewport, Controls & HUD */
    .arena-toolbar {
      display: flex; justify-content: space-between; align-items: center;
      gap: 6px; margin-bottom: 8px; flex-wrap: wrap;
    }
    .toolbar-group { display: flex; gap: 4px; align-items: center; }
    .toolbar-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; margin-right: 2px; }
    .arena-btn {
      padding: 3px 7px; font-size: 0.70rem; border-radius: 3px; font-family: var(--mono-font);
      background: #141924; border: 1px solid #232b3c; color: #a2afc4; cursor: pointer;
      transition: all 0.15s ease;
    }
    .arena-btn:hover { background: #1c2436; color: #fff; border-color: #3b475e; }
    .arena-btn.active {
      background: #1a273b; border-color: var(--accent-cyan); color: var(--accent-cyan); font-weight: 600;
    }
    .arena-viewport-container {
      position: relative; width: 100%; border-radius: 4px; overflow: hidden;
      border: 1px solid #141822; background: #06070a;
    }
    #arenaCanvas {
      width: 100%; height: 300px;
      background: #06070a; display: block; cursor: grab;
    }
    .arena-telemetry-hud {
      position: absolute; top: 8px; right: 8px;
      background: rgba(9, 12, 18, 0.88); backdrop-filter: blur(4px);
      border: 1px solid #1e2638; border-radius: 4px;
      padding: 6px 9px; font-family: var(--mono-font); font-size: 0.68rem;
      color: #9cb0cc; pointer-events: none; line-height: 1.45;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .hud-line { display: flex; justify-content: space-between; gap: 10px; }
    .hud-lbl { color: var(--text-muted); font-size: 0.63rem; text-transform: uppercase; }
    .hud-val { color: #f0f4fc; font-weight: 600; }
    .hud-fresh { color: var(--accent-green); font-weight: 700; }
    .hud-stale { color: var(--accent-red); font-weight: 700; }
    .badge-state { padding: 1px 4px; border-radius: 2px; font-size: 0.63rem; }
    .badge-awake { background: rgba(0, 230, 118, 0.2); color: var(--accent-green); }
    .badge-sleep { background: rgba(156, 100, 255, 0.2); color: var(--accent-purple); }
    .badge-motion { color: var(--accent-cyan); }
    .arena-status-hud {
      position: absolute; bottom: 6px; left: 8px;
      font-size: 0.64rem; font-family: var(--mono-font); color: #62708a;
      background: rgba(8, 11, 16, 0.75); padding: 2px 6px; border-radius: 3px;
      pointer-events: none;
    }
    .arena-hint-hud {
      position: absolute; bottom: 6px; right: 8px;
      font-size: 0.64rem; font-family: var(--mono-font); color: #505c72;
      background: rgba(8, 11, 16, 0.75); padding: 2px 6px; border-radius: 3px;
      pointer-events: none;
    }

    /* Learning & Reinforcement Console */
    .reinforce-controls {
      display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap;
    }
    .magnitude-select { display: flex; gap: 4px; align-items: center; font-size: 0.72rem; color: var(--text-muted); }
    .mag-btn {
      padding: 3px 6px; font-size: 0.7rem; border-radius: 3px;
      background: #141924; border: 1px solid #232a3a; color: #a2afc4; cursor: pointer;
    }
    .mag-btn.active { background: #1f2738; border-color: var(--accent-cyan); color: #fff; }
    .learning-meters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 8px; }
    .meter-card {
      background: #0b0e14; border: 1px solid #1a202d; border-radius: 4px; padding: 6px 8px; text-align: center;
    }
    .meter-title { font-size: 0.67rem; color: var(--text-muted); text-transform: uppercase; }
    .meter-value { font-size: 0.95rem; font-weight: 700; color: var(--accent-green); font-family: var(--mono-font); margin-top: 2px; }
    .episode-history {
      max-height: 110px; overflow-y: auto; background: #080a0e; border: 1px solid #151a24;
      border-radius: 4px; padding: 6px 8px; font-size: 0.72rem;
    }
    .ep-row {
      padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
      display: flex; justify-content: space-between; align-items: flex-start; gap: 6px;
    }

    /* Affect & Internal Diagnostics */
    .affect-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .affect-box {
      background: #0b0e14; border: 1px solid #1a202d; border-radius: 4px; padding: 5px 8px;
    }
    .affect-name { font-size: 0.67rem; color: var(--text-muted); display: flex; justify-content: space-between; }
    .affect-bar-wrap { height: 4px; background: #1c2230; border-radius: 2px; margin-top: 4px; overflow: hidden; }
    .affect-bar-fill { height: 100%; width: 50%; background: var(--accent-cyan); transition: width 0.2s ease; }

    /* Timeline Window (Span 12) */
    .col-timeline { grid-column: span 12; height: 180px; }
    #timelineCanvas {
      width: 100%; height: calc(100% - 28px);
      background: #06070a; border-radius: 4px; display: block;
      border: 1px solid #141822;
    }

    .scientific-notice {
      font-size: 0.67rem; color: #5d677a; line-height: 1.35; margin-top: 6px;
      border-top: 1px solid rgba(255,255,255,0.04); padding-top: 5px;
    }
  </style>
  <script src="/static/three.min.js"></script>
  <script>
    if (typeof THREE === "undefined") {
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
    }
  </script>
</head>
<body>
  <header>
    <div class="header-left">
      <h1>
        Drosophila Connectome Robot
        <span class="tag">138k LIF Model</span>
      </h1>
      <span class="badge badge-awake" id="badgeLifecycle">AWAKE</span>
      <span class="badge badge-awake" id="badgeSafety">NOMINAL</span>
      <span class="badge" id="badgeCheckpoint" style="background:#131826; border:1px solid #2a344d; color:#8ea0bf;">CP: READY</span>
    </div>
    <div class="header-controls">
      <button class="btn btn-sleep" id="btnSleep" onclick="requestSleep()">&#9646;&#9646; Sleep</button>
      <button class="btn btn-wake" id="btnWake" onclick="requestWake()">&#9654; Wake</button>
      <button class="btn btn-mode active" id="btnAuto" onclick="setControlMode('auto')">Connectome Autonomous</button>
      <button class="btn btn-mode" id="btnTeach" onclick="setControlMode('teach')">Teacher Override</button>
    </div>
  </header>

  <div class="grid-container">
    <!-- WINDOW 2: SCIENTIFIC 3D CONNECTOME VIEWER -->
    <div class="card col-viewer">
      <div class="card-title">
        <span>Window 2: Whole-Brain Connectome (3D Spatial Viewer)</span>
        <span id="viewerStats" style="font-size:0.7rem; color:var(--text-muted); font-family:var(--mono-font);">138,639 Neurons &bull; 60 FPS</span>
      </div>
      <div id="viewer3dContainer">
        <div class="viewer-hud">Left-Drag: Orbit &bull; Right-Drag: Pan &bull; Wheel: Zoom &bull; Click: Select</div>
        <div class="viewer-legend">
          <div style="font-weight:700; margin-bottom:3px; color:#fff;">Neuropil Regions:</div>
          <div><span style="color:#1fb8fc;">&#9632;</span> Left Optic Lobe (LC4)</div>
          <div><span style="color:#1ff2e1;">&#9632;</span> Right Optic Lobe</div>
          <div><span style="color:#fcc81f;">&#9632;</span> Central Complex (CX)</div>
          <div><span style="color:#be61f7;">&#9632;</span> Antennal / JO Mechanosensory</div>
          <div><span style="color:#f7663e;">&#9632;</span> Subesophageal Zone (SEZ)</div>
          <div><span style="color:#26f276;">&#9632;</span> Descending Motor Tracts</div>
          <div><span style="color:#ffffff;">&#9679;</span> <span style="color:#fff; font-weight:700;">Live Spike Event</span></div>
        </div>
      </div>
      <div class="viewer-toolbar">
        <div class="layer-toggles">
          <button class="btn-layer active" onclick="setNeuropilFilter('all')">All Neuropils</button>
          <button class="btn-layer" onclick="setNeuropilFilter('optic')">Optic Lobes</button>
          <button class="btn-layer" onclick="setNeuropilFilter('cx')">Central Complex</button>
          <button class="btn-layer" onclick="setNeuropilFilter('sez')">SEZ Taste</button>
          <button class="btn-layer" onclick="setNeuropilFilter('motor')">Descending Motor</button>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="landmarkSelect" class="btn" style="padding:3px 6px; font-size:0.72rem;" onchange="selectLandmark(this.value)">
            <option value="">-- Jump to Biological Neuron --</option>
            <option value="720575940644438551">DNa01_left (Steering Right)</option>
            <option value="720575940627787609">DNa01_right (Steering Left)</option>
            <option value="720575940626730883">P9_oDN1_left (Forward Walking)</option>
            <option value="720575940620300308">P9_oDN1_right (Forward Walking)</option>
            <option value="720575940616026939">MDN_1 (Moonwalker Stop)</option>
            <option value="720575940622838154">Giant_Fiber_1 (Escape Jump)</option>
            <option value="720575940605598892">LC4_left_01 (Visual Looming)</option>
            <option value="720575940616885538">Sugar_GRN_01 (Appetitive Taste)</option>
            <option value="720575940619072513">Bitter_GRN_01 (Aversive Danger)</option>
            <option value="720575940645106376">JO_EV_01 (Antennal Vibration)</option>
          </select>
        </div>
      </div>

      <!-- Neuron Physiological Inspector -->
      <div class="inspector-panel">
        <div class="search-row">
          <input type="text" id="neuronSearchInput" placeholder="Enter exact 64-bit FlyWire ID (e.g. 720575940644438551) or name (DNa01_left)...">
          <button class="btn btn-action" onclick="searchNeuron()">Inspect Neuron</button>
        </div>
        <div class="inspector-grid">
          <div class="inspect-metric">
            <div class="inspect-label">FlyWire ID</div>
            <div class="inspect-val" id="insId" style="color:var(--accent-cyan); font-size:0.75rem;">720575940644438551</div>
          </div>
          <div class="inspect-metric">
            <div class="inspect-label">Biological Name</div>
            <div class="inspect-val" id="insName">DNa01_left</div>
          </div>
          <div class="inspect-metric">
            <div class="inspect-label">Neuropil / Group</div>
            <div class="inspect-val" id="insGroup">Descending Motor</div>
          </div>
          <div class="inspect-metric">
            <div class="inspect-label">Membrane Potential</div>
            <div class="inspect-val" id="insVoltage">-65.0 mV</div>
          </div>
        </div>
        <div style="font-size:0.72rem; color:#8da0ba; margin-bottom:4px;" id="insRole">
          Role: Asymmetric steering / turning (right bias).
        </div>
        <div class="synapse-tables">
          <div class="syn-col">
            <div style="font-weight:700; color:#8fa1be; margin-bottom:3px;">Top Presynaptic Inputs:</div>
            <div id="insPreTable">Querying incoming synapses...</div>
          </div>
          <div class="syn-col">
            <div style="font-weight:700; color:#8fa1be; margin-bottom:3px;">Top Postsynaptic Targets:</div>
            <div id="insPostTable">Querying outgoing targets...</div>
          </div>
        </div>
      </div>
      <div class="scientific-notice">
        <strong>Spatial Architecture Notice:</strong> Topologically partitioned by biological neuropils (Shiu et al. 2024 / FlyWire v783 dataset). Coordinates represent functional anatomical compartments, maintaining exact 64-bit biological identity.
      </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div class="col-right">
      <!-- WINDOW 1: ROBOT & ARENA -->
      <div class="card">
        <div class="card-title">
          <span>Window 1: Simulated Robot &amp; Arena</span>
          <span id="kinematicsReadout" style="font-family:var(--mono-font); font-size:0.7rem; color:var(--accent-green);">v: 0.00 m/s &bull; &omega;: 0.00 rad/s</span>
        </div>
        
        <!-- Professional Compact Toolbar -->
        <div class="arena-toolbar">
          <div class="toolbar-group">
            <button class="arena-btn" id="btnZoomIn" onclick="arenaZoomIn()" title="Zoom In (+ / Wheel Up)">+</button>
            <button class="arena-btn" id="btnZoomOut" onclick="arenaZoomOut()" title="Zoom Out (&minus; / Wheel Down)">&minus;</button>
            <button class="arena-btn" id="btnResetView" onclick="arenaResetView()" title="Reset View to Default (R)">Reset</button>
            <button class="arena-btn" id="btnCenterRobot" onclick="arenaCenterRobot()" title="Center Camera on Robot (C)">Center</button>
            <button class="arena-btn active" id="btnFollowRobot" onclick="arenaToggleFollow()" title="Toggle Camera Follow Robot (F)">Follow: ON</button>
            <button class="arena-btn" id="btnFitArena" onclick="arenaFitArena()" title="Fit 20m x 20m Arena (A)">Fit Arena</button>
          </div>
          <div class="toolbar-group layer-group">
            <span class="toolbar-label">Layers:</span>
            <button class="arena-btn active" id="btnLayerRays" onclick="arenaToggleLayer('rays')" title="Toggle LiDAR scan rays">Rays</button>
            <button class="arena-btn active" id="btnLayerHits" onclick="arenaToggleLayer('hits')" title="Toggle obstacle hit points">Hits</button>
            <button class="arena-btn active" id="btnLayerTrail" onclick="arenaToggleLayer('trail')" title="Toggle trajectory trail">Trail</button>
            <button class="arena-btn active" id="btnLayerGrid" onclick="arenaToggleLayer('grid')" title="Toggle dynamic world grid &amp; coordinates">Grid</button>
            <button class="arena-btn active" id="btnLayerWalls" onclick="arenaToggleLayer('walls')" title="Toggle arena walls and obstacles">Walls</button>
          </div>
        </div>

        <!-- Canvas Container with Interactive HUDs -->
        <div class="arena-viewport-container">
          <canvas id="arenaCanvas"></canvas>
          
          <!-- Top-Right Live Telemetry HUD -->
          <div class="arena-telemetry-hud" id="arenaTelemetryHud">
            <div class="hud-line"><span class="hud-lbl">POSE</span> <span class="hud-val" id="hudPose">X: 0.00m &bull; Y: -6.00m &bull; 90.0&deg;</span></div>
            <div class="hud-line"><span class="hud-lbl">VEL</span> <span class="hud-val" id="hudVel">v: 0.00 m/s &bull; &omega;: 0.00 rad/s</span></div>
            <div class="hud-line"><span class="hud-lbl">STATE</span> <span class="hud-val" id="hudState"><span class="badge-state badge-awake">AWAKE</span> <span class="badge-motion" id="hudMotionState">WAITING_FOR_SENSORS</span></span></div>
            <div class="hud-line"><span class="hud-lbl">LIDAR</span> <span class="hud-val" id="hudLidar"><span id="hudLidarStatus" class="hud-fresh">ACTIVE</span> &bull; <span id="hudLidarHits">0 hits</span></span></div>
            <div class="hud-line"><span class="hud-lbl">SAFETY</span> <span class="hud-val" id="hudSafety">NOMINAL</span></div>
          </div>

          <!-- Bottom-Left Status HUD -->
          <div class="arena-status-hud" id="arenaStatusHud">
            Scale: <span id="hudScale">16.0</span> px/m &bull; <span id="hudMode">Follow: ON</span>
          </div>

          <!-- Bottom-Right Navigation Hint -->
          <div class="arena-hint-hud">
            Drag: Pan &bull; Wheel: Zoom &bull; Dbl-Click: Center
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px; font-size:0.73rem; font-family:var(--mono-font);">
          <span>Front: <strong id="valDistFront">10.0m</strong></span>
          <span>Left: <strong id="valDistLeft">10.0m</strong></span>
          <span>Right: <strong id="valDistRight">10.0m</strong></span>
          <span>Min: <strong id="valDistMin" style="color:var(--accent-amber);">10.0m</strong></span>
        </div>
      </div>

      <!-- WINDOW 4: LEARNING & REINFORCEMENT CONSOLE -->
      <div class="card">
        <div class="card-title">
          <span>Window 4: Learning & Reinforcement Console</span>
          <span id="learningUpdatesBadge" style="font-family:var(--mono-font); font-size:0.7rem; color:var(--accent-green);">Steps: 0</span>
        </div>
        <div class="reinforce-controls">
          <button class="btn btn-reward" onclick="sendReward()">+ Reward</button>
          <button class="btn btn-punish" onclick="sendPunish()">&minus; Punish</button>
          <div class="magnitude-select">
            <span>Mag:</span>
            <button class="mag-btn" onclick="setMag(0.25)">0.25</button>
            <button class="mag-btn" onclick="setMag(0.50)">0.50</button>
            <button class="mag-btn active" onclick="setMag(1.00)">1.00</button>
            <button class="mag-btn" onclick="setMag(1.50)">1.50</button>
          </div>
          <input type="text" id="reinforceNote" placeholder="Note (optional)..." style="flex:1; background:#0c0f16; border:1px solid #202738; color:#fff; font-size:0.72rem; padding:3px 6px; border-radius:3px;">
        </div>
        <div class="learning-meters">
          <div class="meter-card">
            <div class="meter-title">Steer Margin (&Delta;d)</div>
            <div class="meter-value" id="meterDeltaSteer">+0.00m</div>
          </div>
          <div class="meter-card">
            <div class="meter-title">Directional Bias</div>
            <div class="meter-value" id="meterDirBias">0.00</div>
          </div>
          <div class="meter-card">
            <div class="meter-title">Clutter Speed Factor</div>
            <div class="meter-value" id="meterSpeedFactor">1.00</div>
          </div>
        </div>
        <div class="episode-history" id="episodeHistoryList">
          <div style="color:var(--text-muted); text-align:center; padding:12px;">Awaiting reinforcement events and obstacle encounters.</div>
        </div>
      </div>

      <!-- WINDOW 5: AFFECT & SELF-MONITORING -->
      <div class="card">
        <div class="card-title">
          <span>Window 5: Simulated Affect & Internal State</span>
          <span style="font-size:0.68rem; color:var(--text-muted);">Deterministic Computational Metrics</span>
        </div>
        <div class="affect-grid">
          <div class="affect-box">
            <div class="affect-name"><span>Surprise</span><strong id="valSurprise">0.00</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barSurprise" style="width:0%;"></div></div>
          </div>
          <div class="affect-box">
            <div class="affect-name"><span>Vigilance</span><strong id="valVigilance">0.00</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barVigilance" style="width:0%; background:var(--accent-amber);"></div></div>
          </div>
          <div class="affect-box">
            <div class="affect-name"><span>Fatigue</span><strong id="valFatigue">0.00</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barFatigue" style="width:0%; background:var(--accent-purple);"></div></div>
          </div>
          <div class="affect-box">
            <div class="affect-name"><span>Confidence</span><strong id="valConfidence">1.00</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barConfidence" style="width:100%; background:var(--accent-green);"></div></div>
          </div>
          <div class="affect-box">
            <div class="affect-name"><span>Uncertainty</span><strong id="valUncertainty">0.00</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barUncertainty" style="width:0%;"></div></div>
          </div>
          <div class="affect-box">
            <div class="affect-name"><span>Valence</span><strong id="valValence">0.50</strong></div>
            <div class="affect-bar-wrap"><div class="affect-bar-fill" id="barValence" style="width:75%;"></div></div>
          </div>
        </div>
        <div style="margin-top:6px; font-size:0.71rem; color:#8fa1be;">
          <strong>Memory Lesson:</strong> <span id="valMemoryLesson">Nominal open terrain.</span>
        </div>
      </div>
    </div>

    <!-- WINDOW 3: SENSORY -> BRAIN -> MOTOR TIMELINE -->
    <div class="card col-timeline">
      <div class="card-title">
        <span>Window 3: Synchronized Sensory &rarr; Connectome &rarr; Motor Timeline (Rolling History)</span>
        <span style="font-size:0.7rem; color:var(--text-muted); font-family:var(--mono-font);">LiDAR Distance (Front) &bull; LC4 Looming Rate &bull; Motor Spike Count &bull; Commanded Speed</span>
      </div>
      <canvas id="timelineCanvas"></canvas>
    </div>
  </div>

  <script>
    // Global State
    let latestState = null;
    let selectedFeedbackMag = 1.00;
    let selectedNeuropil = "all";
    let inspectedFlywireId = "720575940644438551";

    // -------------------------------------------------------------------------
    // Window 1: Simulated Robot & Arena Camera and Viewport State
    // -------------------------------------------------------------------------
    const arenaCanvas = document.getElementById("arenaCanvas");
    const arenaCtx = arenaCanvas.getContext("2d");

    const arenaCamera = {
      x: 0.0,              // Camera focal center X in world meters
      y: -6.0,             // Camera focal center Y in world meters (starts near robot initial spawn)
      scale: 16.0,         // Pixels per meter (default 16 px/m: 20m arena is 320px)
      minScale: 2.0,       // Max zoom out (150m field)
      maxScale: 120.0,     // Max zoom in (inspect chassis components)
      followRobot: true,   // Camera follow robot enabled by default
      isDragging: false,
      dragStartX: 0,
      dragStartY: 0,
      camStartX: 0,
      camStartY: 0,
      layers: {
        rays: true,        // LiDAR free-space scan rays
        hits: true,        // Obstacle hit endpoints
        trail: true,       // Robot historical odometry trail
        grid: true,        // Dynamic world grid & coordinate labels
        walls: true,       // Arena boundaries & fixed obstacles
        footprint: true    // Robot chassis & wheels footprint
      }
    };

    function updateFollowBtnUI() {
      const btn = document.getElementById("btnFollowRobot");
      const modeHud = document.getElementById("hudMode");
      if (btn) {
        if (arenaCamera.followRobot) {
          btn.innerText = "Follow: ON";
          btn.classList.add("active");
        } else {
          btn.innerText = "Follow: OFF";
          btn.classList.remove("active");
        }
      }
      if (modeHud) {
        modeHud.innerText = arenaCamera.followRobot ? "Follow: ON" : "Free Pan";
      }
    }

    function arenaZoomIn() {
      arenaCamera.scale = Math.min(arenaCamera.maxScale, arenaCamera.scale * 1.25);
      const scaleHud = document.getElementById("hudScale");
      if (scaleHud) scaleHud.innerText = arenaCamera.scale.toFixed(1);
      drawArena(latestState);
    }

    function arenaZoomOut() {
      arenaCamera.scale = Math.max(arenaCamera.minScale, arenaCamera.scale / 1.25);
      const scaleHud = document.getElementById("hudScale");
      if (scaleHud) scaleHud.innerText = arenaCamera.scale.toFixed(1);
      drawArena(latestState);
    }

    function worldToScreen(xw, yw) {
      const dpr = window.devicePixelRatio || 1;
      const w = arenaCanvas ? (arenaCanvas.width / dpr) : 600;
      const h = arenaCanvas ? (arenaCanvas.height / dpr) : 300;
      return {
        x: (w / 2) + (xw - arenaCamera.x) * arenaCamera.scale,
        y: (h / 2) - (yw - arenaCamera.y) * arenaCamera.scale
      };
    }

    function screenToWorld(xs, ys) {
      const dpr = window.devicePixelRatio || 1;
      const w = arenaCanvas ? (arenaCanvas.width / dpr) : 600;
      const h = arenaCanvas ? (arenaCanvas.height / dpr) : 300;
      return {
        x: arenaCamera.x + (xs - w / 2) / arenaCamera.scale,
        y: arenaCamera.y - (ys - h / 2) / arenaCamera.scale
      };
    }

    function arenaResetView() {
      arenaCamera.x = 0.0;
      arenaCamera.y = -6.0;
      arenaCamera.scale = 16.0;
      arenaCamera.followRobot = true;
      updateFollowBtnUI();
      const scaleHud = document.getElementById("hudScale");
      if (scaleHud) scaleHud.innerText = arenaCamera.scale.toFixed(1);
      drawArena(latestState);
    }

    function arenaCenterRobot() {
      if (latestState) {
        arenaCamera.x = (latestState.x !== undefined) ? latestState.x : 0.0;
        arenaCamera.y = (latestState.y !== undefined) ? latestState.y : -6.0;
      }
      drawArena(latestState);
    }

    function arenaToggleFollow(forceState) {
      if (forceState !== undefined) {
        arenaCamera.followRobot = Boolean(forceState);
      } else {
        arenaCamera.followRobot = !arenaCamera.followRobot;
      }
      if (arenaCamera.followRobot && latestState) {
        arenaCamera.x = (latestState.x !== undefined) ? latestState.x : 0.0;
        arenaCamera.y = (latestState.y !== undefined) ? latestState.y : -6.0;
      }
      updateFollowBtnUI();
      drawArena(latestState);
    }

    function arenaFitArena() {
      if (!arenaCanvas) return;
      const dpr = window.devicePixelRatio || 1;
      const w = arenaCanvas.width / dpr;
      const h = arenaCanvas.height / dpr;
      arenaCamera.x = 0.0;
      arenaCamera.y = 0.0;
      // Arena is 20m x 20m (from learning_lab_arena.world)
      const arenaSpan = 22.0; // 20m + 2m margin
      const fitScale = Math.min((w - 40) / arenaSpan, (h - 40) / arenaSpan);
      arenaCamera.scale = Math.max(arenaCamera.minScale, Math.min(arenaCamera.maxScale, fitScale));
      arenaCamera.followRobot = false;
      updateFollowBtnUI();
      const scaleHud = document.getElementById("hudScale");
      if (scaleHud) scaleHud.innerText = arenaCamera.scale.toFixed(1);
      drawArena(latestState);
    }

    function arenaToggleLayer(layerName) {
      if (arenaCamera.layers[layerName] !== undefined) {
        arenaCamera.layers[layerName] = !arenaCamera.layers[layerName];
        const btn = document.getElementById("btnLayer" + layerName.charAt(0).toUpperCase() + layerName.slice(1)) ||
                    document.getElementById("layerBtn_" + layerName);
        if (btn) {
          if (arenaCamera.layers[layerName]) btn.classList.add("active");
          else btn.classList.remove("active");
        }
        drawArena(latestState);
      }
    }

    // Attach Mouse & Touch Event Listeners to arenaCanvas
    if (arenaCanvas) {
      arenaCanvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const dpr = window.devicePixelRatio || 1;
        const rect = arenaCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const w = arenaCanvas.width / dpr;
        const h = arenaCanvas.height / dpr;
        const cx = w / 2;
        const cy = h / 2;

        // Current world coordinate under mouse pointer
        const worldX = arenaCamera.x + (mouseX - cx) / arenaCamera.scale;
        const worldY = arenaCamera.y - (mouseY - cy) / arenaCamera.scale;

        const zoomFactor = e.deltaY < 0 ? 1.15 : (1.0 / 1.15);
        const newScale = Math.max(arenaCamera.minScale, Math.min(arenaCamera.maxScale, arenaCamera.scale * zoomFactor));

        // Adjust camera focal center so world position under mouse stays constant
        arenaCamera.x = worldX - (mouseX - cx) / newScale;
        arenaCamera.y = worldY + (mouseY - cy) / newScale;
        arenaCamera.scale = newScale;

        const scaleHud = document.getElementById("hudScale");
        if (scaleHud) scaleHud.innerText = arenaCamera.scale.toFixed(1);
        drawArena(latestState);
      }, { passive: false });

      arenaCanvas.addEventListener("pointerdown", (e) => {
        arenaCamera.isDragging = true;
        arenaCamera.dragStartX = e.clientX;
        arenaCamera.dragStartY = e.clientY;
        arenaCamera.camStartX = arenaCamera.x;
        arenaCamera.camStartY = arenaCamera.y;
        arenaCanvas.style.cursor = "grabbing";
        try { arenaCanvas.setPointerCapture(e.pointerId); } catch (_) {}
      });

      arenaCanvas.addEventListener("pointermove", (e) => {
        if (!arenaCamera.isDragging) return;
        const dx = e.clientX - arenaCamera.dragStartX;
        const dy = e.clientY - arenaCamera.dragStartY;

        // If dragged significantly, switch Follow Robot OFF so user pan isn't snapped back
        if (Math.hypot(dx, dy) > 4 && arenaCamera.followRobot) {
          arenaCamera.followRobot = false;
          updateFollowBtnUI();
        }

        arenaCamera.x = arenaCamera.camStartX - dx / arenaCamera.scale;
        arenaCamera.y = arenaCamera.camStartY + dy / arenaCamera.scale;
        drawArena(latestState);
      });

      function stopArenaDrag(e) {
        if (arenaCamera.isDragging) {
          arenaCamera.isDragging = false;
          arenaCanvas.style.cursor = "grab";
          try { arenaCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
        }
      }
      arenaCanvas.addEventListener("pointerup", stopArenaDrag);
      arenaCanvas.addEventListener("pointerleave", stopArenaDrag);
      arenaCanvas.addEventListener("pointercancel", stopArenaDrag);

      arenaCanvas.addEventListener("dblclick", (e) => {
        const dpr = window.devicePixelRatio || 1;
        const rect = arenaCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const w = arenaCanvas.width / dpr;
        const h = arenaCanvas.height / dpr;
        const clickedWorldX = arenaCamera.x + (mouseX - w / 2) / arenaCamera.scale;
        const clickedWorldY = arenaCamera.y - (mouseY - h / 2) / arenaCamera.scale;

        const rx = latestState ? (latestState.x || 0.0) : 0.0;
        const ry = latestState ? (latestState.y || -6.0) : -6.0;
        if (Math.hypot(clickedWorldX - rx, clickedWorldY - ry) < 2.0) {
          arenaCamera.x = rx;
          arenaCamera.y = ry;
          arenaCamera.followRobot = true;
          updateFollowBtnUI();
        } else {
          arenaCamera.x = clickedWorldX;
          arenaCamera.y = clickedWorldY;
        }
        drawArena(latestState);
      });

      // Keyboard shortcuts for Arena Viewport
      window.addEventListener("keydown", (e) => {
        if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) return;
        const key = e.key.toLowerCase();
        if (key === "+" || key === "=") { arenaZoomIn(); }
        else if (key === "-" || key === "_") { arenaZoomOut(); }
        else if (key === "f") { arenaToggleFollow(); }
        else if (key === "c") { arenaCenterRobot(); }
        else if (key === "r") { arenaResetView(); }
        else if (key === "a") { arenaFitArena(); }
      });
    }

    // Timeline Canvas
    const timelineCanvas = document.getElementById("timelineCanvas");
    const timeCtx = timelineCanvas.getContext("2d");
    const timelineHistory = [];

    function resizeCanvases() {
      const dpr = window.devicePixelRatio || 1;
      if (arenaCanvas) {
        arenaCanvas.width = Math.round(arenaCanvas.clientWidth * dpr);
        arenaCanvas.height = Math.round(arenaCanvas.clientHeight * dpr);
        if (latestState) drawArena(latestState);
      }
      if (timelineCanvas) {
        timelineCanvas.width = Math.round(timelineCanvas.clientWidth * dpr);
        timelineCanvas.height = Math.round(timelineCanvas.clientHeight * dpr);
        if (latestState) drawTimeline(latestState);
      }
    }
    window.addEventListener("resize", resizeCanvases);

    // -------------------------------------------------------------------------
    // 3D WebGL Connectome Viewer (Three.js with Native OrbitControls)
    // -------------------------------------------------------------------------
    const viewerContainer = document.getElementById("viewer3dContainer");
    let scene, camera, renderer, pointsMesh, spikePointsMesh;
    let rawCoords = null; // Float32Array [x, y, z, neuropil_id]
    let activeSpikesList = []; // Array of {pos: THREE.Vector3, intensity: 1.0}
    let controls;
    let markerMesh;

    class SimpleOrbitControls {
      constructor(cam, dom) {
        this.camera = cam;
        this.dom = dom;
        this.target = new THREE.Vector3(0, 0, 0);
        this.radius = 290;
        this.theta = 0.35;
        this.phi = 1.15;
        this.isDragging = false;
        this.isPanning = false;
        this.prev = { x: 0, y: 0 };

        this.dom.addEventListener("mousedown", (e) => {
          if (e.button === 2 || e.shiftKey) this.isPanning = true;
          else this.isDragging = true;
          this.prev = { x: e.clientX, y: e.clientY };
        });
        window.addEventListener("mousemove", (e) => {
          if (!this.isDragging && !this.isPanning) return;
          const dx = e.clientX - this.prev.x;
          const dy = e.clientY - this.prev.y;
          this.prev = { x: e.clientX, y: e.clientY };

          if (this.isDragging) {
            this.theta -= dx * 0.008;
            this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi + dy * 0.008));
          } else if (this.isPanning) {
            const factor = this.radius * 0.0016;
            const fwd = new THREE.Vector3().subVectors(this.target, this.camera.position).normalize();
            const right = new THREE.Vector3().crossVectors(fwd, this.camera.up).normalize();
            const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
            this.target.addScaledVector(right, -dx * factor);
            this.target.addScaledVector(up, dy * factor);
          }
          this.update();
        });
        window.addEventListener("mouseup", () => {
          this.isDragging = false;
          this.isPanning = false;
        });
        this.dom.addEventListener("wheel", (e) => {
          e.preventDefault();
          const zoom = e.deltaY > 0 ? 1.08 : 0.92;
          this.radius = Math.max(30, Math.min(600, this.radius * zoom));
          this.update();
        }, { passive: false });
        this.dom.addEventListener("contextmenu", (e) => e.preventDefault());
        this.update();
      }
      update() {
        this.camera.position.x = this.target.x + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
        this.camera.position.y = this.target.y + this.radius * Math.cos(this.phi);
        this.camera.position.z = this.target.z + this.radius * Math.sin(this.phi) * Math.cos(this.theta);
        this.camera.lookAt(this.target);
      }
      focusOn(x, y, z, dist = 80) {
        this.target.set(x, y, z);
        this.radius = dist;
        this.update();
      }
    }

    function init3DViewer() {
      if (typeof THREE === "undefined") {
        viewerContainer.innerHTML = "<div style='color:#ff5555; padding:20px;'>WebGL Three.js library could not be initialized.</div>";
        return;
      }
      const width = viewerContainer.clientWidth;
      const height = viewerContainer.clientHeight;

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x06070a);

      camera = new THREE.PerspectiveCamera(45, width / height, 1, 1500);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      viewerContainer.appendChild(renderer.domElement);

      controls = new SimpleOrbitControls(camera, renderer.domElement);

      // Anatomical target marker ring
      const ringGeo = new THREE.RingGeometry(3.0, 3.8, 24);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide });
      markerMesh = new THREE.Mesh(ringGeo, ringMat);
      markerMesh.position.set(-12, -25, -30);
      scene.add(markerMesh);

      // Bounding Grid Floor
      const grid = new THREE.GridHelper(300, 20, 0x1f2738, 0x111622);
      grid.position.y = -90;
      scene.add(grid);

      loadSpatialAtlas();
      animate3D();
    }

    async function loadSpatialAtlas() {
      try {
        const resp = await fetch("/api/connectome/spatial_coords_bin");
        if (!resp.ok) throw new Error("Binary fetch fallback");
        const buffer = await resp.arrayBuffer();
        rawCoords = new Float32Array(buffer);
        buildPointCloud(rawCoords);
      } catch (err) {
        console.warn("Generating procedural point cloud fallback:", err);
        generateProceduralScaffold();
      }
    }

    function buildPointCloud(coords) {
      const N = coords.length / 4;
      const positions = new Float32Array(N * 3);
      const colors = new Float32Array(N * 3);

      const palette = {
        1: [0.12, 0.72, 0.98], // OL_L
        2: [0.12, 0.95, 0.88], // OL_R
        3: [0.98, 0.78, 0.12], // CX
        4: [0.75, 0.38, 0.96], // AL_JO
        5: [0.97, 0.40, 0.24], // SEZ
        6: [0.15, 0.95, 0.46], // DN
        7: [0.42, 0.50, 0.65], // CB
      };

      for (let i = 0; i < N; i++) {
        const x = coords[i * 4];
        const y = coords[i * 4 + 1];
        const z = coords[i * 4 + 2];
        const nid = Math.round(coords[i * 4 + 3]);

        positions[i * 3] = x;
        positions[i * 3 + 1] = z; // Map Z to Three.js Y up
        positions[i * 3 + 2] = -y; // Map Y to -Z

        const col = palette[nid] || [0.4, 0.5, 0.6];
        colors[i * 3] = col[0];
        colors[i * 3 + 1] = col[1];
        colors[i * 3 + 2] = col[2];
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: 1.6,
        vertexColors: true,
        transparent: true,
        opacity: 0.65,
      });

      if (pointsMesh) scene.remove(pointsMesh);
      pointsMesh = new THREE.Points(geometry, material);
      scene.add(pointsMesh);

      // Active Spike Point Cloud Overlay
      const spikeGeo = new THREE.BufferGeometry();
      const maxSpikes = 200;
      spikeGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxSpikes * 3), 3));
      spikeGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(maxSpikes * 3), 3));

      const spikeMat = new THREE.PointsMaterial({
        size: 4.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
      });
      spikePointsMesh = new THREE.Points(spikeGeo, spikeMat);
      scene.add(spikePointsMesh);
    }

    function generateProceduralScaffold() {
      const N = 8000;
      const positions = new Float32Array(N * 3);
      const colors = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 200;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 120;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 150;
        colors[i * 3] = 0.2; colors[i * 3 + 1] = 0.7; colors[i * 3 + 2] = 0.9;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      pointsMesh = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.8, vertexColors: true }));
      scene.add(pointsMesh);
    }

    function animate3D() {
      requestAnimationFrame(animate3D);
      if (markerMesh) {
        markerMesh.rotation.z += 0.02;
        markerMesh.lookAt(camera.position);
      }
      // Update and decay live spikes
      if (spikePointsMesh && activeSpikesList.length > 0) {
        const posAttr = spikePointsMesh.geometry.attributes.position;
        const colAttr = spikePointsMesh.geometry.attributes.color;
        let count = 0;
        for (let i = activeSpikesList.length - 1; i >= 0; i--) {
          const sp = activeSpikesList[i];
          sp.intensity *= 0.92;
          if (sp.intensity < 0.05) {
            activeSpikesList.splice(i, 1);
            continue;
          }
          if (count < 200) {
            posAttr.setXYZ(count, sp.x, sp.z, -sp.y);
            colAttr.setXYZ(count, 1.0 * sp.intensity, 0.95 * sp.intensity, 0.2 * sp.intensity);
            count++;
          }
        }
        spikePointsMesh.geometry.setDrawRange(0, count);
        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
      }
      renderer.render(scene, camera);
    }

    function registerLiveSpike(x, y, z) {
      activeSpikesList.push({ x: x, y: y, z: z, intensity: 1.0 });
    }

    function setNeuropilFilter(filterName) {
      selectedNeuropil = filterName;
      document.querySelectorAll(".btn-layer").forEach(b => b.classList.remove("active"));
      event.target.classList.add("active");
      if (!rawCoords || !pointsMesh) return;

      const N = rawCoords.length / 4;
      const colors = pointsMesh.geometry.attributes.color.array;

      const palette = {
        1: [0.12, 0.72, 0.98], 2: [0.12, 0.95, 0.88],
        3: [0.98, 0.78, 0.12], 4: [0.75, 0.38, 0.96],
        5: [0.97, 0.40, 0.24], 6: [0.15, 0.95, 0.46],
        7: [0.42, 0.50, 0.65],
      };

      for (let i = 0; i < N; i++) {
        const nid = Math.round(rawCoords[i * 4 + 3]);
        let visible = true;
        if (filterName === "optic") visible = (nid === 1 || nid === 2);
        else if (filterName === "cx") visible = (nid === 3);
        else if (filterName === "sez") visible = (nid === 5);
        else if (filterName === "motor") visible = (nid === 6);

        const col = palette[nid] || [0.4, 0.5, 0.6];
        const alpha = visible ? 1.0 : 0.08;
        colors[i * 3] = col[0] * alpha;
        colors[i * 3 + 1] = col[1] * alpha;
        colors[i * 3 + 2] = col[2] * alpha;
      }
      pointsMesh.geometry.attributes.color.needsUpdate = true;
    }

    // -------------------------------------------------------------------------
    // REST API & Lifecycle Requests
    // -------------------------------------------------------------------------
    async function requestSleep() {
      try {
        const res = await fetch("/api/lifecycle/sleep", { method: "POST" });
        const data = await res.json();
        updateLifecycleBadge("ASLEEP");
        fetchState();
      } catch (err) {
        console.error("Sleep request error:", err);
      }
    }

    async function requestWake() {
      try {
        const res = await fetch("/api/lifecycle/wake", { method: "POST" });
        const data = await res.json();
        updateLifecycleBadge("AWAKE");
        fetchState();
      } catch (err) {
        console.error("Wake request error:", err);
      }
    }

    async function setControlMode(mode) {
      await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: mode })
      });
      document.getElementById("btnAuto").className = "btn btn-mode" + (mode === "auto" ? " active" : "");
      document.getElementById("btnTeach").className = "btn btn-mode" + (mode === "teach" ? " active" : "");
      fetchState();
    }

    function setMag(mag) {
      selectedFeedbackMag = mag;
      document.querySelectorAll(".mag-btn").forEach(b => b.classList.remove("active"));
      event.target.classList.add("active");
    }

    async function sendReward() {
      const note = document.getElementById("reinforceNote").value || "Manual appetitive reward";
      await fetch("/api/learning/reward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ magnitude: selectedFeedbackMag, note: note })
      });
      document.getElementById("reinforceNote").value = "";
      fetchState();
    }

    async function sendPunish() {
      const note = document.getElementById("reinforceNote").value || "Manual aversive punishment";
      await fetch("/api/learning/punish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ magnitude: selectedFeedbackMag, note: note })
      });
      document.getElementById("reinforceNote").value = "";
      fetchState();
    }

    async function selectLandmark(flywireId) {
      if (!flywireId) return;
      document.getElementById("neuronSearchInput").value = flywireId;
      searchNeuron();
    }

    async function searchNeuron() {
      const q = document.getElementById("neuronSearchInput").value.trim();
      if (!q) return;

      try {
        const res = await fetch("/api/neuron/" + encodeURIComponent(q));
        const data = await res.json();
        if (data.error) {
          document.getElementById("insId").innerText = "NOT FOUND";
          document.getElementById("insName").innerText = q;
          document.getElementById("insRole").innerText = data.error;
          return;
        }

        const exactFid = String(data.flywire_id || q);
        inspectedFlywireId = exactFid;
        document.getElementById("insId").innerText = exactFid;
        document.getElementById("insName").innerText = data.name || "Neuron";
        document.getElementById("insGroup").innerText = data.functional_group || "Connectome";
        document.getElementById("insRole").innerText = "Role: " + (data.role || "Recurrent LIF connectome interneuron");
        document.getElementById("insVoltage").innerText = (data.voltage !== undefined ? data.voltage : -65.0) + " mV";

        // Place 3D Marker Ring
        if (controls && markerMesh) {
          let x = 0, y = 0, z = 0;
          if (data.x !== undefined) { x = data.x; y = data.y; z = data.z; }
          else { x = -12; y = -25; z = -30; }
          markerMesh.position.set(x, z, -y);
          controls.focusOn(x, z, -y, 65);
        }

        // Render presynaptic inputs table
        const preTable = document.getElementById("insPreTable");
        if (data.presynaptic && data.presynaptic.length > 0) {
          preTable.innerHTML = data.presynaptic.map(syn => `
            <div class="syn-row">
              <span title="${syn.flywire_id}">${syn.name || syn.flywire_id}</span>
              <span style="color:var(--accent-green);">+${syn.weight}</span>
            </div>
          `).join("");
        } else {
          preTable.innerHTML = "<div style='color:var(--text-muted);'>No presynaptic inputs recorded.</div>";
        }

        // Render postsynaptic targets table
        const postTable = document.getElementById("insPostTable");
        if (data.postsynaptic && data.postsynaptic.length > 0) {
          postTable.innerHTML = data.postsynaptic.map(syn => `
            <div class="syn-row">
              <span title="${syn.flywire_id}">${syn.name || syn.flywire_id}</span>
              <span style="color:var(--accent-cyan);">${syn.weight}</span>
            </div>
          `).join("");
        } else {
          postTable.innerHTML = "<div style='color:var(--text-muted);'>No postsynaptic targets recorded.</div>";
        }
      } catch (err) {
        console.error("Neuron query failed:", err);
      }
    }

    function updateLifecycleBadge(state) {
      const badge = document.getElementById("badgeLifecycle");
      if (state === "ASLEEP" || state === "SLEEPING") {
        badge.className = "badge badge-sleep";
        badge.innerText = "ASLEEP (HALTED)";
      } else {
        badge.className = "badge badge-awake";
        badge.innerText = "AWAKE";
      }
    }

    // -------------------------------------------------------------------------
    // Main Periodic Telemetry Polling & UI Update Loop
    // -------------------------------------------------------------------------
    async function fetchState() {
      try {
        const res = await fetch("/api/state");
        latestState = await res.json();
        updateUI(latestState);
        drawArena(latestState);
        drawTimeline(latestState);
      } catch (err) {
        console.warn("Telemetry fetch error:", err);
      }
    }

    function updateUI(st) {
      if (!st) return;

      // Lifecycle status
      const isAsleep = (st.status === "ASLEEP" || st.is_asleep || st.lifecycle_state === "ASLEEP" || st.lifecycle_state === "SLEEPING");
      updateLifecycleBadge(isAsleep ? "ASLEEP" : "AWAKE");

      // Safety and Motion badge
      const sBadge = document.getElementById("badgeSafety");
      const mState = st.motion_state || (isAsleep ? "ASLEEP" : (st.status || "NOMINAL"));
      if (mState === "MOVING" || mState === "NOMINAL") sBadge.className = "badge badge-awake";
      else if (mState === "WAITING_FOR_SENSORS" || mState === "AWAKE_NO_MOTOR_OUTPUT" || mState.includes("SLOWDOWN") || mState.includes("AVOIDANCE")) sBadge.className = "badge badge-warn";
      else sBadge.className = "badge badge-danger";
      sBadge.innerText = mState;

      // Kinematics readout
      document.getElementById("kinematicsReadout").innerText =
        `v: ${Number(st.safe_vx || st.v || 0).toFixed(2)} m/s • ω: ${Number(st.safe_wz || st.omega || 0).toFixed(2)} rad/s`;

      // Distances
      if (st.sector_distances) {
        document.getElementById("valDistFront").innerText = `${Number(st.sector_distances.front || 10).toFixed(1)}m`;
        document.getElementById("valDistLeft").innerText = `${Number(st.sector_distances.left || 10).toFixed(1)}m`;
        document.getElementById("valDistRight").innerText = `${Number(st.sector_distances.right || 10).toFixed(1)}m`;
        document.getElementById("valDistMin").innerText = `${Number(st.sector_distances.min || 10).toFixed(1)}m`;
      }

      // Learning Meters
      if (st.learning) {
        const l = st.learning;
        document.getElementById("meterDeltaSteer").innerText = `+${Number(l.delta_steer_dist || 0).toFixed(2)}m`;
        document.getElementById("meterDirBias").innerText = `${Number(l.w_dir_bias || 0).toFixed(2)}`;
        document.getElementById("meterSpeedFactor").innerText = `${Number(l.speed_factor || 1.0).toFixed(2)}`;
        document.getElementById("learningUpdatesBadge").innerText = `Updates: ${l.total_learning_steps || 0}`;
      }

      // Episodic memory history
      if (st.episodic_memory_recent && st.episodic_memory_recent.length > 0) {
        const histEl = document.getElementById("episodeHistoryList");
        histEl.innerHTML = st.episodic_memory_recent.slice().reverse().map(ep => {
          const isReward = (ep.reward > 0);
          const color = isReward ? "var(--accent-green)" : (ep.reward < 0 ? "var(--accent-red)" : "#a2b4cc");
          return `
            <div class="ep-row">
              <span style="font-weight:700; color:${color}; font-family:var(--mono-font);">
                ${ep.outcome} (${ep.reward > 0 ? "+" : ""}${ep.reward})
              </span>
              <span style="color:#b3c2d6;">${ep.lesson || ""}</span>
            </div>
          `;
        }).join("");
      }

      // Affect & Diagnostics
      if (st.internal_state) {
        const af = st.internal_state;
        setAffectMeter("Surprise", af.sensory_surprise || 0);
        setAffectMeter("Vigilance", af.environmental_vigilance || 0);
        setAffectMeter("Fatigue", af.computational_fatigue || 0);
        setAffectMeter("Confidence", af.confidence || 1.0);
        setAffectMeter("Uncertainty", af.uncertainty || 0);
        setAffectMeter("Valence", ((af.valence_affect !== undefined ? af.valence_affect : 0.5) + 1.0) / 2.0);
        document.getElementById("valMemoryLesson").innerText = af.memory_lesson || "Nominal open terrain.";
      }

      // Spikes in 3D viewer
      if (st.subnetwork && st.subnetwork.active_nodes && st.subnetwork.active_nodes.length > 0) {
        st.subnetwork.active_nodes.forEach(node => {
          registerLiveSpike(
            (Math.random() - 0.5) * 120,
            (Math.random() - 0.5) * 80,
            (Math.random() - 0.5) * 60
          );
        });
      }

      // Checkpoint status
      if (st.lifecycle) {
        const lc = st.lifecycle;
        const cpEl = document.getElementById("badgeCheckpoint");
        if (lc.checkpoint_exists) {
          cpEl.innerText = "CP: PERSISTENT";
          cpEl.style.borderColor = "var(--accent-green)";
          cpEl.style.color = "var(--accent-green)";
        } else {
          cpEl.innerText = "CP: UNINITIALIZED";
        }
      }
    }

    function setAffectMeter(name, val) {
      val = Math.max(0.0, Math.min(1.0, val));
      const elVal = document.getElementById("val" + name);
      const elBar = document.getElementById("bar" + name);
      if (elVal) elVal.innerText = val.toFixed(2);
      if (elBar) elBar.style.width = (val * 100).toFixed(0) + "%";
    }

    // -------------------------------------------------------------------------
    // 2D Arena Canvas Rendering: Accurate Robotics Engineering Visualization
    // -------------------------------------------------------------------------
    function drawArena(st) {
      if (!arenaCanvas) return;
      const dpr = window.devicePixelRatio || 1;
      const w = arenaCanvas.width / dpr;
      const h = arenaCanvas.height / dpr;
      if (w <= 0 || h <= 0) return;

      arenaCtx.save();
      arenaCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      arenaCtx.clearRect(0, 0, w, h);

      // Arena background
      arenaCtx.fillStyle = "#06070a";
      arenaCtx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const scale = arenaCamera.scale;

      const rx = (st && st.x !== undefined) ? st.x : 0.0;
      const ry = (st && st.y !== undefined) ? st.y : -6.0;
      const th = (st && st.theta !== undefined) ? st.theta : 1.5708;
      const isAsleep = (st && st.is_asleep);
      const motionState = (st && st.motion_state) ? st.motion_state : (isAsleep ? "ASLEEP" : "MOVING");

      // Smooth camera follow tracking (exponential smoothing filter)
      if (arenaCamera.followRobot && !arenaCamera.isDragging) {
        const dist = Math.hypot(rx - arenaCamera.x, ry - arenaCamera.y);
        if (dist > 8.0) {
          arenaCamera.x = rx;
          arenaCamera.y = ry;
        } else {
          arenaCamera.x += 0.22 * (rx - arenaCamera.x);
          arenaCamera.y += 0.22 * (ry - arenaCamera.y);
        }
      }

      function toScreen(wx, wy) {
        return [cx + (wx - arenaCamera.x) * scale, cy - (wy - arenaCamera.y) * scale];
      }
      function toWorld(sx, sy) {
        return [arenaCamera.x + (sx - cx) / scale, arenaCamera.y - (sy - cy) / scale];
      }

      const [minWorldX, maxWorldY] = toWorld(0, 0);
      const [maxWorldX, minWorldY] = toWorld(w, h);

      // -----------------------------------------------------------------------
      // 1. Layer: Dynamic World Grid & Coordinate Axes
      // -----------------------------------------------------------------------
      if (arenaCamera.layers.grid) {
        let minorStep = 1.0;
        let majorStep = 5.0;
        if (scale < 6.0) { minorStep = 5.0; majorStep = 20.0; }
        else if (scale < 14.0) { minorStep = 2.0; majorStep = 10.0; }
        else if (scale < 32.0) { minorStep = 1.0; majorStep = 5.0; }
        else if (scale < 75.0) { minorStep = 0.5; majorStep = 2.0; }
        else { minorStep = 0.1; majorStep = 0.5; }

        arenaCtx.lineWidth = 1;
        arenaCtx.font = "9px SFMono-Regular, Consolas, monospace";
        arenaCtx.textAlign = "center";
        arenaCtx.textBaseline = "bottom";

        // Vertical Grid Lines (X)
        const startX = Math.floor(minWorldX / minorStep) * minorStep;
        for (let gx = startX; gx <= maxWorldX + minorStep; gx += minorStep) {
          const sx = cx + (gx - arenaCamera.x) * scale;
          const isMajor = Math.abs(Math.round(gx / minorStep) % Math.round(majorStep / minorStep)) === 0;
          arenaCtx.strokeStyle = isMajor ? "#1a2233" : "#0d111a";
          arenaCtx.beginPath();
          arenaCtx.moveTo(sx, 0);
          arenaCtx.lineTo(sx, h);
          arenaCtx.stroke();

          if (isMajor && sx >= 20 && sx <= w - 20) {
            arenaCtx.fillStyle = "#4a5770";
            arenaCtx.fillText((gx >= 0 ? "+" : "") + gx.toFixed(0) + "m", sx, h - 3);
          }
        }

        // Horizontal Grid Lines (Y)
        arenaCtx.textAlign = "left";
        arenaCtx.textBaseline = "middle";
        const startY = Math.floor(minWorldY / minorStep) * minorStep;
        for (let gy = startY; gy <= maxWorldY + minorStep; gy += minorStep) {
          const sy = cy - (gy - arenaCamera.y) * scale;
          const isMajor = Math.abs(Math.round(gy / minorStep) % Math.round(majorStep / minorStep)) === 0;
          arenaCtx.strokeStyle = isMajor ? "#1a2233" : "#0d111a";
          arenaCtx.beginPath();
          arenaCtx.moveTo(0, sy);
          arenaCtx.lineTo(w, sy);
          arenaCtx.stroke();

          if (isMajor && sy >= 15 && sy <= h - 15) {
            arenaCtx.fillStyle = "#4a5770";
            arenaCtx.fillText((gy >= 0 ? "+" : "") + gy.toFixed(0) + "m", 4, sy);
          }
        }

        // World Origin Crosshair Axes (0,0)
        const [originX, originY] = toScreen(0, 0);
        arenaCtx.lineWidth = 1.2;
        // World +X axis (East - Red vector)
        arenaCtx.strokeStyle = "rgba(255, 68, 85, 0.45)";
        arenaCtx.beginPath();
        arenaCtx.moveTo(originX, originY);
        arenaCtx.lineTo(originX + 35, originY);
        arenaCtx.stroke();
        // World +Y axis (North - Green vector)
        arenaCtx.strokeStyle = "rgba(0, 230, 118, 0.45)";
        arenaCtx.beginPath();
        arenaCtx.moveTo(originX, originY);
        arenaCtx.lineTo(originX, originY - 35);
        arenaCtx.stroke();

        arenaCtx.fillStyle = "#71829e";
        arenaCtx.beginPath();
        arenaCtx.arc(originX, originY, 3, 0, Math.PI * 2);
        arenaCtx.fill();
        arenaCtx.font = "8px monospace";
        arenaCtx.fillText("(0,0)", originX + 5, originY + 12);
      }

      // -----------------------------------------------------------------------
      // 2. Layer: Physical Arena Boundary Walls & Fixed Obstacles
      // -----------------------------------------------------------------------
      const bounds = (st && st.arena_bounds) ? st.arena_bounds : {
        min_x: -10.0, max_x: 10.0, min_y: -10.0, max_y: 10.0,
        fixed_obstacles: [
          { name: "center_pillar", type: "cylinder", x: 0.0, y: 0.0, radius: 1.2 },
          { name: "left_pillar", type: "cylinder", x: -3.5, y: -2.0, radius: 0.8 },
          { name: "right_pillar", type: "cylinder", x: 3.5, y: -2.0, radius: 0.8 },
        ],
        zones: [
          { name: "sucrose_reward", type: "circle", x: 5.5, y: 5.5, radius: 1.8, color: "rgba(0, 230, 118, 0.18)", border: "#00e676" },
          { name: "shock_hazard", type: "circle", x: -5.5, y: 5.5, radius: 1.8, color: "rgba(255, 51, 75, 0.18)", border: "#ff334b" }
        ]
      };

      if (arenaCamera.layers.walls) {
        // Arena Floor Fill (20m x 20m)
        const [nwX, nwY] = toScreen(bounds.min_x, bounds.max_y);
        const [seX, seY] = toScreen(bounds.max_x, bounds.min_y);
        arenaCtx.fillStyle = "rgba(13, 17, 26, 0.45)";
        arenaCtx.fillRect(nwX, nwY, seX - nwX, seY - nwY);

        // Functional Biological Zones (omitting removed Sugar & Shock circle markers)
        if (bounds.zones) {
          const excludedZones = new Set(["sucrose_reward", "shock_hazard", "sugar", "shock"]);
          bounds.zones.forEach(z => {
            if (excludedZones.has(z.name.toLowerCase())) return;
            const [zx, zy] = toScreen(z.x, z.y);
            const zr = z.radius * scale;
            arenaCtx.fillStyle = z.color;
            arenaCtx.strokeStyle = z.border;
            arenaCtx.lineWidth = 1.2;
            arenaCtx.setLineDash([4, 4]);
            arenaCtx.beginPath();
            arenaCtx.arc(zx, zy, zr, 0, Math.PI * 2);
            arenaCtx.fill();
            arenaCtx.stroke();
            arenaCtx.setLineDash([]);
            arenaCtx.font = "8px monospace";
            arenaCtx.fillStyle = z.border;
            arenaCtx.textAlign = "center";
            arenaCtx.fillText(z.name, zx, zy - zr - 3);
          });
        }

        // Fixed World Obstacles (omitting removed Center, Left & Right circle markers)
        if (bounds.fixed_obstacles) {
          const excludedObs = new Set(["center_pillar", "left_pillar", "right_pillar", "center", "left", "right"]);
          bounds.fixed_obstacles.forEach(obs => {
            if (excludedObs.has(obs.name.toLowerCase())) return;
            const [ox, oy] = toScreen(obs.x, obs.y);
            const rPx = obs.radius * scale;
            arenaCtx.fillStyle = "rgba(255, 152, 0, 0.22)";
            arenaCtx.strokeStyle = "#ff9800";
            arenaCtx.lineWidth = 1.5;
            arenaCtx.beginPath();
            arenaCtx.arc(ox, oy, rPx, 0, Math.PI * 2);
            arenaCtx.fill();
            arenaCtx.stroke();
            arenaCtx.fillStyle = "#ffb74d";
            arenaCtx.font = "8px monospace";
            arenaCtx.textAlign = "center";
            arenaCtx.fillText(obs.name.replace("_pillar", ""), ox, oy + 3);
          });
        }

        // 20m x 20m Perimeter Solid Walls
        arenaCtx.strokeStyle = "#3e4c66";
        arenaCtx.lineWidth = Math.max(2.0, 0.20 * scale);
        arenaCtx.strokeRect(nwX, nwY, seX - nwX, seY - nwY);

        arenaCtx.font = "8px monospace";
        arenaCtx.fillStyle = "#5c6d8c";
        arenaCtx.textAlign = "center";
        arenaCtx.fillText("NORTH WALL (20m)", (nwX + seX) / 2, nwY - 6);
        arenaCtx.fillText("SOUTH WALL (20m)", (nwX + seX) / 2, seY + 12);
      }

      // -----------------------------------------------------------------------
      // 3. Layer: Robot Trajectory Trail
      // -----------------------------------------------------------------------
      if (arenaCamera.layers.trail && st && st.trail && st.trail.length > 1) {
        arenaCtx.strokeStyle = "rgba(0, 212, 230, 0.40)";
        arenaCtx.lineWidth = 1.4;
        arenaCtx.setLineDash([3, 3]);
        arenaCtx.beginPath();
        st.trail.forEach((pt, i) => {
          const [tx, ty] = toScreen(pt[0], pt[1]);
          if (i === 0) arenaCtx.moveTo(tx, ty);
          else arenaCtx.lineTo(tx, ty);
        });
        arenaCtx.stroke();
        arenaCtx.setLineDash([]);
      }

      // -----------------------------------------------------------------------
      // 4. Layer: LiDAR Beams & Obstacle Returns
      // -----------------------------------------------------------------------
      const sensorOffsetX = 0.10;
      const sensorWorldX = rx + sensorOffsetX * Math.cos(th);
      const sensorWorldY = ry + sensorOffsetX * Math.sin(th);
      const [sensorSx, sensorSy] = toScreen(sensorWorldX, sensorWorldY);

      if (st && st.rays && st.rays.length > 0) {
        // Free-space & Obstacle Rays
        if (arenaCamera.layers.rays) {
          st.rays.forEach(ray => {
            const [ex, ey] = toScreen(ray.rx, ray.ry);
            const isHit = ray.hit;
            arenaCtx.lineWidth = 1;
            arenaCtx.strokeStyle = isHit ? "rgba(255, 75, 90, 0.35)" : "rgba(0, 212, 230, 0.08)";
            arenaCtx.beginPath();
            arenaCtx.moveTo(sensorSx, sensorSy);
            arenaCtx.lineTo(ex, ey);
            arenaCtx.stroke();
          });
        }

        // Distinct Obstacle Hit Points
        if (arenaCamera.layers.hits) {
          st.rays.forEach(ray => {
            if (ray.hit) {
              const [ex, ey] = toScreen(ray.rx, ray.ry);
              arenaCtx.fillStyle = "#ff334b";
              arenaCtx.beginPath();
              arenaCtx.arc(ex, ey, Math.max(2.0, Math.min(4.5, 0.06 * scale)), 0, Math.PI * 2);
              arenaCtx.fill();
            }
          });
        }
      }

      // -----------------------------------------------------------------------
      // 5. Layer: Robot Model & Footprint (URDF/SDF 4-Wheel Differential Chassis)
      // -----------------------------------------------------------------------
      if (arenaCamera.layers.footprint) {
        const [robSx, robSy] = toScreen(rx, ry);
        arenaCtx.save();
        arenaCtx.translate(robSx, robSy);
        arenaCtx.rotate(-th); // Standard canvas rotation for CCW yaw

        // Wheels: L = 0.17m (dia), W = 0.05m
        const wL = 0.17 * scale;
        const wW = 0.05 * scale;
        const wheelCoords = [
          [0.12 * scale, -0.18 * scale],  // Front Left (left is -Y in screen local)
          [0.12 * scale, 0.18 * scale],   // Front Right
          [-0.12 * scale, -0.18 * scale], // Rear Left
          [-0.12 * scale, 0.18 * scale],  // Rear Right
        ];
        arenaCtx.fillStyle = "#1e2430";
        arenaCtx.strokeStyle = "#475569";
        arenaCtx.lineWidth = 1;
        wheelCoords.forEach(wc => {
          arenaCtx.beginPath();
          arenaCtx.roundRect(wc[0] - wL / 2, wc[1] - wW / 2, wL, wW, Math.max(1, 0.015 * scale));
          arenaCtx.fill();
          arenaCtx.stroke();
        });

        // Chassis Body: 0.40m length x 0.30m width
        const cL = 0.40 * scale;
        const cW = 0.30 * scale;
        let bodyColor = "#00b4d8";
        let bodyBorder = "#e2e8f0";
        if (isAsleep) {
          bodyColor = "#7c4dff";
          bodyBorder = "#b388ff";
        } else if (motionState === "SAFETY_STOP") {
          bodyColor = "#ff3d00";
          bodyBorder = "#ff8a80";
        } else if (motionState === "WAITING_FOR_SENSORS") {
          bodyColor = "#ffab00";
          bodyBorder = "#ffd54f";
        }

        arenaCtx.fillStyle = bodyColor;
        arenaCtx.strokeStyle = bodyBorder;
        arenaCtx.lineWidth = 1.5;
        arenaCtx.beginPath();
        arenaCtx.roundRect(-cL / 2, -cW / 2, cL, cW, Math.max(2, 0.03 * scale));
        arenaCtx.fill();
        arenaCtx.stroke();

        // Inner Electronics Cockpit
        arenaCtx.fillStyle = "rgba(10, 15, 24, 0.6)";
        arenaCtx.fillRect(-cL / 4, -cW / 3, cL / 2, (cW * 2) / 3);

        // LiDAR Sensor Turret at +0.10m forward
        const sensorXLocal = 0.10 * scale;
        const turretR = Math.max(3.0, 0.045 * scale);
        arenaCtx.fillStyle = "#0c1017";
        arenaCtx.strokeStyle = "#00e676";
        arenaCtx.lineWidth = 1.2;
        arenaCtx.beginPath();
        arenaCtx.arc(sensorXLocal, 0, turretR, 0, Math.PI * 2);
        arenaCtx.fill();
        arenaCtx.stroke();

        arenaCtx.fillStyle = "#00ffff";
        arenaCtx.beginPath();
        arenaCtx.arc(sensorXLocal, 0, Math.max(1.5, 0.018 * scale), 0, Math.PI * 2);
        arenaCtx.fill();

        // Forward Heading Indicator Arrow
        arenaCtx.strokeStyle = "#ffffff";
        arenaCtx.lineWidth = 2;
        arenaCtx.beginPath();
        arenaCtx.moveTo(sensorXLocal, 0);
        arenaCtx.lineTo(sensorXLocal + 0.22 * scale, 0);
        arenaCtx.stroke();
        // Arrowhead
        arenaCtx.fillStyle = "#ffffff";
        arenaCtx.beginPath();
        arenaCtx.moveTo(sensorXLocal + 0.22 * scale, 0);
        arenaCtx.lineTo(sensorXLocal + 0.16 * scale, -0.04 * scale);
        arenaCtx.lineTo(sensorXLocal + 0.16 * scale, 0.04 * scale);
        arenaCtx.closePath();
        arenaCtx.fill();

        arenaCtx.restore();
      }

      arenaCtx.restore();

      // -----------------------------------------------------------------------
      // 6. Update Live Telemetry HUD Overlays
      // -----------------------------------------------------------------------
      const poseEl = document.getElementById("hudPose");
      if (poseEl) {
        const deg = (th * 180.0 / Math.PI).toFixed(1);
        poseEl.innerText = `X: ${rx >= 0 ? "+" : ""}${rx.toFixed(2)}m • Y: ${ry >= 0 ? "+" : ""}${ry.toFixed(2)}m • ${deg}°`;
      }
      const velEl = document.getElementById("hudVel");
      if (velEl) {
        const vx = (st && st.safe_vx !== undefined) ? st.safe_vx : 0.0;
        const wz = (st && st.safe_wz !== undefined) ? st.safe_wz : 0.0;
        velEl.innerText = `v: ${vx.toFixed(2)} m/s • ω: ${wz.toFixed(2)} rad/s`;
      }
      const stateEl = document.getElementById("hudState");
      if (stateEl) {
        const stateBadgeClass = isAsleep ? "badge-sleep" : "badge-awake";
        const stateName = isAsleep ? "ASLEEP" : "AWAKE";
        stateEl.innerHTML = `<span class="badge-state ${stateBadgeClass}">${stateName}</span> <span class="badge-motion" id="hudMotionState">${motionState}</span>`;
      }
      const lidarEl = document.getElementById("hudLidar");
      if (lidarEl && st && st.lidar) {
        const freshCls = st.lidar.fresh ? "hud-fresh" : "hud-stale";
        const statusTxt = st.lidar.fresh ? "ACTIVE" : `STALE (${st.lidar.age_sec}s)`;
        lidarEl.innerHTML = `<span class="${freshCls}">${statusTxt}</span> • ${st.lidar.valid_hits} hits / ${st.lidar.total_rays} rays`;
      }
      const safetyEl = document.getElementById("hudSafety");
      if (safetyEl) {
        const safeStat = (st && st.status) ? st.status : "NOMINAL";
        const avoidDir = (st && st.avoidance_direction !== undefined) ? st.avoidance_direction : 0;
        safetyEl.innerText = safeStat + (avoidDir !== 0 ? ` (Avoidance: ${avoidDir > 0 ? "Left" : "Right"})` : "");
      }
      const kinEl = document.getElementById("kinematicsReadout");
      if (kinEl) {
        const vx = (st && st.safe_vx !== undefined) ? st.safe_vx : 0.0;
        const wz = (st && st.safe_wz !== undefined) ? st.safe_wz : 0.0;
        const deg = (th * 180.0 / Math.PI).toFixed(0);
        kinEl.innerText = `v: ${vx.toFixed(2)} m/s • ω: ${wz.toFixed(2)} rad/s • (${rx.toFixed(1)}m, ${ry.toFixed(1)}m, ${deg}°)`;
      }
    }

    // -------------------------------------------------------------------------
    // Synchronized Timeline Strip Chart
    // -------------------------------------------------------------------------
    function drawTimeline(st) {
      if (!timelineCanvas) return;
      const w = timelineCanvas.width;
      const h = timelineCanvas.height;
      timeCtx.clearRect(0, 0, w, h);

      const dFront = (st && st.sector_distances) ? st.sector_distances.front : 10.0;
      const speed = (st && st.safe_vx !== undefined) ? st.safe_vx : 0.0;
      const loomRate = (st && st.motor_rates && st.motor_rates.fwd_left) ? st.motor_rates.fwd_left : 0.0;
      const activeCount = (st && st.network_activity && st.network_activity.mean_active_spikes) ? st.network_activity.mean_active_spikes : 0.0;

      timelineHistory.push({ dFront, speed, loomRate, activeCount });
      if (timelineHistory.length > 180) timelineHistory.shift();

      timeCtx.fillStyle = "#06070a";
      timeCtx.fillRect(0, 0, w, h);

      // Grid
      timeCtx.strokeStyle = "#10141f";
      timeCtx.lineWidth = 1;
      for (let y = 0; y < h; y += 35) {
        timeCtx.beginPath(); timeCtx.moveTo(0, y); timeCtx.lineTo(w, y); timeCtx.stroke();
      }

      const N = timelineHistory.length;
      if (N < 2) return;

      function plot(fn, color, yMax, yMin = 0) {
        timeCtx.strokeStyle = color;
        timeCtx.lineWidth = 1.6;
        timeCtx.beginPath();
        for (let i = 0; i < N; i++) {
          const val = fn(timelineHistory[i]);
          const norm = (val - yMin) / (yMax - yMin);
          const x = (i / (N - 1)) * w;
          const y = h - norm * (h - 10) - 5;
          if (i === 0) timeCtx.moveTo(x, y);
          else timeCtx.lineTo(x, y);
        }
        timeCtx.stroke();
      }

      plot(p => p.dFront, "#00d4e6", 6.0, 0.0);       // Cyan: Front Obstacle Dist (m)
      plot(p => p.speed, "#00e676", 0.4, 0.0);        // Green: Commanded Speed (m/s)
      plot(p => p.loomRate, "#ffb300", 5.0, 0.0);     // Amber: Motor Spike Rate (Hz)
    }

    // Startup
    window.addEventListener("DOMContentLoaded", () => {
      resizeCanvases();
      init3DViewer();
      searchNeuron();
      fetchState();
      setInterval(fetchState, 100);
    });
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Backend REST Endpoints
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(sim.get_state())


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(force=True) if request.data else {}
    cmd = data.get("command")

    global ros_bridge
    is_ros = HAVE_ROS2 and (ros_bridge is not None) and (time.time() - ros_bridge.last_msg_time < 3.0)

    if is_ros:
        ros_bridge.set_control_mode("TELEOP")
        if cmd == "forward":
            ros_bridge.publish_teleop_cmd(0.30, 0.0)
        elif cmd == "backward":
            ros_bridge.publish_teleop_cmd(-0.20, 0.0)
        elif cmd == "left":
            ros_bridge.publish_teleop_cmd(0.0, 0.80)
        elif cmd == "right":
            ros_bridge.publish_teleop_cmd(0.0, -0.80)
        elif cmd == "stop":
            ros_bridge.publish_teleop_cmd(0.0, 0.0)
        return jsonify({"status": "ok", "mode": "TELEOP", "command": cmd})

    return jsonify({"status": "ok", "mode": "STANDALONE", "command": cmd})


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(force=True) if request.data else {}
    mode = data.get("state")
    global ros_bridge
    if HAVE_ROS2 and ros_bridge is not None:
        if mode == "teach":
            ros_bridge.set_control_mode("TELEOP")
        else:
            ros_bridge.set_control_mode("NEURAL")
    return jsonify({"status": "ok", "mode": mode})


@app.route("/api/lifecycle/sleep", methods=["POST"])
def api_lifecycle_sleep():
    global lifecycle_supervisor, ros_bridge
    if HAVE_ROS2 and ros_bridge is not None:
        ros_bridge.publish_teleop_cmd(0.0, 0.0)
        ros_bridge.send_lifecycle_command("SLEEP")

    if lifecycle_supervisor:
        res = lifecycle_supervisor.request_sleep()
        return jsonify(res)
    return jsonify({"lifecycle_state": "SLEEPING", "status": "ASLEEP"})


@app.route("/api/lifecycle/wake", methods=["POST"])
def api_lifecycle_wake():
    global lifecycle_supervisor, ros_bridge
    if HAVE_ROS2 and ros_bridge is not None:
        ros_bridge.send_lifecycle_command("WAKE")

    if lifecycle_supervisor:
        res = lifecycle_supervisor.request_wake(headless=True)
        return jsonify(res)
    return jsonify({"lifecycle_state": "AWAKE", "status": "AWAKE"})


@app.route("/api/lifecycle/status", methods=["GET"])
def api_lifecycle_status():
    global lifecycle_supervisor
    if lifecycle_supervisor:
        return jsonify(lifecycle_supervisor.get_status())
    canonical_cp = get_canonical_checkpoint_path()
    return jsonify({
        "lifecycle_state": "STANDALONE",
        "checkpoint_path": canonical_cp,
        "checkpoint_exists": os.path.exists(canonical_cp),
    })


@app.route("/api/learning/feedback", methods=["POST"])
def api_learning_feedback():
    data = request.get_json(force=True) if request.data else {}
    try:
        reward = float(data.get("reward", 0.0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Reward must be a valid number"}), 400

    if reward < -2.0 or reward > 2.0:
        return jsonify({"status": "error", "message": f"Reward {reward} outside safe bounds [-2.0, +2.0]"}), 400

    note = str(data.get("note", ""))
    global ros_bridge, sim
    is_ros = HAVE_ROS2 and (ros_bridge is not None) and (time.time() - ros_bridge.last_msg_time < 3.0)

    if is_ros:
        ros_bridge.publish_learning_feedback(reward, note)
        with ros_bridge.lock:
            cur_learn = ros_bridge.telemetry.get("learning", {})
        return jsonify({
            "status": "APPLIED",
            "reward": round(reward, 3),
            "note": note,
            "source": "ROS2",
            "learning": cur_learn,
        })
    else:
        with sim.lock:
            if sim.learner is not None:
                res = sim.learner.apply_feedback(reward, note)
                if sim.internal_state:
                    sim.internal_state.process_feedback(reward)
                res["source"] = "STANDALONE"
                return jsonify(res)
            return jsonify({"status": "APPLIED", "reward": reward, "source": "STANDALONE_MINIMAL"})


@app.route("/api/learning/reward", methods=["POST"])
def api_learning_reward():
    data = request.get_json(force=True) if request.data else {}
    mag = float(data.get("magnitude", 1.0))
    mag = max(0.01, min(2.0, mag))
    note = str(data.get("note", "Manual appetitive reward"))
    return _apply_feedback_internal(mag, note)


@app.route("/api/learning/punish", methods=["POST"])
def api_learning_punish():
    data = request.get_json(force=True) if request.data else {}
    mag = float(data.get("magnitude", 1.0))
    mag = max(0.01, min(2.0, mag))
    note = str(data.get("note", "Manual aversive punishment"))
    return _apply_feedback_internal(-mag, note)


def _apply_feedback_internal(reward: float, note: str):
    global ros_bridge, sim
    is_ros = HAVE_ROS2 and (ros_bridge is not None) and (time.time() - ros_bridge.last_msg_time < 3.0)
    if is_ros:
        ros_bridge.publish_learning_feedback(reward, note)
        with ros_bridge.lock:
            cur_learn = ros_bridge.telemetry.get("learning", {})
        return jsonify({"status": "APPLIED", "reward": reward, "note": note, "learning": cur_learn})
    else:
        with sim.lock:
            if sim.learner:
                res = sim.learner.apply_feedback(reward, note)
                if sim.internal_state:
                    sim.internal_state.process_feedback(reward)
                return jsonify(res)
            return jsonify({"status": "APPLIED", "reward": reward, "note": note})


@app.route("/api/learning/reset", methods=["POST"])
def api_learning_reset():
    global ros_bridge, sim
    if HAVE_ROS2 and ros_bridge is not None:
        ros_bridge.send_lifecycle_command("RESET_LEARNING")
    with sim.lock:
        if sim.learner:
            sim.learner.reset_learning()
    return jsonify({"status": "ok", "message": "Learned parameters reset to baseline."})


@app.route("/api/memory/consolidate", methods=["POST"])
def api_memory_consolidate():
    global ros_bridge, sim
    if HAVE_ROS2 and ros_bridge is not None:
        ros_bridge.send_lifecycle_command("CONSOLIDATE")
    with sim.lock:
        if sim.learner:
            sim.learner.episodic_memory.consolidate()
    return jsonify({"status": "ok", "message": "Episodic memory consolidation commanded."})


@app.route("/api/connectome/spatial_layout", methods=["GET"])
def api_connectome_spatial_layout():
    """Returns biological landmark neurons and neuropil region metadata."""
    global cached_spatial_atlas, cached_key_neurons
    if cached_spatial_atlas is None:
        coords, key_neurons = generate_spatial_atlas()
        cached_spatial_atlas = coords
        cached_key_neurons = key_neurons

    neuropils_list = [
        {"id": nid, "name": info["name"], "code": info["code"], "color": info["color"]}
        for nid, info in NEUROPILS.items()
    ] if isinstance(NEUROPILS, dict) else NEUROPILS

    return jsonify({
        "total_neurons": len(cached_spatial_atlas),
        "neuropils": neuropils_list,
        "key_neurons": cached_key_neurons,
        "dataset": "Drosophila Melanogaster Connectome v783 (Shiu et al. 2024 / FlyWire MaleCNS)",
        "spatial_projection": {
            "type": "Topologically Partitioned by Anatomical Neuropil",
            "bounding_box": {
                "x_range": [-160.0, 160.0],
                "y_range": [-110.0, 110.0],
                "z_range": [-90.0, 90.0],
            },
        },
    })


@app.route("/api/connectome/spatial_coords_bin", methods=["GET"])
def api_connectome_spatial_coords_bin():
    """Streams compact Float32 binary buffer for fast WebGL BufferGeometry rendering."""
    global cached_spatial_atlas
    candidates = [
        os.environ.get("FLY_BRAIN_SPATIAL_COORDS_BIN", ""),
        Path(__file__).resolve().parents[1] / "data" / "connectome_spatial_coords.bin",
        Path(__file__).resolve().parents[2] / "fly-brain" / "data" / "connectome_spatial_coords.bin",
        Path(__file__).resolve().parents[2] / "data" / "connectome_spatial_coords.bin",
        Path.cwd() / "data" / "connectome_spatial_coords.bin",
        Path.cwd() / "fly-brain" / "data" / "connectome_spatial_coords.bin",
        Path("/home/om/Desktop/robo_fly/fly-brain/data/connectome_spatial_coords.bin"),
    ]
    for cand in candidates:
        if cand and os.path.exists(str(cand)):
            with open(cand, "rb") as f:
                data = f.read()
            return Response(data, mimetype="application/octet-stream")

    if cached_spatial_atlas is None:
        coords, _ = generate_spatial_atlas()
        cached_spatial_atlas = coords
    return Response(cached_spatial_atlas.tobytes(), mimetype="application/octet-stream")


@app.route("/api/neuron/<identifier>", methods=["GET"])
def api_neuron(identifier):
    ident_str = str(identifier).strip()
    global ros_bridge

    # Check cached ROS inspection
    if HAVE_ROS2 and ros_bridge is not None:
        with ros_bridge.lock:
            if ident_str in ros_bridge.neuron_cache:
                cached = dict(ros_bridge.neuron_cache[ident_str])
                if "flywire_id" in cached:
                    cached["flywire_id"] = str(cached["flywire_id"])
                return jsonify(cached)
            resolved = resolve_neuron_name_or_alias(ident_str)
            if resolved is not None and str(resolved) in ros_bridge.neuron_cache:
                cached = dict(ros_bridge.neuron_cache[str(resolved)])
                if "flywire_id" in cached:
                    cached["flywire_id"] = str(cached["flywire_id"])
                return jsonify(cached)

        is_ros_live = (time.time() - ros_bridge.last_msg_time < 3.0)
        if is_ros_live:
            ros_bridge.query_neuron(ident_str)
            for _ in range(8):
                time.sleep(0.05)
                with ros_bridge.lock:
                    if ident_str in ros_bridge.neuron_cache:
                        cached = dict(ros_bridge.neuron_cache[ident_str])
                        if "flywire_id" in cached:
                            cached["flywire_id"] = str(cached["flywire_id"])
                        return jsonify(cached)
                    resolved = resolve_neuron_name_or_alias(ident_str)
                    if resolved is not None and str(resolved) in ros_bridge.neuron_cache:
                        cached = dict(ros_bridge.neuron_cache[str(resolved)])
                        if "flywire_id" in cached:
                            cached["flywire_id"] = str(cached["flywire_id"])
                        return jsonify(cached)

    # Biological metadata resolution
    val = None
    is_numeric = False
    try:
        val = int(ident_str)
        is_numeric = True
    except ValueError:
        val = resolve_neuron_name_or_alias(ident_str)

    if val is not None and (val in BIOLOGICAL_METADATA or str(val) in BIOLOGICAL_METADATA):
        info = get_neuron_info(val)
        if "flywire_id" in info:
            info["flywire_id"] = str(info["flywire_id"])
        return jsonify(info)

    if is_numeric:
        if val is not None and (val >= 10**9 or len(ident_str) >= 10):
            return jsonify({
                "error": f"FlyWire ID '{ident_str}' not found in the 138,639-neuron connectome dataset or biological metadata.",
                "query": ident_str,
            }), 404
        else:
            return jsonify({
                "error": f"Connectome local matrix index {val} is out of bounds (valid range: 0 to 138638).",
                "query": ident_str,
            }), 404
    else:
        return jsonify({
            "error": f"Neuron '{ident_str}' is neither a recognized biological neuron name/alias, nor a valid local index (0..138638), nor an authenticated FlyWire ID.",
            "query": ident_str,
        }), 404


@app.route("/api/neuron/<identifier>/synapses", methods=["GET"])
def api_neuron_synapses(identifier):
    ident_str = str(identifier).strip()
    global ros_bridge

    # If cached in ros_bridge
    if HAVE_ROS2 and ros_bridge is not None:
        with ros_bridge.lock:
            cached = ros_bridge.neuron_cache.get(ident_str)
            if cached and ("top_presynaptic_inputs" in cached or "presynaptic" in cached):
                pre = cached.get("top_presynaptic_inputs", cached.get("presynaptic", []))
                post = cached.get("top_postsynaptic_targets", cached.get("postsynaptic", []))
                return jsonify({
                    "flywire_id": str(cached.get("flywire_id", ident_str)),
                    "presynaptic": pre,
                    "postsynaptic": post,
                    "total_presynaptic_count": cached.get("total_presynaptic_count", len(pre)),
                    "total_postsynaptic_count": cached.get("total_postsynaptic_count", len(post)),
                })

    # Fallback to biological metadata or query
    val = None
    try:
        val = int(ident_str)
    except ValueError:
        val = resolve_neuron_name_or_alias(ident_str)

    if val is not None and (val in BIOLOGICAL_METADATA or str(val) in BIOLOGICAL_METADATA):
        meta = get_neuron_info(val)
        return jsonify({
            "flywire_id": str(meta.get("flywire_id", val)),
            "name": meta.get("name", "Unknown"),
            "presynaptic": meta.get("top_presynaptic_inputs", []),
            "postsynaptic": meta.get("top_postsynaptic_targets", []),
            "total_presynaptic_count": meta.get("total_presynaptic_count", 0),
            "total_postsynaptic_count": meta.get("total_postsynaptic_count", 0),
        })

    return jsonify({"error": f"Neuron {ident_str} not found", "query": ident_str}), 404


@app.route("/api/ros_telemetry", methods=["GET"])
def api_ros_telemetry():
    global ros_bridge
    if HAVE_ROS2 and ros_bridge is not None:
        with ros_bridge.lock:
            return jsonify(ros_bridge.telemetry)
    return jsonify({"status": "OFFLINE", "is_healthy": False})


def start_server(host: str = "0.0.0.0", port: int = 5000, manage_sim: bool = False, headless: bool = True):
    global lifecycle_supervisor
    if manage_sim and lifecycle_supervisor is not None:
        print("[web_server] Initializing and launching managed simulation stack...")
        lifecycle_supervisor.start_simulation(headless=headless)

    def _sig_handler(sig, frame):
        print("\n[web_server] Clean signal shutdown received. Terminating child simulation stack...")
        if lifecycle_supervisor is not None:
            lifecycle_supervisor.terminate_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"\n=======================================================")
    print(f" DROSOPHILA CONNECTOME ROBOTICS LABORATORY — WEB SERVER")
    print(f" Access live interactive dashboard at:")
    print(f"   http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main():
    parser = argparse.ArgumentParser(description="Drosophila Connectome Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--manage-sim", action="store_true", help="Supervise and launch simulation child process group")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Gazebo simulator headless")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, manage_sim=args.manage_sim, headless=args.headless)


if __name__ == "__main__":
    main()
