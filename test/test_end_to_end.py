"""
End-to-End Closed-Loop Integration Test for Fly Brain Robot Control.

Demonstrates the complete sequence:
1. Environment raycasts LiDAR sensor scan.
2. SensoryEncoder translates ranges to Poisson rates for LC4 & P9 neurons.
3. Persistent PyTorch LIF brain advances biological simulation.
4. MotorDecoder computes linear/angular velocities and 4-wheel speeds.
5. SafetySupervisor validates constraints.
6. MockFourWheelChassis integrates wheel speeds and updates robot pose.
"""

import os
import sys
import unittest
from pathlib import Path
from time import perf_counter as time

# Include package in path
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.brain_engine import FlyBrainEngine
from fly_brain_ros.sensory_encoder import SensoryEncoder
from fly_brain_ros.motor_decoder import MotorDecoder
from fly_brain_ros.safety_supervisor import SafetySupervisor
from fly_brain_ros.mock_chassis import MockFourWheelChassis


class TestEndToEndClosedLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=== [SETUP] Initializing Whole-Brain Fly Connectome for Closed-Loop Test ===")
        t0 = time()
        candidates = [
            os.environ.get("FLY_BRAIN_DATA_DIR", ""),
            pkg_root / "data",
            pkg_root.parent / "fly-brain" / "data",
            pkg_root.parent / "data",
            Path.cwd() / "data",
            Path.cwd() / "fly-brain" / "data",
            Path("/home/om/Desktop/robo_fly/fly-brain/data"),
        ]
        data_dir = None
        for cand in candidates:
            if cand and os.path.exists(str(cand)):
                data_dir = Path(cand)
                break
        if data_dir is None:
            data_dir = pkg_root / "data"

        comp_path = str(data_dir / "2025_Completeness_783.csv")
        conn_path = str(data_dir / "2025_Connectivity_783.parquet")
        cache_dir = str(data_dir)

        cls.engine = FlyBrainEngine(
            comp_path=comp_path,
            conn_path=conn_path,
            cache_dir=cache_dir,
            dt_ms=0.1,
            device='cpu'
        )
        init_duration = cls.engine.initialize()
        print(f"=== [SETUP] Initialized {cls.engine.num_neurons} neurons in {init_duration:.2f}s ===")

        cls.encoder = SensoryEncoder(safe_dist=1.5, crit_dist=0.35, max_loom_rate=200.0, nominal_fwd_rate=100.0)
        cls.decoder = MotorDecoder(kv=0.02, kw=0.03, krev=0.02, max_v=0.4, max_w=1.0, wheel_separation=0.36)
        cls.safety = SafetySupervisor(max_linear_vel=0.4, max_angular_vel=1.0, timeout_sec=1.0)
        cls.chassis = MockFourWheelChassis(x=0.0, y=0.0, yaw=0.0, track_width=0.36)

    def setUp(self):
        self.engine.reset_state()
        self.safety.set_ready(True)
        self.safety.notify_sensor_heartbeat()
        self.safety.notify_brain_heartbeat()

    def test_step_1_to_8_closed_loop_execution(self):
        """Executes full 5-cycle closed-loop simulation loop and records telemetry."""
        print("\n--- Starting Closed-Loop Simulation Sequence ---")
        history = []

        # Run 5 control cycles (each cycle = 10 biological steps = 1.0 ms brain time)
        for cycle in range(5):
            t_cycle_start = time()

            # Step 1: Sensor generates LiDAR scan in virtual world
            scan = self.chassis.generate_laserscan(num_rays=180, fov_rad=3.14159)
            self.safety.notify_sensor_heartbeat()

            # Step 2: Encoder produces Poisson rates
            rates_dict = self.encoder.encode_laserscan(
                scan['ranges'], scan['angle_min'], scan['angle_increment']
            )
            min_dist = min(scan['ranges'])

            # Step 3: Neural runtime accepts input
            self.engine.set_rates_by_id(rates_dict)

            # Step 4: Neural simulation advances
            steps_per_cycle = 10
            spikes = self.engine.step(num_steps=steps_per_cycle)
            self.safety.notify_brain_heartbeat()

            # Step 5: Decoder observes descending motor spikes
            motor_counts = {
                nid: float(spikes[self.engine.flyid2i[nid]])
                for nid in self.decoder.monitored_ids
                if nid in self.engine.flyid2i
            }
            window_sec = (steps_per_cycle * self.engine.dt_ms) / 1000.0
            raw_cmd = self.decoder.decode_spikes(motor_counts, window_sec=window_sec)

            # Step 6: Safety supervisor validates command
            safe_vx, safe_wz, status = self.safety.filter_command(
                raw_cmd['linear_x'], raw_cmd['angular_z'], min_obstacle_dist=min_dist
            )

            # Step 7: Produce independent left and right wheel commands
            b = self.chassis.b
            v_l = safe_vx - (safe_wz * b / 2.0)
            v_r = safe_vx + (safe_wz * b / 2.0)

            # Step 8: Apply command to 4-wheel robot chassis and step physics
            self.chassis.set_cmd_vel(safe_vx, safe_wz)
            pose = self.chassis.step_physics(dt=0.1)  # 100ms real-world control timestep

            cycle_time = time() - t_cycle_start

            record = {
                'cycle': cycle + 1,
                'min_dist': min_dist,
                'status': status,
                'safe_vx': safe_vx,
                'safe_wz': safe_wz,
                'v_left_wheels': v_l,
                'v_right_wheels': v_r,
                'robot_x': pose['x'],
                'robot_y': pose['y'],
                'robot_yaw': pose['yaw'],
                'cycle_latency_ms': cycle_time * 1000.0
            }
            history.append(record)

            print(f"Cycle {cycle+1:02d}: Dist={min_dist:.2f}m | Status={status} | "
                  f"Vx={safe_vx:.3f}m/s, Wz={safe_wz:.3f}rad/s | "
                  f"LeftWheels={v_l:.3f}m/s, RightWheels={v_r:.3f}m/s | "
                  f"Pose=({pose['x']:.3f}, {pose['y']:.3f}) | "
                  f"Latency={cycle_time*1000.0:.1f}ms")

        # Verifications
        self.assertEqual(len(history), 5)
        for h in history:
            self.assertIn(h['status'], ("NOMINAL", "SAFETY_SLOWDOWN"))
            self.assertTrue(math_isfinite(h['robot_x']))
            self.assertTrue(math_isfinite(h['robot_y']))
            self.assertGreater(h['cycle_latency_ms'], 0.0)


def math_isfinite(val):
    import math
    return not (math.isnan(val) or math.isinf(val))


if __name__ == '__main__':
    unittest.main()
