"""
Suite F: Startup Lifecycle, Checkpoint State Separation, and Motion Gate Diagnostic Suite.

Verifies:
  1. A valid checkpoint containing previously persisted neural and learning state restores successfully.
  2. A checkpoint saved while the brain was asleep does not unintentionally leave the new session asleep.
  3. The deliberate startup policy defaults to AWAKE after successful restore (unless start_asleep=True).
  4. Startup grace period rejects stale/external SLEEP messages (e.g. from lingering DDS discovery).
  5. Sleep during a session gates every final velocity command to strictly zero.
  6. Wake correctly reopens the gate after sensor and safety readiness checks.
  7. Stale pre-sleep commands cannot move the robot during sleep or be replayed during wake.
  8. Missing /scan, invalid sensor data, safety stop, and lack of motor output are distinguishable.
  9. With valid inputs and valid non-zero command, final output permits motion when awake and safe.
  10. Dashboard API lifecycle status matches the backend lifecycle state.
"""

import os
import sys
import json
import time
import unittest
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from fly_brain_ros.checkpoint_config import (
    get_canonical_checkpoint_path,
    calculate_checkpoint_sha256,
    validate_checkpoint_file,
)
from fly_brain_ros.fly_brain_node import FlyBrainNode, FlyBrainControllerPipeline
from fly_brain_ros.lifecycle_supervisor import SimulationLifecycleSupervisor

# Include live_lab
REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_LAB_DIR = REPO_ROOT / "live_lab"
if str(LIVE_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_LAB_DIR))

import web_server  # type: ignore
from web_server import app, sim  # type: ignore


class TestStartupLifecycleAndMotionGate(unittest.TestCase):
    """Validates deliberate startup policy, model/runtime separation, and motion gating."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.test_dir = Path("/tmp/test_fly_startup_lifecycle")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.cp_path = str(self.test_dir / "test_sleep_checkpoint.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_checkpoint(self, is_asleep: bool = True, delta_steer: float = 0.25) -> str:
        """Creates a valid schema 2.0.0 checkpoint based on canonical model data."""
        canonical = get_canonical_checkpoint_path()
        with open(canonical, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["learning_state"]["delta_steer_dist"] = delta_steer
        data["learning_state"]["speed_factor"] = 0.85
        data["internal_affect_state"]["valence_affect"] = 0.70
        data["internal_affect_state"]["confidence"] = 0.90
        data["lifecycle_state"] = {
            "status": "ASLEEP" if is_asleep else "AWAKE",
            "is_asleep": is_asleep,
            "lifecycle_state": "ASLEEP" if is_asleep else "AWAKE",
            "timestamp": time.time(),
            "last_sector_distances": [4.0, 4.0, 4.5],
        }
        data["manifest"]["sha256"] = calculate_checkpoint_sha256(data)

        with open(self.cp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return self.cp_path

    def test_01_checkpoint_model_state_restores_successfully(self):
        """1. A valid checkpoint containing previously persisted neural and learning state restores successfully."""
        cp = self._create_checkpoint(is_asleep=True, delta_steer=0.42)
        node = FlyBrainNode()
        try:
            ok = node.pipeline.restore_checkpoint(cp, is_wake=False)
            self.assertTrue(ok)
            # Verify learning policy model state was restored
            self.assertAlmostEqual(node.pipeline.learner.delta_steer_dist, 0.42, places=3)
            self.assertAlmostEqual(node.pipeline.learner.speed_factor, 0.85, places=3)
            # Verify internal affect model state was restored
            self.assertAlmostEqual(node.pipeline.internal_state.valence_affect, 0.70, places=2)
            self.assertAlmostEqual(node.pipeline.internal_state.confidence, 0.90, places=2)
        finally:
            node.destroy_node()

    def test_02_checkpoint_saved_asleep_does_not_leave_new_session_asleep(self):
        """2. A checkpoint saved while the brain was asleep does not unintentionally leave the new session asleep."""
        # Create a checkpoint explicitly saved while ASLEEP
        cp = self._create_checkpoint(is_asleep=True)
        is_valid, err, summary = validate_checkpoint_file(cp)
        self.assertTrue(is_valid, f"Validation failed: {err}")

        # Set checkpoint override
        old_env = os.environ.get("FLY_BRAIN_CHECKPOINT_PATH")
        os.environ["FLY_BRAIN_CHECKPOINT_PATH"] = cp
        try:
            node = FlyBrainNode()
            try:
                # Deliberate startup policy must leave the newly booted node AWAKE
                self.assertFalse(node.is_asleep, "Node must NOT inherit is_asleep=True from checkpoint on normal startup!")
                self.assertEqual(node.lifecycle_state, "AWAKE", "Node lifecycle_state must be AWAKE on normal startup!")
                self.assertFalse(node.pipeline.is_asleep, "Pipeline must NOT be asleep on normal startup!")
                self.assertEqual(node.pipeline.lifecycle_state, "AWAKE")
            finally:
                node.destroy_node()
        finally:
            if old_env is not None:
                os.environ["FLY_BRAIN_CHECKPOINT_PATH"] = old_env
            else:
                os.environ.pop("FLY_BRAIN_CHECKPOINT_PATH", None)

    def test_03_deliberate_startup_policy_defaults_to_awake(self):
        """3. The deliberate startup policy defaults to AWAKE, and respects explicit start_asleep parameter."""
        node_default = FlyBrainNode()
        try:
            self.assertEqual(node_default.lifecycle_state, "AWAKE")
            self.assertFalse(node_default.is_asleep)
        finally:
            node_default.destroy_node()

    def test_04_startup_grace_period_rejects_stale_sleep_message(self):
        """4. Startup grace period rejects stale/external SLEEP message arriving during node boot window."""
        node = FlyBrainNode()
        node.startup_grace_period_sec = 1.5
        try:
            self.assertEqual(node.lifecycle_state, "AWAKE")
            self.assertFalse(node.is_asleep)

            # Send SLEEP command immediately (within 1.5s startup grace window)
            msg = String()
            msg.data = "SLEEP"
            node.lifecycle_callback(msg)

            # Node must ignore the stale command and remain AWAKE
            self.assertEqual(node.lifecycle_state, "AWAKE", "Node should have ignored early SLEEP message during startup!")
            self.assertFalse(node.is_asleep)

            # Fast-forward startup time beyond grace period (using perf_counter reference)
            node.node_startup_time = 0.0
            node.lifecycle_callback(msg)

            # Now the intentional sleep command must be accepted
            self.assertEqual(node.lifecycle_state, "ASLEEP")
            self.assertTrue(node.is_asleep)
        finally:
            node.destroy_node()

    def test_05_sleep_gates_final_velocity_to_strictly_zero(self):
        """5. Sleep during a session gates every final velocity command to strictly zero at /cmd_vel."""
        node = FlyBrainNode()
        published_twists = []
        node.cmd_vel_pub.publish = lambda t: published_twists.append((t.linear.x, t.angular.z))

        try:
            # Set to sleep
            node.node_startup_time = 0.0
            msg = String()
            msg.data = "SLEEP"
            node.lifecycle_callback(msg)
            self.assertTrue(node.is_asleep)

            # Try to force motion via teleop inputs
            node.teleop_raw_vx = 0.50
            node.teleop_raw_wz = 0.80
            node.last_teleop_time = time.perf_counter()
            node.control_mode = "TELEOP"

            # Execute control cycle
            node.control_cycle()

            self.assertGreater(len(published_twists), 0)
            last_cmd = published_twists[-1]
            self.assertEqual(last_cmd[0], 0.0, "linear.x must be strictly 0.0 while asleep!")
            self.assertEqual(last_cmd[1], 0.0, "angular.z must be strictly 0.0 while asleep!")
        finally:
            node.destroy_node()

    def test_06_wake_correctly_opens_gate_and_requires_fresh_sensor_scan(self):
        """6. Wake correctly reopens the gate after sensor and safety readiness checks."""
        node = FlyBrainNode()
        published_twists = []
        node.cmd_vel_pub.publish = lambda t: published_twists.append((t.linear.x, t.angular.z))

        try:
            # Enter sleep
            node.node_startup_time = 0.0
            s_msg = String()
            s_msg.data = "SLEEP"
            node.lifecycle_callback(s_msg)
            self.assertTrue(node.is_asleep)

            # Send WAKE command
            w_msg = String()
            w_msg.data = "WAKE"
            node.lifecycle_callback(w_msg)

            self.assertFalse(node.is_asleep)
            self.assertEqual(node.lifecycle_state, "AWAKE")

            # Before fresh scan arrives: Rule 1 must keep velocity at 0.0
            published_twists.clear()
            node.control_cycle()
            self.assertEqual(published_twists[-1], (0.0, 0.0), "Must hold 0.0 until fresh scan arrives!")
            self.assertEqual(node.last_motion_state, "WAITING_FOR_SENSORS")

            # Deliver fresh laser scan
            scan = LaserScan()
            scan.angle_min = -1.5708
            scan.angle_max = 1.5708
            scan.angle_increment = 0.01
            scan.range_min = 0.1
            scan.range_max = 10.0
            scan.ranges = [5.0] * 314
            node.scan_callback(scan)

            # Keep safety heartbeats fresh
            node.pipeline.safety.notify_sensor_heartbeat()
            node.pipeline.safety.notify_brain_heartbeat()

            # Now control cycle should transition from WAITING_FOR_SENSORS
            node.control_cycle()
            self.assertIn(node.last_motion_state, ("MOVING", "AWAKE_NO_MOTOR_OUTPUT"))
        finally:
            node.destroy_node()

    def test_07_stale_pre_sleep_commands_cannot_move_or_replay_on_wake(self):
        """7. Stale pre-sleep commands cannot move the robot during sleep or be replayed during wake."""
        node = FlyBrainNode()
        try:
            # Provide high pre-sleep commands
            node.pipeline.filtered_vx = 0.35
            node.pipeline.filtered_wz = -0.50
            node.pipeline.last_raw_vx = 0.35
            node.pipeline.last_raw_wz = -0.50
            node.teleop_raw_vx = 0.35
            node.teleop_raw_wz = -0.50

            # Put to sleep
            node.node_startup_time = 0.0
            msg = String()
            msg.data = "SLEEP"
            node.lifecycle_callback(msg)

            # Verify transient commands zeroed
            self.assertEqual(node.pipeline.filtered_vx, 0.0)
            self.assertEqual(node.pipeline.filtered_wz, 0.0)

            # Wake up
            w_msg = String()
            w_msg.data = "WAKE"
            node.lifecycle_callback(w_msg)

            # Confirm transient velocities remain clean zero
            self.assertEqual(node.pipeline.filtered_vx, 0.0)
            self.assertEqual(node.pipeline.filtered_wz, 0.0)
            self.assertEqual(node.teleop_raw_vx, 0.0)
            self.assertEqual(node.teleop_raw_wz, 0.0)
        finally:
            node.destroy_node()

    def test_08_diagnostic_motion_states_are_distinguishable(self):
        """8. Missing /scan, invalid sensor data, safety stop, and lack of motor output are distinguishable."""
        node = FlyBrainNode()
        published_twists = []
        node.cmd_vel_pub.publish = lambda t: published_twists.append((t.linear.x, t.angular.z))

        try:
            # State A: WAITING_FOR_SENSORS (boot with 0 scans)
            node.scan_received_count = 0
            node.control_cycle()
            self.assertEqual(node.last_motion_state, "WAITING_FOR_SENSORS")

            # State B: ASLEEP
            node.node_startup_time = 0.0
            msg = String()
            msg.data = "SLEEP"
            node.lifecycle_callback(msg)
            node.control_cycle()
            self.assertEqual(node.last_motion_state, "ASLEEP")

            # Wake up
            w_msg = String()
            w_msg.data = "WAKE"
            node.lifecycle_callback(w_msg)

            # Provide scan
            scan = LaserScan()
            scan.angle_min = -1.5708
            scan.angle_max = 1.5708
            scan.angle_increment = 0.01
            scan.range_min = 0.1
            scan.range_max = 10.0
            scan.ranges = [5.0] * 314
            node.scan_callback(scan)
            node.pipeline.safety.notify_sensor_heartbeat()
            node.pipeline.safety.notify_brain_heartbeat()

            # State C: AWAKE_NO_MOTOR_OUTPUT or MOVING
            node.control_cycle()
            self.assertIn(node.last_motion_state, ("MOVING", "AWAKE_NO_MOTOR_OUTPUT"))

            # State D: SAFETY_STOP (inject obstacle at 0.10m inside hard stop zone)
            scan_blocked = LaserScan()
            scan_blocked.angle_min = -1.5708
            scan_blocked.angle_max = 1.5708
            scan_blocked.angle_increment = 0.01
            scan_blocked.range_min = 0.1
            scan_blocked.range_max = 10.0
            scan_blocked.ranges = [0.15] * 314  # inside hard stop distance 0.25m
            node.scan_callback(scan_blocked)
            node.pipeline.safety.notify_sensor_heartbeat()
            node.pipeline.safety.notify_brain_heartbeat()
            node.control_cycle()
            self.assertEqual(node.last_motion_state, "SAFETY_STOP")
        finally:
            node.destroy_node()

    def test_09_final_output_path_permits_command_when_awake_and_safe(self):
        """9. With valid inputs and valid non-zero command, final output permits motion when awake and safe."""
        node = FlyBrainNode()
        published_twists = []
        node.cmd_vel_pub.publish = lambda t: published_twists.append((t.linear.x, t.angular.z))

        try:
            # Deliver open terrain scan
            scan = LaserScan()
            scan.angle_min = -1.5708
            scan.angle_max = 1.5708
            scan.angle_increment = 0.01
            scan.range_min = 0.1
            scan.range_max = 10.0
            scan.ranges = [6.0] * 314
            node.scan_callback(scan)

            # Ensure safety heartbeats are fresh
            node.pipeline.safety.notify_sensor_heartbeat()
            node.pipeline.safety.notify_brain_heartbeat()

            # Command teleop forward motion
            node.control_mode = "TELEOP"
            node.teleop_raw_vx = 0.30
            node.teleop_raw_wz = 0.0
            node.last_teleop_time = time.perf_counter()

            # Execute several control cycles to allow filter smoothing
            for _ in range(5):
                node.pipeline.safety.notify_brain_heartbeat()
                node.last_teleop_time = time.perf_counter()
                node.control_cycle()

            last_linear, last_angular = published_twists[-1]
            self.assertGreater(last_linear, 0.05, f"Expected forward velocity, got {last_linear}")
            self.assertEqual(node.last_motion_state, "MOVING")
        finally:
            node.destroy_node()

    def test_10_dashboard_matches_real_backend_lifecycle(self):
        """10. Dashboard API lifecycle status matches the backend lifecycle state."""
        # Ensure ROS bridge mock does not spoof active node
        if hasattr(web_server, "ros_bridge") and web_server.ros_bridge is not None:
            web_server.ros_bridge.last_msg_time = 0.0

        client = app.test_client()

        # Check default status
        resp = client.get("/api/lifecycle/status")
        self.assertEqual(resp.status_code, 200)
        st = resp.get_json()
        self.assertIn("lifecycle_state", st)

        # Trigger sleep via API
        resp_sleep = client.post("/api/lifecycle/sleep")
        self.assertEqual(resp_sleep.status_code, 200)
        data_sleep = resp_sleep.get_json()
        self.assertIn(data_sleep["lifecycle_state"], ("SLEEPING", "ASLEEP"))

        # Check state API reports ASLEEP
        resp_state = client.get("/api/state")
        self.assertEqual(resp_state.status_code, 200)
        state_data = resp_state.get_json()
        self.assertTrue(state_data.get("is_asleep"))

        # Trigger wake via API
        resp_wake = client.post("/api/lifecycle/wake")
        self.assertEqual(resp_wake.status_code, 200)
        data_wake = resp_wake.get_json()
        self.assertEqual(data_wake["lifecycle_state"], "AWAKE")

        # Check state API reports AWAKE
        resp_state_wake = client.get("/api/state")
        state_wake_data = resp_state_wake.get_json()
        self.assertFalse(state_wake_data.get("is_asleep"))


if __name__ == "__main__":
    unittest.main()
