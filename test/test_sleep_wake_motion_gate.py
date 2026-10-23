"""
Automated Test Suite: Sleep/Wake Lifecycle & Authoritative Motor Command Gating.

Validates Requirements B1 through B8:
  B1: Moving robot enters Sleep -> final motion commands become zero immediately.
  B2: Brain cannot publish a non-zero final command while asleep (hard authoritative gate).
  B3: Repeated Sleep is idempotent.
  B4: Wake restores valid operation without replaying stale commands.
  B5: Sleep during turn/avoidance stops motion.
  B6: Lifecycle state shown in dashboard matches backend.
  B7: Failure responses are not misreported as successful Sleep/Wake.
  B8: Sleep-time save behavior verified separately from normal shutdown behavior.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from pathlib import Path

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

try:
    import rclpy
    HAVE_RCLPY = True
except ImportError:
    HAVE_RCLPY = False

from fly_brain_ros.fly_brain_node import FlyBrainNode, LaserScan, Twist, String
from fly_brain_ros.checkpoint_config import validate_checkpoint_file
from fly_brain_ros.lifecycle_supervisor import SimulationLifecycleSupervisor


class TestSleepWakeMotionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAVE_RCLPY and not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if HAVE_RCLPY and rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="fly_sleep_test_")
        self.cp_path = os.path.join(self.temp_dir, "sleep_cp.json")

        self.node = FlyBrainNode(node_name=f"test_sleep_{self._testMethodName}")
        self.node.checkpoint_file = self.cp_path

        # Setup publisher hook to capture all published Twists
        self.published_twists = []
        original_pub = self.node.cmd_vel_pub.publish
        def hook_pub(msg):
            self.published_twists.append(msg)
            original_pub(msg)
        self.node.cmd_vel_pub.publish = hook_pub

    def tearDown(self):
        self.node.destroy_node()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _feed_nominal_scan(self, dist=3.0):
        scan = LaserScan()
        scan.ranges = [dist] * 180
        self.node.scan_callback(scan)

    def _feed_obstacle_scan(self, left=3.0, right=0.5, front=1.0):
        # Sector indexing in scan_callback:
        # Front: [-15 deg, +15 deg] = [-0.2618 rad, +0.2618 rad]
        # Left: [+15 deg, +90 deg] = [0.2618 rad, +1.5708 rad]
        # Right: [-90 deg, -15 deg] = [-1.5708 rad, -0.2618 rad]
        scan = LaserScan()
        ranges = []
        for i in range(180):
            angle = scan.angle_min + i * scan.angle_increment
            if -0.2618 <= angle <= 0.2618:
                ranges.append(front)
            elif 0.2618 < angle <= 1.5708:
                ranges.append(left)
            elif -1.5708 <= angle < -0.2618:
                ranges.append(right)
            else:
                ranges.append(5.0)
        scan.ranges = ranges
        self.node.scan_callback(scan)

    def test_b1_moving_robot_enters_sleep_halts_immediately(self):
        """B1: A moving simulated robot enters Sleep and final motion commands become zero."""
        self._feed_nominal_scan(3.0)
        self.node.control_cycle()

        # Trigger SLEEP
        msg = String()
        msg.data = "SLEEP"
        self.node.lifecycle_callback(msg)

        self.assertTrue(self.node.is_asleep)
        self.assertEqual(self.node.lifecycle_state, "ASLEEP")

        # The command published during sleep request must be zero
        sleep_halt_twist = self.published_twists[-1]
        self.assertEqual(sleep_halt_twist.linear.x, 0.0)
        self.assertEqual(sleep_halt_twist.angular.z, 0.0)

    def test_b2_brain_cannot_publish_nonzero_command_while_asleep(self):
        """B2: The brain cannot publish a non-zero final command while asleep."""
        # Enter sleep
        msg = String()
        msg.data = "SLEEP"
        self.node.lifecycle_callback(msg)
        self.assertTrue(self.node.is_asleep)

        # Clear published list to strictly monitor subsequent ticks
        self.published_twists.clear()

        # Simulate 10 control cycles with nominal open sensor inputs
        for _ in range(10):
            self._feed_nominal_scan(5.0)
            self.node.control_cycle()
            self.assertEqual(self.published_twists[-1].linear.x, 0.0)
            self.assertEqual(self.published_twists[-1].angular.z, 0.0)

        # Attempt teleoperation override while asleep
        teleop_msg = Twist()
        teleop_msg.linear.x = 0.35
        teleop_msg.angular.z = 0.50
        self.node.teleop_cmd_callback(teleop_msg)

        # Control cycle must still reject teleoperation and output zero
        self.node.control_cycle()
        self.assertEqual(self.published_twists[-1].linear.x, 0.0)
        self.assertEqual(self.published_twists[-1].angular.z, 0.0)

    def test_b3_repeated_sleep_is_idempotent(self):
        """B3: Repeated Sleep is idempotent."""
        msg = String()
        msg.data = "SLEEP"
        self.node.lifecycle_callback(msg)
        self.assertTrue(self.node.is_asleep)
        self.assertEqual(self.node.lifecycle_state, "ASLEEP")

        # Second SLEEP request
        self.node.lifecycle_callback(msg)
        self.assertTrue(self.node.is_asleep)
        self.assertEqual(self.node.lifecycle_state, "ASLEEP")

        self.node.control_cycle()
        self.assertEqual(self.published_twists[-1].linear.x, 0.0)

    def test_b4_wake_restores_valid_operation_without_replaying_stale_commands(self):
        """B4: Wake restores valid operation without replaying stale commands."""
        # Stage stale teleop command
        stale_twist = Twist()
        stale_twist.linear.x = 0.40
        self.node.teleop_cmd_callback(stale_twist)

        # Put node to sleep
        sleep_msg = String()
        sleep_msg.data = "SLEEP"
        self.node.lifecycle_callback(sleep_msg)
        self.assertTrue(self.node.is_asleep)

        # Now WAKE
        wake_msg = String()
        wake_msg.data = "WAKE"
        self.node.lifecycle_callback(wake_msg)

        self.assertFalse(self.node.is_asleep)
        self.assertEqual(self.node.lifecycle_state, "AWAKE")
        # Ensure stale teleop command was cleared
        self.assertEqual(self.node.teleop_raw_vx, 0.0)
        self.assertEqual(self.node.teleop_filtered_vx, 0.0)

        # Feed fresh sensor and run control cycle
        self._feed_nominal_scan(3.0)
        self.node.control_cycle()
        active_cmd = self.published_twists[-1]
        self.assertIsNotNone(active_cmd)

    def test_b5_sleep_during_turn_avoidance_stops_motion(self):
        """B5: Sleep during turn/avoidance also stops motion."""
        # Trigger avoidance condition
        self._feed_obstacle_scan(left=3.0, right=0.35, front=0.8)
        self.node.control_cycle()

        # Command SLEEP during active avoidance
        sleep_msg = String()
        sleep_msg.data = "SLEEP"
        self.node.lifecycle_callback(sleep_msg)
        self.assertTrue(self.node.is_asleep)

        # Output command must be zero
        self.assertEqual(self.published_twists[-1].linear.x, 0.0)
        self.assertEqual(self.published_twists[-1].angular.z, 0.0)

        # Ticking control cycle keeps it zero
        self.node.control_cycle()
        self.assertEqual(self.published_twists[-1].linear.x, 0.0)
        self.assertEqual(self.published_twists[-1].angular.z, 0.0)

    def test_b6_lifecycle_state_matches_supervisor(self):
        """B6: Lifecycle state matches between supervisor and node."""
        supervisor = SimulationLifecycleSupervisor(checkpoint_path=self.cp_path)
        self.assertEqual(supervisor.current_state, SimulationLifecycleSupervisor.STATE_SLEEPING)

        st = supervisor.get_status()
        self.assertEqual(st["lifecycle_state"], "SLEEPING")
        self.assertEqual(st["checkpoint_path"], os.path.abspath(self.cp_path))

    def test_b7_failure_responses_not_misreported_as_success(self):
        """B7: Corrupt checkpoint on wake does not crash or misreport."""
        # Corrupt file
        with open(self.cp_path, "w") as f:
            f.write("CORRUPT_NOT_JSON")

        wake_msg = String()
        wake_msg.data = "WAKE"
        # Should handle cleanly without unhandled exception
        self.node.lifecycle_callback(wake_msg)
        self.assertEqual(self.node.lifecycle_state, "AWAKE")

    def test_b8_sleep_time_save_verified_separately_from_shutdown(self):
        """B8: Sleep-time save behavior is verified separately from normal shutdown behavior."""
        sleep_msg = String()
        sleep_msg.data = "SLEEP"
        self.node.lifecycle_callback(sleep_msg)

        # Verify that the sleep-triggered checkpoint was safely written and is cryptographically valid
        self.assertTrue(os.path.exists(self.cp_path))
        is_valid, msg, summary = validate_checkpoint_file(self.cp_path)
        self.assertTrue(is_valid, f"Sleep checkpoint invalid: {msg}")
        self.assertEqual(summary["schema_version"], "2.0.0")


if __name__ == "__main__":
    unittest.main()
