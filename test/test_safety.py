"""
Comprehensive Unit and Regression Tests for SafetySupervisor.

Covers all 15 core safety and avoidance scenarios:
  1. Clear path (NOMINAL velocity pass-through)
  2. Deceleration at slowdown distance (front < 1.8m)
  3. Early avoidance steering (front < 1.5m)
  4. Turn-in-place stop (front < 0.85m, vx=0)
  5. Hard stop override (min_dist <= 0.25m)
  6. Left obstacle priority (avoids right)
  7. Right obstacle priority (avoids left)
  8. Blocked sides condition (SAFETY_BLOCKED when both sides blocked)
  9. Avoidance latching (direction maintained even if front opens)
 10. Side clearance exit condition (does not exit until passing side > 0.8m)
 11. Hysteresis counter (requires N=5 cycles before returning to NOMINAL)
 12. Stale sensor watchdog (timeout -> zero velocity)
 13. Brain timeout watchdog (timeout -> zero velocity)
 14. Emergency stop latching and clearing
 15. Invalid / NaN / Inf values rejection
"""

import math
import sys
import time
import unittest
from pathlib import Path

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.safety_supervisor import SafetySupervisor


class TestSafetySupervisor(unittest.TestCase):

    def setUp(self):
        self.safety = SafetySupervisor(
            max_linear_vel=0.4,
            max_angular_vel=1.0,
            max_linear_accel=1.0,
            timeout_sec=0.2,
            obstacle_hard_stop_dist=0.25,
            obstacle_slow_dist=1.8,
            obstacle_turn_dist=0.85,
            obstacle_steer_start_dist=1.5,
            minimum_side_clearance=0.70,
            avoidance_angular_vel=0.50,
            passing_side_clearance=0.80,
            clear_hysteresis_cycles=5,
        )
        self.safety.set_ready(True)
        self.safety.notify_sensor_heartbeat()
        self.safety.notify_brain_heartbeat()

    def test_startup_inactivity(self):
        """Before set_ready(True), commands must be zero."""
        unready_safety = SafetySupervisor()
        vx, wz, status = unready_safety.filter_command(0.3, 0.5)
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(status, "INITIALIZING")

    def test_01_clear_path_nominal(self):
        """Scenario 1: Open arena path allows nominal velocity pass-through."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.1,
            front_obstacle_dist=5.0,
            left_obstacle_dist=5.0,
            right_obstacle_dist=5.0,
            dt=0.5,
        )
        self.assertEqual(status, "NOMINAL")
        self.assertAlmostEqual(vx, 0.3, delta=0.01)
        self.assertAlmostEqual(wz, 0.1, delta=0.01)

    def test_02_slowdown_distance(self):
        """Scenario 2: Obstacle within slowdown zone (front < 1.8m) reduces forward speed."""
        # Baseline speed at 5.0m
        vx_clear, _, _ = self.safety.filter_command(
            0.4, 0.0,
            front_obstacle_dist=5.0, left_obstacle_dist=5.0, right_obstacle_dist=5.0,
            dt=1.0,
        )

        # Move to front=1.7m (slowdown active, but steer start 1.5m not yet reached)
        vx_slow, wz_slow, status = self.safety.filter_command(
            0.4, 0.0,
            front_obstacle_dist=1.7, left_obstacle_dist=5.0, right_obstacle_dist=5.0,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_SLOWDOWN")
        self.assertLess(vx_slow, vx_clear)
        self.assertGreater(vx_slow, 0.0)

    def test_03_early_avoidance_steering(self):
        """Scenario 3: Approaching front <= 1.5m triggers early avoidance turn."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.4, left_obstacle_dist=2.5, right_obstacle_dist=0.9,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_AVOIDANCE")
        self.assertNotEqual(wz, 0.0)
        # Left side is clearer than right side (2.5m vs 0.9m), so turn left (+wz)
        self.assertGreater(wz, 0.0)
        self.assertEqual(self.safety.avoidance_direction, 1)

    def test_04_turn_in_place_stop(self):
        """Scenario 4: Close obstacle (front <= 0.85m) stops forward motion for in-place turn."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=0.75, left_obstacle_dist=2.0, right_obstacle_dist=1.0,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_AVOIDANCE")
        self.assertEqual(vx, 0.0)
        self.assertGreater(abs(wz), 0.0)

    def test_05_hard_stop_override(self):
        """Scenario 5: Obstacle within hard stop distance (<= 0.25m) halts vehicle instantly."""
        vx, wz, status = self.safety.filter_command(
            0.4, 0.5,
            min_obstacle_dist=0.20,
            front_obstacle_dist=0.20,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_OVERRIDE_OBSTACLE")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_06_left_obstacle_priority(self):
        """Scenario 6: Obstacle on left steers right (negative yaw rate)."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.4, left_obstacle_dist=0.8, right_obstacle_dist=2.5,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_AVOIDANCE")
        self.assertEqual(self.safety.avoidance_direction, -1)
        self.assertLess(wz, 0.0)

    def test_07_right_obstacle_priority(self):
        """Scenario 7: Obstacle on right steers left (positive yaw rate)."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.4, left_obstacle_dist=2.5, right_obstacle_dist=0.8,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_AVOIDANCE")
        self.assertEqual(self.safety.avoidance_direction, 1)
        self.assertGreater(wz, 0.0)

    def test_08_blocked_sides_condition(self):
        """Scenario 8: Front obstacle with both sides restricted triggers SAFETY_BLOCKED."""
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.2, left_obstacle_dist=0.45, right_obstacle_dist=0.50,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_BLOCKED")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_09_avoidance_latching_persistent(self):
        """Scenario 9: Direction latches and does not reset if front opens while side is close."""
        # 1. Trigger left turn avoidance (right side closer)
        self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.3, left_obstacle_dist=2.5, right_obstacle_dist=0.9,
            dt=0.1,
        )
        self.assertEqual(self.safety.avoidance_direction, 1)

        # 2. Next cycle: robot turns, front clears to 3.0m, but passing side (right) is 0.60m
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=3.0, left_obstacle_dist=2.5, right_obstacle_dist=0.60,
            dt=0.1,
        )
        self.assertEqual(status, "SAFETY_AVOIDANCE")
        self.assertEqual(self.safety.avoidance_direction, 1)
        self.assertGreater(wz, 0.0)

    def test_10_side_clearance_exit_condition(self):
        """Scenario 10: Avoidance cannot exit while passing side clearance < 0.8m."""
        # Trigger left avoidance
        self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.3, left_obstacle_dist=2.5, right_obstacle_dist=0.9,
            dt=0.1,
        )

        # Front is open (3.0m), but right passing clearance is 0.75m (< 0.80m)
        for _ in range(10):
            vx, wz, status = self.safety.filter_command(
                0.3, 0.0,
                front_obstacle_dist=3.0, left_obstacle_dist=2.0, right_obstacle_dist=0.75,
                dt=0.1,
            )
            self.assertEqual(status, "SAFETY_AVOIDANCE")
            self.assertEqual(self.safety.avoidance_direction, 1)

    def test_11_hysteresis_counter_5_cycles(self):
        """Scenario 11: Clearing avoidance requires exactly 5 consecutive clear cycles."""
        # Trigger avoidance
        self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=1.3, left_obstacle_dist=2.5, right_obstacle_dist=0.9,
            dt=0.1,
        )
        self.assertEqual(self.safety.avoidance_direction, 1)

        # Supply clear front (3.5m > 1.95m) and clear side (1.2m > 0.80m)
        # Cycles 1 to 4 must remain in AVOIDANCE
        for c in range(1, 5):
            vx, wz, status = self.safety.filter_command(
                0.3, 0.0,
                front_obstacle_dist=3.5, left_obstacle_dist=2.0, right_obstacle_dist=1.2,
                dt=0.1,
            )
            self.assertEqual(status, "SAFETY_AVOIDANCE", f"Premature clearance on cycle {c}")
            self.assertEqual(self.safety.clear_consecutive_cycles, c)

        # Cycle 5: Reaches 5 cycles -> clears to NOMINAL
        vx, wz, status = self.safety.filter_command(
            0.3, 0.0,
            front_obstacle_dist=3.5, left_obstacle_dist=2.0, right_obstacle_dist=1.2,
            dt=0.1,
        )
        self.assertEqual(status, "NOMINAL")
        self.assertEqual(self.safety.avoidance_direction, 0)
        self.assertEqual(self.safety.clear_consecutive_cycles, 0)

    def test_12_stale_sensor_watchdog(self):
        """Scenario 12: Stale sensor data triggers watchdog halt."""
        time.sleep(0.25)  # Exceeds timeout_sec=0.2s
        vx, wz, status = self.safety.filter_command(0.3, 0.0)
        self.assertEqual(status, "FAULT_STALE_SENSOR")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_13_brain_timeout_watchdog(self):
        """Scenario 13: Stale brain updates trigger watchdog halt."""
        self.safety.notify_sensor_heartbeat()
        time.sleep(0.25)
        self.safety.notify_sensor_heartbeat()  # sensor refreshed, brain not
        vx, wz, status = self.safety.filter_command(0.3, 0.0)
        self.assertEqual(status, "FAULT_BRAIN_TIMEOUT")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_14_emergency_stop_latch_and_clear(self):
        """Scenario 14: Emergency stop latches zero velocity until explicitly cleared."""
        self.safety.trigger_emergency_stop(True)
        vx, wz, status = self.safety.filter_command(0.3, 0.5)
        self.assertEqual(status, "EMERGENCY_STOP")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

        # Clear emergency stop
        self.safety.trigger_emergency_stop(False)
        self.safety.notify_sensor_heartbeat()
        self.safety.notify_brain_heartbeat()
        vx, wz, status = self.safety.filter_command(0.3, 0.0, front_obstacle_dist=5.0, dt=0.5)
        self.assertEqual(status, "NOMINAL")

    def test_15_invalid_nan_inf_values(self):
        """Scenario 15: Non-finite and negative distance inputs are safely rejected."""
        # NaN linear velocity
        vx, wz, status = self.safety.filter_command(float('nan'), 0.0)
        self.assertEqual(status, "FAULT_INVALID_VALUES")
        self.assertEqual(vx, 0.0)

        # Inf distance
        vx, wz, status = self.safety.filter_command(0.3, 0.0, min_obstacle_dist=float('inf'))
        self.assertEqual(status, "FAULT_INVALID_VALUES")
        self.assertEqual(vx, 0.0)

        # Negative distance
        vx, wz, status = self.safety.filter_command(0.3, 0.0, front_obstacle_dist=-1.0)
        self.assertEqual(status, "FAULT_INVALID_VALUES")
        self.assertEqual(vx, 0.0)


if __name__ == '__main__':
    unittest.main()
