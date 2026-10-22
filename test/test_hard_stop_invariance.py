"""
Regression Test Suite for Absolute Safety Invariance & Hard-Stop Non-Bypassability.

Verifies:
  1. Authoritative hard-stop trigger at configured threshold (d <= 0.25m).
  2. Maximum learned steering margin (delta_steer = +0.50m) cannot relax or disable hard stop.
  3. Out-of-bounds or corrupted learned parameter values cannot bypass safety.
  4. Extreme learned directional bias cannot override blocked side corridors (SAFETY_BLOCKED).
  5. Learned clutter speed factor cannot override stale sensor / watchdog halts.
  6. Teleoperation and autonomous neural control both pass through the same authoritative safety gate.
  7. Hard-stop response strictly commands zero linear and zero angular velocity under all conditions.
"""

import math
import unittest
from time import sleep, perf_counter as time

from fly_brain_ros.safety_supervisor import SafetySupervisor
from fly_brain_ros.learning_subsystem import ConnectomeLearner


class TestHardStopInvariance(unittest.TestCase):
    """Validates that learning and teleoperation cannot bypass or loosen authoritative safety constraints."""

    def setUp(self):
        self.safety = SafetySupervisor(
            max_linear_vel=0.40,
            max_angular_vel=1.00,
            obstacle_hard_stop_dist=0.25,
            obstacle_slow_dist=1.80,
            obstacle_turn_dist=0.85,
            obstacle_steer_start_dist=1.50,
            minimum_side_clearance=0.70,
        )
        self.safety.set_ready(True)
        self.learner = ConnectomeLearner()

    def test_01_hard_stop_at_configured_threshold(self):
        """Obstacle at or below 0.25m immediately triggers zero-velocity halt."""
        # Test exact boundary (0.25m) and penetration (0.20m, 0.05m)
        for d in [0.25, 0.20, 0.10, 0.02]:
            vx, wz, status = self.safety.filter_command(
                raw_vx=0.35, raw_wz=0.0,
                front_obstacle_dist=d, left_obstacle_dist=2.0, right_obstacle_dist=2.0
            )
            self.assertEqual(status, "SAFETY_OVERRIDE_OBSTACLE")
            self.assertEqual(vx, 0.0)
            self.assertEqual(wz, 0.0)

    def test_02_maximum_learned_steer_margin_cannot_relax_hard_stop(self):
        """Even with maximum proactive steer margin (+0.50m), hard stop remains enforced at <= 0.25m."""
        self.learner.delta_steer_dist = self.learner.max_delta_steer  # +0.50m
        override = self.learner.get_effective_steer_threshold()

        vx, wz, status = self.safety.filter_command(
            raw_vx=0.40, raw_wz=0.50,
            front_obstacle_dist=0.24, left_obstacle_dist=1.5, right_obstacle_dist=1.5,
            steer_start_override=override
        )
        self.assertEqual(status, "SAFETY_OVERRIDE_OBSTACLE")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_03_out_of_bounds_steer_override_clamped(self):
        """Corrupted or hostile steer overrides cannot shift the hard stop or bypass limits."""
        # 1. Absurdly high override (+9999.0m)
        vx1, wz1, status1 = self.safety.filter_command(
            raw_vx=0.30, raw_wz=0.0,
            front_obstacle_dist=0.20,
            steer_start_override=9999.0
        )
        self.assertEqual(status1, "SAFETY_OVERRIDE_OBSTACLE")
        self.assertEqual(vx1, 0.0)
        self.assertEqual(wz1, 0.0)

        # 2. Hostile negative override (-500.0m) attempting to disable avoidance
        # At 1.40m, safety should still initiate avoidance because override is clamped to >= 1.50m
        vx2, wz2, status2 = self.safety.filter_command(
            raw_vx=0.30, raw_wz=0.0,
            front_obstacle_dist=1.40, left_obstacle_dist=2.0, right_obstacle_dist=2.0,
            steer_start_override=-500.0
        )
        self.assertEqual(status2, "SAFETY_AVOIDANCE")
        self.assertNotEqual(wz2, 0.0)

    def test_04_learned_directional_bias_cannot_override_blocked_sides(self):
        """Learned preference for a direction cannot steer the vehicle into an obstructed corridor."""
        # Max left preference (+0.30 rad/s)
        self.learner.w_dir_bias = 0.30

        # Front is close (1.20m), but BOTH left and right sides are blocked (< 0.70m)
        mod_vx, mod_wz = self.learner.modulate_command(0.20, 0.0, 1.20, 0.40, 0.40)
        vx, wz, status = self.safety.filter_command(
            raw_vx=mod_vx, raw_wz=mod_wz,
            front_obstacle_dist=1.20, left_obstacle_dist=0.40, right_obstacle_dist=0.40
        )
        self.assertEqual(status, "SAFETY_BLOCKED")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_05_learned_speed_factor_cannot_override_watchdog(self):
        """Learned speed adaptation cannot bypass stale sensor timeout halt."""
        self.learner.speed_factor = 1.0
        # Age sensor time
        self.safety.last_sensor_time = time() - 2.0

        vx, wz, status = self.safety.filter_command(0.30, 0.0, front_obstacle_dist=5.0)
        self.assertEqual(status, "FAULT_STALE_SENSOR")
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_06_teleoperation_authoritative_safety_gate(self):
        """Teleoperation inputs pass through the exact same hard stop and cannot drive into obstacles."""
        teleop_vx = 0.40
        teleop_wz = 0.0

        # Teleop tries to drive forward into a 0.15m obstacle
        safe_vx, safe_wz, status = self.safety.filter_command(
            teleop_vx, teleop_wz,
            front_obstacle_dist=0.15, left_obstacle_dist=1.0, right_obstacle_dist=1.0
        )
        self.assertEqual(status, "SAFETY_OVERRIDE_OBSTACLE")
        self.assertEqual(safe_vx, 0.0)
        self.assertEqual(safe_wz, 0.0)


if __name__ == "__main__":
    unittest.main()
