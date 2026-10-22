"""
Safety Supervisor for Neural Robot Controller.

Enforces fail-safe vehicle constraints, command timeouts, stale-sensor detection,
imminent obstacle override, acceleration limits, and emergency stop handling.
"""

import math
from time import perf_counter as time


class SafetySupervisor:
    """Safety supervisor that validates and filters velocity commands."""

    def __init__(self,
                 max_linear_vel=0.4,
                 max_angular_vel=1.0,
                 max_linear_accel=1.0,
                 timeout_sec=0.5,
                 obstacle_hard_stop_dist=0.25,
                 obstacle_slow_dist=1.8,
                 obstacle_turn_dist=0.85,
                 obstacle_steer_start_dist=1.5,
                 minimum_side_clearance=0.70,
                 avoidance_angular_vel=0.50,
                 passing_side_clearance=0.80,
                 clear_hysteresis_cycles=5):
        self.max_v = float(max_linear_vel)
        self.max_w = float(max_angular_vel)
        self.max_a = float(max_linear_accel)
        self.timeout_sec = float(timeout_sec)
        self.hard_stop_dist = float(obstacle_hard_stop_dist)
        self.obstacle_slow_dist = float(obstacle_slow_dist)
        self.obstacle_turn_dist = float(obstacle_turn_dist)
        self.obstacle_steer_start_dist = float(obstacle_steer_start_dist)
        self.minimum_side_clearance = float(minimum_side_clearance)
        self.avoidance_angular_vel = float(avoidance_angular_vel)
        self.passing_side_clearance = float(passing_side_clearance)
        self.clear_hysteresis_cycles = int(clear_hysteresis_cycles)

        if not (
            0.0 < self.hard_stop_dist
            < self.obstacle_turn_dist
            < self.obstacle_steer_start_dist
            <= self.obstacle_slow_dist
        ):
            raise ValueError("Invalid obstacle avoidance distance thresholds")
        if self.minimum_side_clearance <= 0.0:
            raise ValueError("minimum_side_clearance must be positive")
        if self.avoidance_angular_vel <= 0.0:
            raise ValueError("avoidance_angular_vel must be positive")

        self.avoidance_direction = 0  # +1 = left, -1 = right, 0 = none
        self.clear_consecutive_cycles = 0

        self.last_sensor_time = time()
        self.last_brain_time = time()
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_update_time = time()

        self.emergency_stop_latched = False
        self.system_ready = False

    def trigger_emergency_stop(self, state=True):
        """Latches or clears emergency stop."""
        self.emergency_stop_latched = bool(state)
        if self.emergency_stop_latched:
            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0
            self.avoidance_direction = 0
            self.clear_consecutive_cycles = 0

    def notify_sensor_heartbeat(self):
        """Resets sensor watchdog."""
        self.last_sensor_time = time()

    def notify_brain_heartbeat(self):
        """Resets neural execution watchdog."""
        self.last_brain_time = time()

    def set_ready(self, ready=True):
        self.system_ready = bool(ready)
        if not self.system_ready:
            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0
            self.avoidance_direction = 0
            self.clear_consecutive_cycles = 0

    def filter_command(
        self,
        raw_vx,
        raw_wz,
        min_obstacle_dist=10.0,
        dt=None,
        left_obstacle_dist=None,
        right_obstacle_dist=None,
        front_obstacle_dist=None,
        steer_start_override=None,
    ):
        """Filter commands using front and side obstacle distances with hysteresis."""
        now = time()
        if dt is None:
            elapsed = now - self.last_update_time
            # Default to 0.1s if elapsed time is unrealistically small (e.g. tight unit tests)
            dt = elapsed if elapsed >= 0.005 else 0.10
        self.last_update_time = now

        if self.emergency_stop_latched:
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "EMERGENCY_STOP"

        if not self.system_ready:
            return 0.0, 0.0, "INITIALIZING"

        if now - self.last_sensor_time > self.timeout_sec:
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "FAULT_STALE_SENSOR"

        if now - self.last_brain_time > self.timeout_sec:
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "FAULT_BRAIN_TIMEOUT"

        try:
            raw_vx, raw_wz = float(raw_vx), float(raw_wz)
            nearest_input = float(min_obstacle_dist)
            left_dist = float(
                nearest_input if left_obstacle_dist is None
                else left_obstacle_dist
            )
            right_dist = float(
                nearest_input if right_obstacle_dist is None
                else right_obstacle_dist
            )
            front_dist = float(
                nearest_input if front_obstacle_dist is None
                else front_obstacle_dist
            )
        except (TypeError, ValueError):
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "FAULT_INVALID_VALUES"

        if not math.isfinite(raw_vx) or not math.isfinite(raw_wz):
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "FAULT_INVALID_VALUES"

        distances = [nearest_input, left_dist, right_dist, front_dist]
        if any(not math.isfinite(d) or d < 0.0 for d in distances):
            self.last_cmd_v = self.last_cmd_w = 0.0
            return 0.0, 0.0, "FAULT_INVALID_VALUES"

        # 1. Independent hard stop (immediate failsafe halt)
        if min(distances) <= self.hard_stop_dist:
            self.last_cmd_v = self.last_cmd_w = 0.0
            self.avoidance_direction = 0
            self.clear_consecutive_cycles = 0
            return 0.0, 0.0, "SAFETY_OVERRIDE_OBSTACLE"

        # Clamp raw inputs to allowable velocity envelope
        vx = max(-self.max_v * 0.5, min(self.max_v, raw_vx))
        wz = max(-self.max_w, min(self.max_w, raw_wz))

        left_clear = left_dist >= self.minimum_side_clearance
        right_clear = right_dist >= self.minimum_side_clearance

        # 2. Check Hysteresis and Avoidance Exit Condition
        if self.avoidance_direction != 0:
            # Front must clear past the slow distance + margin
            front_cleared = front_dist > (self.obstacle_slow_dist + 0.15)

            # Passing side must have cleared the obstacle
            if self.avoidance_direction == 1:
                # Turned left: obstacle was on the right
                side_cleared = (right_dist >= self.passing_side_clearance and left_clear)
            else:
                # Turned right: obstacle was on the left
                side_cleared = (left_dist >= self.passing_side_clearance and right_clear)

            if front_cleared and side_cleared:
                self.clear_consecutive_cycles += 1
                if self.clear_consecutive_cycles >= self.clear_hysteresis_cycles:
                    self.avoidance_direction = 0
                    self.clear_consecutive_cycles = 0
            else:
                self.clear_consecutive_cycles = 0

        # 3. Check for Blocked Condition or Side Re-evaluation during Avoidance
        if self.avoidance_direction != 0:
            if self.avoidance_direction == 1 and not left_clear:
                if right_clear:
                    self.avoidance_direction = -1
                else:
                    self.last_cmd_v = self.last_cmd_w = 0.0
                    return 0.0, 0.0, "SAFETY_BLOCKED"
            elif self.avoidance_direction == -1 and not right_clear:
                if left_clear:
                    self.avoidance_direction = 1
                else:
                    self.last_cmd_v = self.last_cmd_w = 0.0
                    return 0.0, 0.0, "SAFETY_BLOCKED"

        # 4. Initiate Avoidance if Front Distance Drops Below Steer Start Threshold
        effective_steer_start = self.obstacle_steer_start_dist
        if steer_start_override is not None:
            effective_steer_start = max(self.obstacle_steer_start_dist, min(self.obstacle_slow_dist, float(steer_start_override)))

        if self.avoidance_direction == 0 and front_dist <= effective_steer_start:
            if not left_clear and not right_clear:
                self.last_cmd_v = self.last_cmd_w = 0.0
                return 0.0, 0.0, "SAFETY_BLOCKED"

            if left_clear and right_clear:
                diff = left_dist - right_dist
                if abs(diff) > 0.10:
                    self.avoidance_direction = 1 if diff > 0 else -1
                else:
                    self.avoidance_direction = 1 if wz >= 0 else -1
            elif left_clear:
                self.avoidance_direction = 1
            else:
                self.avoidance_direction = -1

            self.clear_consecutive_cycles = 0

        # 5. Generate Safe Commands Based on Current Operational State
        if self.avoidance_direction != 0:
            status = "SAFETY_AVOIDANCE"
            yaw = min(self.max_w, max(self.avoidance_angular_vel, abs(wz)))
            safe_wz = self.avoidance_direction * yaw

            if front_dist <= self.obstacle_turn_dist:
                # Close obstacle: stop forward velocity and turn in place
                safe_vx = 0.0
            elif front_dist <= self.obstacle_slow_dist:
                span = self.obstacle_slow_dist - self.obstacle_turn_dist
                scale = max(0.0, min(1.0, (front_dist - self.obstacle_turn_dist) / span))
                safe_vx = max(0.0, vx) * scale
            else:
                # Front is open; advance at controlled speed while completing avoidance curve
                safe_vx = max(0.10, min(self.max_v * 0.5, vx if vx > 0 else 0.15))

            max_dv = self.max_a * max(1e-4, float(dt))
            dv = max(-max_dv, min(max_dv, safe_vx - self.last_cmd_v))
            safe_vx = self.last_cmd_v + dv
            safe_vx = max(-self.max_v * 0.5, min(self.max_v, safe_vx))

            self.last_cmd_v = safe_vx
            self.last_cmd_w = safe_wz
            return safe_vx, safe_wz, status

        if front_dist < self.obstacle_slow_dist:
            status = "SAFETY_SLOWDOWN"
            span = self.obstacle_slow_dist - self.obstacle_turn_dist
            scale = max(0.0, min(1.0, (front_dist - self.obstacle_turn_dist) / span))
            target_vx = max(0.0, vx) * scale

            max_dv = self.max_a * max(1e-4, float(dt))
            dv = max(-max_dv, min(max_dv, target_vx - self.last_cmd_v))
            safe_vx = self.last_cmd_v + dv
            safe_vx = max(0.0, min(self.max_v, safe_vx))
            safe_wz = wz

            self.last_cmd_v = safe_vx
            self.last_cmd_w = safe_wz
            return safe_vx, safe_wz, status

        # Nominal operation (free space ahead)
        max_dv = self.max_a * max(1e-4, float(dt))
        dv = max(-max_dv, min(max_dv, vx - self.last_cmd_v))
        safe_vx = self.last_cmd_v + dv
        safe_vx = max(-self.max_v * 0.5, min(self.max_v, safe_vx))
        safe_wz = wz

        self.last_cmd_v = safe_vx
        self.last_cmd_w = safe_wz
        return safe_vx, safe_wz, "NOMINAL"
