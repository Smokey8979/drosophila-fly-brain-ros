"""
High-Fidelity Simulated 4-Wheel Robot Chassis & Virtual Sensor Environment.

Simulates planar differential/skid-steer physics, left/right wheel kinematics,
and 2D LiDAR range raycasting for closed-loop testing.
"""

import math
import numpy as np
from time import perf_counter as time


class MockFourWheelChassis:
    """Simulates a 4-wheel differential-drive ground robot in a 2D obstacle world."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, track_width=0.36, wheel_radius=0.085):
        # State: position (x, y) in meters, orientation (yaw) in radians
        self.x = x
        self.y = y
        self.yaw = yaw

        # Physical parameters
        self.b = track_width
        self.r = wheel_radius

        # Current commanded velocities
        self.cmd_v = 0.0
        self.cmd_w = 0.0

        # Actual wheel velocities (m/s)
        self.v_left_front = 0.0
        self.v_left_rear = 0.0
        self.v_right_front = 0.0
        self.v_right_rear = 0.0

        # Virtual 2D obstacles: list of (x, y, radius) circles
        self.obstacles = [
            (2.0, 0.2, 0.3),    # Obstacle slightly to the right of path at 2m
            (4.0, -0.2, 0.3),   # Obstacle slightly to the left at 4m
            (6.0, 0.0, 0.4),    # Obstacle straight ahead at 6m
        ]
        self.last_update = time()

    def set_cmd_vel(self, linear_x, angular_z):
        """Updates commanded linear and angular velocity."""
        self.cmd_v = float(linear_x)
        self.cmd_w = float(angular_z)

        # In a 4-wheel skid-steer vehicle, both left wheels share v_l, both right wheels share v_r
        v_l = self.cmd_v - (self.cmd_w * self.b / 2.0)
        v_r = self.cmd_v + (self.cmd_w * self.b / 2.0)

        self.v_left_front = v_l
        self.v_left_rear = v_l
        self.v_right_front = v_r
        self.v_right_rear = v_r

    def step_physics(self, dt=None):
        """Integrates vehicle kinematics forward in time."""
        now = time()
        if dt is None:
            dt = max(1e-4, now - self.last_update)
        self.last_update = now

        # Planar unicycle kinematics from left/right wheel speeds
        v_l = (self.v_left_front + self.v_left_rear) / 2.0
        v_r = (self.v_right_front + self.v_right_rear) / 2.0

        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / self.b

        self.yaw += w * dt
        # Normalize yaw to [-pi, pi]
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt

        return {
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
            'v': v,
            'w': w,
            'v_left_side': v_l,
            'v_right_side': v_r
        }

    def generate_laserscan(self, num_rays=360, fov_rad=math.pi, max_range=10.0):
        """Generates synthetic 2D LiDAR range scan from current robot pose."""
        angle_min = -fov_rad / 2.0
        angle_increment = fov_rad / num_rays
        ranges = []

        for i in range(num_rays):
            ray_angle = self.yaw + angle_min + i * angle_increment
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            min_hit = max_range
            # Raycast against all obstacles
            for ox, oy, orad in self.obstacles:
                # Vector from robot to circle center
                cx = ox - self.x
                cy = oy - self.y

                # Projection of circle center onto ray
                proj = cx * ray_dx + cy * ray_dy
                if proj > 0:
                    perp_dist_sq = (cx * cx + cy * cy) - (proj * proj)
                    if perp_dist_sq < (orad * orad):
                        # Intersection distance
                        half_chord = math.sqrt(max(0.0, orad * orad - perp_dist_sq))
                        hit_dist = proj - half_chord
                        if 0 < hit_dist < min_hit:
                            min_hit = hit_dist

            ranges.append(float(min_hit))

        return {
            'angle_min': angle_min,
            'angle_max': angle_min + num_rays * angle_increment,
            'angle_increment': angle_increment,
            'range_min': 0.1,
            'range_max': max_range,
            'ranges': ranges
        }
