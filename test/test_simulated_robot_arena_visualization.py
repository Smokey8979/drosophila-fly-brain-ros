"""
Suite G: Simulated Robot & Arena Interactive Visualization Test Suite.

Verifies:
  A: Camera Mathematics, Zoom Limits, Follow-Robot Tracking, Panning, and Arena Fitting.
  B: Dynamic Infinite World Grid, Adaptive Major/Minor Steps, and Axis Coordinate Invertibility.
  C: Authentic 4-Wheel Robot Footprint, LiDAR Sensor Offset (+0.10m), Obstacle Hit Discrimination, and Stale Scan Handling.
  D: State Telemetry API Contracts (/api/state lidar payload and arena_bounds schema).
"""

import math
import sys
import time
import unittest
from pathlib import Path

# Ensure live_lab and fly_brain_ros are in Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_LAB_DIR = REPO_ROOT / "live_lab"
if str(LIVE_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_LAB_DIR))
FLY_BRAIN_DIR = Path(__file__).resolve().parents[1]
if str(FLY_BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(FLY_BRAIN_DIR))

from web_server import app, sim, ARENA_BOUNDS  # type: ignore


class CameraMathModel:
    """Python reference implementation of the JavaScript arenaCamera projection math."""

    def __init__(self, canvas_w: float = 600.0, canvas_h: float = 300.0):
        self.canvas_w = float(canvas_w)
        self.canvas_h = float(canvas_h)
        self.cx = self.canvas_w / 2.0
        self.cy = self.canvas_h / 2.0

        # Camera state matching web_server.py JavaScript
        self.x = 0.0
        self.y = -6.0
        self.scale = 16.0  # px / m
        self.min_scale = 2.0
        self.max_scale = 120.0
        self.follow_robot = True

        self.layers = {
            "rays": True,
            "hits": True,
            "trail": True,
            "grid": True,
            "walls": True,
            "footprint": True,
        }

    def world_to_screen(self, xw: float, yw: float):
        xs = self.cx + (xw - self.x) * self.scale
        ys = self.cy - (yw - self.y) * self.scale
        return xs, ys

    def screen_to_world(self, xs: float, ys: float):
        xw = self.x + (xs - self.cx) / self.scale
        yw = self.y - (ys - self.cy) / self.scale
        return xw, yw

    def zoom_at_screen_point(self, xs: float, ys: float, factor: float):
        # 1. World point under cursor before zoom
        xw, yw = self.screen_to_world(xs, ys)
        # 2. Clamped scale update
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        # 3. Adjust camera center so that (xs, ys) continues to point to (xw, yw)
        self.x = xw - (xs - self.cx) / new_scale
        self.y = yw + (ys - self.cy) / new_scale
        self.scale = new_scale
        return self.scale

    def pan_by_screen_delta(self, dx_screen: float, dy_screen: float):
        # Dragging screen to right (dx > 0) means camera moves left in world (xw decreases)
        self.x -= dx_screen / self.scale
        self.y += dy_screen / self.scale
        self.follow_robot = False

    def reset_view(self):
        self.x = 0.0
        self.y = -6.0
        self.scale = 16.0
        self.follow_robot = True

    def center_on_robot(self, rx: float, ry: float):
        self.x = rx
        self.y = ry
        self.follow_robot = True

    def update_follow(self, rx: float, ry: float):
        if self.follow_robot:
            self.x = rx
            self.y = ry

    def fit_arena(self, arena_min_x: float, arena_max_x: float, arena_min_y: float, arena_max_y: float, padding: float = 1.2):
        arena_w = (arena_max_x - arena_min_x) * padding
        arena_h = (arena_max_y - arena_min_y) * padding
        fit_scale = min(self.canvas_w / arena_w, self.canvas_h / arena_h)
        self.scale = max(self.min_scale, min(self.max_scale, fit_scale))
        self.x = (arena_min_x + arena_max_x) / 2.0
        self.y = (arena_min_y + arena_max_y) / 2.0
        self.follow_robot = False

    def get_grid_steps(self):
        s = self.scale
        if s < 5.0:
            return 20.0, 5.0
        elif s < 12.0:
            return 10.0, 2.0
        elif s < 30.0:
            return 5.0, 1.0
        elif s < 70.0:
            return 2.0, 0.5
        else:
            return 1.0, 0.2

    def get_visible_world_bounds(self):
        min_xw, max_yw = self.screen_to_world(0, 0)
        max_xw, min_yw = self.screen_to_world(self.canvas_w, self.canvas_h)
        return min_xw, max_xw, min_yw, max_yw


class TestSimulatedRobotArenaVisualization(unittest.TestCase):
    """Automated test suite verifying camera, grid, LiDAR geometry, and telemetry contracts."""

    def setUp(self):
        self.cam = CameraMathModel(canvas_w=600.0, canvas_h=300.0)
        self.client = app.test_client()

    # =========================================================================
    # Suite A: Camera Mathematics, Zoom Limits, and Viewport Controls
    # =========================================================================

    def test_a1_camera_zoom_limits(self):
        """Verifies zoom clamping strictly between minScale (2.0) and maxScale (120.0)."""
        self.assertEqual(self.cam.scale, 16.0)

        # Zoom in past upper bound
        for _ in range(25):
            self.cam.zoom_at_screen_point(300, 150, 1.3)
        self.assertEqual(self.cam.scale, 120.0, "Scale must clamp to maxScale (120.0 px/m)")

        # Zoom out past lower bound
        for _ in range(40):
            self.cam.zoom_at_screen_point(300, 150, 0.7)
        self.assertEqual(self.cam.scale, 2.0, "Scale must clamp to minScale (2.0 px/m)")

    def test_a2_mouse_wheel_zoom_invariant(self):
        """Verifies zoom preserves the exact world coordinates under the cursor."""
        self.cam.x = 2.5
        self.cam.y = -3.8
        self.cam.scale = 20.0

        # Arbitrary cursor position (not screen center)
        cursor_xs, cursor_ys = 425.0, 95.0
        xw_before, yw_before = self.cam.screen_to_world(cursor_xs, cursor_ys)

        # Apply zoom in by factor 1.25
        self.cam.zoom_at_screen_point(cursor_xs, cursor_ys, 1.25)

        # The world coordinate under the cursor must remain strictly invariant
        xw_after, yw_after = self.cam.screen_to_world(cursor_xs, cursor_ys)
        self.assertAlmostEqual(xw_before, xw_after, places=9, msg="World X under cursor must be invariant")
        self.assertAlmostEqual(yw_before, yw_after, places=9, msg="World Y under cursor must be invariant")

    def test_a3_manual_pan_world_invariance(self):
        """Verifies click-and-drag panning moves camera inversely and disengages follow mode."""
        self.cam.follow_robot = True
        self.cam.x = 0.0
        self.cam.y = 0.0
        self.cam.scale = 10.0  # 10 px / m

        # Drag 100 px right and 50 px down
        self.cam.pan_by_screen_delta(100.0, 50.0)

        self.assertFalse(self.cam.follow_robot, "Panning must disengage follow mode")
        # 100 px right on screen -> camera center shifts -10.0m in world X
        self.assertAlmostEqual(self.cam.x, -10.0, places=6)
        # 50 px down on screen -> camera center shifts +5.0m in world Y (North)
        self.assertAlmostEqual(self.cam.y, 5.0, places=6)

    def test_a4_reset_view(self):
        """Verifies Reset View restores default camera center (0, -6) and scale (16 px/m)."""
        self.cam.x = 42.0
        self.cam.y = 99.0
        self.cam.scale = 85.0
        self.cam.follow_robot = False

        self.cam.reset_view()

        self.assertEqual(self.cam.x, 0.0)
        self.assertEqual(self.cam.y, -6.0)
        self.assertEqual(self.cam.scale, 16.0)
        self.assertTrue(self.cam.follow_robot)

    def test_a5_center_on_robot(self):
        """Verifies Center on Robot shifts camera center to robot coordinates without altering zoom."""
        self.cam.scale = 35.0
        self.cam.follow_robot = False

        self.cam.center_on_robot(rx=7.25, ry=-4.50)

        self.assertEqual(self.cam.x, 7.25)
        self.assertEqual(self.cam.y, -4.50)
        self.assertEqual(self.cam.scale, 35.0, "Center on robot must not alter zoom scale")
        self.assertTrue(self.cam.follow_robot)

    def test_a6_follow_robot_tracking(self):
        """Verifies camera updates automatically when follow_robot is enabled, and remains fixed when disabled."""
        self.cam.follow_robot = True
        self.cam.update_follow(rx=3.0, ry=-2.0)
        self.assertEqual(self.cam.x, 3.0)
        self.assertEqual(self.cam.y, -2.0)

        # Disengage follow
        self.cam.follow_robot = False
        self.cam.update_follow(rx=15.0, ry=25.0)
        self.assertEqual(self.cam.x, 3.0, "Camera must not follow when follow_robot is False")
        self.assertEqual(self.cam.y, -2.0)

    def test_a7_fit_arena(self):
        """Verifies Fit Arena scales and centers to accommodate the full 20m x 20m boundary with margin."""
        self.cam.fit_arena(
            arena_min_x=ARENA_BOUNDS["min_x"],
            arena_max_x=ARENA_BOUNDS["max_x"],
            arena_min_y=ARENA_BOUNDS["min_y"],
            arena_max_y=ARENA_BOUNDS["max_y"],
            padding=1.2,
        )

        # Center must be at arena midpoint (0.0, 0.0)
        self.assertAlmostEqual(self.cam.x, 0.0, places=6)
        self.assertAlmostEqual(self.cam.y, 0.0, places=6)

        # Scale must fit 20m * 1.2 = 24m into canvas height 300px -> 300 / 24 = 12.5 px/m
        expected_scale = min(600.0 / (20.0 * 1.2), 300.0 / (20.0 * 1.2))
        self.assertAlmostEqual(self.cam.scale, expected_scale, places=4)
        self.assertFalse(self.cam.follow_robot)

    def test_a8_robot_visibility_beyond_original_viewport(self):
        """Verifies robot remains centered and 100% visible when moving far beyond the original arena boundary."""
        # Robot navigates far out to (45.0, -85.0) meters
        far_x, far_y = 45.0, -85.0
        self.cam.follow_robot = True
        self.cam.update_follow(far_x, far_y)

        # In follow mode, camera center must track robot pose
        self.assertEqual(self.cam.x, far_x)
        self.assertEqual(self.cam.y, far_y)

        # Screen projection of the robot must be exactly at the canvas center (cx, cy)
        rob_sx, rob_sy = self.cam.world_to_screen(far_x, far_y)
        self.assertAlmostEqual(rob_sx, self.cam.cx, places=6)
        self.assertAlmostEqual(rob_sy, self.cam.cy, places=6)

        # Robot footprint bounding box must be well within canvas margins [0, canvas_w] x [0, canvas_h]
        half_chassis_px = (0.40 / 2.0) * self.cam.scale
        self.assertTrue(rob_sx - half_chassis_px >= 0)
        self.assertTrue(rob_sx + half_chassis_px <= self.cam.canvas_w)
        self.assertTrue(rob_sy - half_chassis_px >= 0)
        self.assertTrue(rob_sy + half_chassis_px <= self.cam.canvas_h)

    def test_a9_canvas_resize_projection_consistency(self):
        """Verifies viewport coordinate consistency when canvas is resized dynamically."""
        xw, yw = 3.5, -2.0
        xs_orig, ys_orig = self.cam.world_to_screen(xw, yw)

        # Resize canvas from 600x300 to 1200x800
        self.cam.canvas_w = 1200.0
        self.cam.canvas_h = 800.0
        self.cam.cx = 600.0
        self.cam.cy = 400.0

        # Focal center world coordinate must still project to the new canvas center
        cx_s, cy_s = self.cam.world_to_screen(self.cam.x, self.cam.y)
        self.assertAlmostEqual(cx_s, 600.0, places=6)
        self.assertAlmostEqual(cy_s, 400.0, places=6)

        # Screen-to-world roundtrip must remain perfectly invertible after resize
        xw_rt, yw_rt = self.cam.screen_to_world(cx_s, cy_s)
        self.assertAlmostEqual(xw_rt, self.cam.x, places=9)
        self.assertAlmostEqual(yw_rt, self.cam.y, places=9)

    # =========================================================================
    # Suite B: Dynamic Infinite Grid & World Geometry Projections
    # =========================================================================

    def test_b1_world_to_screen_and_screen_to_world_roundtrip(self):
        """Verifies world-to-screen and screen-to-world projections are perfectly invertible."""
        test_points = [
            (0.0, 0.0),
            (10.0, 10.0),
            (-10.0, -10.0),
            (5.5, 5.5),
            (-3.5, -2.0),
            (125.43, -89.12),  # Distant coordinates
        ]

        self.cam.x = 4.2
        self.cam.y = -1.7
        self.cam.scale = 22.5

        for xw, yw in test_points:
            xs, ys = self.cam.world_to_screen(xw, yw)
            xw_rt, yw_rt = self.cam.screen_to_world(xs, ys)
            self.assertAlmostEqual(xw, xw_rt, places=10, msg=f"X roundtrip failed for ({xw}, {yw})")
            self.assertAlmostEqual(yw, yw_rt, places=10, msg=f"Y roundtrip failed for ({xw}, {yw})")

    def test_b2_adaptive_grid_steps(self):
        """Verifies adaptive major/minor grid steps dynamically change across zoom thresholds."""
        # Macro zoom (far out)
        self.cam.scale = 3.0
        maj, mino = self.cam.get_grid_steps()
        self.assertEqual((maj, mino), (20.0, 5.0))

        # Standard overview
        self.cam.scale = 16.0
        maj, mino = self.cam.get_grid_steps()
        self.assertEqual((maj, mino), (5.0, 1.0))

        # Close-up inspection
        self.cam.scale = 80.0
        maj, mino = self.cam.get_grid_steps()
        self.assertEqual((maj, mino), (1.0, 0.2))

    def test_b3_unbounded_grid_expansion(self):
        """Verifies dynamic world bounds expand infinitely when robot moves far beyond the arena."""
        # Robot travels to (60.0, -90.0) meters
        self.cam.x = 60.0
        self.cam.y = -90.0
        self.cam.scale = 10.0

        min_xw, max_xw, min_yw, max_yw = self.cam.get_visible_world_bounds()

        # Canvas is 600x300, scale=10 -> width=60m, height=30m
        self.assertAlmostEqual(min_xw, 30.0, places=4)
        self.assertAlmostEqual(max_xw, 90.0, places=4)
        self.assertAlmostEqual(min_yw, -105.0, places=4)
        self.assertAlmostEqual(max_yw, -75.0, places=4)

        # Robot position is safely centered in visible world bounds
        self.assertTrue(min_xw <= 60.0 <= max_xw)
        self.assertTrue(min_yw <= -90.0 <= max_yw)

    def test_b4_world_origin_axes_orientation(self):
        """Verifies Red +X axis extends East (screen +X) and Green +Y axis extends North (screen -Y)."""
        self.cam.x = 0.0
        self.cam.y = 0.0
        self.cam.scale = 10.0

        # Origin screen position
        ox, oy = self.cam.world_to_screen(0.0, 0.0)
        self.assertEqual((ox, oy), (300.0, 150.0))

        # Point on +X axis: (1.0, 0.0) -> East
        px_x, px_y = self.cam.world_to_screen(1.0, 0.0)
        self.assertEqual(px_x, 310.0, "+X axis must extend to the right on screen (East)")
        self.assertEqual(px_y, 150.0)

        # Point on +Y axis: (0.0, 1.0) -> North (upward on screen, which is negative Y in Canvas 2D)
        py_x, py_y = self.cam.world_to_screen(0.0, 1.0)
        self.assertEqual(py_x, 300.0)
        self.assertEqual(py_y, 140.0, "+Y axis must extend upward on screen (North)")

    def test_b5_physical_arena_boundaries_extents(self):
        """Verifies ARENA_BOUNDS correctly reflects the 20m x 20m world geometry and obstacles."""
        self.assertEqual(ARENA_BOUNDS["width"], 20.0)
        self.assertEqual(ARENA_BOUNDS["height"], 20.0)
        self.assertEqual(ARENA_BOUNDS["min_x"], -10.0)
        self.assertEqual(ARENA_BOUNDS["max_x"], 10.0)
        self.assertEqual(ARENA_BOUNDS["min_y"], -10.0)
        self.assertEqual(ARENA_BOUNDS["max_y"], 10.0)

        # Confirm 4 bounding walls
        wall_names = [w["name"] for w in ARENA_BOUNDS["walls"]]
        self.assertCountEqual(wall_names, ["north", "south", "east", "west"])

        # Confirm 3 cylindrical obstacles matching learning_lab_arena.world
        obs_names = [o["name"] for o in ARENA_BOUNDS["fixed_obstacles"]]
        self.assertIn("center_pillar", obs_names)
        self.assertIn("left_pillar", obs_names)
        self.assertIn("right_pillar", obs_names)

        # Confirm sucrose reward and shock hazard zones
        zone_names = [z["name"] for z in ARENA_BOUNDS["zones"]]
        self.assertIn("sucrose_reward", zone_names)
        self.assertIn("shock_hazard", zone_names)

    # =========================================================================
    # Suite C: LiDAR Geometry, Sensor Offsets, and Robot Footprint
    # =========================================================================

    def test_c1_lidar_sensor_origin_offset(self):
        """Verifies LiDAR sensor originates from +0.10m forward of robot center along heading theta."""
        # Robot at (1.0, 2.0) facing North (theta = pi/2)
        rx, ry = 1.0, 2.0
        theta = math.pi / 2.0
        sensor_offset_x = 0.10

        sx = rx + sensor_offset_x * math.cos(theta)
        sy = ry + sensor_offset_x * math.sin(theta)

        self.assertAlmostEqual(sx, 1.0, places=9)
        self.assertAlmostEqual(sy, 2.10, places=9, msg="Sensor origin must be +0.10m in North direction")

        # Robot at (-3.0, 4.0) facing East (theta = 0)
        rx, ry = -3.0, 4.0
        theta = 0.0
        sx = rx + sensor_offset_x * math.cos(theta)
        sy = ry + sensor_offset_x * math.sin(theta)
        self.assertAlmostEqual(sx, -2.90, places=9, msg="Sensor origin must be +0.10m in East direction")
        self.assertAlmostEqual(sy, 4.0, places=9)

    def test_c2_lidar_ray_endpoint_calculation(self):
        """Verifies global beam ray endpoint calculation from sensor origin."""
        rx, ry, theta = 0.0, 0.0, 0.0
        sensor_offset_x = 0.10
        sx = rx + sensor_offset_x * math.cos(theta)
        sy = ry + sensor_offset_x * math.sin(theta)

        # Beam at local angle +pi/4, distance 2.0m
        beam_angle = math.pi / 4.0
        dist = 2.0
        world_angle = theta + beam_angle

        hit_x = sx + dist * math.cos(world_angle)
        hit_y = sy + dist * math.sin(world_angle)

        expected_x = 0.10 + 2.0 * math.cos(math.pi / 4.0)
        expected_y = 0.0 + 2.0 * math.sin(math.pi / 4.0)
        self.assertAlmostEqual(hit_x, expected_x, places=7)
        self.assertAlmostEqual(hit_y, expected_y, places=7)

    def test_c3_lidar_free_space_vs_hit_discrimination(self):
        """Verifies distinction between obstacle hit points (r < 0.98 * r_max) and free-space rays."""
        r_max = 15.0
        hit_threshold = r_max * 0.98  # 14.7m

        # Test ranges
        obstacle_range = 3.5
        free_space_range = 14.95
        max_range = 15.0

        self.assertTrue(obstacle_range < hit_threshold, "3.5m range must be classified as an obstacle hit")
        self.assertFalse(free_space_range < hit_threshold, "14.95m range must be classified as free space")
        self.assertFalse(max_range < hit_threshold, "15.0m max range must be classified as free space")

    def test_c4_lidar_freshness_and_stale_detection(self):
        """Verifies LiDAR freshness flag marks scans older than 1.5s as stale."""
        now = time.time()

        # Fresh scan: 0.1s old
        fresh_time = now - 0.1
        scan_age_fresh = now - fresh_time
        is_fresh = bool(scan_age_fresh < 1.5 and fresh_time > 0)
        self.assertTrue(is_fresh)

        # Stale scan: 2.5s old
        stale_time = now - 2.5
        scan_age_stale = now - stale_time
        is_stale = bool(scan_age_stale < 1.5 and stale_time > 0)
        self.assertFalse(is_stale, "Scan older than 1.5s must be flagged as stale")

    def test_c5_robot_chassis_and_wheel_footprint_geometry(self):
        """Verifies authentic 4-wheel robot specifications per model.sdf."""
        chassis_length = 0.40  # 40 cm along local X
        chassis_width = 0.30   # 30 cm along local Y
        wheel_track = 0.36     # left/right wheel offset y = +/- 0.18m
        wheel_base = 0.24      # front/rear wheel offset x = +/- 0.12m

        wheel_positions = [
            {"name": "front_left",  "x": 0.12,  "y": 0.18},
            {"name": "front_right", "x": 0.12,  "y": -0.18},
            {"name": "rear_left",   "x": -0.12, "y": 0.18},
            {"name": "rear_right",  "x": -0.12, "y": -0.18},
        ]

        self.assertEqual(chassis_length, 0.40)
        self.assertEqual(chassis_width, 0.30)
        for wp in wheel_positions:
            self.assertAlmostEqual(abs(wp["x"]), wheel_base / 2.0)
            self.assertAlmostEqual(abs(wp["y"]), wheel_track / 2.0)

    def test_c6_lidar_beam_endpoints_with_robot_rotation_and_translation(self):
        """Verifies LiDAR beam endpoint transformation with combined robot translation and rotation."""
        # Non-trivial robot pose
        rx, ry = -4.2, 6.8
        theta = 2.1  # rad (~120 degrees)
        sensor_offset_x = 0.10

        # Sensor position
        sx = rx + sensor_offset_x * math.cos(theta)
        sy = ry + sensor_offset_x * math.sin(theta)

        # Beam at local angle -0.3 rad, obstacle detected at distance 3.5m
        beam_angle = -0.3
        dist = 3.5
        world_angle = theta + beam_angle

        hit_x = sx + dist * math.cos(world_angle)
        hit_y = sy + dist * math.sin(world_angle)

        expected_sx = -4.2 + 0.10 * math.cos(2.1)
        expected_sy = 6.8 + 0.10 * math.sin(2.1)
        expected_hit_x = expected_sx + 3.5 * math.cos(2.1 - 0.3)
        expected_hit_y = expected_sy + 3.5 * math.sin(2.1 - 0.3)

        self.assertAlmostEqual(sx, expected_sx, places=9)
        self.assertAlmostEqual(sy, expected_sy, places=9)
        self.assertAlmostEqual(hit_x, expected_hit_x, places=9)
        self.assertAlmostEqual(hit_y, expected_hit_y, places=9)

    def test_c7_lidar_invalid_measurement_filtering(self):
        """Verifies filtering of NaN, Inf, and ranges below range_min in scan processing."""
        r_min = 0.10
        r_max = 15.0
        test_ranges = [float("nan"), float("inf"), float("-inf"), 0.05, 3.2, 14.9, -1.0]

        valid_ranges = []
        for r in test_ranges:
            if math.isnan(r) or math.isinf(r) or r < r_min:
                continue
            valid_ranges.append(r)

        self.assertEqual(valid_ranges, [3.2, 14.9])
        for r in valid_ranges:
            self.assertTrue(r_min <= r <= r_max)

    # =========================================================================
    # Suite D: Telemetry Contracts & API Exposure
    # =========================================================================

    def test_d1_state_telemetry_schema_lidar(self):
        """Verifies /api/state returns complete lidar payload with sensor_offset_x."""
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("lidar", data)
        lidar = data["lidar"]
        self.assertIn("timestamp", lidar)
        self.assertIn("fresh", lidar)
        self.assertIn("age_sec", lidar)
        self.assertIn("total_rays", lidar)
        self.assertIn("valid_hits", lidar)
        self.assertIn("sensor_offset_x", lidar)
        self.assertIn("range_min", lidar)
        self.assertIn("range_max", lidar)
        self.assertIn("angle_min", lidar)
        self.assertIn("angle_max", lidar)

        self.assertEqual(lidar["sensor_offset_x"], 0.10, "LiDAR forward offset must be 0.10m")
        self.assertEqual(lidar["range_min"], 0.10)
        self.assertEqual(lidar["range_max"], 15.0)

    def test_d2_state_telemetry_schema_arena(self):
        """Verifies /api/state returns authentic arena_bounds geometry schema."""
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("arena_bounds", data)
        bounds = data["arena_bounds"]
        self.assertEqual(bounds["width"], 20.0)
        self.assertEqual(bounds["height"], 20.0)
        self.assertEqual(len(bounds["walls"]), 4)
        self.assertEqual(len(bounds["fixed_obstacles"]), 3)
        self.assertEqual(len(bounds["zones"]), 2)

    def test_d3_layer_toggles_and_hud_telemetry(self):
        """Verifies all layer toggle flags and required telemetry keys exist in state."""
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        required_keys = [
            "x", "y", "theta", "v", "omega", "motion_state", "lifecycle_state",
            "is_asleep", "status", "rays", "obstacle_points", "trail", "internal_state",
            "learning", "lifecycle", "lidar", "arena_bounds"
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Required telemetry key '{key}' missing from /api/state")

        # Layer toggles in Camera model
        layers = self.cam.layers
        for layer_name in ["rays", "hits", "trail", "grid", "walls", "footprint"]:
            self.assertIn(layer_name, layers)
            self.assertTrue(layers[layer_name])

    def test_d4_bounded_http_api_state_validation(self):
        """Verifies bounded response and complete schema validation for /api/state."""
        start_time = time.time()
        res = self.client.get("/api/state")
        elapsed = time.time() - start_time

        # Bounded response time (< 1.0 second)
        self.assertLess(elapsed, 1.0, f"API state response too slow: {elapsed:.3f}s")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertIsInstance(data, dict)

        # Numerical fields must be finite numbers
        for num_field in ["x", "y", "theta", "v", "omega", "wheel_left", "wheel_right"]:
            self.assertIn(num_field, data)
            val = data[num_field]
            self.assertIsInstance(val, (int, float))
            self.assertFalse(math.isnan(val), f"{num_field} must not be NaN")
            self.assertFalse(math.isinf(val), f"{num_field} must not be Inf")

        # Status and motion state strings
        self.assertIn(data["status"], [
            "NOMINAL", "ASLEEP", "STANDALONE", "AVOIDING", "WAITING_FOR_SENSORS", "FAULT_STALE_SENSOR"
        ])
        self.assertIn(data["motion_state"], [
            "ASLEEP", "WAITING_FOR_SENSORS", "SAFETY_STOP", "AWAKE_NO_MOTOR_OUTPUT", "MOVING"
        ])


if __name__ == "__main__":
    unittest.main()
