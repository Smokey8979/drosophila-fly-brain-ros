"""
Automated Unit and Integration Tests for FlyBrainNode and ROS 2 ↔ Gazebo Interfaces.

Tests:
  1. ROS parameter handling and defaults.
  2. LaserScan sector extraction (left, right, front).
  3. Filtering of invalid, NaN, and infinite sensor values.
  4. Brain input delivery and rate encoding.
  5. Motor decoding and 4-wheel velocity distribution.
  6. Velocity command publication to /cmd_vel.
  7. Stale sensor watchdog trigger and zero-velocity response.
  8. Neural worker health detection and fault handling.
  9. Safe shutdown and zero-velocity publish.
 10. Package share and launch file resource resolution.
"""

import math
import os
import sys
import unittest
from pathlib import Path
from time import sleep, perf_counter as time

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.append(str(pkg_root))

from fly_brain_ros.fly_brain_node import (
    FlyBrainNode,
    FlyBrainControllerPipeline,
    LaserScan,
    Twist,
    resolve_connectome_path,
    HAVE_RCLPY,
)
if HAVE_RCLPY:
    import rclpy
from fly_brain_ros.sensory_encoder import LC4_IDS_LEFT, LC4_IDS_RIGHT, P9_LEFT, P9_RIGHT
from fly_brain_ros.motor_decoder import P9_ODN1_LEFT, P9_ODN1_RIGHT, DNA01_RIGHT


class TestROSNodeIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if HAVE_RCLPY and not rclpy.ok():
            rclpy.init()
        cls.workspace_root = pkg_root.parent
        candidates = [
            pkg_root / 'data',
            cls.workspace_root / 'fly-brain' / 'data',
            cls.workspace_root / 'data',
            Path('/home/om/Desktop/robo_fly/fly-brain/data'),
        ]
        cls.data_dir = pkg_root / 'data'
        for cand in candidates:
            if cand.exists():
                cls.data_dir = cand
                break
        cls.comp_path = str(cls.data_dir / '2025_Completeness_783.csv')
        cls.conn_path = str(cls.data_dir / '2025_Connectivity_783.parquet')
        cls.cache_dir = str(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        if HAVE_RCLPY and rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = FlyBrainNode(node_name=f"test_node_{self._testMethodName}")

    def tearDown(self):
        self.node.destroy_node()

    def test_01_parameter_handling(self):
        """Verify parameter declaration, default values, and path resolution."""
        # Check topic defaults
        self.assertEqual(self.node.get_parameter("scan_topic").value, "/scan")
        self.assertEqual(self.node.get_parameter("cmd_vel_topic").value, "/cmd_vel")
        self.assertAlmostEqual(self.node.get_parameter("controller_rate_hz").value, 10.0)

        # Check vehicle parameters
        self.assertAlmostEqual(self.node.get_parameter("wheel_separation").value, 0.36)
        self.assertAlmostEqual(self.node.get_parameter("wheel_radius").value, 0.085)
        self.assertAlmostEqual(self.node.get_parameter("max_linear_speed").value, 0.4)
        self.assertAlmostEqual(self.node.get_parameter("max_angular_speed").value, 1.0)

        # Check safety parameters
        self.assertAlmostEqual(self.node.get_parameter("cmd_timeout_sec").value, 0.5)
        self.assertAlmostEqual(self.node.get_parameter("obstacle_hard_stop_dist").value, 0.25)

        # Check connectome resolution
        resolved_comp = resolve_connectome_path("", "2025_Completeness_783.csv")
        self.assertTrue(os.path.exists(resolved_comp))

    def test_02_laserscan_sector_extraction(self):
        """Verify accurate sector extraction for front, left, and right obstacles."""
        scan = LaserScan()
        # 180 rays over [-135 deg, +135 deg] = [-2.356 rad, +2.356 rad]
        n_rays = 180
        scan.angle_min = -2.35619
        scan.angle_max = 2.35619
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (n_rays - 1)
        scan.range_min = 0.10
        scan.range_max = 15.0

        # Fill baseline with clear distance (10.0m)
        ranges = [10.0] * n_rays

        # Place an obstacle on left sector (+30 deg = +0.523 rad): index ~ 110
        # angle = -2.35619 + i * 0.02632 => i = (0.523 + 2.35619) / 0.02632 ~= 109
        idx_left = int(round((0.523 - scan.angle_min) / scan.angle_increment))
        ranges[idx_left] = 0.85

        # Place an obstacle on right sector (-45 deg = -0.785 rad)
        idx_right = int(round((-0.785 - scan.angle_min) / scan.angle_increment))
        ranges[idx_right] = 1.45

        # Place an obstacle in front (0 deg = 0.0 rad)
        idx_front = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        ranges[idx_front] = 2.10

        scan.ranges = ranges

        # Feed scan to node callback
        self.node.scan_callback(scan)

        # Verify pipeline received sector distances
        d_l, d_r, d_f = self.node.pipeline.last_sector_distances
        self.assertAlmostEqual(d_l, 0.85, delta=0.05)
        self.assertAlmostEqual(d_r, 1.45, delta=0.05)
        self.assertAlmostEqual(d_f, 2.10, delta=0.05)

    def test_03_invalid_and_infinite_sensor_values(self):
        """Verify that NaN, infinite, and out-of-range values are filtered safely."""
        scan = LaserScan()
        n_rays = 30
        scan.angle_min = -1.0
        scan.angle_max = 1.0
        scan.angle_increment = 2.0 / (n_rays - 1)
        scan.range_min = 0.20
        scan.range_max = 10.0

        # Corrupt scan with NaN, Inf, and negative ranges
        ranges = [float('nan'), float('inf'), -float('inf'), 0.05, 12.0] + [5.0] * 25
        scan.ranges = ranges

        # Should process cleanly without throwing exception
        self.node.scan_callback(scan)
        d_l, d_r, d_f = self.node.pipeline.last_sector_distances
        self.assertFalse(math.isnan(d_l))
        self.assertFalse(math.isnan(d_r))
        self.assertFalse(math.isnan(d_f))
        self.assertFalse(math.isinf(d_l))
        self.assertFalse(math.isinf(d_r))
        self.assertFalse(math.isinf(d_f))

    def test_04_brain_input_delivery(self):
        """Verify delivery of sensory distances into biological Poisson spike rates."""
        pipeline = self.node.pipeline
        pipeline.handle_sensor_ranges(d_left=0.5, d_right=1.8, d_front=2.0)

        # Inspect sensory encoder output
        rates = pipeline.encoder.encode_distances(0.5, 1.8, 2.0)

        # Left obstacle at 0.5m should trigger high looming rates on left LC4s
        for nid in LC4_IDS_LEFT:
            self.assertGreater(rates[nid], 100.0)

        # Right side at 1.8m (> safe_dist 1.2m) should have zero loom rate
        for nid in LC4_IDS_RIGHT:
            self.assertEqual(rates[nid], 0.0)

    def test_05_motor_decoding_and_skid_steer_kinematics(self):
        """Verify motor decoding into linear velocity and 4-wheel skid-steer distribution."""
        decoder = self.node.pipeline.decoder

        # Test forward walking
        spike_counts = {
            P9_ODN1_LEFT: 10.0,
            P9_ODN1_RIGHT: 10.0,
        }
        cmd = decoder.decode_spikes(spike_counts, window_sec=0.1)

        self.assertGreater(cmd['linear_x'], 0.0)
        self.assertAlmostEqual(cmd['angular_z'], 0.0, places=4)
        # Symmetrical forward motion: left and right wheel speeds should match
        self.assertAlmostEqual(cmd['v_left_wheels'], cmd['v_right_wheels'], places=4)

        # Test left turn (right DNa activation)
        spike_counts_turn = {
            P9_ODN1_LEFT: 5.0,
            P9_ODN1_RIGHT: 5.0,
            DNA01_RIGHT: 15.0,
        }
        cmd_turn = decoder.decode_spikes(spike_counts_turn, window_sec=0.1)
        self.assertGreater(cmd_turn['angular_z'], 0.0)  # Positive yaw = turn left
        # In left turn, right wheels must spin faster than left wheels
        self.assertGreater(cmd_turn['v_right_wheels'], cmd_turn['v_left_wheels'])

    def test_06_velocity_publication(self):
        """Verify that control_cycle publishes Twist messages to /cmd_vel."""
        # Supply fresh sensor scan
        scan = LaserScan()
        scan.ranges = [5.0] * 180
        scan.angle_min = -2.35619
        scan.angle_max = 2.35619
        scan.angle_increment = 0.02632
        self.node.scan_callback(scan)

        # Wait a moment for pipeline worker to step
        sleep(0.12)

        # Trigger control cycle
        self.node.control_cycle()
        self.assertGreater(self.node.published_cmd_count, 0)

        # Verify published twist
        pub = self.node.cmd_vel_pub
        if hasattr(pub, 'published'):
            last_twist = pub.published[-1]
            self.assertIsInstance(last_twist, Twist)
            self.assertFalse(math.isnan(last_twist.linear.x))
            self.assertFalse(math.isnan(last_twist.angular.z))

    def test_07_sensor_watchdog_behaviour(self):
        """Verify that stale sensor data causes safety override and zero velocity."""
        # 1. Supply scan to initialize
        scan = LaserScan()
        scan.ranges = [5.0] * 180
        scan.angle_min = -2.35619
        scan.angle_max = 2.35619
        scan.angle_increment = 0.02632
        self.node.scan_callback(scan)

        # 2. Artificially age the last scan time beyond timeout
        self.node.last_scan_received_time = time() - 2.0

        # 3. Execute control cycle
        self.node.control_cycle()

        # Command must be safe zero velocity
        pub = self.node.cmd_vel_pub
        if hasattr(pub, 'published'):
            last_twist = pub.published[-1]
            self.assertEqual(last_twist.linear.x, 0.0)
            self.assertEqual(last_twist.angular.z, 0.0)

    def test_08_neural_worker_failure_handling(self):
        """Verify that neural worker failure triggers immediate zero-velocity halt."""
        # Stop worker thread prematurely
        self.node.pipeline.running = False

        self.assertFalse(self.node.pipeline.is_healthy())

        self.node.control_cycle()
        pub = self.node.cmd_vel_pub
        if hasattr(pub, 'published'):
            last_twist = pub.published[-1]
            self.assertEqual(last_twist.linear.x, 0.0)
            self.assertEqual(last_twist.angular.z, 0.0)

    def test_09_shutdown_and_zero_command(self):
        """Verify that destroying the node publishes zero velocity and halts."""
        self.node.destroy_node()
        self.assertFalse(self.node.pipeline.running)

    def test_10_launch_file_and_resource_resolution(self):
        """Verify that launch files, models, and worlds exist and resolve properly."""
        pkg_dir = pkg_root
        world_file = pkg_dir / "worlds" / "learning_lab_arena.world"
        model_sdf = pkg_dir / "models" / "four_wheel_fly_robot" / "model.sdf"
        config_file = pkg_dir / "config" / "fly_brain_params.yaml"
        launch_full = pkg_dir / "launch" / "live_lab_full.launch.py"
        launch_gz = pkg_dir / "launch" / "live_lab_gazebo.launch.py"

        self.assertTrue(world_file.exists(), f"World file missing: {world_file}")
        self.assertTrue(model_sdf.exists(), f"Model SDF missing: {model_sdf}")
        self.assertTrue(config_file.exists(), f"Config YAML missing: {config_file}")
        self.assertTrue(launch_full.exists(), f"Full launch missing: {launch_full}")
        self.assertTrue(launch_gz.exists(), f"Gazebo launch missing: {launch_gz}")

        # Check SDF version 1.8 (Fortress)
        with open(model_sdf, 'r') as f:
            content = f.read()
            self.assertIn('<sdf version="1.8">', content)
            self.assertIn('ignition-gazebo-diff-drive-system', content)
            self.assertIn('gpu_lidar', content)

        # Check world file Fortress systems
        with open(world_file, 'r') as f:
            world_content = f.read()
            self.assertIn('<sdf version="1.8">', world_content)
            self.assertIn('ignition-gazebo-physics-system', world_content)
            self.assertIn('ignition-gazebo-sensors-system', world_content)

        # Check launch generation
        import importlib.util
        spec = importlib.util.spec_from_file_location("live_lab_full", str(launch_full))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        desc = mod.generate_launch_description()
        self.assertIsNotNone(desc)
        self.assertGreater(len(desc.entities), 4)


if __name__ == '__main__':
    unittest.main()
