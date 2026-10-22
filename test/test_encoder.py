"""Unit tests for SensoryEncoder."""

import unittest
from fly_brain_ros.sensory_encoder import (
    SensoryEncoder, LC4_IDS_LEFT, LC4_IDS_RIGHT, P9_LEFT, P9_RIGHT
)


class TestSensoryEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = SensoryEncoder(safe_dist=1.2, crit_dist=0.35, max_loom_rate=200.0, nominal_fwd_rate=100.0)

    def test_clear_path(self):
        """When path is completely clear (> 1.2m), loom rates should be 0 and forward drive active."""
        rates = self.encoder.encode_distances(d_left=3.0, d_right=3.0, d_front=3.0)
        for nid in LC4_IDS_LEFT:
            self.assertEqual(rates[nid], 0.0)
        for nid in LC4_IDS_RIGHT:
            self.assertEqual(rates[nid], 0.0)
        self.assertAlmostEqual(rates[P9_LEFT], 100.0, places=2)
        self.assertAlmostEqual(rates[P9_RIGHT], 100.0, places=2)

    def test_left_obstacle_loom(self):
        """When an obstacle approaches on left (0.5m), Left LC4 neurons should fire strongly."""
        rates = self.encoder.encode_distances(d_left=0.5, d_right=3.0, d_front=2.0)
        for nid in LC4_IDS_LEFT:
            self.assertGreater(rates[nid], 100.0)
        for nid in LC4_IDS_RIGHT:
            self.assertEqual(rates[nid], 0.0)

    def test_right_obstacle_loom(self):
        """When an obstacle approaches on right (0.5m), Right LC4 neurons should fire strongly."""
        rates = self.encoder.encode_distances(d_left=3.0, d_right=0.5, d_front=2.0)
        for nid in LC4_IDS_LEFT:
            self.assertEqual(rates[nid], 0.0)
        for nid in LC4_IDS_RIGHT:
            self.assertGreater(rates[nid], 100.0)

    def test_laserscan_parsing(self):
        """LiDAR range parsing correctly partitions left and right sectors."""
        # 180 degree scan (180 rays, 1 deg each): center 90 is forward, 0-89 right, 91-179 left
        ranges = [5.0] * 180
        ranges[135] = 0.6  # Left obstacle
        rates = self.encoder.encode_laserscan(ranges, angle_min=-1.5708, angle_increment=0.01745)
        for nid in LC4_IDS_LEFT:
            self.assertGreater(rates[nid], 50.0)
        for nid in LC4_IDS_RIGHT:
            self.assertEqual(rates[nid], 0.0)


if __name__ == '__main__':
    unittest.main()
