"""Unit tests for MotorDecoder."""

import unittest
from fly_brain_ros.motor_decoder import (
    MotorDecoder, P9_ODN1_LEFT, P9_ODN1_RIGHT,
    DNA01_LEFT, DNA02_LEFT, DNA01_RIGHT, DNA02_RIGHT, MDN_1
)


class TestMotorDecoder(unittest.TestCase):

    def setUp(self):
        self.decoder = MotorDecoder(
            kv=0.01, kw=0.02, krev=0.02, max_v=0.4, max_w=1.0,
            wheel_separation=0.36, wheel_radius=0.085
        )

    def test_pure_forward_walking(self):
        """Bilateral P9_oDN1 activation produces pure positive forward velocity with zero yaw."""
        spikes = {P9_ODN1_LEFT: 10, P9_ODN1_RIGHT: 10}
        cmd = self.decoder.decode_spikes(spikes, window_sec=0.1)

        self.assertGreater(cmd['linear_x'], 0.1)
        self.assertAlmostEqual(cmd['angular_z'], 0.0, places=3)
        self.assertAlmostEqual(cmd['v_left_wheels'], cmd['v_right_wheels'], places=3)

    def test_right_turn_steering(self):
        """Left DNa01/DNa02 activation produces negative yaw rate (right turn)."""
        spikes = {P9_ODN1_LEFT: 5, P9_ODN1_RIGHT: 5, DNA01_LEFT: 10, DNA02_LEFT: 10}
        cmd = self.decoder.decode_spikes(spikes, window_sec=0.1)

        self.assertLess(cmd['angular_z'], -0.1)
        # In a right turn, left wheels move faster than right wheels
        self.assertGreater(cmd['v_left_wheels'], cmd['v_right_wheels'])

    def test_left_turn_steering(self):
        """Right DNa01/DNa02 activation produces positive yaw rate (left turn)."""
        spikes = {P9_ODN1_LEFT: 5, P9_ODN1_RIGHT: 5, DNA01_RIGHT: 10, DNA02_RIGHT: 10}
        cmd = self.decoder.decode_spikes(spikes, window_sec=0.1)

        self.assertGreater(cmd['angular_z'], 0.1)
        # In a left turn, right wheels move faster than left wheels
        self.assertGreater(cmd['v_right_wheels'], cmd['v_left_wheels'])

    def test_reverse_moonwalker_activation(self):
        """MDN activation suppresses forward drive and commands reverse/stop."""
        spikes = {P9_ODN1_LEFT: 5, P9_ODN1_RIGHT: 5, MDN_1: 20}
        cmd = self.decoder.decode_spikes(spikes, window_sec=0.1)

        self.assertLess(cmd['linear_x'], 0.0)


if __name__ == '__main__':
    unittest.main()
