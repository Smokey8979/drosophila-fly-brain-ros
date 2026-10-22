"""
Sensory Input Encoder for the Fruit-Fly Brain.

Converts physical distance measurements and LiDAR sector ranges into biological
Poisson arrival rates for identified Drosophila sensory populations:
  - LC4 visual projection neurons (left and right looming/obstacle detectors)
  - P9 descending driver interneurons (forward locomotion drive)
"""

# Verified FlyWire biological IDs for Lobula Columellar (LC4) loom detection neurons
# (from Ache et al., 2019 and Shiu et al., 2024 example.ipynb)
LC4_IDS_LEFT = [
    720575940605598892, 720575940611134833, 720575940612580977,
    720575940613256863, 720575940613260959, 720575940614914107,
    720575940615462587, 720575940617176321, 720575940617266722,
    720575940618807105, 720575940620795728, 720575940622108001,
]

LC4_IDS_RIGHT = [
    720575940624017251, 720575940625038090, 720575940625934973,
    720575940625991043, 720575940626605200, 720575940626626895,
    720575940628454522, 720575940628462340, 720575940630851036,
    720575940638496720, 720575940603637438, 720575940610522009,
]

# Forward walking driver neurons
P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101


class SensoryEncoder:
    """Encodes robot perceptual state into biological Poisson stimulation rates."""

    def __init__(self, safe_dist=1.2, crit_dist=0.35, max_loom_rate=200.0, nominal_fwd_rate=100.0):
        self.safe_dist = safe_dist
        self.crit_dist = crit_dist
        self.max_loom_rate = max_loom_rate
        self.nominal_fwd_rate = nominal_fwd_rate

    def encode_distances(self, d_left, d_right, d_front=None):
        """Calculates Poisson arrival rates (Hz) for sensory neuron IDs.

        Args:
            d_left: minimum obstacle distance on left side (meters)
            d_right: minimum obstacle distance on right side (meters)
            d_front: minimum obstacle distance in front (meters)

        Returns:
            rates_dict: dict mapping FlyWire ID (int) -> Poisson rate (Hz)
        """
        rates = {}

        # 1. Left Loom Detection (triggers right steering downstream)
        if d_left < self.safe_dist:
            norm_loom_left = (self.safe_dist - max(d_left, self.crit_dist)) / (self.safe_dist - self.crit_dist)
            rate_left = float(np_clip(norm_loom_left * self.max_loom_rate, 0.0, self.max_loom_rate))
        else:
            rate_left = 0.0

        for nid in LC4_IDS_LEFT:
            rates[nid] = rate_left

        # 2. Right Loom Detection (triggers left steering downstream)
        if d_right < self.safe_dist:
            norm_loom_right = (self.safe_dist - max(d_right, self.crit_dist)) / (self.safe_dist - self.crit_dist)
            rate_right = float(np_clip(norm_loom_right * self.max_loom_rate, 0.0, self.max_loom_rate))
        else:
            rate_right = 0.0

        for nid in LC4_IDS_RIGHT:
            rates[nid] = rate_right

        # 3. Forward Locomotion Drive (attenuated when front is blocked)
        front_dist = d_front if d_front is not None else min(d_left, d_right)
        if front_dist > self.crit_dist:
            fwd_factor = float(np_clip((front_dist - self.crit_dist) / (self.safe_dist - self.crit_dist), 0.0, 1.0))
            fwd_rate = self.nominal_fwd_rate * fwd_factor
        else:
            fwd_rate = 0.0

        rates[P9_LEFT] = fwd_rate
        rates[P9_RIGHT] = fwd_rate

        return rates

    def encode_laserscan(self, scan_ranges, angle_min, angle_increment, range_min=0.10, range_max=15.0):
        """Processes 2D LiDAR range array into left, right, and front sector distances."""
        n_rays = len(scan_ranges)
        if n_rays == 0:
            return self.encode_distances(0.0, 0.0, 0.0)

        left_ranges = []
        right_ranges = []
        front_ranges = []
        valid_count = 0

        for i, r in enumerate(scan_ranges):
            if not np_isfinite(r) or r < 0.0:
                continue
            eff_r = max(0.01, r) if r < range_min else min(float(r), range_max)
            valid_count += 1

            angle = angle_min + i * angle_increment
            # Front sector: [-15 deg, +15 deg]
            if -0.2618 <= angle <= 0.2618:
                front_ranges.append(eff_r)
            # Left sector: [+15 deg, +90 deg]
            if 0.2618 < angle <= 1.5708:
                left_ranges.append(eff_r)
            # Right sector: [-90 deg, -15 deg]
            if -1.5708 <= angle < -0.2618:
                right_ranges.append(eff_r)

        if valid_count == 0:
            return self.encode_distances(0.0, 0.0, 0.0)

        d_l = min(left_ranges) if left_ranges else range_max
        d_r = min(right_ranges) if right_ranges else range_max
        d_f = min(front_ranges) if front_ranges else min(d_l, d_r)

        return self.encode_distances(d_l, d_r, d_f)


def np_clip(val, low, high):
    return max(low, min(high, val))


def np_isfinite(val):
    import math
    return not (math.isnan(val) or math.isinf(val))
