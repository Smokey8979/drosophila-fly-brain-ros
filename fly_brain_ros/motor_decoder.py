"""
Neural Output Decoder for 4-Wheel Differential/Skid-Steer Robot.

Monitors biological descending neuron populations and decodes spike activity
into linear/angular velocity commands (cmd_vel) and independent left/right
wheel pair velocities.
"""

# Biological Descending Neurons (from Shiu et al., Nature 2024 & example.ipynb)
# Forward locomotion descending neurons
P9_ODN1_LEFT = 720575940626730883
P9_ODN1_RIGHT = 720575940620300308

# Steering / turning descending neurons
DNA01_LEFT = 720575940644438551
DNA02_LEFT = 720575940604737708
DNA01_RIGHT = 720575940627787609
DNA02_RIGHT = 720575940629327659

# Backward walking / startle descending neurons
MDN_1 = 720575940616026939
MDN_2 = 720575940631082808


class MotorDecoder:
    """Decodes spike events from descending motor neurons into 4-wheel vehicle commands."""

    def __init__(self,
                 kv=0.015,
                 kw=0.025,
                 krev=0.02,
                 max_v=0.4,
                 max_w=1.0,
                 wheel_separation=0.36,
                 wheel_radius=0.085):
        self.kv = kv
        self.kw = kw
        self.krev = krev
        self.max_v = max_v
        self.max_w = max_w
        self.b = wheel_separation
        self.r = wheel_radius

        self.monitored_ids = [
            P9_ODN1_LEFT, P9_ODN1_RIGHT,
            DNA01_LEFT, DNA02_LEFT,
            DNA01_RIGHT, DNA02_RIGHT,
            MDN_1, MDN_2
        ]

    def decode_spikes(self, spike_counts_by_id, window_sec=0.1):
        """Decodes spike counts across window_sec into chassis velocities and wheel speeds.

        Args:
            spike_counts_by_id: dict mapping FlyWire ID (int) -> spike count (int or float)
            window_sec: duration of observation window in seconds

        Returns:
            dict containing:
                linear_x: forward velocity (m/s)
                angular_z: turning yaw rate (rad/s)
                v_left_wheels: linear velocity for left wheel pair (m/s)
                v_right_wheels: linear velocity for right wheel pair (m/s)
                omega_left_wheels: rotational speed of left wheels (rad/s)
                omega_right_wheels: rotational speed of right wheels (rad/s)
        """
        # Convert spike counts to firing rates (Hz)
        r_fwd_l = spike_counts_by_id.get(P9_ODN1_LEFT, 0) / window_sec
        r_fwd_r = spike_counts_by_id.get(P9_ODN1_RIGHT, 0) / window_sec

        r_steer_r = (spike_counts_by_id.get(DNA01_LEFT, 0) + spike_counts_by_id.get(DNA02_LEFT, 0)) / window_sec
        r_steer_l = (spike_counts_by_id.get(DNA01_RIGHT, 0) + spike_counts_by_id.get(DNA02_RIGHT, 0)) / window_sec

        r_rev = (spike_counts_by_id.get(MDN_1, 0) + spike_counts_by_id.get(MDN_2, 0)) / window_sec

        # Forward velocity computation
        raw_v_fwd = self.kv * (r_fwd_l + r_fwd_r)
        raw_v_rev = self.krev * r_rev
        v_x = float(np_clip(raw_v_fwd - raw_v_rev, -self.max_v * 0.5, self.max_v))

        # Yaw steering rate computation
        # Right asymmetry drives positive angular_z (turn left); Left asymmetry drives negative (turn right)
        raw_w = self.kw * (r_steer_l - r_steer_r)
        omega_z = float(np_clip(raw_w, -self.max_w, self.max_w))

        # Differential drive / skid-steer wheel velocity distribution
        # Both wheels on the left side receive v_l; both wheels on the right receive v_r
        v_left = v_x - (omega_z * self.b / 2.0)
        v_right = v_x + (omega_z * self.b / 2.0)

        omega_left = v_left / self.r
        omega_right = v_right / self.r

        return {
            'linear_x': v_x,
            'angular_z': omega_z,
            'v_left_wheels': v_left,
            'v_right_wheels': v_right,
            'omega_left_wheels': omega_left,
            'omega_right_wheels': omega_right,
            'rates': {
                'fwd_left': r_fwd_l,
                'fwd_right': r_fwd_r,
                'steer_left': r_steer_l,
                'steer_right': r_steer_r,
                'reverse': r_rev,
            }
        }


def np_clip(val, low, high):
    return max(low, min(high, val))
