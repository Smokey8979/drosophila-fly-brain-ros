"""Actuator and descending motor decoder for the embodied Drosophila agent."""

import math
from typing import Dict, Any, Tuple
import numpy as np


class MotorActuator:
    """Decodes spiking descending neuron activity into continuous physical locomotion.
    
    Biological Pathways:
      - DNp09 / P9_oDN1: Bilateral command for forward walking speed.
      - DNa01 / DNa02: Left/right asymmetrical command for steering torque.
      - Feeding MNs: Subesophageal zone motor command triggering proboscis extension.
    """

    def __init__(
        self,
        v_base: float = 1.5,       # Spontaneous baseline walking speed in cm/s
        v_max: float = 8.0,        # Maximum running speed in cm/s
        omega_max: float = 6.0,    # Maximum turning rate in rad/s (~340 deg/s)
        tau_v: float = 0.05,       # Neuromuscular time constant for linear speed (s)
        tau_omega: float = 0.03,   # Neuromuscular time constant for angular velocity (s)
    ):
        self.v_base = v_base
        self.v_max = v_max
        self.omega_max = omega_max
        self.tau_v = tau_v
        self.tau_omega = tau_omega

        # Internal smoothed velocities
        self.current_v = 0.0
        self.current_omega = 0.0
        self.per_extended = False

    def reset(self):
        """Reset internal actuator states."""
        self.current_v = 0.0
        self.current_omega = 0.0
        self.per_extended = False

    def decode_action(
        self,
        descending_spikes: Dict[str, float],
        dt: float = 0.01,
    ) -> Tuple[float, float, bool]:
        """Decode descending neuron firing rates into updated physical velocities.
        
        Args:
            descending_spikes: Dict of instantaneous firing rates (Hz) or spike counts.
                Expected keys:
                  - 'DNp09_left', 'DNp09_right' (or 'DNp09' aggregate)
                  - 'DNa01_left', 'DNa01_right'
                  - 'DNa02_left', 'DNa02_right'
                  - 'PER' (proboscis extension reflex)
            dt: Simulation timestep in seconds.
            
        Returns:
            Tuple of (v, omega, per_extended):
              - v: Current linear speed in cm/s.
              - omega: Current angular turning velocity in rad/s.
              - per_extended: Boolean proboscis extension state.
        """
        # 1. Forward velocity decoding
        # Sum of bilateral forward walking descending neurons
        p9_l = descending_spikes.get("DNp09_left", descending_spikes.get("DNp09", 0.0))
        p9_r = descending_spikes.get("DNp09_right", descending_spikes.get("DNp09", 0.0))
        p9_total = p9_l + p9_r

        # Gain factor: 50 Hz total brings speed towards v_max
        target_v = self.v_base + (self.v_max - self.v_base) * (p9_total / (p9_total + 30.0))
        target_v = float(np.clip(target_v, 0.0, self.v_max))

        # 2. Steering velocity decoding
        # Biological DNa01 and DNa02 drive ipsilateral turning:
        # Left DNa activation drives turn left (+omega), Right DNa drives turn right (-omega)
        dna_l = descending_spikes.get("DNa01_left", 0.0) + descending_spikes.get("DNa02_left", 0.0)
        dna_r = descending_spikes.get("DNa01_right", 0.0) + descending_spikes.get("DNa02_right", 0.0)

        steering_diff = dna_l - dna_r  # Positive -> turn left, Negative -> turn right
        target_omega = self.omega_max * math.tanh(steering_diff / 20.0)

        # 3. Proboscis extension reflex
        per_rate = descending_spikes.get("PER", 0.0)
        self.per_extended = per_rate > 10.0

        # 4. Neuromuscular low-pass filter (LTI dynamics)
        alpha_v = dt / (self.tau_v + dt)
        alpha_w = dt / (self.tau_omega + dt)

        self.current_v += alpha_v * (target_v - self.current_v)
        self.current_omega += alpha_w * (target_omega - self.current_omega)

        return float(self.current_v), float(self.current_omega), self.per_extended
