"""Lobula Columnar (LC4) looming avoidance and visual steering circuit."""

import math
from typing import Dict, Any, Tuple
import numpy as np


class VisualLoomingCircuit:
    """Simulates Lobula Columnar (LC4) collision avoidance pathways.
    
    LC4 neurons in the fly optic lobe fire robustly in response to expanding visual edges
    (looming stimuli) and project directly to descending neurons to elicit escape turns.
    """

    def __init__(
        self,
        num_ommatidia: int = 16,
        looming_threshold: float = 0.15,
        escape_gain: float = 30.0,
    ):
        self.num_ommatidia = num_ommatidia
        self.looming_threshold = looming_threshold
        self.escape_gain = escape_gain

        # Half of ommatidia are left eye, half right eye
        self.half_om = num_ommatidia // 2
        self.lc4_left_rate = 0.0
        self.lc4_right_rate = 0.0

    def step(self, vision_obs: Dict[str, np.ndarray], omega: float = 0.0) -> Dict[str, float]:
        """Compute LC4 looming response and steering bias with corollary discharge.
        
        Args:
            vision_obs: Dict containing 'distances', 'proximity', 'looming'.
            omega: Current angular velocity in rad/s (for corollary discharge cancellation).
            
        Returns:
            Dict containing 'lc4_left', 'lc4_right', 'escape_steer_bias', 'fwd_suppression'.
        """
        looming = vision_obs["looming"]
        proximity = vision_obs["proximity"]

        # Threat requires both looming expansion AND close proximity (within 15 cm)
        threat = looming * np.maximum(0.0, proximity - 0.5) * 2.0

        # First half of ommatidia (indices 0..half_om-1) are negative angles (RIGHT eye)
        # Second half of ommatidia (indices half_om..N-1) are positive angles (LEFT eye)
        right_threat = np.max(threat[:self.half_om]) if self.half_om > 0 else 0.0
        left_threat = np.max(threat[self.half_om:]) if self.half_om > 0 else 0.0

        # Biological Corollary Discharge: self-generated rotation suppresses rotational optical flow
        cd_factor = 1.0 / (1.0 + (abs(omega) / 1.5)**2)

        # LC4 firing rates with threshold and corollary discharge gating
        self.lc4_right_rate = float(np.maximum(0.0, right_threat - self.looming_threshold) * self.escape_gain * cd_factor)
        self.lc4_left_rate = float(np.maximum(0.0, left_threat - self.looming_threshold) * self.escape_gain * cd_factor)

        # Biological escape steering:
        # Threat on right -> turn LEFT (+omega, positive steering)
        # Threat on left -> turn RIGHT (-omega, negative steering)
        escape_steer_bias = (self.lc4_right_rate - self.lc4_left_rate)

        # Symmetric looming directly ahead: elicit escape saccade
        frontal_threat = float(np.max(threat[self.half_om - 2 : self.half_om + 2]))
        if frontal_threat > self.looming_threshold and abs(escape_steer_bias) < 5.0:
            escape_steer_bias += 15.0 * cd_factor  # Break symmetry by veering left

        # Threat directly ahead suppresses forward speed
        fwd_suppression = float(np.clip(frontal_threat * 15.0, 0.0, 30.0))

        return {
            "lc4_left": self.lc4_left_rate,
            "lc4_right": self.lc4_right_rate,
            "escape_steer_bias": escape_steer_bias,
            "fwd_suppression": fwd_suppression,
        }
