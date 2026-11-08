"""Central Complex (CX) ring attractor heading circuit."""

import math
from typing import Dict, Any, Tuple
import numpy as np


class CentralComplexCompass:
    """Computational ring attractor model of the Drosophila Ellipsoid Body (EB).
    
    Represents internal heading through an active bump of activity across E-PG
    (compass) neurons that rotates in response to proprioceptive angular velocity.
    """

    def __init__(
        self,
        num_wedges: int = 16,
        bump_sigma: float = 0.5,
        shift_gain: float = 1.0,
    ):
        self.num_wedges = num_wedges
        self.bump_sigma = bump_sigma
        self.shift_gain = shift_gain

        # Wedge preferred directions from -pi to pi
        self.wedge_angles = np.linspace(-math.pi, math.pi, num_wedges, endpoint=False)

        # Internal heading representation (radians)
        self.internal_heading = 0.0
        # E-PG wedge population firing rates
        self.epg_rates = np.zeros(num_wedges, dtype=np.float32)
        self._update_bump()

    def _update_bump(self):
        """Update population activity bump centered on internal_heading."""
        diff = (self.wedge_angles - self.internal_heading + math.pi) % (2.0 * math.pi) - math.pi
        self.epg_rates = np.exp(-(diff**2) / (2.0 * self.bump_sigma**2))
        self.epg_rates /= np.sum(self.epg_rates) + 1e-6

    def reset(self, heading: float = 0.0):
        self.internal_heading = float((heading + math.pi) % (2.0 * math.pi) - math.pi)
        self._update_bump()

    def step(self, omega: float, visual_cue_angle: float = None, dt: float = 0.02) -> Dict[str, Any]:
        """Integrate angular velocity and optional visual cue into the compass bump.
        
        Args:
            omega: Proprioceptive angular turn rate (rad/s).
            visual_cue_angle: Optional landmark azimuth to anchor compass (radians).
            dt: Timestep in seconds.
            
        Returns:
            Dict containing internal heading and E-PG population firing rates.
        """
        # Proprioceptive path integration
        self.internal_heading += omega * self.shift_gain * dt

        # Visual anchor / cue integration (cue tethering)
        if visual_cue_angle is not None:
            cue_err = (visual_cue_angle - self.internal_heading + math.pi) % (2.0 * math.pi) - math.pi
            self.internal_heading += 0.05 * cue_err

        self.internal_heading = float((self.internal_heading + math.pi) % (2.0 * math.pi) - math.pi)
        self._update_bump()

        return {
            "internal_heading": self.internal_heading,
            "epg_rates": self.epg_rates.copy(),
            "bump_peak_wedge": int(np.argmax(self.epg_rates)),
        }
