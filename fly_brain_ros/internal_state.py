"""
Simulated Internal State, Affect, and Self-Monitoring Subsystem.

Provides explicit, observable computational variables for the Drosophila robot:
  - sensory_surprise: Discrepancy between predicted and observed sensor measurements.
  - environmental_vigilance: Looming approach urgency and spatial threat level.
  - confidence_uncertainty: Unambiguity of navigational paths and motor decisions.
  - computational_fatigue: Cumulative synaptic event and spiking activity load proxy.
  - valence_affect: Task-progress and comfort index (-1.0 to +1.0).

SCIENTIFIC TRANSPARENCY NOTICE:
These indicators are deterministic mathematical variables designed for computational
self-monitoring, diagnostics, and adaptive tuning. They DO NOT prove or imply
biological consciousness, subjective experience, or emotional feelings.
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np


class InternalStateManager:
    """Manages explicit computational self-monitoring and simulated affect variables."""

    def __init__(self, dt: float = 0.10, safe_distance: float = 1.80):
        self.dt = dt
        self.safe_distance = safe_distance

        # Computational State Variables
        self.sensory_surprise: float = 0.0
        self.environmental_vigilance: float = 0.0
        self.confidence: float = 1.0
        self.uncertainty: float = 0.0
        self.computational_fatigue: float = 0.0
        self.valence_affect: float = 0.50

        # Memory Match State
        self.matched_episode_id: Optional[str] = None
        self.memory_match_similarity: float = 0.0
        self.memory_lesson: str = "No historical match active."

        # Internal Prediction State
        self.last_front_dist: float = 10.0
        self.last_front_vel: float = 0.0
        self.cumulative_spikes_recent: float = 0.0

    def step(
        self,
        d_front: float,
        d_left: float,
        d_right: float,
        active_spike_count: int,
        safety_status: str,
        episodic_memory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update all internal state metrics based on latest sensor and brain observations."""
        # Sanitize non-finite inputs
        if math.isnan(d_front) or math.isinf(d_front) or d_front < 0.0:
            d_front = 10.0
        if math.isnan(d_left) or math.isinf(d_left) or d_left < 0.0:
            d_left = 10.0
        if math.isnan(d_right) or math.isinf(d_right) or d_right < 0.0:
            d_right = 10.0

        d_min = min(d_front, d_left, d_right)

        # 1. Sensory Surprise (Prediction Error)
        # Predict where front obstacle should be based on previous velocity
        pred_front = max(0.01, self.last_front_dist + self.last_front_vel * self.dt)
        err = abs(d_front - pred_front)
        # Normalize via tanh: 0.5m error produces surprise ~ 0.76
        raw_surprise = float(math.tanh(err / 0.50))
        # Smooth surprise with rapid onset, slow decay
        if raw_surprise > self.sensory_surprise:
            self.sensory_surprise = raw_surprise
        else:
            self.sensory_surprise = float(self.sensory_surprise * 0.75 + raw_surprise * 0.25)

        # Update front distance velocity estimate
        current_vel = (d_front - self.last_front_dist) / max(1e-4, self.dt)
        self.last_front_dist = d_front
        self.last_front_vel = current_vel

        # 2. Environmental Vigilance
        # Driven by proximity to nearest obstacle (< 1.80m) + approaching looming velocity
        prox_threat = max(0.0, (self.safe_distance - d_min) / self.safe_distance)
        loom_threat = max(0.0, -current_vel) * 0.35 if d_front < 2.0 else 0.0
        raw_vigilance = min(1.0, prox_threat + loom_threat)
        # Instant jump on threat, exponential relaxation
        if raw_vigilance > self.environmental_vigilance:
            self.environmental_vigilance = raw_vigilance
        else:
            self.environmental_vigilance = float(self.environmental_vigilance * 0.85 + raw_vigilance * 0.15)

        # 3. Confidence & Uncertainty
        # High confidence when path is open (>1.8m) or when one side clearly dominates.
        # High uncertainty when front is blocked and left/right are cramped and symmetric.
        if d_front < 1.50:
            asym = abs(d_left - d_right) / max(0.1, max(d_left, d_right))
            # Cramped & symmetric = high uncertainty
            self.uncertainty = float(math.exp(-asym * 2.0) * (1.50 - d_front) / 1.50)
            self.uncertainty = min(1.0, max(0.0, self.uncertainty))
        else:
            self.uncertainty = float(self.uncertainty * 0.70)
        self.confidence = float(1.0 - self.uncertainty)

        # 4. Computational Fatigue Proxy
        # Tracks active connectome firing load: high firing rate increases fatigue; decays smoothly
        spike_load = min(1.0, active_spike_count / 15.0)
        self.computational_fatigue = float(np.clip(
            self.computational_fatigue * 0.98 + spike_load * 0.02, 0.0, 1.0
        ))

        # 5. Valence Affect Proxy (-1.0 to +1.0)
        # Positive in open space, negative when slowed, turning severely, or stopped
        if safety_status == "NOMINAL":
            target_valence = 0.70 if d_min > 2.0 else 0.30
        elif safety_status == "SAFETY_SLOWDOWN":
            target_valence = -0.20
        elif safety_status == "SAFETY_AVOIDANCE":
            target_valence = -0.40
        elif safety_status == "SAFETY_BLOCKED":
            target_valence = -0.80
        elif safety_status == "SAFETY_OVERRIDE_OBSTACLE":
            target_valence = -1.00
        else:
            target_valence = 0.0

        self.valence_affect = float(np.clip(
            self.valence_affect * 0.80 + target_valence * 0.20, -1.0, 1.0
        ))

        # 6. Episodic Memory Pattern Matching
        if episodic_memory is not None and d_min <= 2.0:
            matched_ep, sim = episodic_memory.find_matching_episode(d_front, d_left, d_right)
            if matched_ep is not None and sim >= 0.40:
                self.matched_episode_id = matched_ep["episode_id"]
                self.memory_match_similarity = sim
                self.memory_lesson = matched_ep.get("lesson", "Prior encounter recalled.")
                # Experiencing a recognized obstacle scenario modulates vigilance
                if matched_ep.get("outcome") in ("HARD_STOP_OVERRIDE", "SAFETY_BLOCKED"):
                    self.environmental_vigilance = min(1.0, self.environmental_vigilance + 0.20)
            else:
                self.matched_episode_id = None
                self.memory_match_similarity = 0.0
                self.memory_lesson = "No historical match for current clearances."
        else:
            self.matched_episode_id = None
            self.memory_match_similarity = 0.0
            self.memory_lesson = "Nominal open terrain."

        return self.get_state_dict()

    def get_state_dict(self) -> Dict[str, Any]:
        """Compile complete internal state for telemetry and UI visualization."""
        return {
            "sensory_surprise": round(self.sensory_surprise, 3),
            "environmental_vigilance": round(self.environmental_vigilance, 3),
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "computational_fatigue": round(self.computational_fatigue, 3),
            "valence_affect": round(self.valence_affect, 3),
            "matched_episode_id": self.matched_episode_id,
            "memory_match_similarity": round(self.memory_match_similarity, 2),
            "memory_lesson": self.memory_lesson,
        }

    def sleep_reset(self):
        """During sleep, fatigue clears and emotional arousal returns to baseline."""
        self.computational_fatigue = 0.0
        self.environmental_vigilance = 0.0
        self.sensory_surprise = 0.0
        self.confidence = 1.0
        self.uncertainty = 0.0
        self.valence_affect = 0.50
        self.matched_episode_id = None
        self.memory_match_similarity = 0.0
        self.memory_lesson = "Restored from sleep."

    def to_dict(self) -> Dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return self.get_state_dict()

    def load_dict(self, data: Dict[str, Any]):
        """Restore internal state from checkpoint."""
        self.sensory_surprise = float(data.get("sensory_surprise", 0.0))
        self.environmental_vigilance = float(data.get("environmental_vigilance", 0.0))
        self.confidence = float(data.get("confidence", 1.0))
        self.uncertainty = float(data.get("uncertainty", 0.0))
        self.computational_fatigue = float(data.get("computational_fatigue", 0.0))
        self.valence_affect = float(data.get("valence_affect", 0.50))
        self.matched_episode_id = data.get("matched_episode_id")
        self.memory_match_similarity = float(data.get("memory_match_similarity", 0.0))
        self.memory_lesson = str(data.get("memory_lesson", "Restored from checkpoint."))
