"""Mushroom Body plasticity and three-factor learning implementation."""

import json
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class MushroomBodyCircuit:
    """Computational model of the Drosophila Mushroom Body with dopamine-modulated plasticity.
    
    Circuit Components:
      - Kenyon Cells (KCs): Sparse combinatorial odor representation (~2,000 cells).
      - MBON_approach: Drives attraction / forward approach to odor.
      - MBON_avoid: Drives aversion / avoidance turn away from odor.
      - DAN_PAM: Dopaminergic cluster signaling appetitive sucrose reward.
      - DAN_PPL1: Dopaminergic cluster signaling aversive electric shock / bitter tastant.
    """

    def __init__(
        self,
        num_kcs: int = 2000,
        sparse_factor: float = 0.05,  # 5% of KCs active per odor
        eta_ltd: float = 0.05,         # Long-term depression rate
        eta_ltp: float = 0.02,         # Long-term potentiation rate
        w_init: float = 0.50,          # Initial balanced baseline weight
        w_min: float = 0.05,
        w_max: float = 1.00,
        tau_eligibility: float = 1.0,  # Presynaptic eligibility trace time constant (s)
        decay_rate: float = 0.0005,    # Passive extinction rate per second
        seed: Optional[int] = 42,
    ):
        self.num_kcs = num_kcs
        self.sparse_factor = sparse_factor
        self.eta_ltd = eta_ltd
        self.eta_ltp = eta_ltp
        self.w_init = w_init
        self.w_min = w_min
        self.w_max = w_max
        self.tau_eligibility = tau_eligibility
        self.decay_rate = decay_rate

        if seed is not None:
            np.random.seed(seed)

        # Odor-to-KC projections: fixed random projection matrix for odors
        # Each odor gets a unique sparse subset of KCs
        self.odor_kc_masks: Dict[str, np.ndarray] = {}

        # Plastic synaptic weight vectors from KCs to MBONs
        # Initially balanced so neither approach nor avoidance dominates
        self.w_kc_to_mbon_app = np.full(num_kcs, w_init, dtype=np.float32)
        self.w_kc_to_mbon_av = np.full(num_kcs, w_init, dtype=np.float32)

        # Dynamic state variables
        self.kc_eligibility_trace = np.zeros(num_kcs, dtype=np.float32)
        self.kc_activity = np.zeros(num_kcs, dtype=np.float32)
        self.last_odor_valences: Dict[str, float] = {}

        # MBON outputs
        self.mbon_app_rate = 0.0
        self.mbon_av_rate = 0.0

        # Reinforcement history
        self.cumulative_reward = 0.0
        self.cumulative_punish = 0.0

    def _get_odor_mask(self, odor_id: str) -> np.ndarray:
        """Generate or retrieve fixed sparse KC projection mask for an odor."""
        if odor_id not in self.odor_kc_masks:
            import zlib
            odor_seed = zlib.crc32(odor_id.encode("utf-8")) % (2**31 - 1)
            rng = np.random.RandomState(odor_seed)
            num_active = max(1, int(self.num_kcs * self.sparse_factor))
            indices = rng.choice(self.num_kcs, size=num_active, replace=False)
            mask = np.zeros(self.num_kcs, dtype=np.float32)
            mask[indices] = rng.uniform(0.7, 1.0, size=num_active)
            self.odor_kc_masks[odor_id] = mask
        return self.odor_kc_masks[odor_id]

    def get_odor_valence(self, odor_id: str) -> float:
        """Calculate steady-state valence for an odor based on its current KC->MBON weights."""
        mask = self._get_odor_mask(odor_id)
        total = float(np.sum(mask))
        if total > 1e-6:
            diff_w = self.w_kc_to_mbon_app - self.w_kc_to_mbon_av
            return float(np.dot(diff_w, mask) / total)
        return 0.0

    def reset_state(self, reset_weights: bool = False):
        """Reset dynamic activations while preserving learned synaptic memory."""
        self.kc_eligibility_trace.fill(0.0)
        self.kc_activity.fill(0.0)
        self.mbon_app_rate = 0.0
        self.mbon_av_rate = 0.0
        self.cumulative_reward = 0.0
        self.cumulative_punish = 0.0

        if reset_weights:
            self.w_kc_to_mbon_app.fill(self.w_init)
            self.w_kc_to_mbon_av.fill(self.w_init)

    def step(
        self,
        odor_readings: Dict[str, Dict[str, float]],
        sugar_reward: float = 0.0,
        bitter_punish: float = 0.0,
        dt: float = 0.02,
    ) -> Dict[str, Any]:
        """Step the Mushroom Body circuit and apply dopamine-modulated plasticity.
        
        Args:
            odor_readings: Dict of odor concentrations from antennae.
            sugar_reward: Appetitive reinforcement (PAM-DAN activation).
            bitter_punish: Aversive reinforcement (PPL1-DAN activation).
            dt: Timestep in seconds.
            
        Returns:
            Dict containing KC activity, MBON rates, valence, and weight delta stats.
        """
        # 1. Compute Kenyon Cell population activity from odor inputs
        self.kc_activity.fill(0.0)
        odor_kc_activations: Dict[str, np.ndarray] = {}

        for odor_id, readings in odor_readings.items():
            conc = readings["mean"]
            if conc > 1e-4:
                mask = self._get_odor_mask(odor_id)
                # Firing rate saturates with concentration
                kc_drive = (conc / (conc + 0.15)) * mask
                self.kc_activity += kc_drive
                odor_kc_activations[odor_id] = kc_drive

        # Clip KC activity to [0, 1]
        self.kc_activity = np.clip(self.kc_activity, 0.0, 1.0)

        # 2. Update presynaptic eligibility trace (exponential filter of KC spikes)
        alpha_e = dt / max(1e-4, self.tau_eligibility)
        self.kc_eligibility_trace += alpha_e * (self.kc_activity - self.kc_eligibility_trace)

        # 3. Compute MBON activations
        self.mbon_app_rate = float(np.dot(self.w_kc_to_mbon_app, self.kc_activity))
        self.mbon_av_rate = float(np.dot(self.w_kc_to_mbon_av, self.kc_activity))

        # Compute specific valence per odor based on its unique KC ensemble weights
        odor_valences: Dict[str, float] = {}
        for odor_id, act in odor_kc_activations.items():
            total_act = float(np.sum(act))
            if total_act > 1e-6:
                diff_w = self.w_kc_to_mbon_app - self.w_kc_to_mbon_av
                v = float(np.dot(diff_w, act) / total_act)
                odor_valences[odor_id] = v
            else:
                odor_valences[odor_id] = 0.0
        self.last_odor_valences = odor_valences

        # 4. Neuromodulated Three-Factor Synaptic Plasticity
        # Aversive conditioning (Shock/Bitter): PPL1 fires
        delta_w_app = 0.0
        delta_w_av = 0.0

        if bitter_punish > 0.0:
            self.cumulative_punish += bitter_punish
            # Depress approach synapses for active KCs (LTD)
            dw_app_ltd = -self.eta_ltd * bitter_punish * self.kc_eligibility_trace
            # Potentiate avoidance synapses (LTP)
            dw_av_ltp = +self.eta_ltp * bitter_punish * self.kc_eligibility_trace

            self.w_kc_to_mbon_app += dw_app_ltd
            self.w_kc_to_mbon_av += dw_av_ltp
            delta_w_app += float(np.mean(dw_app_ltd))
            delta_w_av += float(np.mean(dw_av_ltp))

        # Appetitive conditioning (Sugar Reward): PAM fires
        if sugar_reward > 0.0:
            self.cumulative_reward += sugar_reward
            # Depress avoidance synapses for active KCs (LTD)
            dw_av_ltd = -self.eta_ltd * sugar_reward * self.kc_eligibility_trace
            # Potentiate approach synapses (LTP)
            dw_app_ltp = +self.eta_ltp * sugar_reward * self.kc_eligibility_trace

            self.w_kc_to_mbon_av += dw_av_ltd
            self.w_kc_to_mbon_app += dw_app_ltp
            delta_w_app += float(np.mean(dw_app_ltp))
            delta_w_av += float(np.mean(dw_av_ltd))

        # 5. Passive decay / extinction towards baseline
        decay_factor = self.decay_rate * dt
        self.w_kc_to_mbon_app -= decay_factor * (self.w_kc_to_mbon_app - self.w_init)
        self.w_kc_to_mbon_av -= decay_factor * (self.w_kc_to_mbon_av - self.w_init)

        # 6. Strict weight bounds
        np.clip(self.w_kc_to_mbon_app, self.w_min, self.w_max, out=self.w_kc_to_mbon_app)
        np.clip(self.w_kc_to_mbon_av, self.w_min, self.w_max, out=self.w_kc_to_mbon_av)

        # Global valence
        valence = self.mbon_app_rate - self.mbon_av_rate

        return {
            "mbon_approach": self.mbon_app_rate,
            "mbon_avoid": self.mbon_av_rate,
            "valence": valence,
            "odor_valences": odor_valences,
            "active_kcs": int(np.count_nonzero(self.kc_activity > 0.1)),
            "mean_w_app": float(np.mean(self.w_kc_to_mbon_app)),
            "mean_w_av": float(np.mean(self.w_kc_to_mbon_av)),
            "delta_w_app": delta_w_app,
            "delta_w_av": delta_w_av,
        }

    def save_checkpoint(self, filepath: str):
        """Serialize plastic weights to disk."""
        data = {
            "num_kcs": self.num_kcs,
            "w_init": self.w_init,
            "w_kc_to_mbon_app": self.w_kc_to_mbon_app.tolist(),
            "w_kc_to_mbon_av": self.w_kc_to_mbon_av.tolist(),
            "cumulative_reward": self.cumulative_reward,
            "cumulative_punish": self.cumulative_punish,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_checkpoint(self, filepath: str):
        """Restore plastic weights from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.w_kc_to_mbon_app = np.array(data["w_kc_to_mbon_app"], dtype=np.float32)
        self.w_kc_to_mbon_av = np.array(data["w_kc_to_mbon_av"], dtype=np.float32)
        self.cumulative_reward = data.get("cumulative_reward", 0.0)
        self.cumulative_punish = data.get("cumulative_punish", 0.0)
