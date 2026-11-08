"""Integrated Closed-Loop Embodied Fly Agent."""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .arena import FlyArena
from .plasticity import MushroomBodyCircuit
from .visual_circuit import VisualLoomingCircuit
from .central_complex import CentralComplexCompass


class EmbodiedFlyAgent:
    """Persistent, embodied agent coupling the Drosophila neural circuits to FlyArena.
    
    Features:
      - Mushroom Body with dopamine-modulated 3-factor Hebbian plasticity (reward/punishment).
      - Visual LC4 looming avoidance circuit.
      - Central Complex ring attractor compass (heading representation).
      - Descending motor command decoding (DNp09, DNa01/02).
      - State persistence & synaptic weight checkpointing.
    """

    def __init__(
        self,
        arena: Optional[FlyArena] = None,
        num_kcs: int = 2000,
        eta_ltd: float = 0.08,
        eta_ltp: float = 0.02,
        seed: Optional[int] = 42,
    ):
        self.arena = arena if arena is not None else FlyArena(seed=seed)
        self.mb_circuit = MushroomBodyCircuit(
            num_kcs=num_kcs,
            eta_ltd=eta_ltd,
            eta_ltp=eta_ltp,
            seed=seed,
        )
        self.visual_circuit = VisualLoomingCircuit()
        self.compass = CentralComplexCompass()

        # Telemetry history
        self.history: List[Dict[str, Any]] = []
        self.last_obs: Optional[Dict[str, Any]] = None

    def reset(self, x: float = 0.0, y: float = 0.0, theta: Optional[float] = None, reset_memory: bool = False):
        """Reset agent body pose and arena while preserving or resetting learned weights."""
        obs = self.arena.reset(x=x, y=y, theta=theta)
        self.last_obs = obs
        self.mb_circuit.reset_state(reset_weights=reset_memory)
        self.compass.reset(heading=self.arena.theta)
        return obs

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Perform one complete closed-loop sensorimotor step."""
        # 1. Obtain current observation from arena (using persistent observation from previous step)
        if self.last_obs is None:
            self.last_obs = self.arena.get_observation()
        obs = self.last_obs
        odors = obs["odors"]
        vision = obs["vision"]
        taste = obs["taste"]

        # 2. Process Mushroom Body associative learning with dopamine gating
        sugar_reward = taste["reward"]
        bitter_punish = taste["punishment"]
        mb_out = self.mb_circuit.step(
            odor_readings=odors,
            sugar_reward=sugar_reward,
            bitter_punish=bitter_punish,
            dt=self.arena.dt,
        )

        # 3. Process Visual Looming (LC4) circuit with biological corollary discharge
        vis_out = self.visual_circuit.step(vision, omega=self.arena.omega)

        # 4. Integrate Olfactory Valence into steering commands
        # For each odor, compute directional tropotaxis weighted by learned odor-specific valence
        olfactory_steer = 0.0
        odor_valences = mb_out.get("odor_valences", {})
        for odor_id, readings in odors.items():
            norm_grad = readings.get("norm_gradient", readings["gradient"] / (readings["mean"] + 0.02))
            mean_c = readings["mean"]
            if mean_c > 1e-4:
                val = odor_valences.get(odor_id, mb_out["valence"])
                # Proportional steering without bang-bang limit cycles
                steer_drive = float(np.clip(val * norm_grad * 250.0, -12.0, 12.0))
                olfactory_steer += steer_drive

        # 5. Combine Steering Commands into Descending Neurons
        # Steering bias: Looming escape has high priority, followed by olfactory valence + spontaneous exploration
        escape_steer = vis_out["escape_steer_bias"]
        total_steer_bias = escape_steer + olfactory_steer

        # Add small spontaneous exploration wander noise
        wander = float(np.random.normal(0.0, 0.5))
        net_turn = total_steer_bias + wander

        # Convert to left/right DNa steering signals
        if net_turn >= 0:
            dna_left = 10.0 + net_turn
            dna_right = 10.0
        else:
            dna_left = 10.0
            dna_right = 10.0 - net_turn

        # 6. Forward walking command (DNp09)
        # Baseline walking, boosted by appetitive valence, suppressed by looming obstacles
        base_fwd = 25.0
        mean_val = float(np.mean(list(odor_valences.values()))) if odor_valences else mb_out["valence"]
        fwd_boost = max(0.0, mean_val * 6.0)
        fwd_suppression = vis_out["fwd_suppression"]
        p9_rate = max(5.0, base_fwd + fwd_boost - fwd_suppression)

        descending_spikes = {
            "DNp09": p9_rate,
            "DNa01_left": dna_left,
            "DNa01_right": dna_right,
            "DNa02_left": dna_left * 0.5,
            "DNa02_right": dna_right * 0.5,
            "PER": 30.0 if taste["sugar"] else 0.0,
        }

        # 7. Step Arena physics with descending commands
        next_obs, reward, done, info = self.arena.step(descending_spikes)
        self.last_obs = next_obs

        # 8. Update Central Complex compass heading
        compass_out = self.compass.step(self.arena.omega, dt=self.arena.dt)

        # 9. Record step telemetry
        step_log = {
            "step": info["step"],
            "x": info["x"],
            "y": info["y"],
            "theta": info["theta"],
            "v": info["v"],
            "omega": info["omega"],
            "reward": reward,
            "valence": mb_out["valence"],
            "mean_w_app": mb_out["mean_w_app"],
            "mean_w_av": mb_out["mean_w_av"],
            "lc4_left": vis_out["lc4_left"],
            "lc4_right": vis_out["lc4_right"],
            "internal_heading": compass_out["internal_heading"],
        }
        self.history.append(step_log)

        return next_obs, reward, done, info

    def run_episode(self, max_steps: int = 500) -> Dict[str, Any]:
        """Execute a full episode in the arena."""
        episode_reward = 0.0
        for _ in range(max_steps):
            _, rew, done, info = self.step()
            episode_reward += rew
            if done:
                break
        return {
            "steps": info["step"],
            "total_reward": info["total_reward"],
            "total_penalty": info["total_penalty"],
            "food_encounters": info["food_encounters"],
            "shock_encounters": info["shock_encounters"],
            "final_x": info["x"],
            "final_y": info["y"],
        }

    def save_checkpoint(self, filepath: str):
        """Save plastic synaptic weights to file."""
        self.mb_circuit.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str):
        """Load plastic synaptic weights from file."""
        self.mb_circuit.load_checkpoint(filepath)
