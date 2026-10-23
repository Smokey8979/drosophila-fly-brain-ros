"""
Adaptive Learning & Persistent Episodic Memory Subsystem for the Drosophila Robot.

Provides an associative, bounded learning layer that operates alongside the immutable
138k connectome model:
  - ConnectomeLearner: Adapts proactive clearance thresholds and directional biases
    without altering the underlying biological synaptic weights.
  - EpisodicMemory: Stores structured, verifiable episodic records of obstacle encounters,
    clearances, actions, outcomes, and derived empirical lessons.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class EpisodicMemory:
    """Bounded, persistent episodic memory store for significant behavioral experiences."""

    def __init__(self, max_episodes: int = 150):
        self.max_episodes = max_episodes
        self.episodes: List[Dict[str, Any]] = []
        self._current_episode: Optional[Dict[str, Any]] = None

    def start_episode(self, trigger: str, d_front: float, d_left: float, d_right: float) -> str:
        """Initiate tracking of a significant navigational episode."""
        ep_id = f"ep_{int(time.time() * 1000) % 10000000:07d}"
        self._current_episode = {
            "episode_id": ep_id,
            "timestamp": round(time.time(), 3),
            "trigger": trigger,
            "initial_clearances": {
                "front": round(float(d_front), 2),
                "left": round(float(d_left), 2),
                "right": round(float(d_right), 2),
            },
            "min_clearance_reached": round(min(d_front, d_left, d_right), 2),
            "safety_transitions": ["NOMINAL"],
            "actions_taken": [],
            "chosen_direction": 0,  # +1 Left, -1 Right, 0 Straight
            "outcome": "UNRESOLVED",
            "reward": 0.0,
            "parameter_updates": {},
            "lesson": "",
        }
        return ep_id

    def update_current(self, d_front: float, d_left: float, d_right: float,
                       safety_status: str, action: str, direction: int = 0):
        """Update active episode with ongoing measurements and transitions."""
        if self._current_episode is None:
            return

        d_min = min(d_front, d_left, d_right)
        if d_min < self._current_episode["min_clearance_reached"]:
            self._current_episode["min_clearance_reached"] = round(float(d_min), 2)

        if not self._current_episode["safety_transitions"] or \
           self._current_episode["safety_transitions"][-1] != safety_status:
            self._current_episode["safety_transitions"].append(safety_status)

        if action and (not self._current_episode["actions_taken"] or
                       self._current_episode["actions_taken"][-1] != action):
            self._current_episode["actions_taken"].append(action)

        if direction != 0:
            self._current_episode["chosen_direction"] = direction

    def end_episode(self, outcome: str, reward: float, parameter_updates: Dict[str, float],
                    lesson: str = "") -> Optional[Dict[str, Any]]:
        """Conclude active episode, assign outcome and store to bounded memory buffer."""
        if self._current_episode is None:
            return None

        ep = self._current_episode
        ep["outcome"] = outcome
        ep["reward"] = round(float(reward), 3)
        ep["parameter_updates"] = parameter_updates
        
        # Generate empirical lesson if not explicitly provided
        if not lesson:
            dir_str = "left" if ep["chosen_direction"] > 0 else ("right" if ep["chosen_direction"] < 0 else "forward")
            init_f = ep["initial_clearances"]["front"]
            min_c = ep["min_clearance_reached"]
            if outcome == "PASS_SUCCESS":
                lesson = (f"Approached front obstacle ({init_f}m); executed {dir_str} avoidance; "
                          f"cleared successfully with minimum clearance {min_c}m.")
            elif outcome == "HARD_STOP_OVERRIDE":
                lesson = (f"Approached front obstacle ({init_f}m); clearance collapsed to {min_c}m "
                          f"triggering emergency hard stop. Requires earlier steering initiation.")
            elif outcome == "SAFETY_BLOCKED":
                lesson = (f"Front obstructed ({init_f}m) with restricted side clearances "
                          f"(min {min_c}m). Both passage sides inadequate.")
            else:
                lesson = f"Navigational encounter finished with outcome: {outcome}."
        
        ep["lesson"] = lesson
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

        self._current_episode = None
        return ep

    def find_matching_episode(self, d_front: float, d_left: float, d_right: float,
                              dist_threshold: float = 0.50) -> Tuple[Optional[Dict[str, Any]], float]:
        """Find the most similar historical encounter based on normalized clearance features.
        
        Returns:
            (best_matching_episode, similarity_score_0_to_1)
        """
        if not self.episodes:
            return None, 0.0

        best_ep = None
        best_dist = float("inf")

        query = np.array([d_front, d_left, d_right], dtype=np.float32)

        for ep in reversed(self.episodes[-30:]):  # Search recent window
            c = ep.get("initial_clearances", {})
            f, l, r = c.get("front", 5.0), c.get("left", 5.0), c.get("right", 5.0)
            target = np.array([f, l, r], dtype=np.float32)
            dist = float(np.linalg.norm(query - target))

            if dist < best_dist:
                best_dist = dist
                best_ep = ep

        if best_ep is not None and best_dist < dist_threshold:
            sim = max(0.0, 1.0 - (best_dist / dist_threshold))
            return best_ep, round(sim, 2)

        return None, 0.0

    def consolidate(self) -> Dict[str, Any]:
        """Extract recurrent patterns and statistical metrics across stored episodes."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "left_success_count": 0,
                "right_success_count": 0,
                "mean_clearance": 0.0,
                "summary": "No episodic memories recorded yet.",
            }

        n = len(self.episodes)
        successes = [ep for ep in self.episodes if ep.get("outcome") == "PASS_SUCCESS"]
        hard_stops = [ep for ep in self.episodes if ep.get("outcome") == "HARD_STOP_OVERRIDE"]
        blocked = [ep for ep in self.episodes if ep.get("outcome") == "SAFETY_BLOCKED"]

        left_succ = sum(1 for ep in successes if ep.get("chosen_direction", 0) > 0)
        right_succ = sum(1 for ep in successes if ep.get("chosen_direction", 0) < 0)

        min_clearances = [ep.get("min_clearance_reached", 0.0) for ep in self.episodes]
        mean_clearance = float(np.mean(min_clearances)) if min_clearances else 0.0

        summary = (
            f"Consolidated {n} episodes: {len(successes)} successful passes "
            f"({len(successes)/n*100:.1f}%), {len(hard_stops)} hard stops, {len(blocked)} blocked corridors. "
            f"Directional pass ratio: Left={left_succ}, Right={right_succ}. "
            f"Average minimum clearance: {mean_clearance:.2f}m."
        )

        return {
            "total_episodes": n,
            "success_rate": round(len(successes) / n, 3),
            "left_success_count": left_succ,
            "right_success_count": right_succ,
            "hard_stop_count": len(hard_stops),
            "blocked_count": len(blocked),
            "mean_clearance": round(mean_clearance, 2),
            "summary": summary,
        }

    def to_dict(self) -> List[Dict[str, Any]]:
        return list(self.episodes)

    def load_dict(self, episodes_list: List[Dict[str, Any]]):
        self.episodes = list(episodes_list[-self.max_episodes:])
        self._current_episode = None


class ConnectomeLearner:
    """Bounded Adaptive Navigation Policy & Reinforcement Subsystem.
    
    Modulates clearance distances and directional torque biases based on
    reward/punishment signals without mutating the base connectome weights.
    """

    def __init__(
        self,
        eta_steer: float = 0.04,     # Learning rate for proactive steer margin
        eta_dir: float = 0.06,       # Learning rate for directional preference
        max_delta_steer: float = 0.50, # Max proactive clearance increase (m)
        max_dir_bias: float = 0.30,   # Max directional yaw bias
    ):
        self.eta_steer = eta_steer
        self.eta_dir = eta_dir
        self.max_delta_steer = max_delta_steer
        self.max_dir_bias = max_dir_bias

        # Learned Policy Parameters (Persisted)
        self.delta_steer_dist: float = 0.0    # meters added to nominal 1.50m steer threshold
        self.w_dir_bias: float = 0.0          # yaw torque bias (+left / -right)
        self.speed_factor: float = 1.0        # clutter speed adaptation [0.6, 1.0]

        # Accounting & Statistics
        self.total_learning_steps: int = 0
        self.cumulative_reward: float = 0.0
        self.last_update_delta: Dict[str, float] = {}

        # Episode Tracker
        self.episodic_memory = EpisodicMemory()
        self.in_encounter: bool = False

    def reset_learning(self, weights_only: bool = False):
        """Reset learned parameters to pristine factory baseline."""
        self.delta_steer_dist = 0.0
        self.w_dir_bias = 0.0
        self.speed_factor = 1.0
        self.last_update_delta = {}
        if not weights_only:
            self.total_learning_steps = 0
            self.cumulative_reward = 0.0
            self.episodic_memory = EpisodicMemory()
            self.in_encounter = False

    def get_effective_steer_threshold(self, base_steer_dist: float = 1.50) -> float:
        """Returns proactive steering threshold including learned clearance margin."""
        return base_steer_dist + self.delta_steer_dist

    def modulate_command(
        self,
        raw_vx: float,
        raw_wz: float,
        d_front: float,
        d_left: float,
        d_right: float,
    ) -> Tuple[float, float]:
        """Apply learned policy parameters to the connectome output.
        
        Applies learned speed factor in clutter and adds learned directional bias
        when steering is active, strictly bounded before safety filtering.
        """
        # 1. Apply clutter speed factor if any sector is restricted (< 1.5m)
        min_d = min(d_front, d_left, d_right)
        if min_d < 1.50:
            adapted_vx = raw_vx * self.speed_factor
        else:
            adapted_vx = raw_vx

        # 2. Apply directional preference bias during turning
        if abs(raw_wz) > 0.01:
            # If turning left (raw_wz > 0) and w_dir_bias > 0 (left preference), boost turning
            # If turning right (raw_wz < 0) and w_dir_bias < 0 (right preference), boost turning
            adapted_wz = raw_wz + self.w_dir_bias * abs(raw_wz)
        else:
            adapted_wz = raw_wz

        return adapted_vx, adapted_wz

    def step(
        self,
        d_front: float,
        d_left: float,
        d_right: float,
        safety_status: str,
        active_direction: int,
    ) -> Dict[str, Any]:
        """Perform one learning step, evaluate outcomes, and update policy parameters."""
        d_min = min(d_front, d_left, d_right)
        step_reward = 0.0
        updates: Dict[str, float] = {}

        # 1. Detect encounter start / track active encounter
        if not self.in_encounter and d_min <= 1.80:
            self.in_encounter = True
            trigger = "OBSTACLE_APPROACH"
            if safety_status == "SAFETY_OVERRIDE_OBSTACLE":
                trigger = "SUDDEN_PROXIMITY"
            self.episodic_memory.start_episode(trigger, d_front, d_left, d_right)

        if self.in_encounter:
            action_desc = "STEER_LEFT" if active_direction > 0 else ("STEER_RIGHT" if active_direction < 0 else "FORWARD")
            self.episodic_memory.update_current(
                d_front, d_left, d_right, safety_status, action_desc, active_direction
            )

        # 2. Reward / Penalty Computation
        if safety_status == "NOMINAL":
            # Nominal progress reward
            step_reward += 0.02
        elif safety_status == "SAFETY_SLOWDOWN":
            # Mild deceleration penalty
            step_reward -= 0.05
        elif safety_status == "SAFETY_AVOIDANCE":
            # Turning penalty (encourages efficient clean passes)
            step_reward -= 0.08
        elif safety_status == "SAFETY_BLOCKED":
            # Trap penalty
            step_reward -= 0.50
        elif safety_status == "SAFETY_OVERRIDE_OBSTACLE":
            # Severe hard-stop penalty
            step_reward -= 1.00

        # 3. Parameter Update on Near-Collision or Trap (Aversive Learning)
        if safety_status in ("SAFETY_OVERRIDE_OBSTACLE", "SAFETY_BLOCKED"):
            # The agent cut it too close: adapt by increasing proactive steer margin
            old_steer = self.delta_steer_dist
            self.delta_steer_dist = min(self.max_delta_steer, self.delta_steer_dist + self.eta_steer)
            updates["delta_steer_dist"] = round(self.delta_steer_dist - old_steer, 4)

            # Reduce clutter speed factor
            old_sp = self.speed_factor
            self.speed_factor = max(0.60, self.speed_factor - 0.05)
            updates["speed_factor"] = round(self.speed_factor - old_sp, 4)

            # Penalize direction that led to the trap/stop
            if active_direction != 0:
                old_dir = self.w_dir_bias
                # Steer away from trapped direction in future
                self.w_dir_bias = float(np.clip(
                    self.w_dir_bias - self.eta_dir * active_direction,
                    -self.max_dir_bias, self.max_dir_bias
                ))
                updates["w_dir_bias"] = round(self.w_dir_bias - old_dir, 4)

        # 4. Detect Encounter Conclusion & Reward Positive Passage
        if self.in_encounter and d_min > 2.0 and safety_status == "NOMINAL":
            # Successfully cleared obstacle!
            min_c = 2.0
            if self.episodic_memory._current_episode:
                min_c = self.episodic_memory._current_episode.get("min_clearance_reached", 2.0)
            self.in_encounter = False

            if min_c <= 0.50:
                # Near-miss pass: cleared obstacle, but cut it dangerously close!
                # Proactively adapt steer margin based on near-miss information without waiting for a hard stop
                near_miss_margin = round(self.eta_steer * 0.5 * ((0.50 - min_c) / 0.25), 4)
                old_steer = self.delta_steer_dist
                self.delta_steer_dist = min(self.max_delta_steer, self.delta_steer_dist + near_miss_margin)
                updates["delta_steer_dist"] = round(self.delta_steer_dist - old_steer, 4)
                clear_reward = 0.50  # Reduced reward for near-miss
                step_reward += clear_reward
                self.episodic_memory.end_episode(
                    "PASS_SUCCESS", clear_reward, updates,
                    lesson=f"Cleared obstacle with close near-miss (min clearance {min_c}m <= 0.50m). Proactively expanded steering margin."
                )
            else:
                clear_reward = 1.50
                step_reward += clear_reward

                # Reinforce the chosen direction that succeeded
                if active_direction != 0:
                    old_dir = self.w_dir_bias
                    self.w_dir_bias = float(np.clip(
                        self.w_dir_bias + (self.eta_dir * 0.5) * active_direction,
                        -self.max_dir_bias, self.max_dir_bias
                    ))
                    updates["w_dir_bias"] = round(self.w_dir_bias - old_dir, 4)

                # Conclude episode
                self.episodic_memory.end_episode("PASS_SUCCESS", clear_reward, updates)
        elif self.in_encounter and safety_status == "SAFETY_OVERRIDE_OBSTACLE":
            # Conclude episode as failure
            self.in_encounter = False
            self.episodic_memory.end_episode("HARD_STOP_OVERRIDE", -1.0, updates)

        self.cumulative_reward += step_reward
        if updates:
            self.total_learning_steps += 1
            self.last_update_delta = updates

        return {
            "step_reward": round(step_reward, 3),
            "cumulative_reward": round(self.cumulative_reward, 3),
            "delta_steer_dist": round(self.delta_steer_dist, 3),
            "effective_steer_threshold": round(self.get_effective_steer_threshold(), 3),
            "w_dir_bias": round(self.w_dir_bias, 3),
            "speed_factor": round(self.speed_factor, 2),
            "updates": updates,
            "in_encounter": self.in_encounter,
        }

    def apply_feedback(self, reward: float, note: str = "") -> Dict[str, Any]:
        """Applies explicit reinforcement feedback to the adaptive navigation policy.

        Args:
            reward: Signed reinforcement scalar in range [-2.0, +2.0].
                    Positive values (> 0) reinforce efficient traversal and boost clutter velocity.
                    Negative values (< 0) penalize risky navigation, expand the proactive
                    steering margin (delta_steer_dist), and reduce clutter speed factor.
            note: Optional human or automated annotation describing the feedback event.

        Returns:
            Structured dictionary of updated policy parameters and episode outcomes.
        """
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            raise ValueError(f"Invalid reward value: {reward}. Must be a finite number.")

        clipped_reward = float(np.clip(reward, -2.0, 2.0))
        self.cumulative_reward += clipped_reward
        updates: Dict[str, float] = {}

        if clipped_reward > 0:
            old_sp = self.speed_factor
            self.speed_factor = min(1.0, round(self.speed_factor + 0.03 * clipped_reward, 4))
            if abs(self.speed_factor - old_sp) > 1e-4:
                updates["speed_factor"] = round(self.speed_factor - old_sp, 4)
            outcome = "MANUAL_REWARD"
            lesson = note or f"Manual reward ({clipped_reward:+.2f}). Reinforcing efficient traversal."
        elif clipped_reward < 0:
            old_steer = self.delta_steer_dist
            steer_inc = round(self.eta_steer * abs(clipped_reward), 4)
            self.delta_steer_dist = min(self.max_delta_steer, round(self.delta_steer_dist + steer_inc, 4))
            if abs(self.delta_steer_dist - old_steer) > 1e-4:
                updates["delta_steer_dist"] = round(self.delta_steer_dist - old_steer, 4)

            old_sp = self.speed_factor
            self.speed_factor = max(0.60, round(self.speed_factor - 0.04 * abs(clipped_reward), 4))
            if abs(self.speed_factor - old_sp) > 1e-4:
                updates["speed_factor"] = round(self.speed_factor - old_sp, 4)
            outcome = "MANUAL_PUNISH"
            lesson = note or f"Manual punishment ({clipped_reward:+.2f}). Proactively expanded steering margin."
        else:
            outcome = "NEUTRAL_FEEDBACK"
            lesson = note or "Neutral feedback (0.00). No parameter adjustments."

        self.total_learning_steps += 1
        self.last_update_delta = updates

        # Store in episodic memory
        ep_id = self.episodic_memory.start_episode(
            trigger="MANUAL_FEEDBACK",
            d_front=2.0,
            d_left=2.0,
            d_right=2.0,
        )
        self.episodic_memory.end_episode(
            outcome=outcome,
            reward=clipped_reward,
            parameter_updates=updates,
            lesson=lesson,
        )

        return {
            "status": "APPLIED",
            "reward": round(clipped_reward, 3),
            "cumulative_reward": round(self.cumulative_reward, 3),
            "delta_steer_dist": round(self.delta_steer_dist, 4),
            "effective_steer_threshold": round(self.get_effective_steer_threshold(), 4),
            "speed_factor": round(self.speed_factor, 4),
            "w_dir_bias": round(self.w_dir_bias, 4),
            "total_learning_steps": self.total_learning_steps,
            "updates": updates,
            "lesson": lesson,
            "episode_id": ep_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete learning state and episodic memory."""
        return {
            "delta_steer_dist": self.delta_steer_dist,
            "w_dir_bias": self.w_dir_bias,
            "speed_factor": self.speed_factor,
            "total_learning_steps": self.total_learning_steps,
            "cumulative_reward": self.cumulative_reward,
            "last_update_delta": self.last_update_delta,
            "episodes": self.episodic_memory.to_dict(),
        }

    def load_dict(self, data: Dict[str, Any]):
        """Restore learned parameters and episodic memory."""
        self.delta_steer_dist = float(data.get("delta_steer_dist", 0.0))
        self.w_dir_bias = float(data.get("w_dir_bias", 0.0))
        self.speed_factor = float(data.get("speed_factor", 1.0))
        self.total_learning_steps = int(data.get("total_learning_steps", 0))
        self.cumulative_reward = float(data.get("cumulative_reward", 0.0))
        self.last_update_delta = data.get("last_update_delta", {})
        if "episodes" in data:
            self.episodic_memory.load_dict(data["episodes"])
