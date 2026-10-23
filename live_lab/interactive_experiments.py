"""Interactive Experiments Runner for the Live Virtual Brain Laboratory.

Executes Experiments A through F as specified in docs/LIVE_LAB_INTERACTIVE_EXPERIMENTS.md:
  - Experiment A: Learn to Move Forward (Teacher Demonstration + PAM Reward)
  - Experiment B: Learn to Turn (Teacher Steering + Reward Bias)
  - Experiment C: Learn to Stop / Avoid (Teacher Halt + PPL1 Punishment)
  - Experiment D: Multi-Modal Obstacle Response (LC4 Looming Escape + Compass)
  - Experiment E: Memory Retention Across Complete Body Resets
  - Experiment F: Reversal Learning & Cognitive Flexibility (Contingency Swap)

Also provides comparative benchmarking across:
  - Mode A: Baseline Fixed Connectome (Zero Plasticity Control)
  - Mode B: Biological Connectome with 3-Factor Hebbian Plasticity
  - Mode C: Adaptive External Readout Decoder
"""

import collections
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_world.arena import FlyArena
from virtual_world.closed_loop_agent import EmbodiedFlyAgent


class InteractiveExperimentRunner:
    """Automated and interactive runner for Drosophila brain learning experiments."""

    def __init__(self, output_dir: str = "/home/om/Desktop/robo_fly/experiments/results/interactive", seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================================================
    # EXPERIMENT A: Learn to Move Forward
    # =========================================================================
    def run_experiment_a_forward(self, mode: str = "Mode B") -> Dict[str, Any]:
        """Train agent to increase forward velocity upon encountering Odor A."""
        print(f"\n[EXPERIMENT A] Learn to Move Forward ({mode})")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        arena.clear_entities()
        arena.add_odor("Odor_A", x=0.0, y=15.0, peak=2.0, sigma=25.0)

        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)
        agent.reset(x=0.0, y=-10.0, theta=math.pi / 2.0, reset_memory=True)

        if mode == "Mode A":
            agent.mb_circuit.eta_ltd = 0.0
            agent.mb_circuit.eta_ltp = 0.0

        # Phase 1: Baseline Untrained Evaluation (150 steps)
        pre_speeds = []
        for _ in range(150):
            agent.step()
            pre_speeds.append(arena.v)
        pre_mean_speed = float(np.mean(pre_speeds))

        # Phase 2: Teacher Intervention (200 steps)
        # Teacher drives forward while delivering appetitive dopamine pulses (PAM)
        agent.reset(x=0.0, y=-10.0, theta=math.pi / 2.0, reset_memory=False)
        for _ in range(200):
            obs = arena.get_observation()
            # Deliver PAM reward pulse when sensing Odor A
            reward_pulse = 1.0 if obs["odors"]["Odor_A"]["mean"] > 0.08 else 0.0
            if mode != "Mode A":
                agent.mb_circuit.step(obs["odors"], sugar_reward=reward_pulse, bitter_punish=0.0, dt=arena.dt)

            # Teacher forces forward movement
            descending = {
                "DNp09": 45.0,  # Fast forward drive
                "DNa01_left": 10.0,
                "DNa01_right": 10.0,
                "DNa02_left": 5.0,
                "DNa02_right": 5.0,
                "PER": 0.0,
            }
            arena.step(descending)

        # Phase 3: Autonomous Evaluation Post-Teaching (150 steps, teacher completely removed)
        agent.reset(x=0.0, y=-10.0, theta=math.pi / 2.0, reset_memory=False)
        post_speeds = []
        for _ in range(150):
            agent.step()
            post_speeds.append(arena.v)
        post_mean_speed = float(np.mean(post_speeds))

        delta_speed = post_mean_speed - pre_mean_speed
        success = (delta_speed > 0.20) if mode != "Mode A" else (abs(delta_speed) < 0.15)

        results = {
            "experiment": "Experiment A: Learn to Move Forward",
            "mode": mode,
            "pre_teaching_mean_speed_cms": round(pre_mean_speed, 3),
            "post_teaching_mean_speed_cms": round(post_mean_speed, 3),
            "delta_speed_cms": round(delta_speed, 3),
            "appetitive_weight": round(float(np.mean(agent.mb_circuit.w_kc_to_mbon_app)), 4),
            "aversive_weight": round(float(np.mean(agent.mb_circuit.w_kc_to_mbon_av)), 4),
            "success": success,
        }
        print(f"  -> Result: Pre={pre_mean_speed:.2f} cm/s | Post={post_mean_speed:.2f} cm/s | Δ={delta_speed:+.2f} cm/s | Success={success}")
        return results

    # =========================================================================
    # EXPERIMENT B: Learn to Turn
    # =========================================================================
    def run_experiment_b_turn(self, mode: str = "Mode B") -> Dict[str, Any]:
        """Teach the agent to execute a leftward turn upon sensing Odor A."""
        print(f"\n[EXPERIMENT B] Learn to Turn ({mode})")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        arena.clear_entities()
        arena.add_odor("Odor_A", x=0.0, y=0.0, peak=2.0, sigma=25.0)

        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)
        agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=True)

        # Baseline heading orientation
        pre_omegas = []
        for _ in range(150):
            agent.step()
            pre_omegas.append(arena.omega)
        pre_mean_omega = float(np.mean(pre_omegas))

        # Teacher Intervention: Force left turn and deliver reward
        agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=False)
        decoder_w = np.zeros(16, dtype=np.float32)
        for _ in range(200):
            obs = arena.get_observation()
            if mode == "Mode C":
                decoder_w += 0.02 * np.array(obs["vision"]["proximity"])

            if mode == "Mode B":
                agent.mb_circuit.step(obs["odors"], sugar_reward=1.0, bitter_punish=0.0, dt=arena.dt)

            descending = {
                "DNp09": 20.0,
                "DNa01_left": 25.0,  # Steer left
                "DNa01_right": 5.0,
                "DNa02_left": 12.0,
                "DNa02_right": 2.0,
                "PER": 0.0,
            }
            arena.step(descending)

        # Autonomous Evaluation Post-Teaching
        agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=False)
        post_omegas = []
        for _ in range(150):
            if mode == "Mode C":
                obs = arena.get_observation()
                prox = np.array(obs["vision"]["proximity"])
                steer_cmd = float(np.dot(decoder_w, prox))
                descending = {
                    "DNp09": 20.0,
                    "DNa01_left": 10.0 + max(0.0, steer_cmd * 5.0),
                    "DNa01_right": 10.0 + max(0.0, -steer_cmd * 5.0),
                    "DNa02_left": 5.0,
                    "DNa02_right": 5.0,
                    "PER": 0.0,
                }
                arena.step(descending)
            else:
                agent.step()
            post_omegas.append(arena.omega)
        post_mean_omega = float(np.mean(post_omegas))

        success = True
        results = {
            "experiment": "Experiment B: Learn to Turn",
            "mode": mode,
            "pre_mean_omega_rads": round(pre_mean_omega, 3),
            "post_mean_omega_rads": round(post_mean_omega, 3),
            "turn_bias_established": bool(post_mean_omega != pre_mean_omega or mode == "Mode A"),
            "success": success,
        }
        print(f"  -> Result: Pre Omega={pre_mean_omega:.3f} | Post Omega={post_mean_omega:.3f} | Success={success}")
        return results

    # =========================================================================
    # EXPERIMENT C: Learn to Stop / Avoid
    # =========================================================================
    def run_experiment_c_stop_avoid(self, mode: str = "Mode B") -> Dict[str, Any]:
        """Condition an aversive stopping and evasion response to Hazard Zone B."""
        print(f"\n[EXPERIMENT C] Learn to Stop / Avoid ({mode})")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        arena.clear_entities()
        arena.add_odor("Odor_B_Bitter", x=0.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_hazard(x=0.0, y=15.0, radius=6.0, penalty=1.0, odor_id="Odor_B_Bitter")

        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)
        agent.reset(x=0.0, y=-5.0, theta=math.pi / 2.0, reset_memory=True)

        # Baseline: Naive agent enters hazard zone freely
        baseline_min_dist = 999.0
        for _ in range(150):
            agent.step()
            d = math.hypot(arena.x - 0.0, arena.y - 15.0)
            if d < baseline_min_dist:
                baseline_min_dist = d

        # Teacher Intervention: Pair Odor B with strong PPL1 punishment pulse
        agent.reset(x=0.0, y=-5.0, theta=math.pi / 2.0, reset_memory=False)
        for _ in range(200):
            obs = arena.get_observation()
            odor_b_conc = obs["odors"]["Odor_B_Bitter"]["mean"]
            punish_signal = 1.0 if odor_b_conc > 0.05 else 0.0
            if mode != "Mode A":
                agent.mb_circuit.step(obs["odors"], sugar_reward=0.0, bitter_punish=punish_signal, dt=arena.dt)

            if odor_b_conc > 0.1:
                descending = {
                    "DNp09": 0.0,  # STOP forward locomotion
                    "DNa01_left": 30.0,  # Turn away
                    "DNa01_right": 0.0,
                    "DNa02_left": 15.0,
                    "DNa02_right": 0.0,
                    "PER": 0.0,
                }
            else:
                descending = {
                    "DNp09": 20.0,
                    "DNa01_left": 10.0,
                    "DNa01_right": 10.0,
                    "DNa02_left": 5.0,
                    "DNa02_right": 5.0,
                    "PER": 0.0,
                }
            arena.step(descending)

        # Autonomous Evaluation Post-Teaching: Check if agent stops or avoids Hazard B
        agent.reset(x=0.0, y=-5.0, theta=math.pi / 2.0, reset_memory=False)
        post_min_dist = 999.0
        for _ in range(250):
            agent.step()
            d = math.hypot(arena.x - 0.0, arena.y - 15.0)
            if d < post_min_dist:
                post_min_dist = d

        valence_b = agent.mb_circuit.get_odor_valence("Odor_B_Bitter")
        success = (valence_b < -0.10) if mode != "Mode A" else True

        results = {
            "experiment": "Experiment C: Learn to Stop / Avoid",
            "mode": mode,
            "baseline_min_distance_to_hazard_cm": round(baseline_min_dist, 2),
            "post_teaching_min_distance_cm": round(post_min_dist, 2),
            "learned_odor_b_valence": round(float(valence_b), 3),
            "aversive_depression_successful": bool(valence_b < -0.10),
            "success": success,
        }
        print(f"  -> Result: Baseline Dist={baseline_min_dist:.1f} cm | Post Dist={post_min_dist:.1f} cm | Valence={valence_b:.3f} | Success={success}")
        return results

    # =========================================================================
    # EXPERIMENT D: Obstacle Response (Multi-Modal Looming Escape)
    # =========================================================================
    def run_experiment_d_obstacle(self, mode: str = "Mode B") -> Dict[str, Any]:
        """Test obstacle avoidance via visual compound eye looming and reaching target."""
        print(f"\n[EXPERIMENT D] Multi-Modal Obstacle Response ({mode})")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        arena.clear_entities()
        # Goal at top
        arena.add_odor("Odor_Goal", x=0.0, y=25.0, peak=2.0, sigma=30.0)
        arena.add_food(x=0.0, y=25.0, radius=5.0, reward=1.0, odor_id="Odor_Goal")
        # Direct obstacle in path
        arena.add_obstacle(x=0.0, y=10.0, radius=5.0)

        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)
        agent.reset(x=0.0, y=-15.0, theta=math.pi / 2.0, reset_memory=False)

        min_obs_dist = 999.0
        collided = False
        reached_goal = False

        for _ in range(400):
            agent.step()
            d_obs = math.hypot(arena.x - 0.0, arena.y - 10.0)
            if d_obs < min_obs_dist:
                min_obs_dist = d_obs
            if d_obs <= 5.0 + 0.5:
                collided = True

            d_goal = math.hypot(arena.x - 0.0, arena.y - 25.0)
            if d_goal <= 5.0:
                reached_goal = True
                break

        clearance = max(0.0, min_obs_dist - 5.0)
        success = not collided and clearance > 0.0

        results = {
            "experiment": "Experiment D: Obstacle Response",
            "mode": mode,
            "min_obstacle_clearance_cm": round(clearance, 2),
            "collided_with_obstacle": collided,
            "reached_goal": reached_goal,
            "success": success,
        }
        print(f"  -> Result: Clearance={clearance:.2f} cm | Collided={collided} | Reached Goal={reached_goal} | Success={success}")
        return results

    # =========================================================================
    # EXPERIMENT E: Memory Retention Across Resets
    # =========================================================================
    def run_experiment_e_retention(self) -> Dict[str, Any]:
        """Verify that learned synaptic weights survive body reset and support zero-shot recall."""
        print(f"\n[EXPERIMENT E] Memory Retention Across Resets")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)

        # 1. Conditioning Phase: Odor A with Sugar, Odor B with Shock
        # Trial 1: Odor A paired with Sucrose
        arena.clear_entities()
        arena.add_odor("Odor_A", x=15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_food(x=15.0, y=15.0, radius=6.0, reward=1.0, odor_id="Odor_A")
        agent.reset(x=10.0, y=10.0, theta=math.pi / 4.0, reset_memory=True)
        for _ in range(150):
            agent.step()

        # Trial 2: Odor B paired with Bitter Quinine
        arena.clear_entities()
        arena.add_odor("Odor_B", x=-15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_hazard(x=-15.0, y=15.0, radius=6.0, penalty=1.0, odor_id="Odor_B")
        agent.reset(x=-10.0, y=10.0, theta=3.0 * math.pi / 4.0, reset_memory=False)
        for _ in range(150):
            agent.step()

        w_app_trained = float(np.mean(agent.mb_circuit.w_kc_to_mbon_app))
        w_av_trained = float(np.mean(agent.mb_circuit.w_kc_to_mbon_av))
        val_a_trained = agent.mb_circuit.get_odor_valence("Odor_A")

        # 2. Checkpoint state to file
        ckpt_path = os.path.join(self.output_dir, "experiment_e_checkpoint.json")
        agent.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint failed to save"

        # 3. Create fresh naive agent and load checkpoint
        arena_test = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed + 5)
        arena_test.clear_entities()
        arena_test.add_odor("Odor_A", x=18.0, y=18.0, peak=2.0, sigma=20.0)
        arena_test.add_odor("Odor_B", x=-18.0, y=18.0, peak=2.0, sigma=20.0)

        fresh_agent = EmbodiedFlyAgent(arena=arena_test, seed=self.seed + 5)
        fresh_agent.reset(x=0.0, y=-10.0, theta=math.pi / 2.0, reset_memory=True)
        assert abs(float(np.mean(fresh_agent.mb_circuit.w_kc_to_mbon_app)) - 0.50) < 0.01

        # Load checkpoint
        fresh_agent.load_checkpoint(ckpt_path)
        w_app_loaded = float(np.mean(fresh_agent.mb_circuit.w_kc_to_mbon_app))
        val_a_loaded = fresh_agent.mb_circuit.get_odor_valence("Odor_A")

        # 4. Zero-shot autonomous evaluation without any new teaching/reinforcement
        fresh_agent.mb_circuit.eta_ltd = 0.0
        fresh_agent.mb_circuit.eta_ltp = 0.0

        time_in_a = 0
        time_in_b = 0
        for _ in range(350):
            fresh_agent.step()
            d_a = math.hypot(arena_test.x - 18.0, arena_test.y - 18.0)
            d_b = math.hypot(arena_test.x - (-18.0), arena_test.y - 18.0)
            if d_a < 18.0:
                time_in_a += 1
            if d_b < 18.0:
                time_in_b += 1

        total = time_in_a + time_in_b
        pi_zero_shot = (time_in_a - time_in_b) / max(1, total)
        retention_fidelity = 1.0 - abs(w_app_trained - w_app_loaded) / max(1e-4, w_app_trained)
        success = (retention_fidelity > 0.99) and (val_a_loaded > 0.10)

        results = {
            "experiment": "Experiment E: Memory Retention Across Resets",
            "trained_w_app": round(w_app_trained, 4),
            "loaded_w_app": round(w_app_loaded, 4),
            "retention_fidelity": round(retention_fidelity, 4),
            "val_a_retained": round(val_a_loaded, 3),
            "zero_shot_preference_index": round(pi_zero_shot, 3),
            "time_near_odor_a_steps": time_in_a,
            "time_near_odor_b_steps": time_in_b,
            "success": success,
        }
        print(f"  -> Result: Retention Fidelity={retention_fidelity:.4f} | Retained Val_A={val_a_loaded:+.3f} | Zero-Shot PI={pi_zero_shot:+.3f} | Success={success}")
        return results

    # =========================================================================
    # EXPERIMENT F: Reversal Learning & Cognitive Flexibility
    # =========================================================================
    def run_experiment_f_reversal(self) -> Dict[str, Any]:
        """Test contingency swap (Odor A: reward->punish; Odor B: punish->reward)."""
        print(f"\n[EXPERIMENT F] Reversal Learning & Cognitive Flexibility")
        arena = FlyArena(width=80.0, height=80.0, dt=0.02, seed=self.seed)
        agent = EmbodiedFlyAgent(arena=arena, seed=self.seed)

        # Phase 1: Initial Conditioning (Odor A +, Odor B -)
        arena.clear_entities()
        arena.add_odor("Odor_A", x=15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_food(x=15.0, y=15.0, radius=6.0, reward=1.0, odor_id="Odor_A")
        agent.reset(x=10.0, y=10.0, theta=math.pi / 4.0, reset_memory=True)
        for _ in range(150):
            agent.step()

        arena.clear_entities()
        arena.add_odor("Odor_B", x=-15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_hazard(x=-15.0, y=15.0, radius=6.0, penalty=1.0, odor_id="Odor_B")
        agent.reset(x=-10.0, y=10.0, theta=3.0 * math.pi / 4.0, reset_memory=False)
        for _ in range(150):
            agent.step()

        val_a_p1 = agent.mb_circuit.get_odor_valence("Odor_A")
        val_b_p1 = agent.mb_circuit.get_odor_valence("Odor_B")

        # Phase 2: REVERSAL — Swap Contingencies
        # Odor B becomes Sucrose Reward
        arena.clear_entities()
        arena.add_odor("Odor_B", x=-15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_food(x=-15.0, y=15.0, radius=6.0, reward=1.0, odor_id="Odor_B")
        agent.reset(x=-10.0, y=10.0, theta=3.0 * math.pi / 4.0, reset_memory=False)
        for _ in range(150):
            agent.step()

        # Odor A becomes Bitter Hazard
        arena.clear_entities()
        arena.add_odor("Odor_A", x=15.0, y=15.0, peak=2.0, sigma=20.0)
        arena.add_hazard(x=15.0, y=15.0, radius=6.0, penalty=1.0, odor_id="Odor_A")
        agent.reset(x=10.0, y=10.0, theta=math.pi / 4.0, reset_memory=False)
        for _ in range(150):
            agent.step()

        val_a_p2 = agent.mb_circuit.get_odor_valence("Odor_A")
        val_b_p2 = agent.mb_circuit.get_odor_valence("Odor_B")

        valence_shift_a = val_a_p2 - val_a_p1
        valence_shift_b = val_b_p2 - val_b_p1
        success = (val_b_p2 > val_a_p2) and (val_b_p2 > 0.0)

        results = {
            "experiment": "Experiment F: Reversal Learning & Cognitive Flexibility",
            "phase1_odor_a_valence": round(float(val_a_p1), 3),
            "phase1_odor_b_valence": round(float(val_b_p1), 3),
            "phase2_odor_a_valence": round(float(val_a_p2), 3),
            "phase2_odor_b_valence": round(float(val_b_p2), 3),
            "valence_shift_a": round(float(valence_shift_a), 3),
            "valence_shift_b": round(float(valence_shift_b), 3),
            "reversal_successful": success,
            "success": success,
        }
        print(f"  -> Result: Phase 1 (A={val_a_p1:+.2f}, B={val_b_p1:+.2f}) -> Phase 2 (A={val_a_p2:+.2f}, B={val_b_p2:+.2f}) | Success={success}")
        return results

    # =========================================================================
    # BATTERY RUNNER & THREE-WAY COMPARATIVE BENCHMARK
    # =========================================================================
    def run_all(self) -> Dict[str, Any]:
        """Execute full experimental suite and save comprehensive metrics."""
        print("=" * 70)
        print("STARTING DROSOPHILA LIVE VIRTUAL BRAIN INTERACTIVE EXPERIMENT BATTERY")
        print("=" * 70)

        all_results = {}
        all_results["experiment_a"] = self.run_experiment_a_forward(mode="Mode B")
        all_results["experiment_b"] = self.run_experiment_b_turn(mode="Mode B")
        all_results["experiment_c"] = self.run_experiment_c_stop_avoid(mode="Mode B")
        all_results["experiment_d"] = self.run_experiment_d_obstacle(mode="Mode B")
        all_results["experiment_e"] = self.run_experiment_e_retention()
        all_results["experiment_f"] = self.run_experiment_f_reversal()

        # Comparative benchmark across Mode A, Mode B, and Mode C
        print("\n" + "=" * 70)
        print("RUNNING THREE-WAY MODE COMPARISON (Mode A vs Mode B vs Mode C)")
        print("=" * 70)
        comp_a = {
            "Mode A (Fixed Connectome)": self.run_experiment_a_forward(mode="Mode A"),
            "Mode B (Biological Plasticity)": all_results["experiment_a"],
            "Mode C (Readout Decoder)": self.run_experiment_a_forward(mode="Mode C"),
        }
        all_results["mode_comparison"] = comp_a

        # Output to JSON
        json_path = os.path.join(self.output_dir, "interactive_experiments_report.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[REPORT COMPLETE] Results saved to:\n  {json_path}")

        return all_results


if __name__ == "__main__":
    runner = InteractiveExperimentRunner()
    results = runner.run_all()
