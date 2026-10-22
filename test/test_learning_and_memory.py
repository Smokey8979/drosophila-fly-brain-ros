"""
Regression Test Suite for Adaptive Navigation Learning and Persistent Episodic Memory.

Verifies:
  1. Structured episodic memory creation from real sensor measurements.
  2. Outcome evaluation (PASS_SUCCESS, HARD_STOP_OVERRIDE, SAFETY_BLOCKED).
  3. Bounded policy adaptation (proactive clearance margin & directional preference).
  4. Deterministic before-and-after learning scenario:
     - Trial 1: Encounter produces aversive trap and updates steer margin.
     - Trial 2: Restored policy initiates earlier avoidance turn with increased clearance.
  5. Contextual episode retrieval and pattern matching.
  6. Memory consolidation and empirical summary reporting.
"""

import unittest
from fly_brain_ros.learning_subsystem import ConnectomeLearner, EpisodicMemory
from fly_brain_ros.safety_supervisor import SafetySupervisor


class TestLearningAndMemory(unittest.TestCase):
    """Validates bounded associative learning and episodic memory."""

    def setUp(self):
        self.learner = ConnectomeLearner(
            eta_steer=0.05,
            eta_dir=0.08,
            max_delta_steer=0.50,
            max_dir_bias=0.30,
        )
        self.safety = SafetySupervisor()
        self.safety.set_ready(True)

    def test_01_episode_creation_and_recording(self):
        """Obstacle proximity triggers active episode creation and clearance tracking."""
        # Step into encounter zone (d_front = 1.40m)
        out = self.learner.step(
            d_front=1.40, d_left=1.50, d_right=0.70,
            safety_status="SAFETY_AVOIDANCE", active_direction=1
        )
        self.assertTrue(out["in_encounter"])
        self.assertIsNotNone(self.learner.episodic_memory._current_episode)
        ep = self.learner.episodic_memory._current_episode
        self.assertEqual(ep["trigger"], "OBSTACLE_APPROACH")
        self.assertEqual(ep["initial_clearances"]["front"], 1.40)
        self.assertEqual(ep["min_clearance_reached"], 0.70)

    def test_02_pass_success_reward_and_lesson(self):
        """Clearing obstacle awards positive reinforcement and generates empirical lesson."""
        # Initiate encounter
        self.learner.step(1.40, 1.50, 0.70, "SAFETY_AVOIDANCE", active_direction=1)
        # Advance until clear (> 2.0m)
        out = self.learner.step(2.50, 2.50, 2.50, "NOMINAL", active_direction=1)
        self.assertFalse(out["in_encounter"])
        self.assertGreater(out["cumulative_reward"], 1.0)

        # Verify completed episode in memory
        eps = self.learner.episodic_memory.episodes
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["outcome"], "PASS_SUCCESS")
        self.assertIn("cleared successfully with minimum clearance", eps[0]["lesson"])

    def test_03_aversive_update_on_hard_stop(self):
        """Hard stop produces negative reinforcement and increases proactive steer margin."""
        init_delta = self.learner.delta_steer_dist
        # Near collision
        out = self.learner.step(0.20, 0.50, 0.50, "SAFETY_OVERRIDE_OBSTACLE", active_direction=-1)
        self.assertLess(out["step_reward"], 0.0)
        self.assertGreater(self.learner.delta_steer_dist, init_delta)
        self.assertIn("delta_steer_dist", out["updates"])
        # Should bias away from right (-1) towards left (+1)
        self.assertGreater(self.learner.w_dir_bias, 0.0)

    def test_04_bounded_parameters(self):
        """Repeated aversive updates strictly respect defined parameter bounds."""
        for _ in range(50):
            self.learner.step(0.20, 0.50, 0.50, "SAFETY_OVERRIDE_OBSTACLE", active_direction=-1)

        self.assertLessEqual(self.learner.delta_steer_dist, self.learner.max_delta_steer)
        self.assertLessEqual(self.learner.w_dir_bias, self.learner.max_dir_bias)
        self.assertGreaterEqual(self.learner.speed_factor, 0.60)

    def test_05_deterministic_before_after_learning(self):
        """Measurable retained policy update changes behavior in identical subsequent scenario."""
        # --- TRIAL 1 (Baseline) ---
        learner1 = ConnectomeLearner(eta_steer=0.10)
        safety1 = SafetySupervisor()
        safety1.set_ready(True)

        # Baseline steer threshold is 1.50m (slowdown threshold is 1.80m).
        # At 1.55m, baseline is in SAFETY_SLOWDOWN (decelerating, but not avoiding yet).
        _, _, status1 = safety1.filter_command(
            raw_vx=0.20, raw_wz=0.0,
            front_obstacle_dist=1.55, left_obstacle_dist=2.0, right_obstacle_dist=2.0,
            steer_start_override=learner1.get_effective_steer_threshold()
        )
        self.assertEqual(status1, "SAFETY_SLOWDOWN")

        # Now simulate aversive event during Trial 1 (e.g. trapped corner)
        learner1.step(0.20, 0.50, 0.50, "SAFETY_OVERRIDE_OBSTACLE", active_direction=-1)
        self.assertGreater(learner1.delta_steer_dist, 0.0)
        learned_margin = learner1.delta_steer_dist

        # Save and restore learner state to verify persistence
        saved_dict = learner1.to_dict()

        # --- TRIAL 2 (Post-Learning under Identical Sensor Conditions) ---
        learner2 = ConnectomeLearner(eta_steer=0.10)
        learner2.load_dict(saved_dict)
        safety2 = SafetySupervisor()
        safety2.set_ready(True)

        self.assertEqual(learner2.delta_steer_dist, learned_margin)
        effective_steer2 = learner2.get_effective_steer_threshold()
        self.assertGreater(effective_steer2, 1.50)

        # At the exact same distance (1.55m), the learned agent initiates avoidance proactively!
        _, _, status2 = safety2.filter_command(
            raw_vx=0.20, raw_wz=0.0,
            front_obstacle_dist=1.55, left_obstacle_dist=2.0, right_obstacle_dist=2.0,
            steer_start_override=effective_steer2
        )
        self.assertEqual(status2, "SAFETY_AVOIDANCE")

    def test_06_episodic_memory_retrieval(self):
        """Pattern matcher accurately retrieves historical encounter by clearance profile."""
        # Store a past episode at front=1.35m, left=0.60m, right=1.80m
        self.learner.episodic_memory.start_episode("TEST_ENCOUNTER", 1.35, 0.60, 1.80)
        self.learner.episodic_memory.end_episode("PASS_SUCCESS", 1.5, {}, "Left clearance obstacle successfully passed.")

        # Query with similar clearances (front=1.38m, left=0.62m, right=1.75m)
        matched_ep, sim = self.learner.episodic_memory.find_matching_episode(1.38, 0.62, 1.75, dist_threshold=0.50)
        self.assertIsNotNone(matched_ep)
        self.assertGreater(sim, 0.70)
        self.assertIn("Left clearance", matched_ep["lesson"])

    def test_07_consolidation_statistics(self):
        """Consolidation accurately extracts pass success rates and directional counts."""
        mem = EpisodicMemory()
        # Episode 1: Left pass success
        mem.start_episode("E1", 1.4, 2.0, 0.8)
        mem._current_episode["chosen_direction"] = 1
        mem._current_episode["min_clearance_reached"] = 0.8
        mem.end_episode("PASS_SUCCESS", 1.5, {})

        # Episode 2: Right pass success
        mem.start_episode("E2", 1.4, 0.8, 2.0)
        mem._current_episode["chosen_direction"] = -1
        mem._current_episode["min_clearance_reached"] = 0.9
        mem.end_episode("PASS_SUCCESS", 1.5, {})

        # Episode 3: Hard stop override
        mem.start_episode("E3", 1.0, 0.5, 0.5)
        mem._current_episode["chosen_direction"] = 1
        mem._current_episode["min_clearance_reached"] = 0.2
        mem.end_episode("HARD_STOP_OVERRIDE", -1.0, {})

        report = mem.consolidate()
        self.assertEqual(report["total_episodes"], 3)
        self.assertAlmostEqual(report["success_rate"], 2/3, places=2)
        self.assertEqual(report["left_success_count"], 1)
        self.assertEqual(report["right_success_count"], 1)
        self.assertEqual(report["hard_stop_count"], 1)
        self.assertIn("Consolidated 3 episodes", report["summary"])


if __name__ == "__main__":
    unittest.main()
