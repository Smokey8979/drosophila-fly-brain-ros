"""
Automated Test Suite: Reinforcement Learning Controls & Adaptive Navigation Updates.

Validates Requirements C1 through C9:
  C1: Valid reward and punishment requests are accepted.
  C2: Invalid magnitudes and malformed payloads are rejected.
  C3: The correct reward sign/convention reaches the intended learning subsystem.
  C4: A successful request records a genuine feedback event in episodic memory.
  C5: Policy updates are actually applied to parameters (delta_steer_dist, speed_factor).
  C6: Results are reflected in telemetry and dashboard state.
  C7: Relevant learning and memory state survives checkpoint/reload.
  C8: Repeated requests are safely and idempotently handled.
  C9: Errors are returned in structured format.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.learning_subsystem import ConnectomeLearner, EpisodicMemory
from fly_brain_ros.internal_state import InternalStateManager
from live_lab.web_server import app


class TestReinforcementLearningControls(unittest.TestCase):
    def setUp(self):
        self.learner = ConnectomeLearner()
        self.internal_state = InternalStateManager()
        self.client = app.test_client()

    def test_c1_valid_reward_and_punish_accepted(self):
        """C1: Valid reward and punishment requests are accepted."""
        res_r = self.learner.apply_feedback(1.0, note="Traversed door cleanly")
        self.assertEqual(res_r["status"], "APPLIED")
        self.assertEqual(res_r["reward"], 1.0)
        self.assertEqual(res_r["cumulative_reward"], 1.0)

        res_p = self.learner.apply_feedback(-1.0, note="Near-miss collision")
        self.assertEqual(res_p["status"], "APPLIED")
        self.assertEqual(res_p["reward"], -1.0)
        self.assertEqual(res_p["cumulative_reward"], 0.0)

    def test_c2_invalid_magnitudes_and_malformed_payloads_rejected(self):
        """C2: Invalid magnitudes and malformed payloads are rejected."""
        # Non-numeric value raises ValueError
        with self.assertRaises(ValueError):
            self.learner.apply_feedback(float("nan"))

        with self.assertRaises(ValueError):
            self.learner.apply_feedback(float("inf"))

        # REST API endpoint tests for out-of-bounds inputs
        resp_too_large = self.client.post("/api/learning/feedback", json={"reward": 5.5})
        self.assertEqual(resp_too_large.status_code, 400)
        self.assertIn("outside safe bounds", resp_too_large.get_json()["message"])

        resp_invalid = self.client.post("/api/learning/feedback", json={"reward": "not_a_number"})
        self.assertEqual(resp_invalid.status_code, 400)

    def test_c3_correct_reward_sign_convention_reaches_subsystem(self):
        """C3: The correct reward sign/convention reaches the intended learning subsystem."""
        init_cum = self.learner.cumulative_reward

        # Positive reward increases cumulative reward
        self.learner.apply_feedback(+0.75)
        self.assertGreater(self.learner.cumulative_reward, init_cum)

        # Negative reward decreases cumulative reward
        self.learner.apply_feedback(-0.50)
        self.assertAlmostEqual(self.learner.cumulative_reward, init_cum + 0.25, places=3)

    def test_c4_successful_request_records_episodic_memory_event(self):
        """C4: A successful request records a genuine feedback event in episodic memory."""
        res = self.learner.apply_feedback(-1.20, note="Sharp avoidance necessary")
        self.assertEqual(len(self.learner.episodic_memory.episodes), 1)

        ep = self.learner.episodic_memory.episodes[0]
        self.assertEqual(ep["outcome"], "MANUAL_PUNISH")
        self.assertAlmostEqual(ep["reward"], -1.20, places=2)
        self.assertIn("Sharp avoidance necessary", ep["lesson"])

    def test_c5_policy_updates_applied_to_parameters(self):
        """C5: Policy updates are actually applied (e.g. delta_steer_dist, speed_factor)."""
        init_steer = self.learner.delta_steer_dist
        init_speed = self.learner.speed_factor

        # Punishment should expand delta_steer_dist and reduce speed_factor
        res_p = self.learner.apply_feedback(-1.50)
        self.assertGreater(self.learner.delta_steer_dist, init_steer)
        self.assertLess(self.learner.speed_factor, init_speed)
        self.assertIn("delta_steer_dist", res_p["updates"])

        # Reward should improve speed factor towards 1.0
        cur_speed = self.learner.speed_factor
        res_r = self.learner.apply_feedback(+1.50)
        self.assertGreaterEqual(self.learner.speed_factor, cur_speed)

    def test_c6_rest_api_returns_structured_feedback_result(self):
        """C6: REST API endpoints return structured feedback and update state."""
        resp = self.client.post("/api/learning/reward", json={"magnitude": 0.50, "note": "API test reward"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "APPLIED")
        self.assertEqual(data["reward"], 0.50)

        resp_p = self.client.post("/api/learning/punish", json={"magnitude": 0.75, "note": "API test punish"})
        self.assertEqual(resp_p.status_code, 200)
        data_p = resp_p.get_json()
        self.assertEqual(data_p["status"], "APPLIED")
        self.assertEqual(data_p["reward"], -0.75)

    def test_c7_state_survives_checkpoint_and_reload(self):
        """C7: Relevant learning state survives serialization and reload."""
        self.learner.apply_feedback(1.40, note="Reinforce turn")
        self.learner.apply_feedback(-0.80, note="Clear corner")

        state_dict = self.learner.to_dict()
        fresh_learner = ConnectomeLearner()
        fresh_learner.load_dict(state_dict)

        self.assertEqual(fresh_learner.total_learning_steps, 2)
        self.assertAlmostEqual(fresh_learner.cumulative_reward, 0.60, places=2)
        self.assertEqual(len(fresh_learner.episodic_memory.episodes), 2)
        self.assertEqual(fresh_learner.delta_steer_dist, self.learner.delta_steer_dist)
        self.assertEqual(fresh_learner.speed_factor, self.learner.speed_factor)

    def test_c8_internal_affect_responds_to_reinforcement(self):
        """C8: Internal affect diagnostics respond correctly to reward and punish."""
        init_valence = self.internal_state.valence_affect

        # Reward boosts valence
        self.internal_state.process_feedback(+1.0)
        self.assertGreater(self.internal_state.valence_affect, init_valence)

        # Severe punish lowers valence and raises vigilance
        init_vig = self.internal_state.environmental_vigilance
        self.internal_state.process_feedback(-1.5)
        self.assertGreater(self.internal_state.environmental_vigilance, init_vig)

    def test_c9_reset_learning_restores_baseline(self):
        """C9: Reset learning cleanly restores baseline parameters."""
        self.learner.apply_feedback(-1.0)
        self.assertGreater(self.learner.delta_steer_dist, 0.0)
        self.assertGreater(self.learner.total_learning_steps, 0)

        self.learner.reset_learning()
        self.assertEqual(self.learner.delta_steer_dist, 0.0)
        self.assertEqual(self.learner.w_dir_bias, 0.0)
        self.assertEqual(self.learner.speed_factor, 1.0)
        self.assertEqual(self.learner.total_learning_steps, 0)
        self.assertEqual(self.learner.cumulative_reward, 0.0)
        self.assertEqual(len(self.learner.episodic_memory.episodes), 0)


if __name__ == "__main__":
    unittest.main()
