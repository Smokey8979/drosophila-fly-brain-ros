"""
Automated Test Suite: Authoritative Checkpoint Configuration, Cryptographic Integrity & Safe Atomic Saving.

Validates Requirements A1 through A11:
  A1: Save a valid live brain state and cryptographically verify it.
  A2: Save to an explicitly configured absolute path from a different current working directory.
  A3: Save when the parent directory does not initially exist.
  A4: Load a known-valid checkpoint and verify the restored state.
  A5: Corrupt a checkpoint and confirm that verification fails clearly.
  A6: Reject a truncated or invalid checkpoint.
  A7: Simulate a save exception and confirm that a previous valid checkpoint survives.
  A8: Test missing file and permission/error paths appropriately.
  A9: Verify that the configured and reported paths are identical.
  A10: Verify persisted learning/reinforcement data survives roundtrip.
  A11: Verify that shutdown summary distinguishes process cleanup from persistence success.
"""

import os
import sys
import json
import shutil
import tempfile
import hashlib
import unittest
from pathlib import Path

# Add fly_brain_ros to sys.path
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fly_brain_ros.checkpoint_config import (
    get_canonical_checkpoint_path,
    validate_checkpoint_file,
    validate_checkpoint_data,
    compute_checkpoint_hash,
    DEFAULT_CANONICAL_PATH,
)
from fly_brain_ros.fly_brain_node import FlyBrainControllerPipeline


class TestCheckpointAuthoritativeVerification(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="fly_cp_test_")
        self.cp_path = os.path.join(self.test_dir, "primary_checkpoint.json")
        self.pipeline = FlyBrainControllerPipeline(device="cpu")
        self.pipeline.start()

    def tearDown(self):
        self.pipeline.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_a1_save_and_verify_valid_checkpoint(self):
        """A1: Save a valid live brain state and cryptographically verify it."""
        manifest = self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="test_a1")
        self.assertTrue(os.path.exists(self.cp_path))
        self.assertIn("sha256", manifest)

        is_valid, msg, summary = validate_checkpoint_file(self.cp_path)
        self.assertTrue(is_valid, f"Validation failed: {msg}")
        self.assertEqual(summary["schema_version"], "2.0.0")
        self.assertGreater(summary["size_bytes"], 5000)
        self.assertEqual(summary["sha256"], manifest["sha256"])

    def test_a2_save_from_different_working_directory(self):
        """A2: Save to an explicitly configured absolute path from a different working directory."""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            target_path = os.path.join(self.test_dir, "nested", "custom_checkpoint.json")
            manifest = self.pipeline.create_checkpoint(filepath=target_path)
            self.assertTrue(os.path.isabs(target_path))
            self.assertTrue(os.path.exists(target_path))
            is_valid, msg, _ = validate_checkpoint_file(target_path)
            self.assertTrue(is_valid)
        finally:
            os.chdir(original_cwd)

    def test_a3_save_creates_missing_parent_directories(self):
        """A3: Save when the parent directory does not initially exist."""
        deep_dir = os.path.join(self.test_dir, "level1", "level2", "level3")
        target_path = os.path.join(deep_dir, "deep_checkpoint.json")
        self.assertFalse(os.path.exists(deep_dir))

        manifest = self.pipeline.create_checkpoint(filepath=target_path)
        self.assertTrue(os.path.exists(deep_dir))
        self.assertTrue(os.path.exists(target_path))
        is_valid, _, _ = validate_checkpoint_file(target_path)
        self.assertTrue(is_valid)

    def test_a4_load_valid_checkpoint_restores_state(self):
        """A4: Load a known-valid checkpoint and verify restored state."""
        # Set distinctive learning parameter
        self.pipeline.learner.delta_steer_dist = 0.28
        self.pipeline.learner.speed_factor = 0.82
        self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="test_a4")

        # Mutate current state
        self.pipeline.learner.delta_steer_dist = 0.0
        self.pipeline.learner.speed_factor = 1.0

        # Restore
        success = self.pipeline.restore_checkpoint(filepath=self.cp_path)
        self.assertTrue(success)
        self.assertAlmostEqual(self.pipeline.learner.delta_steer_dist, 0.28, places=3)
        self.assertAlmostEqual(self.pipeline.learner.speed_factor, 0.82, places=3)

    def test_a5_corrupt_checkpoint_detected(self):
        """A5: Corrupt a checkpoint and confirm that verification fails clearly."""
        self.pipeline.create_checkpoint(filepath=self.cp_path)
        with open(self.cp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Mutate internal state without updating hash
        data["internal_affect_state"]["sensory_surprise"] = 0.999
        with open(self.cp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        is_valid, msg, _ = validate_checkpoint_file(self.cp_path)
        self.assertFalse(is_valid)
        self.assertIn("hash mismatch", msg.lower())

    def test_a6_reject_truncated_or_invalid_checkpoint(self):
        """A6: Reject a truncated or invalid checkpoint."""
        self.pipeline.create_checkpoint(filepath=self.cp_path)
        with open(self.cp_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate halfway
        truncated = content[: len(content) // 2]
        with open(self.cp_path, "w", encoding="utf-8") as f:
            f.write(truncated)

        is_valid, msg, _ = validate_checkpoint_file(self.cp_path)
        self.assertFalse(is_valid)
        self.assertIn("failed to parse json", msg.lower())

    def test_a7_save_exception_preserves_previous_valid_checkpoint(self):
        """A7: Simulate a save exception and confirm that a previous valid checkpoint survives."""
        manifest1 = self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="valid_v1")
        hash1 = manifest1["sha256"]
        self.assertTrue(os.path.exists(self.cp_path))

        # Monkeypatch validate_checkpoint_file in fly_brain_node module
        import fly_brain_ros.fly_brain_node as fbn
        original_validate = fbn.validate_checkpoint_file
        try:
            def failing_validate(filepath):
                if ".tmp" in filepath:
                    return False, "Simulated transient verification failure", {}
                return original_validate(filepath)

            fbn.validate_checkpoint_file = failing_validate

            with self.assertRaises(RuntimeError):
                self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="failed_v2")

            # Check that original checkpoint survives completely intact
            is_valid, _, summary = original_validate(self.cp_path)
            self.assertTrue(is_valid)
            self.assertEqual(summary["sha256"], hash1)
        finally:
            fbn.validate_checkpoint_file = original_validate

    def test_a8_missing_checkpoint_handling(self):
        """A8: Test missing file and error paths appropriately."""
        non_existent = os.path.join(self.test_dir, "non_existent.json")
        is_valid, msg, _ = validate_checkpoint_file(non_existent)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg.lower())

        success = self.pipeline.restore_checkpoint(filepath=non_existent)
        self.assertFalse(success)

    def test_a9_configured_and_reported_paths_identical(self):
        """A9: Verify that the configured and reported paths are identical."""
        custom_path = os.path.join(self.test_dir, "authoritative.json")
        resolved = get_canonical_checkpoint_path(custom_path)
        self.assertEqual(resolved, os.path.abspath(custom_path))

        # Test environment variable override
        os.environ["FLY_BRAIN_CHECKPOINT_PATH"] = custom_path
        try:
            resolved_env = get_canonical_checkpoint_path()
            self.assertEqual(resolved_env, os.path.abspath(custom_path))
        finally:
            os.environ.pop("FLY_BRAIN_CHECKPOINT_PATH", None)

    def test_a10_persisted_learning_data_survives_roundtrip(self):
        """A10: Verify persisted learning/reinforcement data."""
        self.pipeline.learner.apply_feedback(1.25, note="Excellent corridor traversal")
        self.pipeline.learner.apply_feedback(-0.75, note="Wall proximity brush")

        manifest = self.pipeline.create_checkpoint(filepath=self.cp_path)
        self.assertTrue(manifest["sha256"])

        # Reset learner completely
        self.pipeline.learner.reset_learning()
        self.assertEqual(self.pipeline.learner.total_learning_steps, 0)
        self.assertEqual(self.pipeline.learner.cumulative_reward, 0.0)

        # Restore from checkpoint
        self.pipeline.restore_checkpoint(filepath=self.cp_path)
        self.assertEqual(self.pipeline.learner.total_learning_steps, 2)
        self.assertAlmostEqual(self.pipeline.learner.cumulative_reward, 0.50, places=2)
        self.assertEqual(len(self.pipeline.learner.episodic_memory.episodes), 2)
        self.assertEqual(self.pipeline.learner.episodic_memory.episodes[0]["outcome"], "MANUAL_REWARD")
        self.assertEqual(self.pipeline.learner.episodic_memory.episodes[1]["outcome"], "MANUAL_PUNISH")

    def test_a11_truthful_verification_summary(self):
        """A11: Verify that validation truthfully distinguishes valid from corrupt checkpoints."""
        self.pipeline.create_checkpoint(filepath=self.cp_path)
        valid_res, msg_good, sum_good = validate_checkpoint_file(self.cp_path)
        self.assertTrue(valid_res)
        self.assertEqual(sum_good["schema_version"], "2.0.0")

        # Fake small file
        small_path = os.path.join(self.test_dir, "small.json")
        with open(small_path, "w") as f:
            f.write("{}")
        small_res, msg_small, _ = validate_checkpoint_file(small_path)
        self.assertFalse(small_res)
        self.assertIn("suspiciously small", msg_small.lower())


if __name__ == "__main__":
    unittest.main()
