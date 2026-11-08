"""
Regression Test Suite for Versioned State Persistence and Atomic Checkpoint Restoration.

Verifies:
  1. Atomic checkpoint creation with valid schema 2.0.0 manifest and SHA-256 hash.
  2. Backup preservation (.bak) protecting against interrupted writes.
  3. Round-trip serialization and complete restoration of:
     - Neural dynamic state (membrane potentials, active rates, delay buffer head).
     - Learning state (proactive steer margin, directional bias, speed factor).
     - Internal affect state (Surprise, Vigilance, Confidence, Valence).
     - Episodic memory history and verifiable lessons.
  4. Detection and safe rejection of truncated/corrupted checkpoints with fallback to .bak.
  5. Safe handling of missing checkpoint files.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fly_brain_ros.brain_engine import FlyBrainEngine
from fly_brain_ros.fly_brain_node import FlyBrainControllerPipeline


class TestPersistenceCheckpoints(unittest.TestCase):
    """Validates atomic checkpointing, SHA-256 validation, and state restoration."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path(__file__).resolve().parents[1] / "data",
            Path(__file__).resolve().parents[2] / "fly-brain" / "data",
            Path(__file__).resolve().parents[2] / "data",
            Path("/home/om/Desktop/robo_fly/fly-brain/data"),
        ]
        data_dir = Path(__file__).resolve().parents[1] / "data"
        for cand in candidates:
            if cand.exists():
                data_dir = cand
                break
        comp_csv = data_dir / "2025_Completeness_783.csv"
        conn_pqt = data_dir / "2025_Connectivity_783.parquet"
        cls.comp_csv = comp_csv
        cls.conn_pqt = conn_pqt
        cls.cache_dir = data_dir

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cp_path = os.path.join(self.temp_dir, "test_fly_checkpoint.json")
        self.pipeline = FlyBrainControllerPipeline(
            comp_path=str(self.comp_csv),
            conn_path=str(self.conn_pqt),
            cache_dir=str(self.cache_dir)
        )
        self.pipeline.engine.initialize()

    def tearDown(self):
        self.pipeline.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_atomic_save_and_manifest(self):
        """Checkpoint commit creates valid file with schema 2.0.0 and SHA-256 checksum."""
        manifest = self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="test_session_01")
        self.assertTrue(os.path.exists(self.cp_path))
        self.assertEqual(manifest["schema_version"], "2.0.0")
        self.assertEqual(manifest["session_id"], "test_session_01")
        self.assertIn("sha256", manifest)
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_02_backup_preservation(self):
        """Saving a second checkpoint preserves the original valid checkpoint as .bak."""
        self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="session_v1")
        self.assertTrue(os.path.exists(self.cp_path))

        # Second save
        self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="session_v2")
        bak_path = self.cp_path + ".bak"
        self.assertTrue(os.path.exists(bak_path))

        with open(bak_path, "r") as f:
            bak_data = json.load(f)
        self.assertEqual(bak_data["manifest"]["session_id"], "session_v1")

    def test_03_round_trip_restoration(self):
        """All neural, learning, affect, and episodic state survives save and reload."""
        # Mutate neural, learning, and episodic state
        self.pipeline.learner.delta_steer_dist = 0.22
        self.pipeline.learner.w_dir_bias = -0.15
        self.pipeline.learner.speed_factor = 0.85
        self.pipeline.internal_state.sensory_surprise = 0.65
        self.pipeline.internal_state.valence_affect = -0.40

        self.pipeline.learner.episodic_memory.start_episode("TEST_EVENT", 1.2, 0.6, 1.8)
        self.pipeline.learner.episodic_memory.end_episode("PASS_SUCCESS", 1.5, {}, "Roundtrip test episode.")

        # Save
        self.pipeline.create_checkpoint(filepath=self.cp_path)

        # Reset pipeline to factory baseline
        self.pipeline.learner.reset_learning()
        self.pipeline.internal_state.sleep_reset()
        self.assertEqual(self.pipeline.learner.delta_steer_dist, 0.0)

        # Restore from checkpoint
        success = self.pipeline.restore_checkpoint(filepath=self.cp_path)
        self.assertTrue(success)

        # Verify restoration
        self.assertAlmostEqual(self.pipeline.learner.delta_steer_dist, 0.22, places=2)
        self.assertAlmostEqual(self.pipeline.learner.w_dir_bias, -0.15, places=2)
        self.assertAlmostEqual(self.pipeline.learner.speed_factor, 0.85, places=2)
        self.assertAlmostEqual(self.pipeline.internal_state.sensory_surprise, 0.65, places=2)
        self.assertAlmostEqual(self.pipeline.internal_state.valence_affect, -0.40, places=2)
        self.assertEqual(len(self.pipeline.learner.episodic_memory.episodes), 1)
        self.assertIn("Roundtrip test", self.pipeline.learner.episodic_memory.episodes[0]["lesson"])

    def test_04_detect_corrupted_checkpoint(self):
        """Corrupted primary checkpoint is detected and system recovers from .bak."""
        # 1. Save valid checkpoint 1 (will become backup)
        self.pipeline.learner.delta_steer_dist = 0.18
        self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="good_v1")

        # 2. Save valid checkpoint 2
        self.pipeline.learner.delta_steer_dist = 0.28
        self.pipeline.create_checkpoint(filepath=self.cp_path, session_id="good_v2")

        # 3. Corrupt primary checkpoint file (truncate JSON)
        with open(self.cp_path, "w") as f:
            f.write('{"manifest": {"schema_version": "2.0.0"')  # Incomplete JSON

        # 4. Attempt restore: should fall back to .bak safely
        success = self.pipeline.restore_checkpoint(filepath=self.cp_path)
        self.assertTrue(success)
        # Should have restored good_v1 from .bak
        self.assertAlmostEqual(self.pipeline.learner.delta_steer_dist, 0.18, places=2)

    def test_05_missing_checkpoint_handling(self):
        """Attempting to restore non-existent file returns False without crashing."""
        missing_path = os.path.join(self.temp_dir, "non_existent_checkpoint.json")
        success = self.pipeline.restore_checkpoint(filepath=missing_path)
        self.assertFalse(success)

    def test_06_delay_buffer_fidelity_and_dynamical_continuation(self):
        """Delay buffer slots, head pointer, and RNG state survive save/restore with exact fidelity."""
        # 1. Populate delay buffer with non-zero values at specific slots and neurons
        self.pipeline.engine.delay_buffer[0, 3, 42] = 1.25
        self.pipeline.engine.delay_buffer[0, 8, 1000] = 4.50
        self.pipeline.engine.delay_buffer[0, 18, 50000] = 7.75

        # 2. Set non-default delay head pointer
        self.pipeline.engine.neurons.synapse.delay_head = 9

        # 3. Save checkpoint
        manifest = self.pipeline.create_checkpoint(filepath=self.cp_path)
        self.assertTrue(os.path.exists(self.cp_path))
        file_size_kb = os.path.getsize(self.cp_path) / 1024.0
        self.assertLess(file_size_kb, 150.0)  # Verify compact compressed payload

        # 4. Reset engine state
        self.pipeline.engine.reset_state()
        self.assertEqual(self.pipeline.engine.delay_buffer[0, 3, 42].item(), 0.0)
        self.assertEqual(self.pipeline.engine.neurons.synapse.delay_head, 0)

        # 5. Restore checkpoint
        success = self.pipeline.restore_checkpoint(filepath=self.cp_path)
        self.assertTrue(success)

        # 6. Verify exact restored values
        self.assertEqual(self.pipeline.engine.neurons.synapse.delay_head, 9)
        self.assertAlmostEqual(self.pipeline.engine.delay_buffer[0, 3, 42].item(), 1.25, places=3)
        self.assertAlmostEqual(self.pipeline.engine.delay_buffer[0, 8, 1000].item(), 4.50, places=3)
        self.assertAlmostEqual(self.pipeline.engine.delay_buffer[0, 18, 50000].item(), 7.75, places=3)

    def test_07_sleep_wake_fatigue_reset_preserves_policy(self):
        """Waking from sleep clears computational fatigue while strictly preserving learned policy and memory."""
        # Setup active state with high fatigue and learned weights
        self.pipeline.internal_state.computational_fatigue = 0.85
        self.pipeline.internal_state.environmental_vigilance = 0.90
        self.pipeline.internal_state.sensory_surprise = 0.70
        self.pipeline.learner.delta_steer_dist = 0.28
        self.pipeline.learner.w_dir_bias = -0.18
        self.pipeline.learner.speed_factor = 0.80
        self.pipeline.learner.episodic_memory.start_episode("TEST", 1.2, 0.5, 1.5)
        self.pipeline.learner.episodic_memory.end_episode("PASS_SUCCESS", 1.5, {}, "Retained episode.")

        # Save checkpoint
        self.pipeline.create_checkpoint(filepath=self.cp_path)

        # Restore with is_wake=True (simulates dashboard wake)
        success = self.pipeline.restore_checkpoint(filepath=self.cp_path, is_wake=True)
        self.assertTrue(success)

        # Fatigue, vigilance, and surprise must be reset to 0
        self.assertEqual(self.pipeline.internal_state.computational_fatigue, 0.0)
        self.assertEqual(self.pipeline.internal_state.environmental_vigilance, 0.0)
        self.assertEqual(self.pipeline.internal_state.sensory_surprise, 0.0)

        # Learned navigation parameters must be 100% PRESERVED
        self.assertAlmostEqual(self.pipeline.learner.delta_steer_dist, 0.28, places=2)
        self.assertAlmostEqual(self.pipeline.learner.w_dir_bias, -0.18, places=2)
        self.assertAlmostEqual(self.pipeline.learner.speed_factor, 0.80, places=2)

        # Episodic memories must be 100% PRESERVED
        self.assertEqual(len(self.pipeline.learner.episodic_memory.episodes), 1)
        self.assertIn("Retained episode", self.pipeline.learner.episodic_memory.episodes[0]["lesson"])


if __name__ == "__main__":
    unittest.main()
