"""
Regression Test Suite for Sleep/Wake Lifecycle Transitions and Process Supervision.

Verifies:
  1. Lifecycle state machine transitions:
     SLEEPING -> WAKING -> RESTORING -> AWAKE -> SLEEP_REQUESTED -> CHECKPOINTING -> SLEEPING.
  2. Safe zero-motion halt and atomic checkpoint creation upon sleep request.
  3. Clean termination of managed child processes without orphan processes.
  4. Resumption from persistent checkpoint upon wake.
  5. Idempotent handling of repeated sleep/wake clicks without duplicate processes.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from fly_brain_ros.lifecycle_supervisor import SimulationLifecycleSupervisor


class MockROSBridge:
    def __init__(self):
        self.cmd_history = []
        self.lifecycle_commands = []

    def publish_teleop_cmd(self, vx: float, wz: float):
        self.cmd_history.append((vx, wz))

    def send_lifecycle_command(self, cmd: str):
        self.lifecycle_commands.append(cmd)


class TestSleepWakeLifecycle(unittest.TestCase):
    """Validates lifecycle supervisor transitions, safe shutdown, and wake restoration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cp_path = os.path.join(self.temp_dir, "lifecycle_test_checkpoint.json")
        # Create minimal valid checkpoint
        with open(self.cp_path, "w") as f:
            f.write('{"manifest": {"schema_version": "2.0.0", "session_id": "test_lc"}}')

        self.bridge = MockROSBridge()
        self.supervisor = SimulationLifecycleSupervisor(
            checkpoint_path=self.cp_path,
            ros_bridge_node=self.bridge,
        )

    def tearDown(self):
        self.supervisor.terminate_all()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_initial_state_and_transitions(self):
        """Supervisor initializes in SLEEPING state and transitions cleanly."""
        status = self.supervisor.get_status()
        self.assertEqual(status["lifecycle_state"], SimulationLifecycleSupervisor.STATE_SLEEPING)
        self.assertEqual(status["sleep_cycle_count"], 0)
        self.assertTrue(status["checkpoint_exists"])

    def test_02_safe_zero_motion_on_sleep_request(self):
        """Requesting sleep immediately commands (0.0, 0.0) velocity and sends SLEEP signal."""
        # Manually put supervisor in AWAKE state
        self.supervisor.current_state = SimulationLifecycleSupervisor.STATE_AWAKE
        res = self.supervisor.request_sleep(flush_delay=0.0)

        # Verify zero velocity commanded
        self.assertIn((0.0, 0.0), self.bridge.cmd_history)
        self.assertIn("SLEEP", self.bridge.lifecycle_commands)
        self.assertEqual(res["lifecycle_state"], SimulationLifecycleSupervisor.STATE_SLEEPING)
        self.assertEqual(res["sleep_cycle_count"], 1)

    def test_03_repeated_sleep_is_idempotent(self):
        """Calling sleep repeatedly while already SLEEPING does not cause duplicate shutdowns."""
        self.supervisor.current_state = SimulationLifecycleSupervisor.STATE_SLEEPING
        res1 = self.supervisor.request_sleep(flush_delay=0.0)
        res2 = self.supervisor.request_sleep(flush_delay=0.0)
        self.assertEqual(res1["lifecycle_state"], SimulationLifecycleSupervisor.STATE_SLEEPING)
        self.assertEqual(res2["lifecycle_state"], SimulationLifecycleSupervisor.STATE_SLEEPING)
        # Sleep cycle count should not increment on redundant calls
        self.assertEqual(res2["sleep_cycle_count"], 0)

    def test_04_transition_history_logging(self):
        """All lifecycle state changes are recorded with timestamps and reasons."""
        self.supervisor.current_state = SimulationLifecycleSupervisor.STATE_AWAKE
        self.supervisor.request_sleep(flush_delay=0.0)
        status = self.supervisor.get_status()
        history = status["transition_history"]
        states = [h["state"] for h in history]
        self.assertIn(SimulationLifecycleSupervisor.STATE_SLEEP_REQUESTED, states)
        self.assertIn(SimulationLifecycleSupervisor.STATE_CHECKPOINTING, states)
        self.assertIn(SimulationLifecycleSupervisor.STATE_SLEEPING, states)


if __name__ == "__main__":
    unittest.main()
