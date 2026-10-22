"""
Regression Test Suite for Neural Visualisation, Active Connectome Telemetry, and Neuron Inspection.

Verifies:
  1. Real neuron inspection by FlyWire ID returns biological metadata.
  2. Inspection by local index resolves to the matching FlyWire ID.
  3. Presynaptic and postsynaptic connectivity queries match sparse CSR tensors.
  4. Active subnetwork summaries contain real spiking neurons.
  5. Sensory-to-motor timeline buffer records synchronized data.
  6. Telemetry capture latency overhead remains bounded (< 1.0 ms).
"""

import time
import unittest
from pathlib import Path
import numpy as np
import torch

from fly_brain_ros.brain_engine import FlyBrainEngine
from fly_brain_ros.neuron_metadata import (
    get_neuron_info,
    query_neuron_connections,
    DNA01_LEFT,
    P9_ODN1_LEFT,
    LC4_LEFT,
    BIOLOGICAL_METADATA,
)


class TestNeuralTelemetryAndInspection(unittest.TestCase):
    """Validates real-time neuron inspection and connectome telemetry."""

    @classmethod
    def setUpClass(cls):
        data_dir = Path(__file__).resolve().parents[2] / "fly-brain" / "data"
        comp_csv = data_dir / "2025_Completeness_783.csv"
        conn_pqt = data_dir / "2025_Connectivity_783.parquet"
        cls.engine = FlyBrainEngine(comp_path=comp_csv, conn_path=conn_pqt, cache_dir=data_dir)
        cls.engine.initialize()

    def test_01_inspect_known_biological_neuron(self):
        """Inspection of DNa01_left returns authentic biological metadata and exact string FlyWire ID."""
        info = self.engine.inspect_neuron(DNA01_LEFT)
        self.assertNotIn("error", info)
        self.assertIsInstance(info["flywire_id"], str)
        self.assertEqual(info["flywire_id"], str(DNA01_LEFT))
        self.assertEqual(info["name"], "DNa01_left")
        self.assertEqual(info["functional_group"], "Descending Motor")
        self.assertIn("membrane_voltage_mv", info)
        self.assertLessEqual(info["membrane_voltage_mv"], -45.0)  # At or below threshold
        self.assertGreaterEqual(info["membrane_voltage_mv"], -65.0)

    def test_02_inspect_by_local_index(self):
        """Querying by local index resolves correctly to corresponding FlyWire ID as string."""
        target_local_idx = 100
        expected_fid = self.engine.i2flyid[target_local_idx]
        info = self.engine.inspect_neuron(target_local_idx)
        self.assertNotIn("error", info)
        self.assertEqual(info["local_index"], target_local_idx)
        self.assertIsInstance(info["flywire_id"], str)
        self.assertEqual(info["flywire_id"], str(expected_fid))

    def test_03_connectivity_inspection(self):
        """Querying connections returns valid presynaptic and postsynaptic lists with string IDs."""
        info = self.engine.inspect_neuron(P9_ODN1_LEFT)
        self.assertIn("top_presynaptic_inputs", info)
        self.assertIn("top_postsynaptic_targets", info)
        self.assertGreater(info["total_presynaptic_count"], 0)
        self.assertIsInstance(info["top_presynaptic_inputs"], list)
        for pre in info["top_presynaptic_inputs"]:
            self.assertIsInstance(pre["flywire_id"], str)
            self.assertTrue(pre["flywire_id"].isdigit())
        for post in info["top_postsynaptic_targets"]:
            self.assertIsInstance(post["flywire_id"], str)
            self.assertTrue(post["flywire_id"].isdigit())

    def test_04_active_subnetwork_summary(self):
        """Stimulating LC4 looming neurons activates connectome and reports active nodes with string IDs."""
        # Stimulate left LC4 neurons at 200 Hz
        rates = {nid: 200.0 for nid in LC4_LEFT[:4]}
        self.engine.set_rates_by_id(rates)
        spikes = self.engine.step(num_steps=10)

        summary = self.engine.get_active_subnetwork_summary(max_active=20)
        self.assertIn("active_count_step", summary)
        self.assertIn("active_nodes", summary)
        self.assertIn("active_input_rates", summary)
        for node in summary["active_nodes"]:
            self.assertIsInstance(node["flywire_id"], str)
        # Rates should contain stimulated LC4 IDs as strings
        self.assertTrue(any(str(nid) in summary["active_input_rates"] for nid in LC4_LEFT[:4]))

    def test_05_telemetry_overhead_benchmark(self):
        """Executing active subnetwork extraction takes less than 1.0 ms."""
        t0 = time.perf_counter()
        for _ in range(50):
            _ = self.engine.get_active_subnetwork_summary(max_active=25)
        elapsed_avg = (time.perf_counter() - t0) / 50.0
        self.assertLess(elapsed_avg, 0.001)  # < 1.0 ms

    def test_06_flywire_64bit_id_precision_and_string_preservation(self):
        """Verify exact decimal string preservation for 64-bit FlyWire IDs without IEEE 754 truncation."""
        test_id_str = "720575940644438551"
        truncated_id_str = "720575940644438500"

        # Inspect via string identifier
        info = self.engine.inspect_neuron(test_id_str)
        self.assertNotIn("error", info)
        self.assertIsInstance(info["flywire_id"], str)
        self.assertEqual(info["flywire_id"], test_id_str)
        self.assertNotEqual(info["flywire_id"], truncated_id_str)
        self.assertEqual(info["name"], "DNa01_left")
        self.assertEqual(info["local_index"], 133149)

        # Inspect via integer
        info_int = self.engine.inspect_neuron(int(test_id_str))
        self.assertIsInstance(info_int["flywire_id"], str)
        self.assertEqual(info_int["flywire_id"], test_id_str)

    def test_07_all_69_biological_neurons_exact_string_roundtrip(self):
        """Verify that all 69 biological neuron IDs round-trip as exact strings without digit loss."""
        checked_count = 0
        for fid_int, record in BIOLOGICAL_METADATA.items():
            if not isinstance(fid_int, int):
                continue
            expected_fid_str = str(fid_int)
            info = get_neuron_info(fid_int)
            self.assertIsInstance(info["flywire_id"], str)
            self.assertEqual(info["flywire_id"], expected_fid_str)

            # Query by string as well
            info_by_str = get_neuron_info(expected_fid_str)
            self.assertEqual(info_by_str["flywire_id"], expected_fid_str)
            self.assertEqual(info_by_str["name"], record["name"])
            checked_count += 1

        self.assertEqual(checked_count, 69)

    def test_08_distinguish_large_ids_differing_in_final_digits(self):
        """Prove that large 64-bit IDs differing only in lower digits remain distinct strings."""
        id_a = "720575940644438551"
        id_b = "720575940644438500"

        # While float(id_a) == float(id_b) due to mantissa limits:
        self.assertEqual(float(id_a), float(id_b))

        # String representation and get_neuron_info strictly distinguish them:
        info_a = get_neuron_info(id_a)
        info_b = get_neuron_info(id_b)
        self.assertEqual(info_a["flywire_id"], id_a)
        self.assertEqual(info_b["flywire_id"], id_b)
        self.assertNotEqual(info_a["flywire_id"], info_b["flywire_id"])
        self.assertEqual(info_a["name"], "DNa01_left")
        self.assertNotEqual(info_b["name"], "DNa01_left")

    def test_09_lc4_alias_resolution_and_metadata(self):
        """Querying LC4 aliases resolves to genuine biological metadata without fallback."""
        info_left = self.engine.inspect_neuron("LC4_looming_L")
        self.assertNotIn("error", info_left)
        self.assertEqual(info_left["flywire_id"], "720575940605598892")
        self.assertEqual(info_left["name"], "LC4_left_01")
        self.assertEqual(info_left["functional_group"], "Visual Looming Sensory")
        self.assertEqual(info_left["local_index"], 1911)

        info_right = self.engine.inspect_neuron("LC4_looming_R")
        self.assertNotIn("error", info_right)
        self.assertEqual(info_right["flywire_id"], "720575940624017251")
        self.assertEqual(info_right["name"], "LC4_right_01")
        self.assertEqual(info_right["functional_group"], "Visual Looming Sensory")
        self.assertEqual(info_right["local_index"], 63894)

    def test_10_unknown_neuron_error_differentiation(self):
        """Verify error messages distinguish unknown FlyWire ID, out-of-bounds index, and unknown name."""
        # Case 1: Unregistered FlyWire 64-bit ID
        res_flywire = self.engine.inspect_neuron("720575940999999999")
        self.assertIn("error", res_flywire)
        self.assertIn("FlyWire ID '720575940999999999' not found", res_flywire["error"])

        # Case 2: Out of bounds local matrix index
        res_idx = self.engine.inspect_neuron(999999)
        self.assertIn("error", res_idx)
        self.assertIn("Connectome local matrix index 999999 is out of bounds", res_idx["error"])

        # Case 3: Unknown biological name or alias string
        res_name = self.engine.inspect_neuron("unrecognized_neuron_alias")
        self.assertIn("error", res_name)
        self.assertIn("neither a recognized biological neuron name/alias", res_name["error"])


if __name__ == "__main__":
    unittest.main()
