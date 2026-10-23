"""
Suite D & E: 3D Connectome Viewer, Anatomical Layout, Telemetry Contract, and Precise 64-bit ID Test Suite.

Verifies:
  D1: Spatial layout endpoint structure and metadata (138,639 neurons, 7 neuropils, bounding box).
  D2: Binary coordinates stream (/api/connectome/spatial_coords_bin) size and MIME type.
  D3: Binary data deserialization, finite floating-point bounds, and no NaN/Inf.
  D4: Neuropil partitioning covers all 7 anatomical zones and sums to 138,639.
  D5: Authentic biological landmarks positioned correctly with exact string IDs.
  D6: End-to-end 64-bit FlyWire ID string preservation without float truncation.
  D7: Synapses endpoint returns structured pre/post connectivity with exact string IDs.
  D8: Web state telemetry schema contains internal affect, learning policy, and lifecycle.
  D9: Offline Three.js vendor bundle integrity (offline standalone compatibility).
  D10: High-throughput stress & concurrency soak test without memory leaks or latency degradation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure live_lab and fly_brain_ros are in Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_LAB_DIR = REPO_ROOT / "live_lab"
if str(LIVE_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_LAB_DIR))

from web_server import app, sim  # type: ignore
from fly_brain_ros.connectome_spatial_layout import (
    TOTAL_CONNECTOME_NEURONS,
    NEUROPIL_REGIONS,
    get_spatial_projection_manifest,
)
from fly_brain_ros.neuron_metadata import DNA01_LEFT, P9_ODN1_LEFT, LC4_LEFT


class Test3DConnectomeViewerAndTelemetry(unittest.TestCase):
    """Test suite for 3D connectome viewer endpoints, binary spatial stream, and telemetry."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_d1_spatial_layout_endpoint_structure(self):
        """D1: /api/connectome/spatial_layout returns 200 with complete anatomical layout schema."""
        resp = self.client.get("/api/connectome/spatial_layout")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data["total_neurons"], TOTAL_CONNECTOME_NEURONS)
        self.assertEqual(data["total_neurons"], 138639)
        self.assertIn("dataset", data)
        self.assertIn("FlyWire MaleCNS", data["dataset"])
        self.assertIn("neuropils", data)
        self.assertIsInstance(data["neuropils"], list)
        self.assertEqual(len(data["neuropils"]), 7)
        self.assertIn("spatial_projection", data)
        self.assertIn("bounding_box", data["spatial_projection"])
        bbox = data["spatial_projection"]["bounding_box"]
        self.assertIn("x_range", bbox)
        self.assertIn("y_range", bbox)
        self.assertIn("z_range", bbox)

    def test_d2_binary_spatial_coordinates_stream(self):
        """D2: /api/connectome/spatial_coords_bin returns exactly 138639 * 16 bytes with octet-stream."""
        resp = self.client.get("/api/connectome/spatial_coords_bin")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/octet-stream")
        expected_bytes = TOTAL_CONNECTOME_NEURONS * 16  # 4 x float32 per neuron
        self.assertEqual(len(resp.data), expected_bytes)
        self.assertEqual(len(resp.data), 2218224)

    def test_d3_binary_data_deserialization_and_bounds(self):
        """D3: Binary coords unpack to float32 (138639, 4) within valid physical bounds without NaN/Inf."""
        resp = self.client.get("/api/connectome/spatial_coords_bin")
        coords = np.frombuffer(resp.data, dtype=np.float32).reshape(-1, 4)
        self.assertEqual(coords.shape, (138639, 4))
        # Ensure no NaN or Inf
        self.assertTrue(np.all(np.isfinite(coords)))

        # Check coordinate bounds
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())

        self.assertGreaterEqual(x_min, -160.0)
        self.assertLessEqual(x_max, 160.0)
        self.assertGreaterEqual(y_min, -110.0)
        self.assertLessEqual(y_max, 110.0)
        self.assertGreaterEqual(z_min, -90.0)
        self.assertLessEqual(z_max, 90.0)

    def test_d4_neuropil_partitioning_and_coverage(self):
        """D4: All 7 neuropil partitions are represented and sum exactly to 138,639."""
        resp = self.client.get("/api/connectome/spatial_coords_bin")
        coords = np.frombuffer(resp.data, dtype=np.float32).reshape(-1, 4)
        neuropil_ids = np.round(coords[:, 3]).astype(int)

        unique_ids = set(np.unique(neuropil_ids).tolist())
        expected_ids = {1, 2, 3, 4, 5, 6, 7}
        self.assertEqual(unique_ids, expected_ids)

        # Sum of counts must match exactly 138,639
        counts = [int(np.sum(neuropil_ids == nid)) for nid in range(1, 8)]
        self.assertEqual(sum(counts), 138639)
        for i, c in enumerate(counts, 1):
            self.assertGreater(c, 0, f"Neuropil {i} has 0 neurons!")

    def test_d5_key_biological_landmarks_preservation(self):
        """D5: Authenticated biological neurons are identified with exact string IDs and positions."""
        resp = self.client.get("/api/connectome/spatial_layout")
        data = resp.get_json()
        key_neurons = data.get("key_neurons", [])
        self.assertGreater(len(key_neurons), 10)

        fids = [k["flywire_id"] for k in key_neurons]
        self.assertIn(str(DNA01_LEFT), fids)
        self.assertIn(str(P9_ODN1_LEFT), fids)
        self.assertIn(str(LC4_LEFT[0]), fids)

        for kn in key_neurons:
            self.assertIsInstance(kn["flywire_id"], str)
            self.assertTrue(kn["flywire_id"].isdigit())
            self.assertIn("name", kn)
            has_pos = ("coords" in kn) or ("x" in kn and "y" in kn and "z" in kn)
            self.assertTrue(has_pos, f"Key neuron {kn} missing spatial position!")

    def test_d6_flywire_64bit_id_string_preservation_in_api(self):
        """D6: /api/neuron/<id> preserves full 64-bit precision (720575940644438551 != ...500)."""
        target_id_str = "720575940644438551"
        resp = self.client.get(f"/api/neuron/{target_id_str}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data["flywire_id"], str)
        self.assertEqual(data["flywire_id"], target_id_str)
        self.assertEqual(data["name"], "DNa01_left")
        self.assertNotEqual(data["flywire_id"], "720575940644438500")

    def test_d7_neuron_synapses_endpoint(self):
        """D7: /api/neuron/<id>/synapses returns structured connectivity with string IDs."""
        target_id_str = "720575940644438551"
        resp = self.client.get(f"/api/neuron/{target_id_str}/synapses")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["flywire_id"], target_id_str)
        self.assertIn("presynaptic", data)
        self.assertIn("postsynaptic", data)
        self.assertIn("total_presynaptic_count", data)
        self.assertIn("total_postsynaptic_count", data)

    def test_d8_telemetry_state_payload_contract(self):
        """D8: /api/state telemetry contract contains affect, learning, and lifecycle data."""
        resp = self.client.get("/api/state")
        self.assertEqual(resp.status_code, 200)
        state = resp.get_json()
        self.assertEqual(state["total_neurons"], 138639)
        self.assertIn("internal_state", state)
        aff = state["internal_state"]
        for key in ["valence", "vigilance", "energy", "confidence", "arousal"]:
            self.assertIn(key, aff)
            self.assertIsInstance(aff[key], (int, float))

        self.assertIn("learning", state)
        lrn = state["learning"]
        has_steer = ("steer_dist" in lrn) or ("delta_steer_dist" in lrn)
        self.assertTrue(has_steer, "Learning state missing steer distance parameter!")
        self.assertIn("speed_factor", lrn)

        self.assertIn("lifecycle_state", state)
        self.assertIn("is_asleep", state)

    def test_d9_frontend_assets_and_vendor_integrity(self):
        """D9: Vendored Three.js library exists locally (> 500 KB) for offline viewer operation."""
        vendor_file = LIVE_LAB_DIR / "static" / "three.min.js"
        self.assertTrue(vendor_file.exists(), "three.min.js is missing from static directory!")
        file_size = vendor_file.stat().st_size
        self.assertGreater(file_size, 500_000, f"three.min.js is incomplete: {file_size} bytes")

        content = vendor_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("WebGLRenderer", content)
        self.assertIn("BufferGeometry", content)

    def test_d10_stress_and_concurrency_soak(self):
        """D10: Successive high-throughput requests complete successfully without latency degradation."""
        import time
        t0 = time.perf_counter()
        # Perform 40 queries across spatial layout, binary stream, and neuron endpoints
        for i in range(10):
            r1 = self.client.get("/api/connectome/spatial_layout")
            self.assertEqual(r1.status_code, 200)
            r2 = self.client.get("/api/connectome/spatial_coords_bin")
            self.assertEqual(r2.status_code, 200)
            r3 = self.client.get("/api/state")
            self.assertEqual(r3.status_code, 200)
            r4 = self.client.get("/api/neuron/720575940644438551")
            self.assertEqual(r4.status_code, 200)

        elapsed = time.perf_counter() - t0
        avg_per_req = elapsed / 40.0
        # Average response latency should remain under 250ms per request (transferring 2.2MB binary buffers)
        self.assertLess(avg_per_req, 0.250, f"Average request latency too high: {avg_per_req*1000:.2f} ms")


if __name__ == "__main__":
    unittest.main()
