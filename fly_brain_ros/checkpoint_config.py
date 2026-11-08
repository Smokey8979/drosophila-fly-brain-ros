"""
Authoritative Checkpoint Configuration & Cryptographic Integrity Verification.

Provides a single canonical path and validation routines for Drosophila Fly Brain checkpoints:
  - Canonical checkpoint path resolution (configurable via FLY_BRAIN_CHECKPOINT_PATH)
  - Cryptographic SHA-256 hash calculation and verification
  - Schema 2.x structure and required sections validation
  - Safe atomic write and backup management
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def _resolve_default_canonical_path() -> str:
    env_path = os.environ.get("FLY_BRAIN_CHECKPOINT_PATH", "").strip()
    if env_path:
        return os.path.abspath(env_path)

    pkg_root = Path(__file__).resolve().parents[1]
    candidates = [
        pkg_root / "checkpoints" / "fly_brain_checkpoint.json",
        pkg_root.parent / "checkpoints" / "fly_brain_checkpoint.json",
        Path.cwd() / "checkpoints" / "fly_brain_checkpoint.json",
        Path("/home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    # Default to repository checkpoints directory
    return str((pkg_root / "checkpoints" / "fly_brain_checkpoint.json").resolve())


# Canonical authoritative checkpoint path (dynamically evaluated)
DEFAULT_CANONICAL_PATH = _resolve_default_canonical_path()

REQUIRED_SECTIONS = [
    "manifest",
    "neural_state",
    "learning_state",
    "internal_affect_state",
    "episodic_memory",
]


def get_canonical_checkpoint_path(custom_path: Optional[str] = None) -> str:
    """Returns the authoritative absolute checkpoint path.
    
    Priority:
      1. Explicit custom_path argument (if provided and non-empty)
      2. Environment variable FLY_BRAIN_CHECKPOINT_PATH
      3. Candidate checkpoint locations (repo or workspace checkpoints/ directory)
    """
    if custom_path and str(custom_path).strip():
        return os.path.abspath(str(custom_path).strip())
    env_path = os.environ.get("FLY_BRAIN_CHECKPOINT_PATH", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    return _resolve_default_canonical_path()


def compute_checkpoint_hash(data: Dict[str, Any]) -> str:
    """Computes the canonical SHA-256 digest of checkpoint data excluding the stored hash."""
    copied = dict(data)
    manifest = dict(copied.get("manifest", {}))
    manifest.pop("sha256", None)
    copied["manifest"] = manifest
    canonical_bytes = json.dumps(copied, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


calculate_checkpoint_sha256 = compute_checkpoint_hash


def validate_checkpoint_data(data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates checkpoint structure, schema version, and SHA-256 integrity.
    
    Returns:
        (is_valid, error_or_success_message, manifest_dict)
    """
    if not isinstance(data, dict):
        return False, "Checkpoint root must be a JSON dictionary.", {}

    for sec in REQUIRED_SECTIONS:
        if sec not in data:
            return False, f"Missing required checkpoint section: '{sec}'.", {}

    manifest = data.get("manifest", {})
    schema = str(manifest.get("schema_version", ""))
    if not schema.startswith("2."):
        return False, f"Unsupported schema version '{schema}' (expected 2.x).", manifest

    stored_hash = manifest.get("sha256")
    if not stored_hash:
        return False, "Manifest missing 'sha256' cryptographic hash.", manifest

    calculated_hash = compute_checkpoint_hash(data)
    if stored_hash != calculated_hash:
        return (
            False,
            f"SHA-256 hash mismatch: stored={stored_hash[:16]}... calc={calculated_hash[:16]}...",
            manifest,
        )

    return True, "Valid checkpoint.", manifest


def validate_checkpoint_file(filepath: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates a checkpoint file on disk without corrupting state.
    
    Returns:
        (is_valid, message, metadata_summary)
    """
    if not os.path.exists(filepath):
        return False, f"Checkpoint file does not exist: {filepath}", {}

    try:
        size_bytes = os.path.getsize(filepath)
    except OSError as e:
        return False, f"Cannot access checkpoint file: {e}", {}

    if size_bytes < 5000:
        return False, f"Checkpoint file is suspiciously small ({size_bytes} bytes).", {"size_bytes": size_bytes}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse JSON checkpoint: {e}", {"size_bytes": size_bytes}

    is_valid, msg, manifest = validate_checkpoint_data(data)
    summary = {
        "filepath": os.path.abspath(filepath),
        "size_bytes": size_bytes,
        "schema_version": manifest.get("schema_version", "unknown"),
        "sha256": manifest.get("sha256", "none"),
        "session_id": manifest.get("session_id", "none"),
        "timestamp": manifest.get("timestamp", 0),
        "neuron_count": manifest.get("neuron_count", 138639),
    }
    return is_valid, msg, summary
