"""Verify the original experiment artifacts stay bit-for-bit unchanged."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESERVATION_FILE = ROOT / "experiments/corrected_evaluation/preserved_artifacts_sha256.json"


def test_all_baseline_experiment_and_forensic_artifacts_match_fingerprints():
    manifest = json.loads(PRESERVATION_FILE.read_text())
    assert manifest["baseline_commit"] == "8c15d156798bffb1befc7e85c06870c4981f9a72"
    assert len(manifest["files"]) == manifest["protected_file_count"]
    assert manifest["protected_file_count"] >= 100

    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Protected artifact missing: {item['path']}"
        raw = path.read_bytes()
        assert len(raw) == item["size_bytes"], f"Protected artifact size changed: {item['path']}"
        assert hashlib.sha256(raw).hexdigest() == item["sha256"], (
            f"Protected artifact SHA-256 changed: {item['path']}"
        )
