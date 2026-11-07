import json
import shutil

import pytest

from visualize_comparison import (
    REPOSITORY_ROOT,
    benchmark_for_loaded_weights,
    load_verified_benchmark_registry,
    verify_condition_weight,
)


def test_corrected_visualizer_loads_only_hash_bound_completed_results():
    registry = load_verified_benchmark_registry()
    assert len(registry) == 9
    assert list(registry)[0:6] == ["astra", "argon", "sonnet55", "trained", "gpt6", "gpt6_luna"]
    assert registry["argon"]["episodes"] == 3000
    assert registry["argon"]["protocol"] == "numpy-seedsequence-spawn-two-randomstate-v1"
    assert "deepseek_corrected" not in registry
    assert benchmark_for_loaded_weights(registry["argon"], registry["argon"]["weight_sha256"])


def test_custom_weights_cannot_inherit_canonical_score():
    registry = load_verified_benchmark_registry()
    assert benchmark_for_loaded_weights(registry["argon"], "0" * 64) is None
    assert benchmark_for_loaded_weights(registry["random"], "0" * 64) is None


def test_missing_corrected_metrics_fail_clearly(tmp_path):
    source = REPOSITORY_ROOT / "experiments/corrected_evaluation/full_run/run_manifest.json"
    shutil.copy(source, tmp_path / "run_manifest.json")
    with pytest.raises(FileNotFoundError, match="Required corrected metrics are missing"):
        load_verified_benchmark_registry(run_dir=tmp_path)


def test_changed_corrected_metrics_fail_hash_check(tmp_path):
    source = REPOSITORY_ROOT / "experiments/corrected_evaluation/full_run/run_manifest.json"
    manifest = json.loads(source.read_text())
    manifest["analysis_artifact_sha256"]["comparative_metrics.json"] = "0" * 64
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    metrics_source = REPOSITORY_ROOT / "experiments/corrected_evaluation/full_run/analysis/comparative_metrics.json"
    (tmp_path / "analysis").mkdir()
    shutil.copy(metrics_source, tmp_path / "analysis/comparative_metrics.json")
    with pytest.raises(ValueError, match="comparative metrics SHA-256 mismatch"):
        load_verified_benchmark_registry(run_dir=tmp_path)


def test_canonical_weight_hash_mismatch_is_an_error(tmp_path):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"changed weights")
    condition = {
        "id": "gemini_argon_4_medium",
        "weight_path": "weight.bin",
        "weight_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="Weight SHA-256 mismatch"):
        verify_condition_weight(condition, tmp_path)
