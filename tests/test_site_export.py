"""Integrity checks for the generated, static research data bundle."""

import base64
import hashlib
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs/data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_site_bundle_is_current_against_hash_checked_sources():
    result = subprocess.run(
        [sys.executable, "scripts/export_research_site_data.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Verified" in result.stdout


def test_site_result_records_match_raw_csv_and_keep_condition_classes():
    data = json.loads((DOCS_DATA / "research.json").read_text())
    conditions = {item["id"]: item for item in data["conditions"]}
    assert len(conditions) == 9
    assert all(item["metrics"]["episodes"] == 3000 for item in conditions.values())
    assert {key for key, item in conditions.items() if item["is_primary_zero_shot"]} == {
        "gemini_argon_4_medium", "claude_sonnet_55", "gpt6_original", "gpt6_luna_extra_high", "deepseek_original"
    }
    assert "deepseek_corrected" not in conditions
    assert conditions["astra_local_reference"]["condition_class"] == "reference"
    assert conditions["ppo_trained_baseline"]["condition_class"] == "trained_baseline"
    assert conditions["random_action_control"]["condition_class"] == "random_control"
    assert conditions["zero_weight_control"]["condition_class"] == "zero_weight_control"
    for condition_id, item in conditions.items():
        assert len(item["metrics"]["episode_records"]) == 3000
        csv_mean = sum(row[0] for row in item["metrics"]["episode_records"]) / 3000
        assert abs(item["metrics"]["mean_score"] - csv_mean) < 1e-12
        original = ROOT / "experiments/corrected_evaluation/full_run/conditions" / condition_id / "episode_scores.csv"
        published = DOCS_DATA / item["csv_download"]
        assert sha256(original) == sha256(published)


def test_primary_model_archives_and_responses_are_present_and_hash_bound():
    data = json.loads((DOCS_DATA / "models.json").read_text())
    primary = [model for model in data["models"] if model["condition_class"] == "primary_zero_shot"]
    assert len(primary) == 5
    for model in primary:
        weights = model["weights"]
        assert weights["total_parameters"] == 20486
        assert len(weights["source_sha256"]) == 64
        original = ROOT / weights["source_path"]
        published = DOCS_DATA / weights["download_file"]
        assert sha256(original) == weights["source_sha256"] == sha256(published)
        response = model["generation"]["response_file"]
        assert response
        assert (DOCS_DATA / response).is_file()
    assert data["separate_intervention"]["included_in_primary_results"] is False
    assert data["provider_attestation_available"] is False


def test_iterative_argon_export_is_separate_and_matches_all_three_raw_csvs():
    group = json.loads((DOCS_DATA / "iterative_development.json").read_text())
    primary = json.loads((DOCS_DATA / "research.json").read_text())
    models = json.loads((DOCS_DATA / "models.json").read_text())
    assert group["group_id"] == "gemini_argon_high_iterative_development"
    assert group["separate_from_primary_zero_shot_ranking"] is True
    assert [item["id"] for item in group["models"]] == [
        "argon_high_original", "argon_high_followup_1", "argon_high_followup_2"
    ]
    assert len(primary["conditions"]) == 9
    assert [item["id"] for item in primary["conditions"] if item["is_primary_zero_shot"]] == [
        "gemini_argon_4_medium", "claude_sonnet_55", "gpt6_original", "gpt6_luna_extra_high", "deepseek_original"
    ]
    assert not {item["id"] for item in group["models"]} & {item["id"] for item in primary["conditions"]}
    assert not {item["id"] for item in group["models"]} & {item["id"] for item in models["models"]}

    signatures = []
    expected_rounded = {
        "argon_high_original": (59.998333333333335, 1.1953226147561218, 38.0),
        "argon_high_followup_1": (62.367666666666665, 1.2028755450723287, 40.0),
        "argon_high_followup_2": (62.188, 1.1822247828334997, 40.0),
    }
    for item in group["models"]:
        source = ROOT / "experiments/argon_high_iterative_development" / item["benchmark_csv"]
        published = DOCS_DATA / item["csv_download"]
        assert source.read_bytes() == published.read_bytes()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["csv_sha256"]
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        scores = [float(row["score"]) for row in rows]
        lengths = [int(row["length"]) for row in rows]
        assert len(rows) == 3000 == item["metrics"]["episodes"]
        assert math.isclose(statistics.fmean(scores), item["metrics"]["mean_score"], abs_tol=1e-12)
        assert math.isclose(statistics.stdev(scores) / math.sqrt(len(scores)), item["metrics"]["standard_error_score"], abs_tol=1e-12)
        assert statistics.median(scores) == item["metrics"]["median_score"]
        assert min(scores) == item["metrics"]["minimum_score"]
        assert max(scores) == item["metrics"]["maximum_score"]
        assert statistics.fmean(lengths) == item["metrics"]["mean_episode_length"]
        assert sum(row["ball_lost"].lower() in {"true", "1"} for row in rows) == 3000
        assert sum(row["step_cutoff"].lower() in {"true", "1"} for row in rows) == 0
        signatures.append(sorted((row["seed_group"], row["episode_index"], row["base_seed"], row["environment_seed"], row["policy_seed"]) for row in rows))
        mean, se, median = expected_rounded[item["id"]]
        assert math.isclose(item["metrics"]["mean_score"], mean, abs_tol=1e-12)
        assert math.isclose(item["metrics"]["standard_error_score"], se, abs_tol=1e-12)
        assert item["metrics"]["median_score"] == median
        for source_info in (item["source"], item["weights"]):
            original = ROOT / "experiments/argon_high_iterative_development" / source_info["path"]
            exported = DOCS_DATA / source_info["download_file"]
            assert original.read_bytes() == exported.read_bytes()
            assert hashlib.sha256(original.read_bytes()).hexdigest() == source_info["sha256"]
        assert item["weights"]["total_parameters"] == 20486
        assert item["weights"]["nonzero_parameters"] == item["nonzero_parameter_count"]
        assert set(item["weights"]["tensors"]) == {"W1", "b1", "W2", "b2", "W3", "b3"}
        tensor_key_map = {
            "actor/dense1/kernel": "W1", "actor/dense1/bias": "b1",
            "actor/dense2/kernel": "W2", "actor/dense2/bias": "b2",
            "actor/logits/kernel": "W3", "actor/logits/bias": "b3",
        }
        import numpy as np
        with np.load(ROOT / "experiments/argon_high_iterative_development" / item["weights"]["path"], allow_pickle=False) as archive:
            for archive_key, tensor_name in tensor_key_map.items():
                tensor = item["weights"]["tensors"][tensor_name]
                expected_bytes = np.asarray(archive[archive_key], dtype="<f4", order="C").tobytes(order="C")
                assert base64.b64decode(tensor["data_base64"]) == expected_bytes
    assert signatures[0] == signatures[1] == signatures[2]
    assert all(sum(bin_record["count"] for bin_record in model["score_histogram"]) == 3000 for model in group["models"])

    comparisons = group["paired_comparisons"]
    assert len(comparisons) == 3
    assert math.isclose(comparisons[0]["mean_paired_difference"], -2.3693333333333335, abs_tol=1e-12)
    assert math.isclose(comparisons[1]["mean_paired_difference"], -2.1896666666666667, abs_tol=1e-12)
    assert math.isclose(comparisons[2]["mean_paired_difference"], 0.17966666666666667, abs_tol=1e-12)
    assert math.isclose(comparisons[2]["effect_size_cohen_dz"], 0.006462736326960453, abs_tol=1e-12)
    assert math.isclose(comparisons[2]["p_value_holm_adjusted_all_pairwise_conditions"], 0.72455, abs_tol=1e-12)
    assert "not evidence of equivalence" in group["statistical_interpretation"].lower()
    assert group["artifact_registry"]["registered_file_count"] == 35
    assert group["seed_matching_verified"] is True

    report_source = ROOT / "experiments/argon_high_iterative_development/report.md"
    report_export = DOCS_DATA / group["downloads"]["report"]["download_file"]
    assert report_source.read_bytes() == report_export.read_bytes()
