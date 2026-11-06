"""Tests for descriptive metrics and paired inference."""

import numpy as np
import pytest
import csv
from pathlib import Path

from evaluation.corrected_analysis import (
    ORIGINAL_EVALUATION_PATHS,
    _holm_adjust,
    compare_corrected_to_original,
    paired_comparisons,
    summarize_condition,
    validate_matched_conditions,
)


def _row(group, index, base, score):
    row = {
        "seed_group": group,
        "episode_index": index,
        "base_seed": base,
        "environment_seed": base + 10_000,
        "policy_seed": base + 20_000,
        "score": float(score),
        "length": index + 1,
        "bricks_destroyed": int(score),
        "ball_lost": True,
        "step_cutoff": False,
        "trajectory_sha256": f"trace-{group}-{index}-{score}",
    }
    for action in range(6):
        row[f"action_{action}_selected_count"] = row["length"] if action == (index % 6) else 0
        row[f"action_{action}_executed_count"] = row["length"] if action == ((index + 1) % 6) else 0
    return row


def test_condition_summary_reports_sample_se_ci_cutoffs_lengths_and_action_labels():
    rows = [_row(42, 0, 111, 0), _row(42, 1, 112, 2), _row(42, 2, 113, 4)]
    rows[1]["step_cutoff"] = True
    summary = summarize_condition(rows)
    assert summary["episodes"] == 3
    assert summary["mean_score"] == 2.0
    assert summary["median_score"] == 2.0
    assert np.isclose(summary["sample_std_score"], 2.0)
    assert np.isclose(summary["standard_error_score"], 2.0 / np.sqrt(3))
    assert summary["score_mean_ci_95"][0] < 2.0 < summary["score_mean_ci_95"][1]
    assert summary["minimum_score"] == 0.0 and summary["maximum_score"] == 4.0
    assert summary["mean_episode_length"] == 2.0
    assert summary["median_episode_length"] == 2.0
    assert summary["maximum_episode_length"] == 3
    assert summary["cutoff_count"] == 1 and summary["cutoff_percentage"] == pytest.approx(100 / 3)
    assert summary["selected_action_distribution"]["0"]["name"] == "no_op"
    assert sum(item["count"] for item in summary["selected_action_distribution"].values()) == 6
    assert sum(item["count"] for item in summary["executed_action_distribution_after_sticky_handling"].values()) == 6


def test_pairing_requires_same_base_and_derived_seed_triplets():
    a = [_row(42, i, 100 + i, i) for i in range(4)]
    b = [_row(42, i, 100 + i, i + 1) for i in range(4)]
    keys = validate_matched_conditions({"a": a, "b": b})
    assert len(keys) == 4
    b[2]["policy_seed"] += 1
    with pytest.raises(ValueError, match="exact matched seed triplets"):
        validate_matched_conditions({"a": a, "b": b})


def test_paired_analysis_reports_difference_ci_effect_and_nonzero_p_value():
    a = [_row(42, i, 100 + i, i + 3) for i in range(12)]
    b = [_row(42, i, 100 + i, i) for i in range(12)]
    comparisons = paired_comparisons(
        {"a": a, "b": b}, {"a": "A", "b": "B"},
        bootstrap_resamples=199, permutation_resamples=999, random_seed=11,
    )
    item = comparisons[0]
    assert item["matched_episode_pairs"] == 12
    assert item["mean_paired_difference"] == 3.0
    assert item["paired_difference_ci_95_percentile_bootstrap"][0] <= 3.0
    assert item["paired_difference_ci_95_percentile_bootstrap"][1] >= 3.0
    assert item["effect_size_cohen_dz"] is None  # constant differences: zero variance
    assert 0.0 < item["paired_sign_flip_p_value"] <= 1.0
    assert 0.0 < item["p_value_holm_adjusted_all_pairwise_conditions"] <= 1.0


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = _holm_adjust([0.01, 0.04, 0.03, 0.2])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    order = np.argsort([0.01, 0.04, 0.03, 0.2])
    ordered_adjusted = [adjusted[i] for i in order]
    assert ordered_adjusted == sorted(ordered_adjusted)


def test_analysis_rejects_unequal_episode_keys():
    a = [_row(42, i, 100 + i, i) for i in range(3)]
    b = [_row(42, i, 100 + i, i) for i in range(2)]
    with pytest.raises(ValueError, match="exact matched seed triplets"):
        validate_matched_conditions({"a": a, "b": b})


def test_original_comparison_checks_historical_seed_vectors_and_control_overlap():
    root = Path(__file__).resolve().parents[1]
    corrected_rows = {}
    for condition_id, template in ORIGINAL_EVALUATION_PATHS.items():
        groups = [42] if condition_id in {"random_action_control", "zero_weight_control"} else [42, 100, 200]
        rows = []
        for group in groups:
            relative = template if "{group}" not in template else template.format(group=group)
            with (root / relative).open(newline="") as fh:
                for source in csv.DictReader(fh):
                    rows.append({
                        "seed_group": group,
                        "episode_index": int(source["episode_index"]),
                        "base_seed": int(source["seed"]),
                        "environment_seed": 500_000 + int(source["seed"]),
                        "policy_seed": 900_000 + int(source["seed"]),
                        "score": float(source["score"]),
                    })
        corrected_rows[condition_id] = rows

    comparison = compare_corrected_to_original(root / "experiments/corrected_evaluation/full_run", corrected_rows, resamples=25)
    assert comparison["astra_local_reference"]["matched_episode_count"] == 3000
    assert comparison["random_action_control"]["original_csv_sample_size"] == 1000
    assert comparison["random_action_control"]["matched_episode_count"] == 1000
    assert comparison["zero_weight_control"]["matched_episode_count"] == 1000
    assert comparison["astra_historical_4000_note"]["text"].find("group 300") >= 0
