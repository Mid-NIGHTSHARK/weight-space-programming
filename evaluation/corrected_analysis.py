"""Metrics and episode-matched inference for Phase 2 corrected evaluations."""

import csv
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t

from evaluation.corrected_evaluate import ACTION_NAMES


ANALYSIS_METHOD = {
    "condition_ci": "two-sided Student-t interval for the episode mean; sample standard deviation uses ddof=1",
    "paired_ci": "percentile paired bootstrap interval for the mean score difference, resampling matched episode pairs",
    "paired_test": "two-sided paired sign-flip permutation test of the mean difference under within-pair exchangeability",
    "permutation_resamples": 19_999,
    "bootstrap_resamples": 9_999,
    "p_value_correction": "Holm step-down family-wise correction across every pairwise condition comparison",
    "effect_size": "paired Cohen dz: mean within-pair score difference divided by its sample standard deviation",
    "random_seed": 2_026_110_6,
    "equivalence_testing": "omitted; no independently justified, pre-registered margin",
}

ORIGINAL_EVALUATION_PATHS = {
    "gemini_argon_4_medium": "experiments/strict_replication/generation_001/eval_seed{group}/episode_scores.csv",
    "gpt6_original": "experiments/strict_replication/generation_002_gpt6/evaluation/eval_seed{group}/episode_scores.csv",
    "claude_sonnet_55": "experiments/strict_replication/generation_003_sonnet55/evaluation/eval_seed{group}/episode_scores.csv",
    "deepseek_original": "experiments/strict_replication/generation_004_deepseek/evaluation/eval_seed{group}/episode_scores.csv",
    "gpt6_luna_extra_high": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/evaluation/eval_seed{group}/episode_scores.csv",
    "astra_local_reference": "experiments/reference_eval_runs/run_seed{group}/episode_scores.csv",
    "ppo_trained_baseline": "experiments/trained_baseline/eval_seed{group}/episode_scores.csv",
    "random_action_control": "results/baselines/random/episode_scores.csv",
    "zero_weight_control": "results/baselines/zero/episode_scores.csv",
}


def read_episode_csv(path: Path) -> List[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No episode rows in {path}")
    numeric = {
        "seed_group": int,
        "episode_index": int,
        "base_seed": int,
        "environment_seed": int,
        "policy_seed": int,
        "score": float,
        "length": int,
        "bricks_destroyed": int,
        "ball_lost": lambda x: str(x).lower() in {"true", "1"},
        "step_cutoff": lambda x: str(x).lower() in {"true", "1"},
    }
    for row in rows:
        for field, converter in numeric.items():
            row[field] = converter(row[field])
        for action_id in range(6):
            row[f"action_{action_id}_selected_count"] = int(row[f"action_{action_id}_selected_count"])
            row[f"action_{action_id}_executed_count"] = int(row[f"action_{action_id}_executed_count"])
    return rows


def _mean_ci(scores: np.ndarray) -> Tuple[float, float]:
    n = len(scores)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n))
    radius = float(student_t.ppf(0.975, df=n - 1) * se)
    return mean - radius, mean + radius


def summarize_condition(rows: Sequence[dict], metadata: dict | None = None) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty condition")
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.int64)
    n = len(scores)
    sd = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    se = sd / float(np.sqrt(n)) if n > 1 else 0.0
    score_low, score_high = _mean_ci(scores)
    cutoffs = sum(bool(r["step_cutoff"]) for r in rows)
    selected_totals = {
        action: sum(r[f"action_{action}_selected_count"] for r in rows)
        for action in range(6)
    }
    executed_totals = {
        action: sum(r[f"action_{action}_executed_count"] for r in rows)
        for action in range(6)
    }
    for row in rows:
        selected_count = sum(row[f"action_{a}_selected_count"] for a in range(6))
        executed_count = sum(row[f"action_{a}_executed_count"] for a in range(6))
        if selected_count != row["length"]:
            raise ValueError("Selected-action counts do not sum to the recorded episode length")
        if executed_count != row["length"]:
            raise ValueError("Executed-action counts do not sum to the recorded episode length")
    selected_denominator = sum(selected_totals.values())
    executed_denominator = sum(executed_totals.values())
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['seed_group']}:{row['episode_index']}:{row['base_seed']}:{row['score']}:".encode()
        )
        digest.update(str(row["trajectory_sha256"]).encode())

    return {
        "episodes": n,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "sample_std_score": sd,
        "standard_error_score": se,
        "score_mean_ci_95": [float(score_low), float(score_high)],
        "minimum_score": float(np.min(scores)),
        "maximum_score": float(np.max(scores)),
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "maximum_episode_length": int(np.max(lengths)),
        "cutoff_count": int(cutoffs),
        "cutoff_percentage": float(cutoffs / n * 100.0),
        "selected_action_distribution": {
            str(a): {
                "name": ACTION_NAMES[a],
                "count": int(selected_totals[a]),
                "proportion": float(selected_totals[a] / selected_denominator) if selected_denominator else 0.0,
            }
            for a in range(6)
        },
        "executed_action_distribution_after_sticky_handling": {
            str(a): {
                "name": ACTION_NAMES[a],
                "count": int(executed_totals[a]),
                "proportion": float(executed_totals[a] / executed_denominator) if executed_denominator else 0.0,
            }
            for a in range(6)
        },
        "episode_trajectory_set_sha256": digest.hexdigest(),
        "metadata": metadata or {},
    }


def _bootstrap_mean_ci(differences: np.ndarray, resamples: int, seed: int, batch_size: int = 256):
    rng = np.random.default_rng(seed)
    n = differences.size
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _paired_sign_flip_p(differences: np.ndarray, resamples: int, seed: int, batch_size: int = 256) -> float:
    rng = np.random.default_rng(seed)
    n = differences.size
    observed = float(np.mean(differences))
    extreme = 0
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(count, n), dtype=np.int8) * 2 - 1
        permuted_means = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted_means) >= abs(observed)))
    return float((extreme + 1) / (resamples + 1))


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    m = len(p_values)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p_values[int(original_index)]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=np.float64)
    for rank, original_index in enumerate(order):
        adjusted[int(original_index)] = adjusted_sorted[rank]
    return [float(v) for v in adjusted]


def _pair_key(row: dict) -> tuple:
    return int(row["seed_group"]), int(row["episode_index"])


def validate_matched_conditions(condition_rows: Dict[str, Sequence[dict]]) -> List[tuple]:
    if len(condition_rows) < 2:
        raise ValueError("At least two conditions are required for paired comparisons")
    condition_ids = list(condition_rows)
    first_rows = condition_rows[condition_ids[0]]
    expected = {
        _pair_key(row): (row["base_seed"], row["environment_seed"], row["policy_seed"])
        for row in first_rows
    }
    if len(expected) != len(first_rows):
        raise ValueError(f"Duplicate episode keys in condition {condition_ids[0]}")
    ordered_keys = sorted(expected)
    for condition_id in condition_ids[1:]:
        rows = condition_rows[condition_id]
        observed = {
            _pair_key(row): (row["base_seed"], row["environment_seed"], row["policy_seed"])
            for row in rows
        }
        if len(observed) != len(rows) or observed != expected:
            raise ValueError(f"Condition {condition_id} does not have the exact matched seed triplets")
    return ordered_keys


def paired_comparisons(
    condition_rows: Dict[str, Sequence[dict]],
    labels: Dict[str, str],
    bootstrap_resamples: int = ANALYSIS_METHOD["bootstrap_resamples"],
    permutation_resamples: int = ANALYSIS_METHOD["permutation_resamples"],
    random_seed: int = ANALYSIS_METHOD["random_seed"],
) -> List[dict]:
    validate_matched_conditions(condition_rows)
    results = []
    for pair_index, (condition_a, condition_b) in enumerate(itertools.combinations(condition_rows, 2)):
        a_by_key = {_pair_key(row): row for row in condition_rows[condition_a]}
        b_by_key = {_pair_key(row): row for row in condition_rows[condition_b]}
        keys = sorted(a_by_key)
        differences = np.asarray(
            [a_by_key[key]["score"] - b_by_key[key]["score"] for key in keys],
            dtype=np.float64,
        )
        pair_seed = int(random_seed + pair_index * 2 + 1)
        ci_low, ci_high = _bootstrap_mean_ci(differences, bootstrap_resamples, pair_seed)
        p_value = _paired_sign_flip_p(differences, permutation_resamples, pair_seed + 1)
        sd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
        mean_difference = float(np.mean(differences))
        effect = mean_difference / sd if sd > 0 else None
        results.append({
            "condition_a_id": condition_a,
            "condition_a_label": labels[condition_a],
            "condition_b_id": condition_b,
            "condition_b_label": labels[condition_b],
            "difference_definition": f"{labels[condition_a]} minus {labels[condition_b]}",
            "matched_episode_pairs": int(differences.size),
            "mean_paired_difference": mean_difference,
            "paired_difference_ci_95_percentile_bootstrap": [ci_low, ci_high],
            "paired_difference_sample_std": sd,
            "effect_size_cohen_dz": effect,
            "paired_sign_flip_p_value": p_value,
            "pairwise_resamples": {
                "bootstrap": int(bootstrap_resamples),
                "permutation": int(permutation_resamples),
                "random_seed": pair_seed,
            },
        })
    adjusted = _holm_adjust([item["paired_sign_flip_p_value"] for item in results])
    for item, p_holm in zip(results, adjusted):
        item["p_value_holm_adjusted_all_pairwise_conditions"] = p_holm
    return results


def compare_corrected_to_original(run_dir: Path, condition_rows: Dict[str, Sequence[dict]], resamples: int = 9_999) -> dict:
    """Descriptive matched-seed changes; no causal claim about coupling alone."""
    run_dir = Path(run_dir)
    repository_root = Path(__file__).resolve().parents[1]
    result = {}
    for condition_index, (condition_id, template) in enumerate(ORIGINAL_EVALUATION_PATHS.items()):
        original_rows = []
        groups = [42] if condition_id in {"random_action_control", "zero_weight_control"} else [42, 100, 200]
        original_paths = []
        original_path_hashes = {}
        for group in groups:
            relative_path = template if "{group}" not in template else template.format(group=group)
            path = repository_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Original evaluation CSV missing: {path}")
            original_paths.append(relative_path)
            original_path_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(newline="") as fh:
                group_rows = list(csv.DictReader(fh))
            original_rows.extend((group, row) for row in group_rows)

        old_by_key = {}
        for group, row in original_rows:
            key = (int(group), int(row["episode_index"]))
            if key in old_by_key:
                raise ValueError(f"Duplicate original episode key for {condition_id}: {key}")
            old_by_key[key] = {"seed": int(row["seed"]), "score": float(row["score"])}
        corrected_by_key = {_pair_key(row): row for row in condition_rows[condition_id]}
        common_keys = sorted(set(old_by_key).intersection(corrected_by_key))
        if len(common_keys) != len(old_by_key):
            raise ValueError(f"Not every original episode has a corrected matched-seed counterpart for {condition_id}")
        for key in common_keys:
            if old_by_key[key]["seed"] != corrected_by_key[key]["base_seed"]:
                raise ValueError(f"Base-seed mismatch between original and corrected {condition_id} at {key}")

        old_scores = np.asarray([old_by_key[key]["score"] for key in common_keys], dtype=np.float64)
        new_scores = np.asarray([corrected_by_key[key]["score"] for key in common_keys], dtype=np.float64)
        differences = new_scores - old_scores
        ci_low, ci_high = _bootstrap_mean_ci(
            differences,
            resamples,
            ANALYSIS_METHOD["random_seed"] + 10_000 + condition_index,
        )
        result[condition_id] = {
            "original_csv_sample_size": len(old_scores),
            "matched_episode_count": len(common_keys),
            "original_mean_score": float(np.mean(old_scores)),
            "corrected_mean_score_on_matched_seeds": float(np.mean(new_scores)),
            "mean_paired_score_change_corrected_minus_original": float(np.mean(differences)),
            "paired_bootstrap_ci_95_for_change": [ci_low, ci_high],
            "original_csv_paths": original_paths,
            "original_csv_sha256": original_path_hashes,
            "interpretation_limit": "The paired change reflects the RNG protocol change and new random draws together; it does not isolate the causal contribution of stream coupling alone.",
        }
    result["astra_historical_4000_note"] = {
        "text": "Astra's 4,000-episode local historical result includes an additional group 300. This matched comparison uses only groups 42, 100, and 200 (3,000 episodes). The paper's published 1,000-episode result is a separate sample and is not substituted here."
    }
    return result


def analyze_run(
    run_dir: Path,
    bootstrap_resamples: int = ANALYSIS_METHOD["bootstrap_resamples"],
    permutation_resamples: int = ANALYSIS_METHOD["permutation_resamples"],
) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_kind") != "full":
        raise ValueError("Only a full corrected benchmark can enter final analysis")
    if manifest.get("status") != "evaluation_complete_analysis_pending":
        raise ValueError(f"Run is not ready for analysis (status={manifest.get('status')!r})")

    condition_rows = {}
    labels = {}
    metrics = {}
    registry = {item["id"]: item for item in manifest["conditions"]}
    for condition_id, condition in registry.items():
        csv_path = run_dir / "conditions" / condition_id / "episode_scores.csv"
        expected_hash = manifest["condition_csv_sha256"].get(condition_id)
        actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Episode CSV hash mismatch for {condition_id}")
        rows = read_episode_csv(csv_path)
        condition_rows[condition_id] = rows
        labels[condition_id] = condition["label"]
        metrics[condition_id] = summarize_condition(rows, metadata={
            "condition_class": condition["class"],
            "weight_sha256": condition.get("weight_sha256"),
            "episode_scores_sha256": actual_hash,
        })

    matched_keys = validate_matched_conditions(condition_rows)
    if len(matched_keys) != 3000:
        raise ValueError(f"Expected 3,000 matched episodes per condition; found {len(matched_keys)}")
    with (run_dir / manifest["episode_seeds_file"]).open(newline="") as fh:
        seed_schedule = list(csv.DictReader(fh))
    scheduled = {
        (int(row["seed_group"]), int(row["episode_index"])): (
            int(row["base_seed"]), int(row["environment_seed"]), int(row["policy_seed"])
        )
        for row in seed_schedule
    }
    if len(scheduled) != len(seed_schedule) or len(scheduled) != len(matched_keys):
        raise ValueError("Episode seed schedule is incomplete or contains duplicate keys")
    for condition_id, rows in condition_rows.items():
        observed = {
            _pair_key(row): (row["base_seed"], row["environment_seed"], row["policy_seed"])
            for row in rows
        }
        if observed != scheduled:
            raise ValueError(f"Condition {condition_id} does not match the recorded episode seed schedule")
    paired = paired_comparisons(
        condition_rows,
        labels,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
    )
    original_comparison = compare_corrected_to_original(run_dir, condition_rows)
    conditions = []
    for condition_id, item in registry.items():
        row = {
            "condition_id": condition_id,
            "model": item["label"],
            "condition_class": item["class"],
            "dataset": manifest["dataset"],
            "protocol": manifest["evaluation_config"]["rng_implementation"]["rng_protocol"],
            "weight_sha256": item.get("weight_sha256"),
            "mean_score": metrics[condition_id]["mean_score"],
            "se_score": metrics[condition_id]["standard_error_score"],
            "episodes": metrics[condition_id]["episodes"],
            "cutoff_percentage": metrics[condition_id]["cutoff_percentage"],
        }
        conditions.append(row)
    conditions.sort(key=lambda item: item["mean_score"], reverse=True)

    method = dict(ANALYSIS_METHOD)
    method["bootstrap_resamples"] = int(bootstrap_resamples)
    method["permutation_resamples"] = int(permutation_resamples)
    analysis = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "run_manifest_sha256_before_analysis": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "analysis_method": method,
        "matched_episode_count": len(matched_keys),
        "conditions": metrics,
        "pairwise_comparisons": paired,
        "corrected_vs_original_matched_seed_comparison": original_comparison,
        "ranking_all_conditions": conditions,
        "ranking_primary_zero_shot": [
            item for item in conditions if item["condition_class"] == "primary_zero_shot"
        ],
        "excluded_from_primary_zero_shot_ranking": [
            condition_id for condition_id, item in registry.items()
            if item["class"] != "primary_zero_shot"
        ],
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    _write_json(analysis_dir / "condition_metrics.json", {
        "dataset": manifest["dataset"], "analysis_method": method, "conditions": metrics,
    })
    _write_json(analysis_dir / "paired_comparisons.json", {
        "dataset": manifest["dataset"], "analysis_method": method, "matched_episode_count": len(matched_keys),
        "pairwise_comparisons": paired,
    })
    _write_json(analysis_dir / "comparative_metrics.json", conditions)
    _write_json(analysis_dir / "corrected_vs_original.json", original_comparison)
    _write_json(analysis_dir / "analysis.json", analysis)
    (analysis_dir / "analysis_report.md").write_text(render_markdown(analysis) + "\n")

    artifact_hashes = {}
    for path in sorted(analysis_dir.iterdir()):
        if path.is_file():
            artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["status"] = "complete"
    manifest["analysis_completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["analysis_source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["analysis_method"] = method
    manifest["analysis_artifact_sha256"] = artifact_hashes
    manifest["primary_zero_shot_ranking"] = [row["condition_id"] for row in analysis["ranking_primary_zero_shot"]]
    _write_json(manifest_path, manifest)
    return analysis


def render_markdown(analysis: dict) -> str:
    lines = [
        "# Corrected MinAtar Breakout Evaluation Analysis",
        "",
        f"Dataset: `{analysis['dataset']}`; matched episodes per condition: {analysis['matched_episode_count']:,}.",
        "",
        "## Condition metrics",
        "",
        "| Rank | Condition | Type | N | Mean | Median | SD | SE | 95% CI (mean) | Min | Max | Mean length | Median length | Max length | Cutoffs | Selected-action counts |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(analysis["ranking_all_conditions"], 1):
        item = analysis["conditions"][row["condition_id"]]
        selected = ", ".join(
            f"{ACTION_NAMES[a]}={item['selected_action_distribution'][str(a)]['count']}"
            for a in range(6)
        )
        lines.append(
            f"| {rank} | {row['model']} | {row['condition_class']} | {item['episodes']} | "
            f"{item['mean_score']:.4f} | {item['median_score']:.1f} | {item['sample_std_score']:.4f} | "
            f"{item['standard_error_score']:.4f} | [{item['score_mean_ci_95'][0]:.4f}, {item['score_mean_ci_95'][1]:.4f}] | "
            f"{item['minimum_score']:.1f} | {item['maximum_score']:.1f} | {item['mean_episode_length']:.2f} | "
            f"{item['median_episode_length']:.1f} | {item['maximum_episode_length']} | "
            f"{item['cutoff_count']} ({item['cutoff_percentage']:.2f}%) | {selected} |"
        )
    lines.extend([
        "",
        "Selected-action counts are requests sent to MinAtar. Executed-action counts are separately recorded after sticky-action handling in `condition_metrics.json`.",
        "",
        "## Matched comparisons",
        "",
        "Differences are condition A minus condition B. Confidence intervals use paired bootstrap resampling of the matched episode pairs. P-values use two-sided sign-flip permutation tests under within-pair exchangeability and are Holm-adjusted across all condition pairs.",
        "",
        "| A | B | Mean paired difference | 95% bootstrap CI | Cohen dz | p | Holm-adjusted p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for item in analysis["pairwise_comparisons"]:
        ci = item["paired_difference_ci_95_percentile_bootstrap"]
        effect = item["effect_size_cohen_dz"]
        effect_s = "undefined" if effect is None else f"{effect:.4f}"
        lines.append(
            f"| {item['condition_a_label']} | {item['condition_b_label']} | {item['mean_paired_difference']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {effect_s} | {item['paired_sign_flip_p_value']:.6g} | "
            f"{item['p_value_holm_adjusted_all_pairwise_conditions']:.6g} |"
        )
    lines.extend([
        "",
        "## Corrected versus original historical results",
        "",
        "These comparisons use the same base-seed vectors where historical episode CSVs exist. Changes are descriptive paired changes; they combine stream decoupling with different random draws and cannot isolate coupling as the sole cause. Astra's 3,000 matched episodes are distinct from its 4,000-episode local historical evaluation and the paper's separate 1,000-episode published result.",
        "",
        "| Condition | Original N | Original mean | Corrected matched N | Corrected mean | Mean paired change | 95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition_id, item in analysis["corrected_vs_original_matched_seed_comparison"].items():
        if condition_id == "astra_historical_4000_note":
            continue
        ci = item["paired_bootstrap_ci_95_for_change"]
        lines.append(
            f"| {condition_id} | {item['original_csv_sample_size']} | {item['original_mean_score']:.4f} | "
            f"{item['matched_episode_count']} | {item['corrected_mean_score_on_matched_seeds']:.4f} | "
            f"{item['mean_paired_score_change_corrected_minus_original']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines.extend([
        "",
        "No confirmatory equivalence test was conducted. The earlier Astra ±25-point margin is retained only as post-hoc exploratory work and is not used here.",
        "",
    ])
    return "\n".join(lines)


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
