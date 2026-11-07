#!/usr/bin/env python3
"""Analyze raw Argon High episode CSVs with the Phase 2 statistical code."""

import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.corrected_analysis import (  # noqa: E402
    ANALYSIS_METHOD,
    paired_comparisons,
    read_episode_csv,
    summarize_condition,
    validate_matched_conditions,
)


def raw_score_summary(csv_path):
    """Independent direct CSV calculation, separate from summarize_condition."""
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    cutoffs = np.asarray([row["step_cutoff"].strip().lower() in {"true", "1"} for row in rows])
    losses = np.asarray([row["ball_lost"].strip().lower() in {"true", "1"} for row in rows])
    unique_scores, score_counts = np.unique(scores, return_counts=True)
    if scores.size == 0:
        raise ValueError(f"No raw episode rows in {csv_path}")
    return {
        "episodes": int(scores.size),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "sample_std_score": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
        "standard_error_score": float(np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0,
        "minimum_score": float(np.min(scores)),
        "maximum_score": float(np.max(scores)),
        "score_quantiles": {
            str(q): float(np.quantile(scores, q / 100.0)) for q in (10, 25, 50, 75, 90, 95, 99)
        },
        "exact_score_counts": {str(float(score)): int(count) for score, count in zip(unique_scores, score_counts)},
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "cutoff_count": int(np.count_nonzero(cutoffs)),
        "cutoff_percentage": float(np.mean(cutoffs) * 100.0),
        "ball_lost_count": int(np.count_nonzero(losses)),
        "episode_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }


def _same_number(a, b):
    return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12))


def run(run_dir, overwrite_analysis=False):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_kind") != "full" or manifest.get("status") not in {
        "evaluation_complete_analysis_pending", "analysis_complete"
    }:
        raise ValueError("Analysis requires a complete 3,000-episode-per-condition run")
    if manifest.get("episodes_per_condition") != 3000:
        raise ValueError("Full analysis requires exactly 3,000 episodes per condition")
    analysis_dir = run_dir / "analysis"
    if analysis_dir.exists() and not overwrite_analysis:
        raise FileExistsError(
            f"Refusing to overwrite derived analysis artifacts at {analysis_dir}; "
            "pass --overwrite-analysis only to regenerate analysis from the unchanged raw CSVs"
        )

    condition_rows = {}
    labels = {}
    summaries = {}
    raw_recomputed = {}
    per_seed_metrics = {}
    registry = {item["id"]: item for item in manifest["conditions"]}
    for condition_id, record in registry.items():
        csv_path = run_dir / "conditions" / condition_id / "episode_scores.csv"
        actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if actual_hash != manifest["condition_csv_sha256"].get(condition_id):
            raise ValueError(f"Episode CSV hash mismatch for {condition_id}")
        rows = read_episode_csv(csv_path)
        if len(rows) != 3000:
            raise ValueError(f"Expected 3,000 rows for {condition_id}, found {len(rows)}")
        condition_rows[condition_id] = rows
        labels[condition_id] = record["label"]
        summaries[condition_id] = summarize_condition(rows, metadata={
            "condition_class": record["condition_class"],
            "weight_sha256": record["weight_sha256"],
            "episode_scores_sha256": actual_hash,
        })
        raw_recomputed[condition_id] = raw_score_summary(csv_path)
        for key in (
            "episodes", "mean_score", "median_score", "sample_std_score",
            "standard_error_score", "minimum_score", "maximum_score",
            "mean_episode_length", "median_episode_length", "cutoff_count", "cutoff_percentage",
        ):
            if key in {"episodes", "cutoff_count"}:
                if summaries[condition_id][key] != raw_recomputed[condition_id][key]:
                    raise AssertionError(f"Raw summary mismatch: {condition_id} {key}")
            elif not _same_number(summaries[condition_id][key], raw_recomputed[condition_id][key]):
                raise AssertionError(f"Raw summary mismatch: {condition_id} {key}")
        by_group = {}
        for group in (42, 100, 200):
            subset = [row for row in rows if row["seed_group"] == group]
            if len(subset) != 1000:
                raise ValueError(f"Expected 1,000 rows for {condition_id}, seed group {group}")
            by_group[str(group)] = summarize_condition(subset)
        per_seed_metrics[condition_id] = by_group

    keys = validate_matched_conditions(condition_rows)
    if len(keys) != 3000:
        raise ValueError(f"Expected 3,000 matched pairs, found {len(keys)}")
    paired = paired_comparisons(condition_rows, labels)
    run_manifest_schedule = run_dir / "episode_seeds.csv"
    with run_manifest_schedule.open(newline="") as stream:
        schedule = list(csv.DictReader(stream))
    expected_schedule = {
        (int(row["seed_group"]), int(row["episode_index"])): (
            int(row["base_seed"]), int(row["environment_seed"]), int(row["policy_seed"])
        ) for row in schedule
    }
    if len(expected_schedule) != 3000:
        raise ValueError("Recorded episode seed schedule is incomplete or duplicated")
    for condition_id, rows in condition_rows.items():
        actual = {
            (row["seed_group"], row["episode_index"]): (
                row["base_seed"], row["environment_seed"], row["policy_seed"]
            ) for row in rows
        }
        if actual != expected_schedule:
            raise ValueError(f"Seed schedule mismatch for {condition_id}")

    analysis = {
        "dataset": manifest["dataset"],
        "software_versions_observed_at_analysis": {
            "python": sys.version,
            "numpy": np.__version__,
            "minatar": version("minatar"),
            "scipy": version("scipy"),
            "torch": version("torch"),
            "pandas": version("pandas"),
            "pytest": version("pytest"),
            "note": "Observed during final audit in the same project virtual environment; the runner did not capture versions in the run-start manifest.",
        },
        "episodes_per_condition": 3000,
        "matched_episode_pairs_per_comparison": len(keys),
        "statistical_method": ANALYSIS_METHOD,
        "analysis_code": {
            "repository_analysis_file": "evaluation/corrected_analysis.py",
            "repository_analysis_sha256": hashlib.sha256((ROOT / "evaluation/corrected_analysis.py").read_bytes()).hexdigest(),
            "experiment_analysis_wrapper": "experiments/argon_high_iterative_development/analyze_evaluation.py",
            "experiment_analysis_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "conditions": summaries,
        "raw_csv_independent_recomputation": raw_recomputed,
        "per_seed_group_metrics": per_seed_metrics,
        "paired_comparisons": paired,
        "interpretation": "All three conditions are iterative-development stages. Pairwise differences are descriptive/inferential under matched seed identifiers; nonsignificance is not evidence of equivalence.",
    }
    analysis_dir.mkdir(exist_ok=True)
    for filename, content in (
        ("analysis.json", analysis),
        ("condition_metrics.json", summaries),
        ("raw_recomputed_metrics.json", raw_recomputed),
        ("per_seed_metrics.json", per_seed_metrics),
        ("paired_comparisons.json", paired),
    ):
        (analysis_dir / filename).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Gemini Argon High — Iterative Development Evaluation",
        "",
        "This additional comparison is separate from the project's primary zero-shot model ranking.",
        "All follow-ups are researcher-directed development stages. Model/provider identity is recorded, not independently authenticated.",
        "",
        "## Corrected benchmark results",
        "",
        "| Version | Episodes | Mean ± SE | 95% t interval | Median | Min–max | Mean / median / max length | Cutoffs | Ball lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id, record in registry.items():
        m = summaries[condition_id]
        ci = m["score_mean_ci_95"]
        raw = raw_recomputed[condition_id]
        lines.append(
            f"| {record['label']} | {m['episodes']} | {m['mean_score']:.4f} ± {m['standard_error_score']:.4f} "
            f"| [{ci[0]:.4f}, {ci[1]:.4f}] | {m['median_score']:.1f} | "
            f"{m['minimum_score']:.0f}–{m['maximum_score']:.0f} | {m['mean_episode_length']:.2f} / "
            f"{m['median_episode_length']:.0f} / {m['maximum_episode_length']} | "
            f"{m['cutoff_count']} ({m['cutoff_percentage']:.3f}%) | {raw['ball_lost_count']} |"
        )
    lines.extend([
        "",
        "Seed-group mean scores (1,000 episodes per group):",
        "",
        "| Version | Group 42 | Group 100 | Group 200 |",
        "|---|---:|---:|---:|",
    ])
    for condition_id, record in registry.items():
        groups = per_seed_metrics[condition_id]
        lines.append(
            f"| {record['label']} | {groups['42']['mean_score']:.3f} | "
            f"{groups['100']['mean_score']:.3f} | {groups['200']['mean_score']:.3f} |"
        )
    lines.extend([
        "",
        "Selected action distribution (percent of all selected actions; only no-op/left/right had nonzero counts):",
        "",
        "| Version | Selected no-op / left / right | Executed no-op / left / right |",
        "|---|---:|---:|",
    ])
    for condition_id, record in registry.items():
        m = summaries[condition_id]
        selected = m["selected_action_distribution"]
        executed = m["executed_action_distribution_after_sticky_handling"]
        lines.append(
            f"| {record['label']} | {selected['0']['proportion']*100:.2f}% / "
            f"{selected['1']['proportion']*100:.2f}% / {selected['3']['proportion']*100:.2f}% "
            f"| {executed['0']['proportion']*100:.2f}% / {executed['1']['proportion']*100:.2f}% / "
            f"{executed['3']['proportion']*100:.2f}% |"
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "Differences are condition A minus condition B. Confidence intervals use paired episode bootstrap; p-values use a two-sided sign-flip test; Holm correction covers all three pairwise comparisons.",
        "",
        "| A − B | Mean difference | Paired bootstrap 95% CI | Raw p | Holm p | Cohen's dz |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in paired:
        ci = item["paired_difference_ci_95_percentile_bootstrap"]
        dz = item["effect_size_cohen_dz"]
        lines.append(
            f"| {item['condition_a_label']} − {item['condition_b_label']} | {item['mean_paired_difference']:.4f} "
            f"| [{ci[0]:.4f}, {ci[1]:.4f}] | {item['paired_sign_flip_p_value']:.6g} "
            f"| {item['p_value_holm_adjusted_all_pairwise_conditions']:.6g} "
            f"| {'undefined' if dz is None else f'{dz:.4f}'} |"
        )
    lines.extend([
        "",
        "## Analysis limits",
        "",
        "Episodes are paired by seed group and episode index, with exact equality of base, environment, and policy seed identifiers checked. A shared seed schedule does not make resulting trajectories identical when model actions differ. The confidence intervals and tests do not establish equivalence. The supplied second-follow-up mathematical script contains a floating-point assertion that cannot pass because `1.0 - 1e-40` rounds to `1.0`; the source was preserved unchanged, and this analysis uses the repository's existing statistical routines.",
        "",
    ])
    (analysis_dir / "analysis_report.md").write_text("\n".join(lines))
    manifest["status"] = "analysis_complete"
    manifest["analysis"] = {
        "path": "analysis/analysis.json",
        "statistical_method": ANALYSIS_METHOD,
        "analysis_code": analysis["analysis_code"],
        "conditions": list(registry),
    }
    manifest["runtime_versions_observed_after_run"] = analysis["software_versions_observed_at_analysis"]
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Analyzed {len(keys)} matched episode pairs for {len(registry)} conditions.")
    for condition_id, record in registry.items():
        metrics = summaries[condition_id]
        print(f"{condition_id}: mean={metrics['mean_score']:.4f}, SE={metrics['standard_error_score']:.4f}, median={metrics['median_score']:.1f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Evaluation directory containing run_manifest.json")
    parser.add_argument("--overwrite-analysis", action="store_true", help="Replace only derived analysis files after raw CSV hashes are rechecked")
    args = parser.parse_args()
    run(args.run_dir, overwrite_analysis=args.overwrite_analysis)


if __name__ == "__main__":
    main()
