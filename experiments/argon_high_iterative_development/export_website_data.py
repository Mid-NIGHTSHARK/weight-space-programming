#!/usr/bin/env python3
"""Export the verified Argon High development group as website-ready JSON."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "model_registry.json"
RECONSTRUCTION = HERE / "reconstruction_manifest.json"
ANALYSIS = HERE / "evaluation/full_run/analysis/analysis.json"


def build_records():
    registry = json.loads(REGISTRY.read_text())
    reconstruction = json.loads(RECONSTRUCTION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    models = []
    for record in registry["conditions"]:
        version = reconstruction["versions"][record["id"]]
        metrics = analysis["conditions"][record["evaluation_metrics_id"]]
        raw = analysis["raw_csv_independent_recomputation"][record["evaluation_metrics_id"]]
        models.append({
            "id": record["id"],
            "name": record["label"],
            "stage": record["stage"],
            "condition_class": record["condition_class"],
            "attribution": record["recorded_model_attribution"],
            "provider_identity_status": record["provider_identity_status"],
            "generation_provenance_limit": record["generation_provenance_limit"],
            "source": {
                "path": record["source_file"],
                "sha256": record["source_sha256"],
                "weight_path": record["weight_path"],
                "weight_sha256": record["weight_sha256"],
            },
            "architecture": [600, 32, 32, 6],
            "parameter_count": version["total_parameters"],
            "nonzero_parameter_count": version["nonzero_by_tensor"]["total"],
            "sparsity_percent": 100.0 * (
                version["total_parameters"] - version["nonzero_by_tensor"]["total"]
            ) / version["total_parameters"],
            "benchmark_status": "complete",
            "benchmark_csv": record["evaluation_csv_path"],
            "metrics": {
                "episodes": metrics["episodes"],
                "mean_score": metrics["mean_score"],
                "standard_error_score": metrics["standard_error_score"],
                "score_mean_ci_95": metrics["score_mean_ci_95"],
                "median_score": metrics["median_score"],
                "minimum_score": metrics["minimum_score"],
                "maximum_score": metrics["maximum_score"],
                "mean_episode_length": metrics["mean_episode_length"],
                "median_episode_length": metrics["median_episode_length"],
                "maximum_episode_length": metrics["maximum_episode_length"],
                "cutoff_count": metrics["cutoff_count"],
                "cutoff_percentage": metrics["cutoff_percentage"],
                "ball_lost_count": raw["ball_lost_count"],
                "selected_action_distribution": metrics["selected_action_distribution"],
                "executed_action_distribution": metrics["executed_action_distribution_after_sticky_handling"],
                "score_quantiles": raw["score_quantiles"],
                "exact_score_counts": raw["exact_score_counts"],
                "per_seed_group_metrics": analysis["per_seed_group_metrics"][record["id"]],
            },
        })
    return {
        "schema_version": 1,
        "group_id": registry["group_id"],
        "title": registry["group_label"],
        "separate_from_primary_zero_shot_ranking": registry["not_part_of_primary_zero_shot_ranking"],
        "evaluation_status": registry["evaluation_status"],
        "evaluation_dataset": registry["evaluation_dataset"],
        "evaluation_protocol": "Phase 2 corrected MinAtar evaluator: 3,000 matched episodes per version, seed groups 42/100/200, independent environment and policy RNG streams, categorical softmax actions, 0.1 sticky actions, 10,000-step cap.",
        "researcher_contribution": registry["researcher_contribution"],
        "recorded_ai_contribution": registry["recorded_ai_contribution"],
        "development_summary": registry["development_summary"],
        "models": models,
        "paired_comparisons": analysis["paired_comparisons"],
        "statistical_interpretation": analysis["interpretation"],
        "limitations": [
            "The first-prompt source is a user-supplied assignment excerpt, not a provider-authenticated complete response.",
            "The repository also contains a distinct archived Gemini Argon run; it was not substituted for these tensors.",
            "Provider identity and generation settings were not independently authenticated.",
            "All follow-ups are researcher-directed and are not additional independent zero-shot generations.",
            "The second-follow-up source verifier has a floating-point assertion defect documented in the audit report.",
        ],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="website_model_records.json", help="New JSON path relative to this experiment directory")
    args = parser.parse_args()
    output_path = (HERE / args.output).resolve()
    if HERE not in output_path.parents:
        raise ValueError("--output must remain inside this experiment directory")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite website data: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_records(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
