#!/usr/bin/env python3
"""Export hash-checked research artifacts for the static GitHub Pages site.

The exporter only writes beneath docs/data/. It never rewrites experiment files.
Run with --check in CI to verify that the published bundle is current without
changing the worktree.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments/corrected_evaluation"
RUN_DIR = DATA_DIR / "full_run"
ARGON_DIR = ROOT / "experiments/argon_high_iterative_development"
SITE_DATA = ROOT / "docs/data"
PRIMARY_ORDER = [
    "gemini_argon_4_medium",
    "claude_sonnet_55",
    "gpt6_original",
    "gpt6_luna_extra_high",
    "deepseek_original",
]
RESPONSE_PATHS = {
    "gemini_argon_4_medium": "experiments/strict_replication/generation_001/output/raw_response.txt",
    "gpt6_original": "experiments/strict_replication/generation_002_gpt6/output/gpt6_raw_response.txt",
    "claude_sonnet_55": "experiments/strict_replication/generation_003_sonnet55/input/sonnet55_raw_response.txt",
    "deepseek_original": "experiments/strict_replication/generation_004_deepseek/input/deepseek_raw_response.txt",
    "gpt6_luna_extra_high": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/input/gpt6_luna_raw_response.txt",
}
METADATA_PATHS = {
    "gemini_argon_4_medium": "experiments/strict_replication/generation_001/output/generation_metadata.json",
    "gpt6_original": "experiments/strict_replication/generation_002_gpt6/output/generation_metadata.json",
    "claude_sonnet_55": "experiments/strict_replication/generation_003_sonnet55/output/generation_metadata.json",
    "deepseek_original": "experiments/strict_replication/generation_004_deepseek/output/generation_metadata.json",
    "gpt6_luna_extra_high": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/manifest.json",
}
PROMPT_PATH = "experiments/strict_replication/generation_001/input/experiment_5_prompt.txt"
TENSOR_SHAPES = {
    "W1": (600, 32), "b1": (32,),
    "W2": (32, 32), "b2": (32,),
    "W3": (32, 6), "b3": (6,),
}
ARCHIVE_KEY_MAP = {
    "actor/dense1/kernel": "W1",
    "actor/dense1/bias": "b1",
    "actor/dense2/kernel": "W2",
    "actor/dense2/bias": "b2",
    "actor/logits/kernel": "W3",
    "actor/logits/bias": "b3",
}
WEIGHT_PATHS = {
    "gemini_argon_4_medium": "experiments/strict_replication/generation_001/output/argon_weights.npz",
    "gpt6_original": "experiments/strict_replication/generation_002_gpt6/output/gpt6_weights.npz",
    "claude_sonnet_55": "experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz",
    "deepseek_original": "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz",
    "gpt6_luna_extra_high": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz",
    "astra_local_reference": "experiments/reference/hand.npz",
    "ppo_trained_baseline": "experiments/trained_baseline/trained_actor_weights.npz",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_mean(values: list[float]) -> float:
    """Use a stable summation order so generated JSON is Python-version agnostic."""
    return math.fsum(float(value) for value in values) / len(values)


def deterministic_sample_std(values: list[float], mean: float | None = None) -> float:
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least two values")
    center = deterministic_mean(values) if mean is None else mean
    squared_deviations = [((float(value) - center) ** 2) for value in values]
    return math.sqrt(math.fsum(squared_deviations) / (len(values) - 1))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preserved_file(relative_path: str, preserved_records: dict[str, dict[str, Any]]) -> str:
    path = ROOT / relative_path
    record = preserved_records.get(relative_path)
    if record is None:
        raise ValueError(f"No preserved-artifact hash record for {relative_path}")
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != record.get("sha256") or len(raw) != record.get("size_bytes"):
        raise ValueError(f"Preserved artifact mismatch for {relative_path}")
    return actual


def read_verified_csv(condition_id: str, expected_hash: str) -> tuple[list[dict[str, str]], bytes]:
    relative = Path("conditions") / condition_id / "episode_scores.csv"
    path = RUN_DIR / relative
    raw = path.read_bytes()
    actual_hash = sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise ValueError(f"Episode CSV hash mismatch for {condition_id}: {actual_hash} != {expected_hash}")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if len(rows) != 3000:
        raise ValueError(f"Expected 3,000 episodes for {condition_id}; found {len(rows)}")
    required = {
        "seed_group", "episode_index", "base_seed", "environment_seed", "policy_seed",
        "score", "length", "ball_lost", "step_cutoff", "trajectory_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0] if rows else {}))
        raise ValueError(f"Episode CSV schema is incomplete for {condition_id}: {missing}")
    return rows, raw


def bool_value(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"Invalid boolean value in episode data: {value!r}")
    return value.lower() in {"true", "1"}


def summarize_rows(rows: list[dict[str, str]], artifact_metrics: dict[str, Any]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Episode scores must be finite")
    n = len(scores)
    mean = deterministic_mean(scores)
    sd = deterministic_sample_std(scores, mean)
    se = sd / math.sqrt(n)
    median = statistics.median(scores)
    observed = {
        "episodes": n,
        "mean_score": mean,
        "standard_error_score": se,
        "median_score": median,
        "minimum_score": min(scores),
        "maximum_score": max(scores),
    }
    for key, actual in observed.items():
        expected = artifact_metrics[key]
        if isinstance(actual, float):
            if not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Raw CSV and analysis metric disagree for {key}: {actual} != {expected}")
        elif actual != expected:
            raise ValueError(f"Raw CSV and analysis metric disagree for {key}: {actual} != {expected}")

    cutoffs = sum(bool_value(row["step_cutoff"]) for row in rows)
    losses = sum(bool_value(row["ball_lost"]) for row in rows)
    selected_counts = {str(i): 0 for i in range(6)}
    executed_counts = {str(i): 0 for i in range(6)}
    for row in rows:
        for action in range(6):
            selected_key = f"action_{action}_selected_count"
            executed_key = f"action_{action}_executed_count"
            if selected_key in row:
                selected_counts[str(action)] += int(row[selected_key])
            if executed_key in row:
                executed_counts[str(action)] += int(row[executed_key])

    by_group: dict[str, list[float]] = {}
    episodes: list[list[Any]] = []
    for row, score, length in zip(rows, scores, lengths):
        group = str(int(row["seed_group"]))
        by_group.setdefault(group, []).append(score)
        if length < 1 or length > 10_000:
            raise ValueError(f"Invalid episode length for {group}:{row['episode_index']}: {length}")
        episodes.append([
            score, length, int(group), int(row["episode_index"]),
            bool_value(row["step_cutoff"]),
        ])
    group_means = {group: deterministic_mean(values) for group, values in sorted(by_group.items())}
    return {
        **observed,
        "sample_std_score": sd,
        "mean_episode_length": deterministic_mean(lengths),
        "median_episode_length": statistics.median(lengths),
        "maximum_episode_length": max(lengths),
        "cutoff_count": cutoffs,
        "cutoff_percentage": 100 * cutoffs / n,
        "ball_lost_count": losses,
        "ball_lost_percentage": 100 * losses / n,
        "score_mean_ci_95": artifact_metrics["score_mean_ci_95"],
        "selected_action_counts": selected_counts,
        "executed_action_counts": executed_counts,
        "mean_score_by_seed_group": group_means,
        "episode_records": episodes,
    }


def metadata_for(condition_id: str, preserved_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path_text = METADATA_PATHS.get(condition_id)
    if path_text:
        data = read_json(ROOT / path_text)
    elif condition_id == "astra_local_reference":
        data = {"model_name": "Astra reference weights", "condition_type": "handcrafted_reference"}
    elif condition_id == "ppo_trained_baseline":
        data = read_json(ROOT / "experiments/trained_baseline/training_config.json")
    else:
        data = {}
    response_path = RESPONSE_PATHS.get(condition_id)
    if response_path:
        raw = (ROOT / response_path).read_bytes()
        expected_hash = verify_preserved_file(response_path, preserved_records)
        return {
            "recorded": data,
            "response_file": f"responses/{condition_id}.txt",
            "response_sha256": expected_hash,
        }
    return {"recorded": data, "response_file": None, "response_sha256": None}


def model_tensors(condition_id: str, expected_hash: str | None, preserved_records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if condition_id == "zero_weight_control":
        arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in TENSOR_SHAPES.items()}
        archive_sha = None
        path_text = None
    else:
        path_text = WEIGHT_PATHS.get(condition_id)
        if path_text is None:
            return None
        path = ROOT / path_text
        raw = path.read_bytes()
        archive_sha = sha256_bytes(raw)
        if expected_hash and archive_sha != expected_hash:
            raise ValueError(f"Weight archive hash mismatch for {condition_id}")
        verify_preserved_file(path_text, preserved_records)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {}
            for key in archive.files:
                tensor = ARCHIVE_KEY_MAP.get(key, key)
                if tensor in arrays:
                    raise ValueError(f"Duplicate tensor {tensor} in {path_text}")
                arrays[tensor] = np.asarray(archive[key], dtype=np.float32, order="C")
    if set(arrays) != set(TENSOR_SHAPES):
        raise ValueError(f"Tensor set mismatch for {condition_id}: {sorted(arrays)}")
    encoded: dict[str, Any] = {}
    tensor_stats: dict[str, Any] = {}
    nonzero = 0
    total = 0
    for name, shape in TENSOR_SHAPES.items():
        array = np.asarray(arrays[name], dtype="<f4", order="C")
        if tuple(array.shape) != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} tensor for {condition_id}: {array.shape}")
        nonzero += int(np.count_nonzero(array))
        total += int(array.size)
        encoded[name] = {
            "shape": list(shape),
            "dtype": "float32-le",
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
        flat = array.reshape(-1)
        nonzero_indices = np.flatnonzero(flat)
        largest = sorted(nonzero_indices.tolist(), key=lambda i: abs(float(flat[i])), reverse=True)[:8]
        tensor_stats[name] = {
            "nonzero_parameters": int(nonzero_indices.size),
            "minimum": float(array.min()) if array.size else 0.0,
            "maximum": float(array.max()) if array.size else 0.0,
            "mean_absolute_value": deterministic_mean([abs(float(value)) for value in flat]) if array.size else 0.0,
            "largest_absolute_entries": [
                {"flat_index": int(i), "value": float(flat[i])} for i in largest
            ],
        }
    return {
        "source_path": path_text,
        "source_sha256": archive_sha,
        "download_file": f"weights/{condition_id}.npz" if path_text else None,
        "total_parameters": total,
        "nonzero_parameters": nonzero,
        "sparsity_percent": (100 * (total - nonzero) / total),
        "tensor_stats": tensor_stats,
        "tensors": encoded,
    }


def verify_argon_artifact(relative_path: str, records: dict[str, dict[str, Any]]) -> bytes:
    """Read one experiment file only after checking its size and registered hash."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid Argon experiment artifact path: {relative_path}")
    record = records.get(relative.as_posix())
    if record is None:
        raise ValueError(f"Argon artifact has no registry entry: {relative_path}")
    raw = (ARGON_DIR / relative).read_bytes()
    if sha256_bytes(raw) != record.get("sha256") or len(raw) != record.get("size_bytes"):
        raise ValueError(f"Argon artifact registry mismatch: {relative_path}")
    return raw


def _assert_metric_match(label: str, observed: Any, expected: Any) -> None:
    if isinstance(observed, (float, int)) and isinstance(expected, (float, int)):
        if not math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Argon raw CSV and analysis disagree for {label}: {observed} != {expected}")
    elif observed != expected:
        raise ValueError(f"Argon raw CSV and analysis disagree for {label}: {observed} != {expected}")


def build_iterative_development() -> tuple[dict[str, Any], dict[Path, bytes]]:
    """Build the distinct Argon High iterative-development view from preserved sources."""
    artifact_registry_path = ARGON_DIR / "artifact_registry.json"
    artifact_registry_raw = artifact_registry_path.read_bytes()
    artifact_registry = json.loads(artifact_registry_raw)
    artifact_records_list = artifact_registry.get("files", [])
    artifact_records = {item["path"]: item for item in artifact_records_list}
    if len(artifact_records) != len(artifact_records_list) or len(artifact_records) != artifact_registry.get("file_count"):
        raise ValueError("Argon artifact registry has duplicate paths or a stale file count")
    for relative in artifact_records:
        verify_argon_artifact(relative, artifact_records)

    website_records_path = "website_model_records.json"
    website_records_raw = verify_argon_artifact(website_records_path, artifact_records)
    website_records = json.loads(website_records_raw)
    registry_raw = verify_argon_artifact("model_registry.json", artifact_records)
    model_registry = json.loads(registry_raw)
    manifest_raw = verify_argon_artifact("evaluation/full_run/run_manifest.json", artifact_records)
    run_manifest = json.loads(manifest_raw)
    analysis_raw = verify_argon_artifact("evaluation/full_run/analysis/analysis.json", artifact_records)
    analysis = json.loads(analysis_raw)
    pairwise_raw = verify_argon_artifact("evaluation/full_run/analysis/paired_comparisons.json", artifact_records)
    pairwise_file = json.loads(pairwise_raw)

    if website_records.get("group_id") != "gemini_argon_high_iterative_development" or website_records.get("separate_from_primary_zero_shot_ranking") is not True:
        raise ValueError("Argon iterative-development classification is missing or invalid")
    if model_registry.get("not_part_of_primary_zero_shot_ranking") is not True:
        raise ValueError("Argon model registry does not keep the iterative group out of the primary ranking")
    if run_manifest.get("status") != "analysis_complete" or run_manifest.get("run_kind") != "full":
        raise ValueError("Refusing to export an incomplete Argon iterative-development evaluation")
    if run_manifest.get("episodes_per_condition") != 3000 or run_manifest.get("episodes_total") != 9000:
        raise ValueError("Argon full-run episode counts do not match the declared three-by-3,000 design")

    records_by_id = {item["id"]: item for item in website_records["models"]}
    registry_by_id = {item["id"]: item for item in model_registry["conditions"]}
    expected_order = ["argon_high_original", "argon_high_followup_1", "argon_high_followup_2"]
    if list(records_by_id) != expected_order or set(records_by_id) != set(registry_by_id):
        raise ValueError("Argon iterative-development stages are missing, reordered, or merged")
    csv_hashes = run_manifest.get("condition_csv_sha256", {})
    seed_signatures: dict[str, list[tuple[str, str, str, str]]] = {}
    csv_outputs: dict[Path, bytes] = {}
    model_outputs: list[dict[str, Any]] = []
    maximum_score = 0.0
    score_values_by_id: dict[str, list[float]] = {}
    verified_rows: dict[str, list[dict[str, str]]] = {}

    for model_id in expected_order:
        record = records_by_id[model_id]
        registry_record = registry_by_id[model_id]
        source = record.get("source", {})
        for source_key, registry_key in (("path", "source_file"), ("weight_path", "weight_path")):
            if source.get(source_key) != registry_record.get(registry_key):
                raise ValueError(f"Argon source mapping differs between registries for {model_id}: {source_key}")
        if source.get("sha256") != registry_record.get("source_sha256") or source.get("weight_sha256") != registry_record.get("weight_sha256"):
            raise ValueError(f"Argon source/weight hashes differ between registries for {model_id}")
        source_raw = verify_argon_artifact(source["path"], artifact_records)
        weight_raw = verify_argon_artifact(source["weight_path"], artifact_records)
        if sha256_bytes(source_raw) != source["sha256"] or sha256_bytes(weight_raw) != source["weight_sha256"]:
            raise ValueError(f"Argon source or weight file hash mismatch for {model_id}")
        encoded_tensors: dict[str, Any] = {}
        tensor_nonzero = 0
        tensor_total = 0
        with np.load(ARGON_DIR / source["weight_path"], allow_pickle=False) as archive:
            if set(archive.files) != set(ARCHIVE_KEY_MAP):
                raise ValueError(f"Unexpected Argon weight tensor set for {model_id}: {archive.files}")
            for archive_key, tensor_name in ARCHIVE_KEY_MAP.items():
                array = np.asarray(archive[archive_key], dtype="<f4", order="C")
                if tuple(array.shape) != TENSOR_SHAPES[tensor_name] or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {tensor_name} tensor in {source['weight_path']}")
                tensor_nonzero += int(np.count_nonzero(array))
                tensor_total += int(array.size)
                encoded_tensors[tensor_name] = {
                    "shape": list(TENSOR_SHAPES[tensor_name]),
                    "dtype": "float32-le",
                    "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
                }
        if tensor_total != record["parameter_count"] or tensor_nonzero != record["nonzero_parameter_count"]:
            raise ValueError(f"Argon weight parameter counts disagree with the registry for {model_id}")

        csv_relative = record["benchmark_csv"]
        csv_raw = verify_argon_artifact(csv_relative, artifact_records)
        if sha256_bytes(csv_raw) != csv_hashes.get(model_id):
            raise ValueError(f"Argon run manifest CSV hash mismatch for {model_id}")
        rows = list(csv.DictReader(csv_raw.decode("utf-8-sig").splitlines()))
        if len(rows) != 3000:
            raise ValueError(f"Expected 3,000 Argon episodes for {model_id}; found {len(rows)}")
        required = {"seed_group", "episode_index", "base_seed", "environment_seed", "policy_seed", "score", "length", "ball_lost", "step_cutoff"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Argon episode CSV schema is incomplete for {model_id}")
        signatures = sorted((row["seed_group"], row["episode_index"], row["base_seed"], row["environment_seed"], row["policy_seed"]) for row in rows)
        seed_signatures[model_id] = signatures
        scores = [float(row["score"]) for row in rows]
        lengths = [int(row["length"]) for row in rows]
        if not all(math.isfinite(value) for value in scores) or any(length < 1 or length > 10000 for length in lengths):
            raise ValueError(f"Invalid Argon score or episode length for {model_id}")
        metric = record["metrics"]
        n = len(scores)
        raw_summary = {
            "episodes": n,
            "mean_score": deterministic_mean(scores),
            "standard_error_score": deterministic_sample_std(scores) / math.sqrt(n),
            "median_score": statistics.median(scores),
            "minimum_score": min(scores),
            "maximum_score": max(scores),
            "mean_episode_length": deterministic_mean(lengths),
            "median_episode_length": statistics.median(lengths),
            "maximum_episode_length": max(lengths),
            "cutoff_count": sum(bool_value(row["step_cutoff"]) for row in rows),
            "ball_lost_count": sum(bool_value(row["ball_lost"]) for row in rows),
        }
        source_analysis = analysis["raw_csv_independent_recomputation"][model_id]
        official_analysis = analysis["conditions"][model_id]
        for key, observed in raw_summary.items():
            _assert_metric_match(f"{model_id}.{key}", observed, metric[key])
            if key in source_analysis:
                _assert_metric_match(f"{model_id}.raw.{key}", observed, source_analysis[key])
            if key in official_analysis:
                _assert_metric_match(f"{model_id}.official.{key}", observed, official_analysis[key])
        if raw_summary["cutoff_count"] != 0 or raw_summary["ball_lost_count"] != n:
            raise ValueError(f"Argon termination outcomes differ from the documented all-ball-loss result for {model_id}")

        score_counts: dict[str, int] = {}
        for score in scores:
            key = str(float(score))
            score_counts[key] = score_counts.get(key, 0) + 1
        if score_counts != source_analysis["exact_score_counts"] or score_counts != metric["exact_score_counts"]:
            raise ValueError(f"Argon exact score histogram mismatch for {model_id}")
        maximum_score = max(maximum_score, max(scores))
        score_values_by_id[model_id] = scores
        verified_rows[model_id] = rows
        csv_destination = f"iterative-development/episodes/{model_id}.csv"
        csv_outputs[SITE_DATA / csv_destination] = csv_raw

        model_outputs.append({
            "id": model_id,
            "name": record["name"],
            "stage": record["stage"],
            "condition_class": record["condition_class"],
            "attribution": record["attribution"],
            "provider_identity_status": record["provider_identity_status"],
            "generation_provenance_limit": record["generation_provenance_limit"],
            "architecture": record["architecture"],
            "parameter_count": record["parameter_count"],
            "nonzero_parameter_count": record["nonzero_parameter_count"],
            "sparsity_percent": record["sparsity_percent"],
            "source": {
                "path": source["path"], "sha256": source["sha256"],
                "download_file": f"iterative-development/source/{Path(source['path']).name}",
            },
            "weights": {
                "path": source["weight_path"], "sha256": source["weight_sha256"],
                "download_file": f"iterative-development/weights/{Path(source['weight_path']).name}",
                "total_parameters": tensor_total,
                "nonzero_parameters": tensor_nonzero,
                "sparsity_percent": 100.0 * (tensor_total - tensor_nonzero) / tensor_total,
                "tensors": encoded_tensors,
            },
            "benchmark_csv": csv_relative,
            "csv_sha256": sha256_bytes(csv_raw),
            "csv_download": csv_destination,
            "metrics": raw_summary,
        })
        csv_outputs[SITE_DATA / model_outputs[-1]["source"]["download_file"]] = source_raw
        csv_outputs[SITE_DATA / model_outputs[-1]["weights"]["download_file"]] = weight_raw

    if not seed_signatures or len({tuple(value) for value in seed_signatures.values()}) != 1:
        raise ValueError("Argon models do not share the exact matched seed identifiers")

    comparisons = analysis["paired_comparisons"]
    if comparisons != pairwise_file or comparisons != website_records["paired_comparisons"]:
        raise ValueError("Argon paired-comparison records disagree across the analysis files")
    if len(comparisons) != 3 or any(item.get("matched_episode_pairs") != 3000 for item in comparisons):
        raise ValueError("Argon paired-comparison family is incomplete")

    histogram_bins = 20
    histogram_max = max(1.0, maximum_score)
    for model in model_outputs:
        scores = score_values_by_id[model["id"]]
        counts = [0] * histogram_bins
        for score in scores:
            index = min(histogram_bins - 1, int(score / histogram_max * histogram_bins))
            counts[index] += 1
        model["score_histogram"] = [
            {"bin": index, "lower": index * histogram_max / histogram_bins,
             "upper": (index + 1) * histogram_max / histogram_bins, "count": count}
            for index, count in enumerate(counts)
        ]

    report_paths = {
        "report": "report.md",
        "analysis_report": "evaluation/full_run/analysis/analysis_report.md",
        "analysis_json": "evaluation/full_run/analysis/analysis.json",
        "episode_seed_schedule": "evaluation/full_run/episode_seeds.csv",
    }
    download_records = {}
    for name, source_path in report_paths.items():
        raw = verify_argon_artifact(source_path, artifact_records)
        destination = f"iterative-development/{'reports' if name in {'report', 'analysis_report'} else 'data'}/{Path(source_path).name}"
        csv_outputs[SITE_DATA / destination] = raw
        download_records[name] = {
            "source_path": source_path, "download_file": destination,
            "sha256": sha256_bytes(raw), "size_bytes": len(raw),
        }

    payload = {
        "schema_version": 1,
        "group_id": website_records["group_id"],
        "title": website_records["title"],
        "classification": "Iterative-development experiment; excluded from the primary zero-shot ranking.",
        "separate_from_primary_zero_shot_ranking": True,
        "evaluation_status": website_records["evaluation_status"],
        "evaluation_dataset": website_records["evaluation_dataset"],
        "evaluation_protocol": website_records["evaluation_protocol"],
        "development_summary": website_records["development_summary"],
        "statistical_interpretation": website_records["statistical_interpretation"],
        "researcher_contribution": website_records["researcher_contribution"],
        "recorded_ai_contribution": website_records["recorded_ai_contribution"],
        "limitations": website_records["limitations"],
        "models": model_outputs,
        "paired_comparisons": comparisons,
        "downloads": download_records,
        "artifact_registry": {
            "path": "experiments/argon_high_iterative_development/artifact_registry.json",
            "sha256": sha256_bytes(artifact_registry_raw),
            "registered_file_count": len(artifact_records),
            "source_commit": artifact_registry.get("source_commit"),
        },
        "seed_matching_verified": True,
    }
    payload_bytes = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    csv_outputs[SITE_DATA / "iterative_development.json"] = payload_bytes
    return payload, csv_outputs


def build_outputs() -> dict[Path, bytes]:
    config = read_json(DATA_DIR / "evaluation_config.json")
    registry = read_json(DATA_DIR / "condition_registry.json")
    manifest = read_json(RUN_DIR / "run_manifest.json")
    metrics_path = RUN_DIR / "analysis/condition_metrics.json"
    metrics = read_json(metrics_path)
    comparisons_path = RUN_DIR / "analysis/paired_comparisons.json"
    comparisons = read_json(comparisons_path)
    historical_path = RUN_DIR / "analysis/corrected_vs_original.json"
    historical = read_json(historical_path)
    if manifest.get("status") != "complete" or manifest.get("run_kind") != "full":
        raise ValueError("Refusing to export an incomplete or non-full corrected run")
    for rel, key in [
        ("analysis/condition_metrics.json", "condition_metrics.json"),
        ("analysis/paired_comparisons.json", "paired_comparisons.json"),
        ("analysis/corrected_vs_original.json", "corrected_vs_original.json"),
    ]:
        actual = sha256_bytes((RUN_DIR / rel).read_bytes())
        if actual != manifest["analysis_artifact_sha256"][key]:
            raise ValueError(f"Analysis artifact hash mismatch: {rel}")

    conditions_by_id = {item["id"]: item for item in manifest["conditions"]}
    registry_by_id = {item["id"]: item for item in registry["conditions"]}
    if set(conditions_by_id) != set(registry_by_id):
        raise ValueError("Run manifest and condition registry contain different conditions")
    if "deepseek_corrected" in conditions_by_id:
        raise ValueError("Researcher intervention unexpectedly entered the primary dataset")

    preserved_registry_path = DATA_DIR / "preserved_artifacts_sha256.json"
    preserved_registry = read_json(preserved_registry_path)
    preserved_records = {item["path"]: item for item in preserved_registry["files"]}
    verify_preserved_file(PROMPT_PATH, preserved_records)

    condition_records = []
    csv_files: dict[Path, bytes] = {}
    for condition_id, condition in conditions_by_id.items():
        expected_csv_hash = manifest["condition_csv_sha256"][condition_id]
        rows, raw_csv = read_verified_csv(condition_id, expected_csv_hash)
        raw_metrics = metrics["conditions"][condition_id]
        summary = summarize_rows(rows, raw_metrics)
        csv_files[SITE_DATA / "episodes" / f"{condition_id}.csv"] = raw_csv
        class_name = condition["class"]
        condition_records.append({
            "id": condition_id,
            "label": condition["label"],
            "condition_class": class_name,
            "is_primary_zero_shot": class_name == "primary_zero_shot",
            "weight_sha256": condition.get("weight_sha256"),
            "source_weight_path": condition.get("weight_path"),
            "metrics": summary,
            "csv_download": f"episodes/{condition_id}.csv",
        })
    rank = {condition_id: index for index, condition_id in enumerate(PRIMARY_ORDER)}
    condition_records.sort(key=lambda c: (not c["is_primary_zero_shot"], rank.get(c["id"], 99), -c["metrics"]["mean_score"]))

    research = {
        "schema_version": 1,
        "project": "Weight-Space Programming — Experiment 5 / MinAtar Breakout",
        "dataset_id": manifest["dataset"],
        "dataset_label": "Phase 2 corrected local reevaluation",
        "protocol": {
            "environment": config["environment"],
            "policy": config["policy"],
            "seed_groups": config["seed_groups"],
            "episodes_per_seed_group": config["episodes_per_seed_group"],
            "rng_protocol": config["episode_rng_derivation"],
            "matching": config["matching"],
            "source_commit": manifest["source_commit"],
            "analysis_source_commit": manifest["analysis_source_commit"],
        },
        "conditions": condition_records,
        "pairwise_analysis": comparisons["pairwise_comparisons"],
        "pairwise_analysis_method": comparisons["analysis_method"],
        "corrected_vs_original": historical,
        "provenance": {
            "manifest_sha256": sha256_bytes((RUN_DIR / "run_manifest.json").read_bytes()),
            "preserved_artifact_registry_sha256": sha256_bytes(preserved_registry_path.read_bytes()),
            "condition_registry_sha256": sha256_bytes((DATA_DIR / "condition_registry.json").read_bytes()),
            "evaluation_config_sha256": sha256_bytes((DATA_DIR / "evaluation_config.json").read_bytes()),
            "condition_metrics_sha256": sha256_bytes(metrics_path.read_bytes()),
            "paired_comparisons_sha256": sha256_bytes(comparisons_path.read_bytes()),
            "historical_comparison_sha256": sha256_bytes(historical_path.read_bytes()),
        },
        "audit": {
            "verdict": "Verified with limitations",
            "date": "2026-11-06",
            "verified": [
                "Corrected raw episode summaries were independently recomputed from CSV data.",
                "Preserved artifact hashes and selected replay traces were checked.",
                "Corrected environment and policy streams use separate RNG instances.",
            ],
            "limitations": [
                "Provider identities and some generation settings are recorded metadata, not independently authenticated.",
                "The corrected 3,000-episode evaluation is a local reevaluation, not the paper's original 1,000-episode sample.",
                "Six GUI tests failed and three errored at Tk startup; interactive desktop rendering was not verified.",
                "One generation per model and one MinAtar environment do not establish general capability.",
                "The browser arena uses a separate demonstration RNG and is not benchmark-equivalent.",
            ],
        },
    }
    research_bytes = (json.dumps(research, ensure_ascii=False, separators=(",", ":")) + "\n").encode()

    models = []
    for condition in condition_records:
        condition_id = condition["id"]
        generation = metadata_for(condition_id, preserved_records)
        model = {
            "id": condition_id,
            "label": condition["label"],
            "condition_class": condition["condition_class"],
            "weight_sha256": condition["weight_sha256"],
            "generation": generation,
            "metrics": {key: value for key, value in condition["metrics"].items() if key != "episode_records"},
            "weights": model_tensors(condition_id, condition["weight_sha256"], preserved_records),
        }
        models.append(model)
        if model["weights"] and model["weights"].get("source_path"):
            weight_path = ROOT / model["weights"]["source_path"]
            weight_raw = weight_path.read_bytes()
            if sha256_bytes(weight_raw) != model["weights"]["source_sha256"]:
                raise ValueError(f"Weight archive changed while exporting {condition_id}")
            csv_files[SITE_DATA / model["weights"]["download_file"]] = weight_raw
        response_path = RESPONSE_PATHS.get(condition_id)
        if response_path:
            response_raw = (ROOT / response_path).read_bytes()
            if sha256_bytes(response_raw) != generation["response_sha256"]:
                raise ValueError(f"Response hash changed while exporting {condition_id}")
            csv_files[SITE_DATA / generation["response_file"]] = response_raw

    intervention_manifest_path = ROOT / "experiments/controlled_interventions/deepseek_corrected/intervention_manifest.json"
    intervention = read_json(intervention_manifest_path)
    intervention_record = {
        "id": "deepseek_corrected",
        "label": "DeepSeek Corrected — researcher intervention",
        "condition_class": "researcher_intervention",
        "classification": intervention.get("condition_type"),
        "base_condition_id": intervention.get("base_condition_id"),
        "weight_sha256": intervention.get("output_artifacts", {}).get("weights_sha256"),
        "manifest_sha256": sha256_bytes(intervention_manifest_path.read_bytes()),
        "parameter_metrics": intervention.get("parameter_metrics"),
        "included_in_primary_results": False,
    }
    models_data = {
        "schema_version": 1,
        "architecture": {"layers": [600, 32, 32, 6], "hidden_activation": "tanh", "output": "linear logits", "total_parameters": 20486},
        "primary_zero_shot_order": PRIMARY_ORDER,
        "models": models,
        "separate_intervention": intervention_record,
        "prompt": {"file": "experiment_5_prompt.txt", "sha256": sha256_bytes((ROOT / PROMPT_PATH).read_bytes())},
        "provider_attestation_available": False,
        "provider_attestation_record": read_json(DATA_DIR / "phase2_artifact_registry.json").get("provider_attestation"),
    }
    prompt_raw = (ROOT / PROMPT_PATH).read_bytes()
    output: dict[Path, bytes] = {
        SITE_DATA / "research.json": research_bytes,
        SITE_DATA / "models.json": (json.dumps(models_data, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        SITE_DATA / "experiment_5_prompt.txt": prompt_raw,
    }
    output.update(csv_files)
    _, iterative_files = build_iterative_development()
    output.update(iterative_files)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated site data without writing files")
    args = parser.parse_args()
    outputs = build_outputs()
    failures = []
    for path, content in outputs.items():
        relative = path.relative_to(ROOT)
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                failures.append(str(relative))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            print(f"wrote {relative} ({len(content)} bytes)")
    if failures:
        print("Generated site data is stale or missing:", file=sys.stderr)
        print("\n".join(f"  {item}" for item in failures), file=sys.stderr)
        return 1
    if args.check:
        print(f"Verified {len(outputs)} hash-bound site data files.")
    else:
        print(f"Exported {len(outputs)} hash-bound site data files under docs/data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
