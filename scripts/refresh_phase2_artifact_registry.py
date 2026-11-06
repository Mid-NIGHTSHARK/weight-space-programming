#!/usr/bin/env python3
"""Refresh the additive Phase 2 artifact and provenance registry."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/corrected_evaluation/phase2_artifact_registry.json"
PRESERVED = ROOT / "experiments/corrected_evaluation/preserved_artifacts_sha256.json"
CONDITIONS = ROOT / "experiments/corrected_evaluation/condition_registry.json"
REPORT = ROOT / "PHASE_2_CORRECTED_EVALUATION_REPORT.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: str) -> dict:
    full_path = ROOT / path
    if not full_path.is_file():
        raise FileNotFoundError(f"Required Phase 2 provenance artifact is missing: {full_path}")
    return {
        "path": path,
        "sha256": digest(full_path),
        "size_bytes": full_path.stat().st_size,
    }


def main() -> None:
    conditions = json.loads(CONDITIONS.read_text())["conditions"]
    preserved = json.loads(PRESERVED.read_text())
    mismatches = []
    for item in preserved["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or digest(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            mismatches.append(item["path"])
    if mismatches:
        raise RuntimeError(f"Protected baseline artifacts changed: {mismatches}")

    model_sources = {
        "gemini_argon_4_medium": [
            "experiments/strict_replication/generation_001/input/experiment_5_prompt.txt",
            "experiments/strict_replication/generation_001/output/raw_response.txt",
            "experiments/strict_replication/generation_001/output/argon_response_original.docx",
            "experiments/strict_replication/generation_001/output/generation_metadata.json",
        ],
        "gpt6_original": [
            "experiments/strict_replication/generation_002_gpt6/input/experiment_5_prompt.txt",
            "experiments/strict_replication/generation_002_gpt6/output/gpt6_raw_response.txt",
            "experiments/strict_replication/generation_002_gpt6/output/gpt6_raw_response.txt.rtf",
            "experiments/strict_replication/generation_002_gpt6/output/generation_metadata.json",
            "experiments/strict_replication/generation_002_gpt6/manifest.json",
        ],
        "claude_sonnet_55": [
            "experiments/strict_replication/generation_003_sonnet55/input/experiment_5_prompt.txt",
            "experiments/strict_replication/generation_003_sonnet55/input/sonnet55_raw_response.txt",
            "experiments/strict_replication/generation_003_sonnet55/output/generation_metadata.json",
            "experiments/strict_replication/generation_003_sonnet55/manifest.json",
        ],
        "deepseek_original": [
            "experiments/strict_replication/generation_004_deepseek/input/experiment_5_prompt.txt",
            "experiments/strict_replication/generation_004_deepseek/input/deepseek_raw_response.txt",
            "experiments/strict_replication/generation_004_deepseek/output/generation_metadata.json",
            "experiments/strict_replication/generation_004_deepseek/manifest.json",
        ],
        "gpt6_luna_extra_high": [
            "experiments/strict_replication/generation_005_gpt6_luna_extra_high/input/experiment_5_prompt.txt",
            "experiments/strict_replication/generation_005_gpt6_luna_extra_high/input/gpt6_luna_raw_response.txt",
            "experiments/strict_replication/generation_005_gpt6_luna_extra_high/manifest.json",
        ],
    }

    model_records = []
    for condition in conditions:
        condition_id = condition["id"]
        record = {
            "condition_id": condition_id,
            "label": condition["label"],
            "class": condition["class"],
            "weight": artifact(condition["weight_path"]) if condition.get("weight_path") else None,
        }
        if condition_id in model_sources:
            record["source_artifacts"] = [artifact(path) for path in model_sources[condition_id]]
            record["provider_provenance"] = {
                "status": "locally recorded; not independently provider-attested",
                "model_and_reasoning_labels_are_claims_in_local_metadata": True,
            }
        model_records.append(record)

    corrected_files = []
    for path in sorted((ROOT / "experiments/corrected_evaluation").rglob("*")):
        if path.is_file() and path != REGISTRY:
            corrected_files.append(artifact(path.relative_to(ROOT).as_posix()))

    runtime = {}
    for package in ("MinAtar", "numpy", "pandas", "scipy", "torch", "pytest", "matplotlib"):
        runtime[package] = importlib.metadata.version(package)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sonnet_raw_path = ROOT / model_sources["claude_sonnet_55"][1]
    sonnet_manifest = json.loads((ROOT / "experiments/strict_replication/generation_003_sonnet55/manifest.json").read_text())
    manifest_sonnet_bytes = sonnet_manifest["files"]["raw_response_txt"]["size_bytes"]

    registry = {
        "schema_version": 1,
        "registry_type": "additive_phase2_provenance_snapshot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_generation": git_head,
        "baseline_commit": preserved["baseline_commit"],
        "evaluation_source_commit": "b8be6dc0134b80d6fefb69005ff364f799d1d793",
        "analysis_source_commit": "3fa7ba5c786454f6bd71f375e7bf952edd9870a6",
        "historical_artifact_hash_registry": artifact("research/forensic_audit/artifact_hashes.json"),
        "protected_artifact_registry": {
            "path": PRESERVED.relative_to(ROOT).as_posix(),
            "sha256": digest(PRESERVED),
            "protected_file_count": len(preserved["files"]),
        },
        "runtime_versions": runtime,
        "condition_registry_sha256": digest(CONDITIONS),
        "evaluation_config_sha256": digest(ROOT / "experiments/corrected_evaluation/evaluation_config.json"),
        "sonnet_raw_response_size_reconciliation": {
            "path": model_sources["claude_sonnet_55"][1],
            "observed_size_bytes": sonnet_raw_path.stat().st_size,
            "observed_sha256": digest(sonnet_raw_path),
            "original_manifest_declared_size_bytes": manifest_sonnet_bytes,
            "resolution": "preserve original manifest; record observed size and digest here",
        },
        "model_artifacts": model_records,
        "corrected_evaluation_artifacts": corrected_files,
        "phase2_report": artifact(REPORT.relative_to(ROOT).as_posix()),
        "phase2_code_artifacts": [
            artifact(path)
            for path in [
                "environment/actions.py",
                "environment/breakout.py",
                "environment/observations.py",
                "network/architecture.py",
                "network/inference.py",
                "evaluation/rng_streams.py",
                "evaluation/corrected_evaluate.py",
                "evaluation/corrected_analysis.py",
                "evaluation/evaluate.py",
                "evaluation/legacy_trace_replay.py",
                "scripts/run_corrected_evaluation.py",
                "scripts/analyze_corrected_evaluation.py",
                "scripts/replay_original_failure_traces.py",
                "scripts/refresh_phase2_artifact_registry.py",
                "tests/test_corrected_rng.py",
                "tests/test_corrected_analysis.py",
                "tests/test_phase2_preservation.py",
                "tests/test_legacy_trace_replay.py",
                "tests/test_visualizer_data_integrity.py",
                "tests/test_deepseek_correction.py",
                "visualize_comparison.py",
            ]
        ],
        "provider_attestation": "No provider-side evidence was available during Phase 2; all provider/model/version/reasoning labels remain locally recorded claims.",
    }
    REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {REGISTRY.relative_to(ROOT)} with {len(corrected_files)} corrected evaluation artifacts and {len(preserved['files'])} protected baseline entries.")


if __name__ == "__main__":
    main()
