#!/usr/bin/env python3
"""Evaluate the three Argon High versions with the existing corrected evaluator."""

import argparse
import csv
import datetime as dt
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.corrected_evaluate import (  # noqa: E402
    CorrectedPolicyEvaluator,
    corrected_run_metadata,
)
from evaluation.rng_streams import derive_episode_stream_seeds  # noqa: E402
from evaluation.seeds import generate_evaluation_seeds  # noqa: E402
from network.architecture import MinAtarNetworkNumPy  # noqa: E402
from network.inference import PolicyAgent  # noqa: E402
from network.weight_loader import compute_file_sha256, load_weights_npz  # noqa: E402


REGISTRY_PATH = HERE / "model_registry.json"
CONFIG_PATH = HERE / "evaluation/evaluation_config.json"
RECONSTRUCTION_PATH = HERE / "reconstruction_manifest.json"
CSV_FIELDS = [
    "seed_group", "episode_index", "base_seed", "environment_seed", "policy_seed",
    "score", "length", "bricks_destroyed", "ball_lost", "step_cutoff",
    "trajectory_sha256",
] + [f"action_{i}_{kind}_count" for i in range(6) for kind in ("selected", "executed")]


def runtime_versions():
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "minatar": version("minatar"),
        "scipy": version("scipy"),
        "torch": version("torch"),
        "pandas": version("pandas"),
    }


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def build_schedule(config, pilot):
    if pilot:
        counts = {int(group): int(count) for group, count in config["pilot_episodes_by_seed_group"].items()}
        expected_total = int(config["pilot_episodes_per_condition"])
    else:
        counts = {int(group): int(config["episodes_per_seed_group"]) for group in config["seed_groups"]}
        expected_total = sum(counts.values())
    schedule = []
    for group, count in counts.items():
        for index, base_seed in enumerate(generate_evaluation_seeds(count, group)):
            streams = derive_episode_stream_seeds(base_seed)
            schedule.append({
                "seed_group": group,
                "episode_index": index,
                "base_seed": int(base_seed),
                "environment_seed": streams.environment_seed,
                "policy_seed": streams.policy_seed,
            })
    if len(schedule) != expected_total:
        raise ValueError(f"Schedule has {len(schedule)} episodes, expected {expected_total}")
    return schedule


def load_conditions(registry):
    reconstruction = json.loads(RECONSTRUCTION_PATH.read_text())
    conditions = []
    for record in registry["conditions"]:
        weight_path = HERE / record["weight_path"]
        actual_hash = compute_file_sha256(str(weight_path))
        if actual_hash != record["weight_sha256"]:
            raise ValueError(f"Weight hash mismatch for {record['id']}: {actual_hash}")
        reconstruction_record = reconstruction["versions"][record["id"]]
        if reconstruction_record["weight_sha256"] != actual_hash:
            raise ValueError(f"Reconstruction registry mismatch for {record['id']}")
        agent = PolicyAgent(MinAtarNetworkNumPy(load_weights_npz(str(weight_path))))

        def action_fn(obs, rng, active_agent=agent):
            active_agent.rng = rng
            return active_agent.get_action(obs)[0]

        conditions.append({**record, "weight_sha256": actual_hash, "action_fn": action_fn})
    return conditions


def run(args):
    output_dir = (HERE / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation directory: {output_dir}")
    config = json.loads(CONFIG_PATH.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    schedule = build_schedule(config, args.pilot)
    conditions = load_conditions(registry)
    run_kind = "pilot" if args.pilot else "full"
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "group_id": registry["group_id"],
        "dataset": f"argon_high_iterative_{run_kind}",
        "run_kind": run_kind,
        "status": "running",
        "pilot_is_software_validation_only": bool(args.pilot),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "episodes_per_condition": len(schedule),
        "episodes_total": len(schedule) * len(conditions),
        "seed_groups": sorted({row["seed_group"] for row in schedule}),
        "evaluation_config": {**config, "rng_implementation": corrected_run_metadata()},
        "runtime_versions": runtime_versions(),
        "evaluation_config_sha256": sha256_bytes(CONFIG_PATH.read_bytes()),
        "condition_registry_sha256": sha256_bytes(REGISTRY_PATH.read_bytes()),
        "conditions": [
            {key: value for key, value in record.items() if key != "action_fn"}
            for record in conditions
        ],
        "failure_policy": "failed output is retained; no overwrite or implicit resume",
        "replay_checks": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        with (output_dir / "episode_seeds.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(schedule[0]))
            writer.writeheader()
            writer.writerows(schedule)
        for condition in conditions:
            condition_dir = output_dir / "conditions" / condition["id"]
            condition_dir.mkdir(parents=True)
            csv_path = condition_dir / "episode_scores.csv"
            replay_rows = []
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row_number, seed_row in enumerate(schedule):
                    evaluator = CorrectedPolicyEvaluator(
                        seed_group=seed_row["seed_group"],
                        base_seeds=[seed_row["base_seed"]],
                    )
                    result = evaluator.evaluate_episode(
                        condition["action_fn"], seed_row["base_seed"], seed_row["episode_index"]
                    )
                    record = result.to_record()
                    writer.writerow(record)
                    stream.flush()
                    if args.pilot and seed_row["episode_index"] == 0:
                        replay = evaluator.replay_record(condition["action_fn"], record)
                        if replay.trajectory_sha256 != result.trajectory_sha256:
                            raise RuntimeError(
                                f"Replay mismatch for {condition['id']} group {seed_row['seed_group']}"
                            )
                        replay_rows.append({
                            "seed_group": seed_row["seed_group"],
                            "base_seed": seed_row["base_seed"],
                            "trajectory_sha256": replay.trajectory_sha256,
                            "verified": True,
                        })
                    if args.pilot or (row_number + 1) % 100 == 0:
                        print(
                            f"{condition['id']}: {row_number + 1}/{len(schedule)} episodes",
                            flush=True,
                        )
            manifest["replay_checks"].extend(
                {"condition_id": condition["id"], **item} for item in replay_rows
            )
            manifest.setdefault("condition_csv_sha256", {})[condition["id"]] = sha256_bytes(
                csv_path.read_bytes()
            )
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest["status"] = "pilot_complete" if args.pilot else "evaluation_complete_analysis_pending"
        manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Completed {manifest['episodes_total']} episodes; status={manifest['status']}")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (output_dir / "failure_traceback.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the configured 10-episode software pilot")
    parser.add_argument("--output-dir", required=True, help="New path relative to this experiment directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
