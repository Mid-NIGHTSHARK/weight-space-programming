#!/usr/bin/env python3
"""Run the isolated corrected MinAtar evaluation pilot or full benchmark."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from importlib.metadata import version

from baselines.random_agent import RandomAgent
from evaluation.corrected_evaluate import CorrectedPolicyEvaluator, corrected_run_metadata
from evaluation.rng_streams import derive_episode_stream_seeds
from evaluation.seeds import generate_evaluation_seeds
from network.architecture import MinAtarNetworkNumPy, create_zero_weights
from network.inference import PolicyAgent
from network.weight_loader import compute_file_sha256, load_weights_npz


REGISTRY_PATH = ROOT / "experiments/corrected_evaluation/condition_registry.json"
CONFIG_PATH = ROOT / "experiments/corrected_evaluation/evaluation_config.json"


def load_evaluation_config():
    with CONFIG_PATH.open() as fh:
        return json.load(fh)


def load_conditions():
    with REGISTRY_PATH.open() as fh:
        registry = json.load(fh)
    conditions = []
    for item in registry["conditions"]:
        condition = dict(item)
        if condition["weight_path"]:
            full_path = ROOT / condition["weight_path"]
            if not full_path.is_file():
                raise FileNotFoundError(f"Required condition weights missing: {full_path}")
            actual = compute_file_sha256(str(full_path))
            if actual != condition["weight_sha256"]:
                raise ValueError(
                    f"Weight hash mismatch for {condition['id']}: expected "
                    f"{condition['weight_sha256']}, observed {actual}"
                )
            weights = load_weights_npz(str(full_path))
            network = MinAtarNetworkNumPy(weights)
            agent = PolicyAgent(network)

            def action_fn(obs, rng, agent=agent):
                agent.rng = rng
                return agent.get_action(obs)[0]

            condition["action_fn"] = action_fn
            condition["weight_sha256"] = actual
        elif condition["id"] == "random_action_control":
            condition["action_fn"] = RandomAgent()
        elif condition["id"] == "zero_weight_control":
            zero_agent = PolicyAgent(MinAtarNetworkNumPy(create_zero_weights()))

            def action_fn(obs, rng, agent=zero_agent):
                agent.rng = rng
                return agent.get_action(obs)[0]

            condition["action_fn"] = action_fn
            condition["weight_sha256"] = "in-memory-zero-tensors"
        else:
            raise ValueError(f"Unrecognized control condition: {condition['id']}")
        conditions.append(condition)
    return conditions


def build_schedule(kind: str, config: dict):
    if kind == "pilot":
        counts = {int(group): int(count) for group, count in config["pilot_episodes_by_seed_group"].items()}
        if sum(counts.values()) != int(config["pilot_episodes_per_condition"]):
            raise ValueError("Pilot seed-group allocation does not match configured episode count")
    else:
        counts = {int(group): int(config["episodes_per_seed_group"]) for group in config["seed_groups"]}
    schedule = []
    for seed_group, count in counts.items():
        base_seeds = generate_evaluation_seeds(count, seed_group)
        for episode_index, base_seed in enumerate(base_seeds):
            streams = derive_episode_stream_seeds(base_seed)
            schedule.append({
                "seed_group": seed_group,
                "episode_index": episode_index,
                "base_seed": base_seed,
                "environment_seed": streams.environment_seed,
                "policy_seed": streams.policy_seed,
            })
    return schedule


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def runtime_versions():
    return {
        "python": sys.version,
        "numpy": version("numpy"),
        "minatar": version("minatar"),
        "matplotlib": version("matplotlib"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "torch": version("torch"),
        "pytest": version("pytest"),
    }


def record_failure(output_dir: Path, logs_dir: Path, manifest: dict, kind: str, exc: Exception):
    manifest["status"] = "failed"
    manifest["failed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_dir / "run_manifest.json", manifest)
    with (logs_dir / "error.txt").open("w") as fh:
        fh.write(traceback.format_exc())
    logging.exception("Corrected %s run failed", kind)


def run(args):
    kind = "pilot" if args.pilot else "full"
    output_dir = (ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    restarted_from = None
    if args.restart_of:
        failed_manifest_path = (ROOT / args.restart_of).resolve() / "run_manifest.json"
        if not failed_manifest_path.is_file():
            raise FileNotFoundError(f"Restart source manifest not found: {failed_manifest_path}")
        failed_manifest = json.loads(failed_manifest_path.read_text())
        if failed_manifest.get("status") != "failed":
            raise ValueError("--restart-of must reference a run whose manifest status is failed")
        restarted_from = str(Path(args.restart_of))
    output_dir.mkdir(parents=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    logging.basicConfig(
        filename=logs_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_version": 1,
        "dataset": f"phase2_corrected_{kind}",
        "status": "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": git_commit,
        "run_kind": kind,
        "pilot_is_software_validation_only": kind == "pilot",
        "run_attempt": "restart_from_failed_run" if restarted_from else "fresh_start",
        "resumed": False,
        "restarted_from": restarted_from,
        "failure_policy": "failed run is retained; no implicit resume; reruns require a new output directory",
    }
    write_json(output_dir / "run_manifest.json", manifest)

    try:
        config = load_evaluation_config()
        schedule = build_schedule(kind, config)
        conditions = load_conditions()
        manifest.update({
            "episode_count_per_condition": len(schedule),
            "seed_groups": config["seed_groups"],
            "evaluation_config": {**config, "rng_implementation": corrected_run_metadata()},
            "evaluation_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "runtime_versions": runtime_versions(),
            "condition_registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
            "conditions": [
                {k: v for k, v in condition.items() if k not in {"action_fn"}}
                for condition in conditions
            ],
            "episode_seeds_file": "episode_seeds.csv",
        })
        write_json(output_dir / "run_manifest.json", manifest)

        with (output_dir / "episode_seeds.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(schedule[0].keys()))
            writer.writeheader()
            writer.writerows(schedule)
    except Exception as exc:
        record_failure(output_dir, logs_dir, manifest, kind, exc)
        raise

    csv_columns = [
        "seed_group", "episode_index", "base_seed", "environment_seed", "policy_seed",
        "score", "length", "bricks_destroyed", "ball_lost", "step_cutoff", "trajectory_sha256",
    ] + [f"action_{a}_selected_count" for a in range(6)] + [
        f"action_{a}_executed_count" for a in range(6)
    ]
    replay_verified = []
    try:
        for condition in conditions:
            condition_dir = output_dir / "conditions" / condition["id"]
            condition_dir.mkdir(parents=True)
            csv_path = condition_dir / "episode_scores.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=csv_columns)
                writer.writeheader()
                fh.flush()
                for row_number, seed_record in enumerate(schedule):
                    evaluator = CorrectedPolicyEvaluator(
                        seed_group=seed_record["seed_group"],
                        base_seeds=[seed_record["base_seed"]],
                    )
                    result = evaluator.evaluate_episode(
                        condition["action_fn"],
                        base_seed=seed_record["base_seed"],
                        episode_index=seed_record["episode_index"],
                    )
                    result_record = result.to_record()
                    writer.writerow(result_record)
                    fh.flush()
                    if kind == "pilot" and seed_record["episode_index"] == 0:
                        replay = evaluator.replay_record(condition["action_fn"], result_record)
                        if replay.trajectory_sha256 != result.trajectory_sha256:
                            raise RuntimeError(f"Replay mismatch for {condition['id']} seed group {seed_record['seed_group']}")
                        replay_verified.append({
                            "condition_id": condition["id"],
                            "seed_group": seed_record["seed_group"],
                            "base_seed": seed_record["base_seed"],
                            "trajectory_sha256": result.trajectory_sha256,
                            "verified": True,
                        })
                    if row_number % 100 == 0:
                        logging.info("%s episode %s/%s", condition["id"], row_number + 1, len(schedule))
                    if kind == "pilot" or (row_number + 1) % 100 == 0:
                        print(
                            f"{condition['label']}: {row_number + 1}/{len(schedule)} episodes",
                            flush=True,
                        )

            condition["episode_scores_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            logging.info("Completed %s (%s episodes)", condition["id"], len(schedule))

        manifest["status"] = "pilot_complete" if kind == "pilot" else "evaluation_complete_analysis_pending"
        manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["replay_checks"] = replay_verified
        manifest["condition_csv_sha256"] = {
            condition["id"]: condition["episode_scores_sha256"] for condition in conditions
        }
        write_json(output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        record_failure(output_dir, logs_dir, manifest, kind, exc)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Run the configured 10-episode-per-condition validation pilot.")
    mode.add_argument("--full", action="store_true", help="Run the configured 3,000-episode benchmark.")
    parser.add_argument("--output-dir", help="New output directory; refuses to overwrite existing runs.")
    parser.add_argument("--restart-of", help="Relative path to a failed run; records an explicit fresh restart.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "experiments/corrected_evaluation/pilot" if args.pilot else "experiments/corrected_evaluation/full_run"
    return args


if __name__ == "__main__":
    run(parse_args())
