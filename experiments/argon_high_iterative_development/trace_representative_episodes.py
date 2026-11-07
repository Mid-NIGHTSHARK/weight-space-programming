#!/usr/bin/env python3
"""Replay matched seeds that start from each MinAtar Breakout side."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.corrected_evaluate import CorrectedPolicyEvaluator  # noqa: E402
from evaluation.rng_streams import derive_episode_stream_seeds, make_policy_rng  # noqa: E402
from environment.breakout import MinAtarBreakoutEnv  # noqa: E402
from network.architecture import MinAtarNetworkNumPy  # noqa: E402
from network.inference import PolicyAgent  # noqa: E402
from network.weight_loader import load_weights_npz  # noqa: E402


RUN_DIR = HERE / "evaluation/full_run"
ACTION_NAMES = ["no_op", "left", "up", "right", "down", "fire"]


def find_initial_sides():
    with (RUN_DIR / "episode_seeds.csv").open(newline="") as stream:
        schedule = list(csv.DictReader(stream))
    chosen = {}
    for row in schedule:
        group = int(row["seed_group"])
        index = int(row["episode_index"])
        if group != 42:
            continue
        environment_seed = int(row["environment_seed"])
        env = MinAtarBreakoutEnv(sticky_action_prob=0.1, max_steps=10_000)
        env.reset(seed=environment_seed)
        side = "left" if env.internal_env.ball_x == 0 else "right"
        chosen.setdefault(side, {
            "seed_group": group,
            "episode_index": index,
            "base_seed": int(row["base_seed"]),
            "environment_seed": environment_seed,
            "policy_seed": int(row["policy_seed"]),
            "initial_ball_x": int(env.internal_env.ball_x),
            "initial_ball_y": int(env.internal_env.ball_y),
            "initial_ball_dir": int(env.internal_env.ball_dir),
        })
        if len(chosen) == 2:
            break
    if set(chosen) != {"left", "right"}:
        raise RuntimeError(f"Could not find both initial sides in seed group 42: {chosen}")
    return chosen


def run_trace(condition, seed_record):
    weights_path = HERE / condition["weight_path"]
    agent = PolicyAgent(MinAtarNetworkNumPy(load_weights_npz(str(weights_path))))
    evaluator = CorrectedPolicyEvaluator(
        seed_group=seed_record["seed_group"],
        base_seeds=[seed_record["base_seed"]],
    )
    streams = derive_episode_stream_seeds(seed_record["base_seed"])
    env = MinAtarBreakoutEnv(sticky_action_prob=0.1, max_steps=10_000)
    obs = env.reset(seed=streams.environment_seed)
    policy_rng = make_policy_rng(streams.policy_seed)
    digest = evaluator._digest_start(obs)
    start_state = {
        "ball_x": int(env.internal_env.ball_x),
        "ball_y": int(env.internal_env.ball_y),
        "ball_dir": int(env.internal_env.ball_dir),
        "trail_x": int(env.internal_env.last_x),
        "trail_y": int(env.internal_env.last_y),
        "paddle_x": int(env.internal_env.pos),
        "bricks_remaining": int(env.internal_env.brick_map.sum()),
    }
    steps = []
    total_score = 0.0
    last_info = {"ball_lost": False, "step_cutoff": False}
    while not env.terminal:
        before = {
            "ball_x": int(env.internal_env.ball_x),
            "ball_y": int(env.internal_env.ball_y),
            "ball_dir": int(env.internal_env.ball_dir),
            "trail_x": int(env.internal_env.last_x),
            "trail_y": int(env.internal_env.last_y),
            "paddle_x": int(env.internal_env.pos),
        }
        agent.rng = policy_rng
        selected, probabilities, logits = agent.get_action(obs)
        obs, reward, done, info = env.step(selected)
        last_info = info
        total_score += reward
        evaluator._digest_transition(digest, obs, selected, reward, done, info)
        steps.append({
            "step": int(info["step"]),
            "before": before,
            "selected_action": int(selected),
            "selected_action_name": ACTION_NAMES[selected],
            "action_probabilities": [float(x) for x in probabilities],
            "logits": [float(x) for x in logits],
            "executed_action": int(info["executed_action"]),
            "sticky_override": bool(info["sticky_override"]),
            "reward": float(reward),
            "after": {
                "ball_x": int(info["ball_x"]),
                "ball_y": int(info["ball_y"]),
                "ball_dir": int(info["ball_dir"]),
                "paddle_x": int(info["paddle_pos"]),
                "bricks_remaining": int(info["bricks_remaining"]),
            },
            "done": bool(done),
            "step_cutoff": bool(info["step_cutoff"]),
        })
    trace_hash = digest.hexdigest()
    episode_csv = RUN_DIR / "conditions" / condition["id"] / "episode_scores.csv"
    expected = None
    with episode_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["seed_group"]) == seed_record["seed_group"] and int(row["episode_index"]) == seed_record["episode_index"]:
                expected = row
                break
    if expected is None:
        raise RuntimeError(f"Full-run CSV has no episode for {condition['id']} {seed_record}")
    if trace_hash != expected["trajectory_sha256"]:
        raise RuntimeError(f"Trajectory replay differs for {condition['id']} {seed_record}")
    return {
        "condition_id": condition["id"],
        "label": condition["label"],
        "seed_group": seed_record["seed_group"],
        "episode_index": seed_record["episode_index"],
        "base_seed": seed_record["base_seed"],
        "environment_seed": seed_record["environment_seed"],
        "policy_seed": seed_record["policy_seed"],
        "start_side": "left" if start_state["ball_x"] == 0 else "right",
        "start_state": start_state,
        "score": total_score,
        "length": len(steps),
        "ball_lost": bool(last_info["ball_lost"]),
        "step_cutoff": bool(last_info["step_cutoff"]),
        "trajectory_sha256": trace_hash,
        "replay_matches_full_run_csv": True,
        "steps": steps,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/representative_trajectories.json", help="New output path relative to this experiment directory")
    args = parser.parse_args()
    output_path = (HERE / args.output).resolve()
    if HERE not in output_path.parents:
        raise ValueError("--output must remain inside this experiment directory")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite representative traces: {output_path}")
    manifest = json.loads((RUN_DIR / "run_manifest.json").read_text())
    if manifest.get("status") not in {"evaluation_complete_analysis_pending", "analysis_complete"}:
        raise ValueError("Representative traces require a completed full run")
    registry = json.loads((HERE / "model_registry.json").read_text())
    sides = find_initial_sides()
    traces = []
    for condition in registry["conditions"]:
        for side in ("left", "right"):
            traces.append(run_trace(condition, sides[side]))
    output = {
        "description": "Complete matched episode replays for one left-start and one right-start seed, each verified against its full-run trajectory digest.",
        "initial_side_seed_records": sides,
        "traces": traces,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for trace in traces:
        print(
            f"{trace['condition_id']} {trace['start_side']} start: score={trace['score']}, "
            f"length={trace['length']}, steps={len(trace['steps'])}, replay hash verified"
        )


if __name__ == "__main__":
    main()
