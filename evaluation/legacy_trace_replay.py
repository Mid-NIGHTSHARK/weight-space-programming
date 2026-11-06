"""Deterministically reconstruct saved legacy evaluation episodes for audit."""

import csv
import hashlib
from pathlib import Path
from typing import Dict

import numpy as np

from environment.breakout import MinAtarBreakoutEnv
from network.architecture import MinAtarNetworkNumPy
from network.inference import PolicyAgent
from network.weight_loader import compute_file_sha256, load_weights_npz


ROOT = Path(__file__).resolve().parents[1]
TRACE_CONDITIONS = {
    "gpt6_original": {
        "label": "GPT-6 Original",
        "weights": "experiments/strict_replication/generation_002_gpt6/output/gpt6_weights.npz",
        "scores": "experiments/strict_replication/generation_002_gpt6/evaluation/eval_seed{group}/episode_scores.csv",
    },
    "claude_sonnet_55": {
        "label": "Claude Sonnet 5.5",
        "weights": "experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz",
        "scores": "experiments/strict_replication/generation_003_sonnet55/evaluation/eval_seed{group}/episode_scores.csv",
    },
    "deepseek_original": {
        "label": "DeepSeek Original",
        "weights": "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz",
        "scores": "experiments/strict_replication/generation_004_deepseek/evaluation/eval_seed{group}/episode_scores.csv",
    },
    "gpt6_luna_extra_high": {
        "label": "GPT-6 Luna Extra High",
        "weights": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz",
        "scores": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/evaluation/eval_seed{group}/episode_scores.csv",
    },
    "gemini_argon_4_medium": {
        "label": "Gemini Argon 4 Medium",
        "weights": "experiments/strict_replication/generation_001/output/argon_weights.npz",
        "scores": "experiments/strict_replication/generation_001/eval_seed{group}/episode_scores.csv",
    },
}
DIRECTION_OFFSETS = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}


def replay_legacy_episode(condition_id: str, seed_group: int = 42, episode_index: int = 0) -> dict:
    """Replay one legacy same-seed episode and require an exact CSV score/length match."""
    condition = TRACE_CONDITIONS[condition_id]
    score_relative_path = condition["scores"].format(group=seed_group)
    score_path = ROOT / score_relative_path
    weight_path = ROOT / condition["weights"]
    with score_path.open(newline="") as fh:
        expected = next((row for row in csv.DictReader(fh) if int(row["episode_index"]) == episode_index), None)
    if expected is None:
        raise ValueError(f"Historical episode index {episode_index} missing in {score_path}")

    seed = int(expected["seed"])
    weight_hash = compute_file_sha256(str(weight_path))
    score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    net = MinAtarNetworkNumPy(load_weights_npz(str(weight_path)))
    agent = PolicyAgent(net, rng=np.random.RandomState(seed))
    env = MinAtarBreakoutEnv(sticky_action_prob=0.1, max_steps=10_000)
    obs = env.reset(seed=seed)

    trajectory = []
    selected_counts = {action: 0 for action in range(6)}
    executed_counts = {action: 0 for action in range(6)}
    initial_logits = None
    total_reward = 0.0
    while not env.terminal:
        state = env.internal_env
        before = {
            "ball_x": int(state.ball_x),
            "ball_y": int(state.ball_y),
            "trail_x": int(state.last_x),
            "trail_y": int(state.last_y),
            "paddle_x": int(state.pos),
            "ball_dir": int(state.ball_dir),
        }
        dx, dy = DIRECTION_OFFSETS[before["ball_dir"]]
        attempted_next = [before["ball_x"] + dx, before["ball_y"] + dy]
        selected, probabilities, logits = agent.get_action(obs)
        if initial_logits is None:
            initial_logits = [float(value) for value in logits]
        selected_counts[int(selected)] += 1
        obs, reward, done, info = env.step(selected)
        total_reward += float(reward)
        executed_counts[int(info["executed_action"])] += 1
        trajectory.append({
            "step": int(info["step"]),
            "before": before,
            "selected_action": int(info["selected_action"]),
            "executed_action": int(info["executed_action"]),
            "sticky_override": bool(info["sticky_override"]),
            "attempted_next_xy_before_wall_or_collision_resolution": attempted_next,
            "after": {
                "ball_x": int(info["ball_x"]),
                "ball_y": int(info["ball_y"]),
                "paddle_x": int(info["paddle_pos"]),
                "ball_dir": int(info["ball_dir"]),
                "reward": float(reward),
                "bricks_remaining": int(info["bricks_remaining"]),
            },
            "ball_lost": bool(info["ball_lost"]),
            "step_cutoff": bool(info["step_cutoff"]),
        })

    observed_length = len(trajectory)
    if observed_length != int(expected["length"]) or total_reward != float(expected["score"]):
        raise RuntimeError(
            f"Legacy replay mismatch for {condition_id}: CSV=({expected['score']}, {expected['length']}), "
            f"replay=({total_reward}, {observed_length})"
        )

    terminal = trajectory[-1]
    next_x = max(0, min(9, terminal["attempted_next_xy_before_wall_or_collision_resolution"][0]))
    capture_columns = sorted({terminal["before"]["ball_x"], next_x})
    terminal["collision_audit"] = {
        "attempted_y": terminal["attempted_next_xy_before_wall_or_collision_resolution"][1],
        "current_x_checked_by_minatar": terminal["before"]["ball_x"],
        "next_x_after_wall_clamp_checked_by_minatar": next_x,
        "paddle_x_after_selected_action_and_sticky_handling": terminal["after"]["paddle_x"],
        "valid_capture_columns": capture_columns,
        "miss_confirmed": bool(
            terminal["ball_lost"]
            and terminal["attempted_next_xy_before_wall_or_collision_resolution"][1] == 9
            and terminal["after"]["paddle_x"] not in capture_columns
        ),
    }
    return {
        "condition_id": condition_id,
        "model_label_locally_recorded": condition["label"],
        "protocol": "legacy_evaluation_replay_same_integer_seed_for_environment_and_policy",
        "seed_group": int(seed_group),
        "episode_index": int(episode_index),
        "base_episode_seed": seed,
        "weight_path": condition["weights"],
        "weight_sha256": weight_hash,
        "historical_episode_csv": score_relative_path,
        "historical_episode_csv_sha256": score_hash,
        "historical_csv_score": float(expected["score"]),
        "historical_csv_length": int(expected["length"]),
        "replayed_score": total_reward,
        "replayed_length": observed_length,
        "replay_matches_historical_csv": True,
        "terminal_event": "step_cutoff" if terminal["step_cutoff"] else "ball_lost" if terminal["ball_lost"] else "unknown",
        "selected_action_counts": {str(k): v for k, v in selected_counts.items()},
        "executed_action_counts_after_sticky_handling": {str(k): v for k, v in executed_counts.items()},
        "initial_logits": initial_logits,
        "trajectory": trajectory,
    }


def replay_all_legacy_failure_traces() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "diagnostic_replay_not_a_new_benchmark",
        "description": "Five seed-42 episode-0 legacy-policy replays. Each trajectory is checked against the original CSV score and length; no historical artifact is rewritten.",
        "conditions": {
            condition_id: replay_legacy_episode(condition_id)
            for condition_id in TRACE_CONDITIONS
        },
    }
