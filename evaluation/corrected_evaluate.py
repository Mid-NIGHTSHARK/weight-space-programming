"""Corrected MinAtar evaluator with independent environment and policy RNGs.

This module is intentionally separate from ``evaluation.evaluate``.  The
existing PolicyEvaluator remains the historical same-seed implementation;
corrected runs must opt into this evaluator explicitly.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Dict, Iterable, List, Optional

import minatar
import numpy as np

from environment.breakout import MinAtarBreakoutEnv
from evaluation.rng_streams import (
    EpisodeStreamSeeds,
    RNG_PROTOCOL,
    derive_episode_stream_seeds,
    make_policy_rng,
)


ActionFunction = Callable[[np.ndarray, np.random.RandomState], int]
ACTION_NAMES = {0: "no_op", 1: "left", 2: "up", 3: "right", 4: "down", 5: "fire"}


@dataclass
class CorrectedEpisodeResult:
    seed_group: int
    episode_index: int
    base_seed: int
    environment_seed: int
    policy_seed: int
    score: float
    length: int
    bricks_destroyed: int
    ball_lost: bool
    step_cutoff: bool
    selected_action_counts: Dict[int, int]
    executed_action_counts: Dict[int, int]
    trajectory_sha256: str

    def to_record(self) -> dict:
        record = asdict(self)
        for action_id in range(6):
            record[f"action_{action_id}_selected_count"] = self.selected_action_counts.get(action_id, 0)
            record[f"action_{action_id}_executed_count"] = self.executed_action_counts.get(action_id, 0)
        del record["selected_action_counts"]
        del record["executed_action_counts"]
        return record


class CorrectedPolicyEvaluator:
    """Run episodes where each RNG stream is derived independently per episode."""

    def __init__(
        self,
        seed_group: int,
        base_seeds: Iterable[int],
        sticky_action_prob: float = 0.1,
        max_steps: int = 10_000,
    ):
        if not np.isclose(sticky_action_prob, 0.1):
            raise ValueError("Corrected protocol preserves sticky_action_prob=0.1")
        if max_steps != 10_000:
            raise ValueError("Corrected protocol preserves the 10,000-step cutoff")
        self.seed_group = int(seed_group)
        self.base_seeds = [int(seed) for seed in base_seeds]
        self.sticky_action_prob = float(sticky_action_prob)
        self.max_steps = int(max_steps)

    @staticmethod
    def _digest_start(obs: np.ndarray):
        digest = hashlib.sha256()
        digest.update(np.asarray(obs, dtype=np.uint8).tobytes(order="C"))
        return digest

    @staticmethod
    def _digest_transition(digest, obs: np.ndarray, selected: int, reward: float, done: bool, info: dict):
        ball_dir = info["ball_dir"]
        if np.isscalar(ball_dir):
            ball_dir_value = int(ball_dir)
        else:
            ball_dir_value = [int(v) for v in ball_dir]
        transition = {
            "selected_action": int(selected),
            "executed_action": int(info["executed_action"]),
            "sticky_override": bool(info["sticky_override"]),
            "reward_hex": float(reward).hex(),
            "done": bool(done),
            "ball_lost": bool(info["ball_lost"]),
            "step_cutoff": bool(info["step_cutoff"]),
            "paddle_pos": int(info["paddle_pos"]),
            "ball_x": int(info["ball_x"]),
            "ball_y": int(info["ball_y"]),
            "ball_dir": ball_dir_value,
            "bricks_remaining": int(info["bricks_remaining"]),
        }
        digest.update(json.dumps(transition, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(np.asarray(obs, dtype=np.uint8).tobytes(order="C"))

    def evaluate_episode(
        self,
        action_fn: ActionFunction,
        base_seed: int,
        episode_index: int,
        stream_seeds: Optional[EpisodeStreamSeeds] = None,
    ) -> CorrectedEpisodeResult:
        streams = stream_seeds or derive_episode_stream_seeds(base_seed)
        if streams.base_seed != int(base_seed):
            raise ValueError("recorded stream seeds do not match the requested base seed")

        env = MinAtarBreakoutEnv(
            sticky_action_prob=self.sticky_action_prob,
            max_steps=self.max_steps,
        )
        obs = env.reset(seed=streams.environment_seed)
        policy_rng = make_policy_rng(streams.policy_seed)
        digest = self._digest_start(obs)
        selected_counts = {action: 0 for action in range(6)}
        executed_counts = {action: 0 for action in range(6)}
        score = 0.0
        length = 0
        bricks_destroyed = 0
        last_info = {"ball_lost": False, "step_cutoff": False}

        while not env.terminal:
            selected_action = int(action_fn(obs, policy_rng))
            if selected_action not in selected_counts:
                raise ValueError(f"action_fn returned invalid action {selected_action}")
            selected_counts[selected_action] += 1
            obs, reward, done, last_info = env.step(selected_action)
            executed_counts[int(last_info["executed_action"])] += 1
            score += float(reward)
            length += 1
            bricks_destroyed += int(reward) if reward > 0 else 0
            self._digest_transition(digest, obs, selected_action, reward, done, last_info)

        return CorrectedEpisodeResult(
            seed_group=self.seed_group,
            episode_index=int(episode_index),
            base_seed=streams.base_seed,
            environment_seed=streams.environment_seed,
            policy_seed=streams.policy_seed,
            score=score,
            length=length,
            bricks_destroyed=bricks_destroyed,
            ball_lost=bool(last_info.get("ball_lost", False)),
            step_cutoff=bool(last_info.get("step_cutoff", False)),
            selected_action_counts=selected_counts,
            executed_action_counts=executed_counts,
            trajectory_sha256=digest.hexdigest(),
        )

    def evaluate(self, action_fn: ActionFunction) -> List[CorrectedEpisodeResult]:
        return [
            self.evaluate_episode(action_fn, base_seed, episode_index)
            for episode_index, base_seed in enumerate(self.base_seeds)
        ]

    def replay_record(self, action_fn: ActionFunction, record: dict) -> CorrectedEpisodeResult:
        """Replay from a saved base seed and verify its recorded derived seeds."""
        streams = derive_episode_stream_seeds(int(record["base_seed"]))
        expected_seeds = (
            int(record["environment_seed"]),
            int(record["policy_seed"]),
        )
        if (streams.environment_seed, streams.policy_seed) != expected_seeds:
            raise ValueError("recorded environment/policy seeds do not match the RNG derivation")
        return self.evaluate_episode(
            action_fn,
            int(record["base_seed"]),
            int(record["episode_index"]),
            stream_seeds=streams,
        )


def corrected_run_metadata() -> dict:
    """Stable protocol metadata shared by pilot and full run manifests."""
    return {
        "rng_protocol": RNG_PROTOCOL,
        "environment_rng": "MinAtar 1.0.15 RandomState seeded by environment_seed",
        "policy_rng": "separate numpy.random.RandomState(policy_seed); np.random.choice(p=softmax, temperature=1.0)",
        "sticky_action_probability": 0.1,
        "max_episode_steps": 10_000,
        "action_distribution": "categorical softmax sampling, temperature 1.0",
        "minatar_version": getattr(minatar, "__version__", "1.0.15"),
    }
