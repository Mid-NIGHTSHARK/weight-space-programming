"""Authoritative evaluation harness for MinAtar Breakout policies."""

import datetime
import os
import time
from typing import Dict, Optional, Any, Callable
import minatar
import numpy as np

from environment.breakout import MinAtarBreakoutEnv
from evaluation.metrics import (
    EpisodeResult,
    EvaluationSummary,
    compute_evaluation_metrics,
)
from evaluation.seeds import generate_evaluation_seeds
from evaluation.statistical_analysis import (
    generate_markdown_report,
    save_episode_scores_csv,
    save_summary_json,
)
from network.architecture import MinAtarNetworkNumPy
from network.inference import PolicyAgent
from network.weight_loader import compute_file_sha256


class PolicyEvaluator:
    """Evaluates neural network policies or baseline agents under controlled MinAtar conditions."""

    def __init__(
        self,
        num_episodes: int = 1000,
        sticky_action_prob: float = 0.1,
        max_steps: int = 10_000,
        base_seed: int = 42,
    ):
        self.num_episodes = num_episodes
        self.sticky_action_prob = sticky_action_prob
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.seeds = generate_evaluation_seeds(num_episodes, base_seed)

    def evaluate_agent(
        self,
        action_fn: Callable[[np.ndarray, np.random.RandomState], int],
        agent_id: str = "custom_agent",
        weight_sha256: str = "N/A",
        output_dir: Optional[str] = None,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> EvaluationSummary:
        """Run standard evaluation protocol on an agent action function."""
        env = MinAtarBreakoutEnv(
            sticky_action_prob=self.sticky_action_prob,
            max_steps=self.max_steps,
        )

        episode_results = []
        t0 = time.time()

        if verbose:
            print(f"Starting evaluation of '{agent_id}' across {self.num_episodes} episodes...")

        for idx, seed in enumerate(self.seeds):
            obs = env.reset(seed=seed)
            agent_rng = np.random.RandomState(seed)

            action_counts: Dict[int, int] = {a: 0 for a in range(6)}
            ep_reward = 0.0
            ep_steps = 0
            ep_bricks = 0

            while not env.terminal:
                action = action_fn(obs, agent_rng)
                action_counts[action] = action_counts.get(action, 0) + 1

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if reward > 0:
                    ep_bricks += int(reward)

            ep_res = EpisodeResult(
                episode_index=idx,
                seed=seed,
                score=ep_reward,
                length=ep_steps,
                bricks_destroyed=ep_bricks,
                ball_lost=info.get("ball_lost", False),
                step_cutoff=info.get("step_cutoff", False),
                action_counts=action_counts,
            )
            episode_results.append(ep_res)

            if verbose and ((idx + 1) % log_interval == 0 or idx == self.num_episodes - 1):
                interim_scores = [e.score for e in episode_results]
                elapsed = time.time() - t0
                print(
                    f"  [Episode {idx + 1}/{self.num_episodes}] "
                    f"Mean: {np.mean(interim_scores):.2f} ± {np.std(interim_scores)/np.sqrt(idx+1):.2f} | "
                    f"Last score: {ep_reward:.0f} (len {ep_steps}) | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        total_time = time.time() - t0

        metadata = {
            "model_id": agent_id,
            "weight_sha256": weight_sha256,
            "episodes": self.num_episodes,
            "sticky_action_prob": self.sticky_action_prob,
            "max_steps": self.max_steps,
            "base_seed": self.base_seed,
            "minatar_version": getattr(minatar, "__version__", "1.0.15"),
            "total_eval_time_seconds": round(total_time, 2),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        summary = compute_evaluation_metrics(episode_results, metadata=metadata)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_episode_scores_csv(
                episode_results,
                os.path.join(output_dir, "episode_scores.csv"),
            )
            save_summary_json(
                summary,
                os.path.join(output_dir, "summary.json"),
            )
            generate_markdown_report(
                summary,
                os.path.join(output_dir, "evaluation_report.md"),
                experiment_title=f"MinAtar Breakout Evaluation: {agent_id}",
            )
            if verbose:
                print(f"Results successfully saved to {output_dir}")

        return summary

    def evaluate_network(
        self,
        network: MinAtarNetworkNumPy,
        agent_id: str = "network_policy",
        weight_sha256: str = "N/A",
        output_dir: Optional[str] = None,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> EvaluationSummary:
        """Evaluate a MinAtarNetworkNumPy policy."""
        agent = PolicyAgent(network)

        def action_fn(obs: np.ndarray, rng: np.random.RandomState) -> int:
            agent.rng = rng
            action, probs, logits = agent.get_action(obs)
            return action

        return self.evaluate_agent(
            action_fn=action_fn,
            agent_id=agent_id,
            weight_sha256=weight_sha256,
            output_dir=output_dir,
            verbose=verbose,
            log_interval=log_interval,
        )

    def evaluate_weight_file(
        self,
        weight_filepath: str,
        agent_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> EvaluationSummary:
        """Load and evaluate a .npz weight archive."""
        from network.weight_loader import load_weights_npz

        if agent_id is None:
            agent_id = os.path.basename(weight_filepath)

        sha256 = compute_file_sha256(weight_filepath)
        weights = load_weights_npz(weight_filepath)
        network = MinAtarNetworkNumPy(weights)

        return self.evaluate_network(
            network=network,
            agent_id=agent_id,
            weight_sha256=sha256,
            output_dir=output_dir,
            verbose=verbose,
            log_interval=log_interval,
        )

