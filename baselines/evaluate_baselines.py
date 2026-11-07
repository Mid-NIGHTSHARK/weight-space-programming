"""Evaluate random and zero-weight baselines across the 1,000-episode protocol."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.random_agent import RandomAgent
from baselines.zero_agent import ZeroWeightAgent
from evaluation.evaluate import PolicyEvaluator


def evaluate_baselines(num_episodes: int = 1000, output_root: str = "results/baselines"):
    evaluator = PolicyEvaluator(num_episodes=num_episodes, sticky_action_prob=0.1, max_steps=10_000)

    print(f"\n{'='*70}")
    print(f"EVALUATING BASELINES ({num_episodes} EPISODES EACH)")
    print(f"{'='*70}\n")

    # 1. Random Agent
    print(">>> Baseline 1: Random Agent")
    random_agent = RandomAgent()
    random_summary = evaluator.evaluate_agent(
        action_fn=random_agent,
        agent_id="random_baseline",
        output_dir=os.path.join(output_root, "random"),
        verbose=True,
    )

    # 2. Zero-Weight Agent
    print("\n>>> Baseline 2: Zero-Weight Network Agent")
    zero_agent = ZeroWeightAgent()
    zero_summary = evaluator.evaluate_agent(
        action_fn=zero_agent,
        agent_id="zero_weight_baseline",
        output_dir=os.path.join(output_root, "zero"),
        verbose=True,
    )

    print(f"\n{'='*70}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent':<25} | {'Mean Score':<12} | {'Std Error':<10} | {'Max Score':<10} | {'Mean Length':<12}")
    print(f"{'-'*25}-|-{'-'*12}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}")
    print(
        f"{'Random Baseline':<25} | "
        f"{random_summary.mean_score:<12.3f} | "
        f"±{random_summary.se_score:<9.3f} | "
        f"{random_summary.max_score:<10.1f} | "
        f"{random_summary.mean_length:<12.1f}"
    )
    print(
        f"{'Zero-Weight Baseline':<25} | "
        f"{zero_summary.mean_score:<12.3f} | "
        f"±{zero_summary.se_score:<9.3f} | "
        f"{zero_summary.max_score:<10.1f} | "
        f"{zero_summary.mean_length:<12.1f}"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    evaluate_baselines(num_episodes=episodes)

