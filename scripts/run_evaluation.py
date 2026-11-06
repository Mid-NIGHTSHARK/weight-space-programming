"""CLI runner for MinAtar Breakout policy evaluation."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import PolicyEvaluator
from baselines.random_agent import RandomAgent
from baselines.zero_agent import ZeroWeightAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MinAtar Breakout Policies")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/reference/hand.npz",
        help="Path to .npz weights archive",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random", "zero"],
        default=None,
        help="Run baseline agent instead of weights file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes (default: 1000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode cutoff (default: 10000)",
    )
    parser.add_argument(
        "--sticky-prob",
        type=float,
        default=0.1,
        help="Sticky action probability (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Identifier for the agent (default derived from source)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save CSV, JSON, and Markdown reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = PolicyEvaluator(
        num_episodes=args.episodes,
        sticky_action_prob=args.sticky_prob,
        max_steps=args.max_steps,
        base_seed=args.seed,
    )

    if args.baseline == "random":
        agent_id = args.agent_id or "random_baseline"
        agent = RandomAgent()
        evaluator.evaluate_agent(
            action_fn=agent,
            agent_id=agent_id,
            output_dir=args.output_dir,
            verbose=True,
        )
    elif args.baseline == "zero":
        agent_id = args.agent_id or "zero_weight_baseline"
        agent = ZeroWeightAgent()
        evaluator.evaluate_agent(
            action_fn=agent,
            agent_id=agent_id,
            output_dir=args.output_dir,
            verbose=True,
        )
    else:
        if not os.path.exists(args.weights):
            print(f"Error: Weights file not found: {args.weights}", file=sys.stderr)
            sys.exit(1)
        agent_id = args.agent_id or os.path.splitext(os.path.basename(args.weights))[0]
        evaluator.evaluate_weight_file(
            weight_filepath=args.weights,
            agent_id=agent_id,
            output_dir=args.output_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()

