#!/usr/bin/env python3
"""Create paired metrics and inferential artifacts for the corrected run."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.corrected_analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="experiments/corrected_evaluation/full_run")
    parser.add_argument("--bootstrap-resamples", type=int, default=9_999)
    parser.add_argument("--permutation-resamples", type=int, default=19_999)
    args = parser.parse_args()
    analysis = analyze_run(
        Path(args.run_dir),
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    )
    print(f"Completed analysis of {analysis['matched_episode_count']} matched episodes per condition.")
    print("Primary zero-shot ranking:")
    for rank, condition in enumerate(analysis["ranking_primary_zero_shot"], 1):
        print(f"{rank}. {condition['model']}: {condition['mean_score']:.3f}")


if __name__ == "__main__":
    main()
