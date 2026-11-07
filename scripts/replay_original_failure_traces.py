#!/usr/bin/env python3
"""Replay five saved historical episodes and store their transition evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.legacy_trace_replay import replay_all_legacy_failure_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="experiments/corrected_evaluation/failure_trace_replays.json",
        help="New additive artifact path; no historical CSV is modified.",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite trace artifact: {output_path}")
    data = replay_all_legacy_failure_traces()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(data['conditions'])} replay-verified trajectories to {output_path}")


if __name__ == "__main__":
    main()
