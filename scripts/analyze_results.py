"""Compile and analyze evaluation results from multiple experimental runs."""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.statistical_analysis import PAPER_REFERENCE_SCORES


def load_run_summary(summary_path: str) -> Dict[str, Any]:
    with open(summary_path, "r") as f:
        return json.load(f)


def compare_runs(summary_paths: List[str]) -> pd.DataFrame:
    rows = []
    # Add paper references
    for name, ref in PAPER_REFERENCE_SCORES.items():
        rows.append({
            "Run ID": f"[Paper] {name}",
            "Episodes": 1000,
            "Mean Score": ref["mean"],
            "SE": ref["se"],
            "Max Score": ref["max"],
            "Mean Length": ref["mean_length"],
            "Source": "Paper Reference",
        })

    for p in summary_paths:
        if not os.path.exists(p):
            continue
        data = load_run_summary(p)
        meta = data.get("metadata", {})
        rows.append({
            "Run ID": meta.get("model_id", os.path.basename(os.path.dirname(p))),
            "Episodes": data.get("episodes", 0),
            "Mean Score": round(data.get("mean", 0.0), 3),
            "SE": round(data.get("se", 0.0), 3),
            "Max Score": round(data.get("max", 0.0), 1),
            "Mean Length": round(data.get("mean_length", 0.0), 1),
            "Source": "Reproduction Lab",
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare evaluation runs")
    parser.add_argument("summary_files", nargs="+", help="Paths to summary.json files")
    args = parser.parse_args()

    df = compare_runs(args.summary_files)
    print("\n" + "=" * 80)
    print("EXPERIMENTAL REPLICATION & COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

