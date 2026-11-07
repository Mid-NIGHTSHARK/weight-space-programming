"""Statistical analysis and reporting utilities for evaluation results."""

import json
import os
from typing import Dict, Any, List
import pandas as pd

from evaluation.metrics import EvaluationSummary, EpisodeResult

# Ground-truth values reported in the original paper minatar_eval.json
PAPER_REFERENCE_SCORES: Dict[str, Dict[str, float]] = {
    "astra_hand": {
        "mean": 517.411,
        "se": 7.417,
        "max": 800.0,
        "mean_length": 7417.202,
    },
    "ppo_100m_best": {
        "mean": 15.779,
        "se": 0.093,
        "max": 17.0,
        "mean_length": 178.42,
    },
    "ppo_500m_best": {
        "mean": 25.344,
        "se": 0.121,
        "max": 29.0,
        "mean_length": 281.852,
    },
}


def save_episode_scores_csv(episodes: List[EpisodeResult], filepath: str) -> None:
    """Save individual episode records to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    records = []
    for ep in episodes:
        rec = {
            "episode_index": ep.episode_index,
            "seed": ep.seed,
            "score": ep.score,
            "length": ep.length,
            "bricks_destroyed": ep.bricks_destroyed,
            "ball_lost": ep.ball_lost,
            "step_cutoff": ep.step_cutoff,
        }
        for a in range(6):
            rec[f"action_{a}_count"] = ep.action_counts.get(a, 0)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)


def save_summary_json(summary: EvaluationSummary, filepath: str) -> None:
    """Save evaluation summary metrics to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)


def generate_markdown_report(
    summary: EvaluationSummary,
    filepath: str,
    experiment_title: str = "MinAtar Breakout Policy Evaluation Report",
) -> str:
    """Generate a GitHub-flavored Markdown evaluation report."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    meta = summary.metadata
    report_lines = [
        f"# {experiment_title}",
        "",
        f"**Timestamp:** {meta.get('timestamp', 'N/A')}",
        f"**Agent / Model Identifier:** `{meta.get('model_id', 'Unknown')}`",
        f"**Weight Hash (SHA-256):** `{meta.get('weight_sha256', 'N/A')}`",
        f"**Environment:** MinAtar Breakout (`minatar=={meta.get('minatar_version', '1.0.15')}`)",
        f"**Evaluation Protocol:** {summary.episodes} episodes, sticky actions 0.1, max 10,000 steps",
        "",
        "## Executive Summary",
        "",
        "| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |",
        "|---|---|---|---|",
        f"| **Mean Score** | **{summary.mean_score:.3f}** | 517.411 | 25.344 |",
        f"| **Standard Error** | ±{summary.se_score:.3f} | ±7.417 | ±0.121 |",
        f"| **Median Score** | {summary.median_score:.1f} | N/A | N/A |",
        f"| **Score Range [Min, Max]** | [{summary.min_score:.1f}, {summary.max_score:.1f}] | [N/A, 800.0] | [N/A, 29.0] |",
        f"| **Mean Episode Length** | {summary.mean_length:.1f} steps | 7,417.2 steps | 281.9 steps |",
        f"| **Total Bricks Destroyed** | {summary.total_bricks_destroyed:,} | N/A | N/A |",
        f"| **Step Cutoffs (>=10,000)** | {summary.step_cutoffs_count} ({summary.step_cutoffs_count / summary.episodes * 100:.1f}%) | N/A | N/A |",
        "",
        "## Action Selection Distribution",
        "",
        "| Action Index | Action Name | Proportion |",
        "|---|---|---|",
    ]

    action_names = {0: "No-op", 1: "Left", 2: "Up", 3: "Right", 4: "Down", 5: "Fire"}
    for a in range(6):
        pct = summary.action_distribution.get(a, 0.0) * 100.0
        report_lines.append(f"| {a} | {action_names[a]} | {pct:.2f}% |")

    report_lines.extend([
        "",
        "## Scientific Verdict",
        "",
    ])

    if summary.mean_score > 100.0:
        report_lines.append(
            "> [!NOTE]\n"
            "> **High-Competence Policy:** The evaluated network exhibits strategic, highly competent play "
            f"(mean score: {summary.mean_score:.2f}), far exceeding conventional reinforcement learning baselines "
            "(PPO 500M: 25.344) and demonstrating non-trivial weight-space programming capability."
        )
    elif summary.mean_score > 20.0:
        report_lines.append(
            "> [!NOTE]\n"
            "> **Moderate Policy:** The evaluated network plays above random/zero baselines and achieves performance "
            f"comparable to multi-million step PPO training (mean score: {summary.mean_score:.2f})."
        )
    else:
        report_lines.append(
            "> [!WARNING]\n"
            "> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with "
            f"untrained or random behavior (mean score: {summary.mean_score:.2f})."
        )

    content = "\n".join(report_lines) + "\n"
    with open(filepath, "w") as f:
        f.write(content)
    return content

