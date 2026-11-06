"""Statistical evaluation metrics for MinAtar Breakout experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    score: float
    length: int
    bricks_destroyed: int
    ball_lost: bool
    step_cutoff: bool
    action_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    episodes: int
    mean_score: float
    median_score: float
    std_score: float
    se_score: float
    min_score: float
    max_score: float
    mean_length: float
    median_length: float
    std_length: float
    total_bricks_destroyed: int
    action_distribution: Dict[int, float]
    step_cutoffs_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "mean": self.mean_score,
            "median": self.median_score,
            "std": self.std_score,
            "se": self.se_score,
            "min": self.min_score,
            "max": self.max_score,
            "mean_length": self.mean_length,
            "median_length": self.median_length,
            "std_length": self.std_length,
            "total_bricks_destroyed": self.total_bricks_destroyed,
            "action_distribution": self.action_distribution,
            "step_cutoffs_count": self.step_cutoffs_count,
            "metadata": self.metadata,
        }


def compute_evaluation_metrics(
    episodes: List[EpisodeResult],
    metadata: Dict[str, Any] = None,
) -> EvaluationSummary:
    """Compute aggregate statistical metrics across evaluation episodes."""
    if not episodes:
        raise ValueError("Cannot compute metrics over empty episode list")

    n = len(episodes)
    scores = np.array([ep.score for ep in episodes], dtype=np.float64)
    lengths = np.array([ep.length for ep in episodes], dtype=np.float64)

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    se_score = float(std_score / np.sqrt(n)) if n > 1 else 0.0
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    mean_length = float(np.mean(lengths))
    median_length = float(np.median(lengths))
    std_length = float(np.std(lengths, ddof=1)) if n > 1 else 0.0

    total_bricks = int(sum(ep.bricks_destroyed for ep in episodes))
    step_cutoffs = int(sum(1 for ep in episodes if ep.step_cutoff))

    # Action counts aggregate
    total_actions = sum(sum(ep.action_counts.values()) for ep in episodes)
    action_dist: Dict[int, float] = {}
    for a in range(6):
        count = sum(ep.action_counts.get(a, 0) for ep in episodes)
        action_dist[a] = float(count / total_actions) if total_actions > 0 else 0.0

    return EvaluationSummary(
        episodes=n,
        mean_score=mean_score,
        median_score=median_score,
        std_score=std_score,
        se_score=se_score,
        min_score=min_score,
        max_score=max_score,
        mean_length=mean_length,
        median_length=median_length,
        std_length=std_length,
        total_bricks_destroyed=total_bricks,
        action_distribution=action_dist,
        step_cutoffs_count=step_cutoffs,
        metadata=metadata or {},
    )

