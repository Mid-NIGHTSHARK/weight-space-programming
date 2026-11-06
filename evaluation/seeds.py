"""Standardized seed generation for evaluation protocols."""

from typing import List


def generate_evaluation_seeds(num_episodes: int = 1000, base_seed: int = 42) -> List[int]:
    """Generate a deterministic sequence of random seeds for evaluation episodes."""
    import numpy as np
    rng = np.random.RandomState(base_seed)
    # Generate unique positive 31-bit integers
    seeds = rng.randint(1, 2**31 - 1, size=num_episodes).tolist()
    return seeds
