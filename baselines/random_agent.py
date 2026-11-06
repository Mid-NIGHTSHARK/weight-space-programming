"""Random baseline agent for MinAtar Breakout."""

import numpy as np


class RandomAgent:
    """Agent that samples uniformly at random from the 6 MinAtar actions."""

    def __init__(self, rng: np.random.RandomState = None):
        self.rng = rng if rng is not None else np.random.RandomState()

    def get_action(self, obs: np.ndarray) -> int:
        return int(self.rng.choice(6))

    def __call__(self, obs: np.ndarray, rng: np.random.RandomState) -> int:
        return int(rng.choice(6))
