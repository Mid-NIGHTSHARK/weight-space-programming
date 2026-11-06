"""Zero-weight baseline agent for MinAtar Breakout."""

import numpy as np
from network.architecture import MinAtarNetworkNumPy, create_zero_weights
from network.inference import PolicyAgent


class ZeroWeightAgent:
    """Agent using a policy network with all weights and biases initialized to zero.
    
    Zero weights produce zero logits for all 6 actions, which translates to a uniform
    probability distribution (1/6 for each action) under softmax. Actions are sampled
    stochastically, interacting with sticky actions and game physics.
    """

    def __init__(self):
        self.network = MinAtarNetworkNumPy(create_zero_weights())
        self.agent = PolicyAgent(self.network)

    def __call__(self, obs: np.ndarray, rng: np.random.RandomState) -> int:
        self.agent.rng = rng
        action, probs, logits = self.agent.get_action(obs)
        return action
