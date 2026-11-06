"""Inference and action sampling engine for MinAtar Breakout policy."""

from typing import Optional, Tuple
import numpy as np

from network.architecture import MinAtarNetworkNumPy


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax probabilities over logits.
    
    Formula: exp(z - max(z)) / sum(exp(z - max(z)))
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = logits / temperature
    # Shift for numerical stability
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs


def sample_action(
    probs: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """Sample a discrete action from the softmax probability distribution.
    
    Argmax is explicitly avoided to preserve stochastic evaluation fidelity.
    """
    if rng is None:
        rng = np.random.RandomState()
    action = rng.choice(len(probs), p=probs)
    return int(action)


class PolicyAgent:
    """Complete inference policy agent that evaluates observations and returns sampled actions."""

    def __init__(
        self,
        network: MinAtarNetworkNumPy,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.network = network
        self.rng = rng if rng is not None else np.random.RandomState()

    def get_action(self, obs: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """Run forward inference and sample an action.
        
        Args:
            obs: 600-element observation vector.
            
        Returns:
            action: Selected discrete action in [0, 5].
            probs: 6-element action probability distribution.
            logits: 6-element raw logits.
        """
        logits = self.network.forward(obs)
        probs = softmax(logits)
        action = sample_action(probs, rng=self.rng)
        return action, probs, logits
