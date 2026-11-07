"""Neural network architecture specifications for MinAtar Breakout policy.

Architecture: 600 -> 32 -> 32 -> 6 (tanh activations for hidden layers, linear output).
Total parameters: 20,486.
"""

from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

# Parameter shapes and names matching the paper and hand.npz
TENSOR_SHAPES: Dict[str, Tuple[int, ...]] = {
    "W1": (600, 32),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (32, 6),
    "b3": (6,),
}

# Mapping between short names and hand.npz archive keys
NPZ_KEY_MAP: Dict[str, str] = {
    "W1": "actor/dense1/kernel",
    "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel",
    "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel",
    "b3": "actor/logits/bias",
}

REVERSE_NPZ_KEY_MAP: Dict[str, str] = {v: k for k, v in NPZ_KEY_MAP.items()}

TOTAL_PARAMETERS: int = 600 * 32 + 32 + 32 * 32 + 32 + 32 * 6 + 6  # 20,486


def create_zero_weights() -> Dict[str, np.ndarray]:
    """Create zero-initialized weight dictionary with exact shapes and float32 dtype."""
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in TENSOR_SHAPES.items()
    }


def count_parameters(weights: Dict[str, np.ndarray]) -> int:
    """Calculate the total number of parameters in a weight dictionary."""
    return sum(arr.size for arr in weights.values())


def count_nonzero_parameters(weights: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Calculate the number of nonzero parameters per tensor and in total."""
    stats = {name: int(np.count_nonzero(arr)) for name, arr in weights.items()}
    stats["total"] = sum(v for k, v in stats.items() if k != "total")
    return stats


class MinAtarNetworkNumPy:
    """Authoritative NumPy implementation of the 600-32-32-6 policy network."""

    def __init__(self, weights: Dict[str, np.ndarray] = None):
        if weights is None:
            self.weights = create_zero_weights()
        else:
            self.weights = {k: v.copy().astype(np.float32) for k, v in weights.items()}
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate all tensor shapes, types, and finiteness."""
        for name, expected_shape in TENSOR_SHAPES.items():
            if name not in self.weights:
                raise KeyError(f"Missing parameter tensor: {name}")
            arr = self.weights[name]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Tensor {name} must be numpy ndarray, got {type(arr)}")
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Tensor {name} shape mismatch: expected {expected_shape}, got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Tensor {name} contains non-finite values (NaN or Inf)")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward inference pass.
        
        Args:
            x: Input array of shape (600,) or (batch_size, 600).
            
        Returns:
            logits: Output array of shape (6,) or (batch_size, 6).
        """
        W1 = self.weights["W1"]
        b1 = self.weights["b1"]
        W2 = self.weights["W2"]
        b2 = self.weights["b2"]
        W3 = self.weights["W3"]
        b3 = self.weights["b3"]

        # Use np.dot for mathematically identical, BLAS-stable matmul
        h1 = np.tanh(np.dot(x, W1) + b1)
        h2 = np.tanh(np.dot(h1, W2) + b2)
        logits = np.dot(h2, W3) + b3
        return logits


class MinAtarNetworkPyTorch(nn.Module):
    """PyTorch reference model used for cross-validation and baseline comparison."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(600, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 6, bias=True)
        self.act = nn.Tanh()
        self.reset_parameters_to_zero()

    def reset_parameters_to_zero(self) -> None:
        """Initialize all weights and biases to zero."""
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            self.fc3.weight.zero_()
            self.fc3.bias.zero_()

    def load_from_numpy(self, weights: Dict[str, np.ndarray]) -> None:
        """Load weights from NumPy dict.
        
        Note: PyTorch nn.Linear stores weights as (out_features, in_features),
        whereas the paper/NumPy weights are stored as (in_features, out_features).
        We transpose kernel weights accordingly for PyTorch.
        """
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(weights["W1"].T))
            self.fc1.bias.copy_(torch.from_numpy(weights["b1"]))
            self.fc2.weight.copy_(torch.from_numpy(weights["W2"].T))
            self.fc2.bias.copy_(torch.from_numpy(weights["b2"]))
            self.fc3.weight.copy_(torch.from_numpy(weights["W3"].T))
            self.fc3.bias.copy_(torch.from_numpy(weights["b3"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        logits = self.fc3(h2)
        return logits

