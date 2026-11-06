"""Validation and analysis routines for neural network weight tensors."""

from typing import Dict, Any, Tuple
import numpy as np

from network.architecture import TENSOR_SHAPES, TOTAL_PARAMETERS


def validate_weight_dict(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Perform comprehensive structural and numerical validation of a weight dictionary."""
    if not isinstance(weights, dict):
        raise TypeError(f"weights must be a dictionary, got {type(weights)}")

    missing = set(TENSOR_SHAPES) - set(weights)
    if missing:
        raise KeyError(f"Missing required weight tensors: {missing}")

    unexpected = set(weights) - set(TENSOR_SHAPES)
    if unexpected:
        raise KeyError(f"Unexpected weight tensors: {unexpected}")

    stats: Dict[str, Any] = {
        "nonzero": {},
        "max_abs": {},
        "min": {},
        "max": {},
        "total_parameters": 0,
        "nonzero_total": 0,
    }

    for name, expected_shape in TENSOR_SHAPES.items():
        arr = weights[name]
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Tensor {name} must be numpy ndarray, got {type(arr)}")
        if arr.shape != expected_shape:
            raise ValueError(
                f"Tensor {name} shape mismatch: expected {expected_shape}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Tensor {name} contains NaN or Inf values")

        num_params = arr.size
        nonzero_count = int(np.count_nonzero(arr))
        stats["total_parameters"] += num_params
        stats["nonzero_total"] += nonzero_count
        stats["nonzero"][name] = nonzero_count
        stats["max_abs"][name] = float(np.max(np.abs(arr))) if num_params > 0 else 0.0
        stats["min"][name] = float(np.min(arr)) if num_params > 0 else 0.0
        stats["max"][name] = float(np.max(arr)) if num_params > 0 else 0.0

    assert stats["total_parameters"] == TOTAL_PARAMETERS, (
        f"Parameter count mismatch: expected {TOTAL_PARAMETERS}, got {stats['total_parameters']}"
    )
    stats["sparsity_pct"] = (
        (stats["total_parameters"] - stats["nonzero_total"]) / stats["total_parameters"]
    ) * 100.0

    return stats


def compare_weight_dicts(
    weights_a: Dict[str, np.ndarray],
    weights_b: Dict[str, np.ndarray],
    tolerance: float = 1e-6,
) -> Tuple[bool, Dict[str, float]]:
    """Compare two weight dictionaries.
    
    Returns:
        is_match: True if all tensors match within tolerance.
        diffs: Maximum absolute difference per tensor.
    """
    diffs = {}
    is_match = True

    for name in TENSOR_SHAPES:
        a = weights_a[name]
        b = weights_b[name]
        max_diff = float(np.max(np.abs(a - b)))
        diffs[name] = max_diff
        if max_diff > tolerance:
            is_match = False

    return is_match, diffs

