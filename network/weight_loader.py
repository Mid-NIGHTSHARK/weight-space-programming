"""Utilities for loading, saving, and verifying weight archives."""

import hashlib
import os
from typing import Dict, Tuple, Optional
import numpy as np

from network.architecture import (
    NPZ_KEY_MAP,
    REVERSE_NPZ_KEY_MAP,
    TENSOR_SHAPES,
    create_zero_weights,
)


def compute_file_sha256(filepath: str) -> str:
    """Calculate the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_weights_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load neural network weights from an .npz archive.
    
    Supports both paper keys ('actor/dense1/kernel') and short keys ('W1').
    Returns a dictionary keyed by standard short names ('W1', 'b1', 'W2', 'b2', 'W3', 'b3').
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weight archive not found: {filepath}")

    weights = create_zero_weights()
    with np.load(filepath) as data:
        files = set(data.files)
        # Check if archive uses paper keys
        uses_paper_keys = any(k in NPZ_KEY_MAP.values() for k in files)

        for short_name, shape in TENSOR_SHAPES.items():
            if uses_paper_keys:
                key = NPZ_KEY_MAP[short_name]
            else:
                key = short_name

            if key not in files:
                raise KeyError(f"Archive missing expected key: {key} (keys present: {list(files)})")

            arr = data[key].astype(np.float32)
            if arr.shape != shape:
                raise ValueError(f"Shape mismatch for {key}: expected {shape}, got {arr.shape}")
            weights[short_name] = arr

    return weights


def save_weights_npz(
    filepath: str,
    weights: Dict[str, np.ndarray],
    use_paper_keys: bool = True,
) -> str:
    """Save weights dictionary to an .npz archive.
    
    Returns the SHA-256 hash of the generated file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    export_dict = {}
    for short_name, arr in weights.items():
        if short_name not in TENSOR_SHAPES:
            raise KeyError(f"Unknown weight tensor: {short_name}")
        arr_f32 = np.asarray(arr, dtype=np.float32)
        if arr_f32.shape != TENSOR_SHAPES[short_name]:
            raise ValueError(
                f"Shape mismatch for {short_name}: expected {TENSOR_SHAPES[short_name]}, got {arr_f32.shape}"
            )
        key = NPZ_KEY_MAP[short_name] if use_paper_keys else short_name
        export_dict[key] = arr_f32

    np.savez(filepath, **export_dict)
    return compute_file_sha256(filepath)
