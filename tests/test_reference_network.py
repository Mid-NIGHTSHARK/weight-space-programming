"""Unit tests for the reference network (GPT-6 Astra Breakout weights)."""

import os
import numpy as np
import pytest
import torch

from network.weight_loader import load_weights_npz, compute_file_sha256
from network.weight_validation import validate_weight_dict
from network.architecture import (
    MinAtarNetworkNumPy,
    MinAtarNetworkPyTorch,
    TOTAL_PARAMETERS,
)

REFERENCE_PATH = "experiments/reference/hand.npz"
EXPECTED_SHA256 = "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"


def test_reference_file_exists_and_hash():
    assert os.path.exists(REFERENCE_PATH), f"Missing {REFERENCE_PATH}"
    actual_hash = compute_file_sha256(REFERENCE_PATH)
    assert actual_hash == EXPECTED_SHA256


def test_reference_tensor_properties():
    weights = load_weights_npz(REFERENCE_PATH)
    stats = validate_weight_dict(weights)

    assert stats["total_parameters"] == TOTAL_PARAMETERS
    assert stats["nonzero_total"] == 815
    assert np.isclose(stats["sparsity_pct"], 96.02, atol=0.01)

    # Check exact per-tensor nonzero counts from paper apply.json
    assert stats["nonzero"]["W1"] == 769
    assert stats["nonzero"]["b1"] == 1
    assert stats["nonzero"]["W2"] == 24
    assert stats["nonzero"]["b2"] == 8
    assert stats["nonzero"]["W3"] == 8
    assert stats["nonzero"]["b3"] == 5

    # Check max abs values
    assert np.isclose(stats["max_abs"]["W1"], 180.0)
    assert np.isclose(stats["max_abs"]["b1"], 5.0)
    assert np.isclose(stats["max_abs"]["W2"], 1000.0)
    assert np.isclose(stats["max_abs"]["b2"], 415.0)
    assert np.isclose(stats["max_abs"]["W3"], 30.0)
    assert np.isclose(stats["max_abs"]["b3"], 100.0)


def test_reference_numpy_pytorch_equivalence():
    weights = load_weights_npz(REFERENCE_PATH)
    net_np = MinAtarNetworkNumPy(weights)
    net_pt = MinAtarNetworkPyTorch()
    net_pt.load_from_numpy(weights)
    net_pt.eval()

    rng = np.random.RandomState(42)
    x = rng.randn(600).astype(np.float32)

    logits_np = net_np.forward(x)
    with torch.no_grad():
        logits_pt = net_pt.forward(torch.from_numpy(x)).numpy()

    assert np.allclose(logits_np, logits_pt, atol=1e-4)

