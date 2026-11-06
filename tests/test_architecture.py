"""Unit tests for neural network architecture, forward pass, and inference."""

import numpy as np
import pytest
import torch

from network.architecture import (
    MinAtarNetworkNumPy,
    MinAtarNetworkPyTorch,
    create_zero_weights,
    count_parameters,
    count_nonzero_parameters,
    TENSOR_SHAPES,
    TOTAL_PARAMETERS,
)
from network.inference import softmax, sample_action, PolicyAgent
from network.weight_validation import validate_weight_dict, compare_weight_dicts


def test_tensor_shapes_and_parameter_count():
    assert TOTAL_PARAMETERS == 20_486
    weights = create_zero_weights()
    assert count_parameters(weights) == 20_486

    assert weights["W1"].shape == (600, 32)
    assert weights["b1"].shape == (32,)
    assert weights["W2"].shape == (32, 32)
    assert weights["b2"].shape == (32,)
    assert weights["W3"].shape == (32, 6)
    assert weights["b3"].shape == (6,)


def test_zero_initialization():
    weights = create_zero_weights()
    stats = count_nonzero_parameters(weights)
    assert stats["total"] == 0
    for name in TENSOR_SHAPES:
        assert stats[name] == 0


def test_numpy_forward_pass_zero_network():
    net = MinAtarNetworkNumPy()
    x = np.ones(600, dtype=np.float32)
    logits = net.forward(x)

    assert logits.shape == (6,)
    # With zero weights, tanh(0) = 0, so logits = 0
    assert np.allclose(logits, np.zeros(6, dtype=np.float32))

    probs = softmax(logits)
    assert probs.shape == (6,)
    # Equal logits must produce uniform distribution: 1/6
    assert np.allclose(probs, np.full(6, 1.0 / 6.0, dtype=np.float32))


def test_numpy_forward_pass_batch():
    net = MinAtarNetworkNumPy()
    x = np.ones((5, 600), dtype=np.float32)
    logits = net.forward(x)
    assert logits.shape == (5, 6)


def test_pytorch_reference_equivalence():
    """Verify exact numerical agreement between NumPy and PyTorch implementations."""
    rng = np.random.RandomState(42)
    weights = {
        name: rng.randn(*shape).astype(np.float32)
        for name, shape in TENSOR_SHAPES.items()
    }

    net_np = MinAtarNetworkNumPy(weights)
    net_pt = MinAtarNetworkPyTorch()
    net_pt.load_from_numpy(weights)
    net_pt.eval()

    x_np = rng.randn(600).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    logits_np = net_np.forward(x_np)
    with torch.no_grad():
        logits_pt = net_pt.forward(x_pt).numpy()

    assert np.allclose(logits_np, logits_pt, atol=1e-5), (
        f"NumPy and PyTorch logits mismatch: max diff = {np.max(np.abs(logits_np - logits_pt))}"
    )


def test_softmax_numerical_stability():
    """Test that extreme logits do not trigger NaN or overflow in softmax."""
    large_logits = np.array([1000.0, -1000.0, 500.0, 0.0, -500.0, 250.0], dtype=np.float32)
    probs = softmax(large_logits)
    assert np.all(np.isfinite(probs))
    assert np.isclose(np.sum(probs), 1.0)
    assert np.argmax(probs) == 0


def test_policy_agent_action_sampling():
    net = MinAtarNetworkNumPy()
    rng = np.random.RandomState(42)
    agent = PolicyAgent(net, rng=rng)

    obs = np.zeros(600, dtype=np.float32)
    action, probs, logits = agent.get_action(obs)

    assert 0 <= action < 6
    assert probs.shape == (6,)
    assert logits.shape == (6,)
    assert np.isclose(np.sum(probs), 1.0)


def test_invalid_tensor_rejection():
    weights = create_zero_weights()
    # Missing tensor
    del weights["W1"]
    with pytest.raises(KeyError):
        MinAtarNetworkNumPy(weights)

    # Wrong shape
    weights = create_zero_weights()
    weights["W1"] = np.zeros((500, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        MinAtarNetworkNumPy(weights)

    # NaN value
    weights = create_zero_weights()
    weights["W1"][0, 0] = np.nan
    with pytest.raises(ValueError):
        MinAtarNetworkNumPy(weights)
