"""Automated test suite for DeepSeek generated neural network weights and forensic audit."""

import os
import sys
import numpy as np
import pytest

from environment.breakout import MinAtarBreakoutEnv
from environment.actions import NUM_ACTIONS
from network.architecture import MinAtarNetworkNumPy, MinAtarNetworkPyTorch
from network.inference import PolicyAgent, softmax
from network.weight_loader import load_weights_npz, compute_file_sha256
from network.weight_parser import SafeASTWeightParser
from visualize_comparison import (
    VisualizerState,
    BreakoutVisualizerGUI,
    ASTRA_REFERENCE_PATH,
    GPT6_DEFAULT_PATH,
    SONNET55_DEFAULT_PATH,
    DEEPSEEK_DEFAULT_PATH,
)

DEEPSEEK_RAW_PATH = "experiments/strict_replication/generation_004_deepseek/input/deepseek_raw_response.txt"
DEEPSEEK_PROMPT_PATH = "experiments/strict_replication/generation_004_deepseek/input/experiment_5_prompt.txt"
DEEPSEEK_WEIGHTS_PATH = "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz"

EXPECTED_DEEPSEEK_RAW_SHA256 = "e16d5a6fc4cfff6aa0184211d983bf53274c34b04af6f0501d2f91ed0d1e9de0"
EXPECTED_DEEPSEEK_PROMPT_SHA256 = "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876"
EXPECTED_DEEPSEEK_WEIGHTS_SHA256 = "5ea84b9b5d04292aef19f5b4e12bf8c553145714017753b9a843625f19d969d9"


@pytest.fixture(scope="module")
def deepseek_weights():
    assert os.path.exists(DEEPSEEK_WEIGHTS_PATH), f"Weights not found: {DEEPSEEK_WEIGHTS_PATH}"
    return load_weights_npz(DEEPSEEK_WEIGHTS_PATH)


def test_deepseek_raw_integrity():
    """Verify cryptographic SHA-256 integrity of the raw ingested DeepSeek response."""
    assert os.path.exists(DEEPSEEK_RAW_PATH)
    actual_hash = compute_file_sha256(DEEPSEEK_RAW_PATH)
    assert actual_hash == EXPECTED_DEEPSEEK_RAW_SHA256


def test_deepseek_prompt_integrity():
    """Verify cryptographic SHA-256 integrity of the canonical experiment prompt."""
    assert os.path.exists(DEEPSEEK_PROMPT_PATH)
    actual_hash = compute_file_sha256(DEEPSEEK_PROMPT_PATH)
    assert actual_hash == EXPECTED_DEEPSEEK_PROMPT_SHA256


def test_deepseek_weights_integrity():
    """Verify cryptographic SHA-256 integrity of parsed DeepSeek weights .npz."""
    assert os.path.exists(DEEPSEEK_WEIGHTS_PATH)
    actual_hash = compute_file_sha256(DEEPSEEK_WEIGHTS_PATH)
    assert actual_hash == EXPECTED_DEEPSEEK_WEIGHTS_SHA256


def test_deepseek_ast_parsing_reproducibility():
    """Verify SafeASTWeightParser parses raw response with exact fidelity (0 errors, 97 statements)."""
    with open(DEEPSEEK_RAW_PATH, "r") as f:
        content = f.read()

    parser = SafeASTWeightParser(strict=True)
    weights, report = parser.parse_text(content)

    assert report.block_found is True
    assert report.statements_applied == 97
    assert len(report.errors) == 0
    assert report.nonzero_total == 786
    assert report.nonzero["W1"] == 569
    assert report.nonzero["b1"] == 12
    assert report.nonzero["W2"] == 80
    assert report.nonzero["b2"] == 20
    assert report.nonzero["W3"] == 100
    assert report.nonzero["b3"] == 5


def test_deepseek_tensor_dimensions_and_sparsity(deepseek_weights):
    """Verify tensor dimensions and 96.16% parameter sparsity rate."""
    expected_shapes = {
        "W1": (600, 32),
        "b1": (32,),
        "W2": (32, 32),
        "b2": (32,),
        "W3": (32, 6),
        "b3": (6,),
    }
    total_params = 0
    nz_params = 0
    for name, shape in expected_shapes.items():
        assert name in deepseek_weights
        assert deepseek_weights[name].shape == shape
        assert deepseek_weights[name].dtype == np.float32
        total_params += deepseek_weights[name].size
        nz_params += int(np.count_nonzero(deepseek_weights[name]))

    assert total_params == 20486
    assert nz_params == 786
    sparsity = 1.0 - (nz_params / total_params)
    assert np.isclose(sparsity, 0.961632, atol=1e-4)


def test_deepseek_numpy_pytorch_equivalence(deepseek_weights):
    """Verify forward pass output equivalence between NumPy and PyTorch implementations."""
    import torch
    net_np = MinAtarNetworkNumPy(deepseek_weights)
    net_pt = MinAtarNetworkPyTorch()
    net_pt.load_from_numpy(deepseek_weights)

    rng = np.random.RandomState(42)
    for _ in range(50):
        obs = rng.uniform(0.0, 1.0, size=(600,)).astype(np.float32)
        out_np = net_np.forward(obs)
        with torch.no_grad():
            out_pt = net_pt(torch.from_numpy(obs)).numpy()
        np.testing.assert_allclose(out_np, out_pt, atol=1e-4)


def test_deepseek_symbolic_policy_reconstruction(deepseek_weights):
    """Verify that DeepSeek's symbolic derivation matches MinAtarNetworkNumPy forward pass."""
    net = MinAtarNetworkNumPy(deepseek_weights)
    W1 = deepseek_weights["W1"]
    b1 = deepseek_weights["b1"]
    W2 = deepseek_weights["W2"]
    b2 = deepseek_weights["b2"]
    W3 = deepseek_weights["W3"]
    b3 = deepseek_weights["b3"]

    rng = np.random.RandomState(42)
    for _ in range(50):
        obs = np.zeros((10, 10, 6), dtype=np.float32)
        bx = rng.randint(0, 10)
        by = rng.randint(1, 9)
        obs[by, bx, 3] = 1.0  # Ball
        px = rng.randint(0, 10)
        obs[9, px, 0] = 1.0   # Paddle
        tx = rng.randint(0, 10)
        ty = rng.randint(1, 9)
        obs[ty, tx, 4] = 1.0  # Trail

        flat_obs = obs.reshape(-1)
        net_logits = net.forward(flat_obs)

        # Symbolic calculation:
        # Layer 1
        h1 = np.tanh(np.dot(flat_obs, W1) + b1)
        # Layer 2
        h2 = np.tanh(np.dot(h1, W2) + b2)
        # Layer 3
        expected_logits = np.dot(h2, W3) + b3

        np.testing.assert_allclose(net_logits, expected_logits, atol=1e-5)


def test_deepseek_inverted_tanh_suppression_flaw(deepseek_weights):
    """Forensic verification: Inverted tanh suppression causes massive positive logits for Up, Down, Fire."""
    net = MinAtarNetworkNumPy(deepseek_weights)
    rng = np.random.RandomState(42)

    # Test across 50 valid Breakout game states
    for _ in range(50):
        obs = np.zeros((10, 10, 6), dtype=np.float32)
        bx = rng.randint(0, 10)
        by = rng.randint(1, 9)
        obs[by, bx, 3] = 1.0  # Ball
        px = rng.randint(0, 10)
        obs[9, px, 0] = 1.0   # Paddle
        tx = rng.randint(0, 10)
        ty = rng.randint(1, 9)
        obs[ty, tx, 4] = 1.0  # Trail

        logits = net.forward(obs.reshape(-1))
        probs = softmax(logits)

        # Logits for Up (2), Down (4), Fire (5) must be > +1700
        assert logits[2] > 1700.0, f"Up logit should be > 1700, got {logits[2]}"
        assert logits[4] > 1700.0, f"Down logit should be > 1700, got {logits[4]}"
        assert logits[5] > 1700.0, f"Fire logit should be > 1700, got {logits[5]}"

        # Logits for Left (1) and Right (3) must be small (< 20)
        assert logits[1] < 20.0, f"Left logit should be < 20, got {logits[1]}"
        assert logits[3] < 20.0, f"Right logit should be < 20, got {logits[3]}"
        assert logits[0] == 0.0, f"No-op logit should be 0, got {logits[0]}"

        # Probabilities: Up, Down, Fire each get ~33.33%; Left, Right, No-op get 0.0%
        assert np.isclose(probs[1], 0.0, atol=1e-7), f"Left prob should be 0.0, got {probs[1]}"
        assert np.isclose(probs[3], 0.0, atol=1e-7), f"Right prob should be 0.0, got {probs[3]}"
        assert np.isclose(probs[0], 0.0, atol=1e-7), f"No-op prob should be 0.0, got {probs[0]}"

        inert_sum = probs[2] + probs[4] + probs[5]
        assert np.isclose(inert_sum, 1.0, atol=1e-5), f"Inert actions should sum to 1.0, got {inert_sum}"
        for act in (2, 4, 5):
            assert 0.30 <= probs[act] <= 0.36, f"Action {act} prob should be ~0.333, got {probs[act]}"


def test_deepseek_wall_bounce_direction_detector(deepseek_weights):
    """Verify DeepSeek's direction units 10 and 11 on wall-bounce frames."""
    net = MinAtarNetworkNumPy(deepseek_weights)

    # 1. Right wall bounce: ball at col 9, trail at col 9
    obs_right_wall = np.zeros((10, 10, 6), dtype=np.float32)
    obs_right_wall[5, 9, 3] = 1.0  # Ball at (5, 9)
    obs_right_wall[5, 9, 4] = 1.0  # Trail at (5, 9)
    obs_right_wall[9, 4, 0] = 1.0  # Paddle at col 4

    h1_right = np.tanh(obs_right_wall.reshape(-1) @ deepseek_weights["W1"] + deepseek_weights["b1"])
    # Unit 11 is D_minus: on col 9 bounce, it should activate strongly (~+1.0)
    assert h1_right[11] > 0.99, f"D_minus should be +1 on right wall bounce, got {h1_right[11]}"
    # Unit 10 is D_plus: should be inactive (-1.0)
    assert h1_right[10] < -0.99, f"D_plus should be -1 on right wall bounce, got {h1_right[10]}"

    # 2. Left wall bounce: ball at col 0, trail at col 0
    obs_left_wall = np.zeros((10, 10, 6), dtype=np.float32)
    obs_left_wall[5, 0, 3] = 1.0  # Ball at (5, 0)
    obs_left_wall[5, 0, 4] = 1.0  # Trail at (5, 0)
    obs_left_wall[9, 4, 0] = 1.0  # Paddle at col 4

    h1_left = np.tanh(obs_left_wall.reshape(-1) @ deepseek_weights["W1"] + deepseek_weights["b1"])
    # Unit 10 is D_plus: on col 0 bounce, it should activate strongly (~+1.0)
    assert h1_left[10] > 0.99, f"D_plus should be +1 on left wall bounce, got {h1_left[10]}"
    # Unit 11 is D_minus: should be inactive (-1.0)
    assert h1_left[11] < -0.99, f"D_minus should be -1 on left wall bounce, got {h1_left[11]}"


def test_deepseek_visualizer_state_6way():
    """Verify VisualizerState operates correctly with all six models in lockstep."""
    argon_path = "experiments/strict_replication/generation_001/output/argon_weights.npz"
    trained_path = "experiments/trained_baseline/trained_actor_weights.npz"

    state = VisualizerState(
        argon_weights_path=argon_path,
        trained_weights_path=trained_path,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_DEFAULT_PATH,
        deepseek_weights_path=DEEPSEEK_WEIGHTS_PATH,
        initial_seed=42,
    )

    assert state.has_deepseek is True
    assert state.has_sonnet55 is True
    assert state.has_gpt6 is True
    assert state.has_astra is True
    assert len(state.deepseek_probs) == NUM_ACTIONS
    assert np.isclose(np.sum(state.deepseek_probs), 1.0, atol=1e-6)

    # Step simulation
    dones = state.step()
    assert len(dones) == 6
    assert state.env_deepseek.step_count == 1
    assert state.env_sonnet55.step_count == 1
    assert state.env_gpt6.step_count == 1
    assert state.env_astra.step_count == 1
    assert state.env_argon.step_count == 1
    assert state.env_trained.step_count == 1
