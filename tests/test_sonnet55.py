"""Automated test suite for Claude Sonnet 5.5 generated neural network weights."""

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
    SONNET55_DEFAULT_PATH,
    GPT6_DEFAULT_PATH,
    ASTRA_REFERENCE_PATH,
)

SONNET55_WEIGHTS_PATH = "experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz"
SONNET55_RAW_PATH = "experiments/strict_replication/generation_003_sonnet55/input/sonnet55_raw_response.txt"
EXPECTED_SONNET55_WEIGHTS_SHA256 = "2fb65f893087fa93eff087a3d8af2605c5c834b069c83bc8c8e08bf2fb60993b"
EXPECTED_SONNET55_RAW_SHA256 = "7b6a0461c2f775cc08604c65672f145aa3efc1adbb5ce51ac59244e036cb85ee"


@pytest.fixture(scope="module")
def sonnet55_weights():
    assert os.path.exists(SONNET55_WEIGHTS_PATH), f"Weights not found: {SONNET55_WEIGHTS_PATH}"
    return load_weights_npz(SONNET55_WEIGHTS_PATH)


def test_sonnet55_raw_integrity():
    """Verify SHA-256 hash of raw ingested Claude Sonnet 5.5 response."""
    assert os.path.exists(SONNET55_RAW_PATH)
    actual_hash = compute_file_sha256(SONNET55_RAW_PATH)
    assert actual_hash == EXPECTED_SONNET55_RAW_SHA256


def test_sonnet55_weights_integrity():
    """Verify SHA-256 hash of parsed Claude Sonnet 5.5 weights .npz."""
    assert os.path.exists(SONNET55_WEIGHTS_PATH)
    actual_hash = compute_file_sha256(SONNET55_WEIGHTS_PATH)
    assert actual_hash == EXPECTED_SONNET55_WEIGHTS_SHA256


def test_sonnet55_ast_parsing_reproducibility():
    """Verify SafeASTWeightParser reproduces Claude Sonnet 5.5 weights with 0 errors."""
    with open(SONNET55_RAW_PATH, "r") as f:
        content = f.read()

    parser = SafeASTWeightParser(strict=True)
    weights, report = parser.parse_text(content)

    assert report.block_found is True
    assert report.statements_applied == 65
    assert len(report.errors) == 0
    assert report.nonzero_total == 551
    assert report.nonzero["W1"] == 540
    assert report.nonzero["b1"] == 2
    assert report.nonzero["W2"] == 2
    assert report.nonzero["b2"] == 0
    assert report.nonzero["W3"] == 4
    assert report.nonzero["b3"] == 3


def test_sonnet55_tensor_dimensions_and_sparsity(sonnet55_weights):
    """Verify tensor dimensions and exact sparsity rate."""
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
        assert name in sonnet55_weights
        assert sonnet55_weights[name].shape == shape
        assert sonnet55_weights[name].dtype == np.float32
        total_params += sonnet55_weights[name].size
        nz_params += int(np.count_nonzero(sonnet55_weights[name]))

    assert total_params == 20486
    assert nz_params == 551
    sparsity = 1.0 - (nz_params / total_params)
    assert np.isclose(sparsity, 0.9731, atol=1e-3)


def test_sonnet55_symbolic_policy_equivalence(sonnet55_weights):
    """Verify exact equivalence between Claude's mathematical formula and the neural network forward pass."""
    net = MinAtarNetworkNumPy(sonnet55_weights)

    rng = np.random.RandomState(42)
    # Test across 200 diverse state combinations
    for _ in range(200):
        obs = np.zeros((10, 10, 6), dtype=np.float32)
        bx = rng.randint(0, 10)
        by = rng.randint(1, 9)
        obs[by, bx, 3] = 1.0  # Ball in Channel 3

        # Paddle at row 9 in Channel 0
        px = rng.randint(0, 10)
        obs[9, px, 0] = 1.0   # Paddle in Channel 0

        # Trail in Channel 4
        tx = rng.randint(0, 10)
        ty = rng.randint(1, 9)
        obs[ty, tx, 4] = 1.0  # Trail in Channel 4

        flat_obs = obs.reshape(-1)
        net_logits = net.forward(flat_obs)

        # Theoretical closed-form formula
        d = 2 * bx - tx - px
        h1_0 = np.tanh(6.0 * d - 3.0)
        h1_1 = np.tanh(6.0 * d + 3.0)
        h2_0 = np.tanh(3.0 * h1_0)
        h2_1 = np.tanh(3.0 * h1_1)

        expected_logits = np.zeros(6, dtype=np.float32)
        expected_logits[0] = 10.0 * (h2_1 - h2_0)
        expected_logits[1] = -10.0 * h2_1
        expected_logits[2] = -10.0
        expected_logits[3] = 10.0 * h2_0
        expected_logits[4] = -10.0
        expected_logits[5] = -10.0

        np.testing.assert_allclose(net_logits, expected_logits, atol=1e-5)


def test_sonnet55_wall_bounce_deficit_mechanism(sonnet55_weights):
    """Verify that when a ball hits a wall (last_x == ball_x == 9), Claude's formula yields d=0 and commands No-op."""
    net = MinAtarNetworkNumPy(sonnet55_weights)

    # State: ball at col 9, trail at col 9 (wall reflection frame), paddle at col 9
    # Channel 0: Paddle, Channel 3: Ball, Channel 4: Trail
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[5, 9, 3] = 1.0  # Ball at (y=5, x=9) in Channel 3
    obs[9, 9, 0] = 1.0  # Paddle at (y=9, x=9) in Channel 0
    obs[5, 9, 4] = 1.0  # Trail at (y=5, x=9) in Channel 4 -> MinAtar sets last_x=ball_x on bounce

    flat_obs = obs.reshape(-1)
    logits = net.forward(flat_obs)
    probs = softmax(logits)

    # In this state d = 2*9 - 9 - 9 = 0
    # Expected logits: Right ~ -10, Left ~ -10, No-op ~ +19.9
    assert logits[0] > 19.0, f"Expected strong No-op logit, got {logits[0]}"
    assert logits[1] < -9.0, f"Expected suppressed Left logit, got {logits[1]}"
    assert logits[3] < -9.0, f"Expected suppressed Right logit, got {logits[3]}"
    assert probs[0] > 0.9999, f"Expected No-op prob > 0.9999, got {probs[0]}"


def test_sonnet55_visualizer_state_5way():
    """Verify VisualizerState correctly steps 5 models in lockstep with Sonnet 5.5."""
    argon_path = "experiments/strict_replication/generation_001/output/argon_weights.npz"
    trained_path = "experiments/trained_baseline/trained_actor_weights.npz"

    state = VisualizerState(
        argon_weights_path=argon_path,
        trained_weights_path=trained_path,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_WEIGHTS_PATH,
        initial_seed=42,
    )

    assert state.has_sonnet55 is True
    assert state.has_gpt6 is True
    assert state.has_astra is True
    assert len(state.sonnet55_probs) == NUM_ACTIONS
    assert np.isclose(np.sum(state.sonnet55_probs), 1.0, atol=1e-6)

    # Step simulation
    dones = state.step()
    assert len(dones) == 5
    assert state.env_sonnet55.step_count == 1
    assert state.env_gpt6.step_count == 1
    assert state.env_astra.step_count == 1
    assert state.env_argon.step_count == 1
    assert state.env_trained.step_count == 1
