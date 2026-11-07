"""Automated test suite for GPT-6 Luna Extra High generated weights and policy mechanics."""

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
    GPT6_LUNA_DEFAULT_PATH,
    SONNET55_DEFAULT_PATH,
    DEEPSEEK_DEFAULT_PATH,
)

GPT6_LUNA_RAW_PATH = "experiments/strict_replication/generation_005_gpt6_luna_extra_high/input/gpt6_luna_raw_response.txt"
GPT6_LUNA_PROMPT_PATH = "experiments/strict_replication/generation_005_gpt6_luna_extra_high/input/experiment_5_prompt.txt"
GPT6_LUNA_WEIGHTS_PATH = "experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz"

EXPECTED_GPT6_LUNA_RAW_SHA256 = "e037cc069e092820da6e0f08e54a7fad82af4eabe396695b559311bac068601c"
EXPECTED_GPT6_LUNA_PROMPT_SHA256 = "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876"
EXPECTED_GPT6_LUNA_WEIGHTS_SHA256 = "1f275bbbecaf05bc16cc3d6c791768235ccc65c3c91285fe442836c2c27adce9"


@pytest.fixture(scope="module")
def gpt6_luna_weights():
    assert os.path.exists(GPT6_LUNA_WEIGHTS_PATH), f"Weights not found: {GPT6_LUNA_WEIGHTS_PATH}"
    return load_weights_npz(GPT6_LUNA_WEIGHTS_PATH)


def test_gpt6_luna_raw_integrity():
    """Verify cryptographic SHA-256 integrity of the raw ingested GPT-6 Luna response."""
    assert os.path.exists(GPT6_LUNA_RAW_PATH)
    actual_hash = compute_file_sha256(GPT6_LUNA_RAW_PATH)
    assert actual_hash == EXPECTED_GPT6_LUNA_RAW_SHA256


def test_gpt6_luna_prompt_integrity():
    """Verify cryptographic SHA-256 integrity of the canonical experiment prompt."""
    assert os.path.exists(GPT6_LUNA_PROMPT_PATH)
    actual_hash = compute_file_sha256(GPT6_LUNA_PROMPT_PATH)
    assert actual_hash == EXPECTED_GPT6_LUNA_PROMPT_SHA256


def test_gpt6_luna_weights_integrity():
    """Verify cryptographic SHA-256 integrity of parsed GPT-6 Luna weights .npz."""
    assert os.path.exists(GPT6_LUNA_WEIGHTS_PATH)
    actual_hash = compute_file_sha256(GPT6_LUNA_WEIGHTS_PATH)
    assert actual_hash == EXPECTED_GPT6_LUNA_WEIGHTS_SHA256


def test_gpt6_luna_ast_parsing_reproducibility():
    """Verify SafeASTWeightParser parses raw response with exact fidelity (0 errors, 39 statements)."""
    with open(GPT6_LUNA_RAW_PATH, "r") as f:
        content = f.read()

    parser = SafeASTWeightParser(strict=True)
    weights, report = parser.parse_text(content)

    assert report.block_found is True
    assert report.statements_applied == 39
    assert len(report.errors) == 0
    assert report.nonzero_total == 309
    assert report.nonzero["W1"] == 300
    assert report.nonzero["b1"] == 0
    assert report.nonzero["W2"] == 3
    assert report.nonzero["b2"] == 0
    assert report.nonzero["W3"] == 2
    assert report.nonzero["b3"] == 4


def test_gpt6_luna_tensor_dimensions_and_sparsity(gpt6_luna_weights):
    """Verify tensor dimensions and 98.49% parameter sparsity rate."""
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
        assert name in gpt6_luna_weights
        assert gpt6_luna_weights[name].shape == shape
        assert gpt6_luna_weights[name].dtype == np.float32
        total_params += gpt6_luna_weights[name].size
        nz_params += int(np.count_nonzero(gpt6_luna_weights[name]))

    assert total_params == 20486
    assert nz_params == 309
    sparsity = 1.0 - (nz_params / total_params)
    assert np.isclose(sparsity, 0.984917, atol=1e-4)


def test_gpt6_luna_numpy_pytorch_equivalence(gpt6_luna_weights):
    """Verify forward pass output equivalence between NumPy and PyTorch implementations."""
    import torch

    net_np = MinAtarNetworkNumPy(gpt6_luna_weights)
    net_pt = MinAtarNetworkPyTorch()
    net_pt.load_from_numpy(gpt6_luna_weights)

    rng = np.random.RandomState(42)
    for _ in range(50):
        obs = rng.uniform(0.0, 1.0, size=(600,)).astype(np.float32)
        out_np = net_np.forward(obs)
        with torch.no_grad():
            out_pt = net_pt(torch.from_numpy(obs)).numpy()
        np.testing.assert_allclose(out_np, out_pt, atol=1e-5)


def test_gpt6_luna_symbolic_policy_reconstruction(gpt6_luna_weights):
    """Verify mathematical policy mechanics: 2*ball_x - trail_x - paddle_x."""
    net = MinAtarNetworkNumPy(gpt6_luna_weights)

    # Test states across different configurations
    test_cases = [
        # (ball_x, trail_x, paddle_x, expected_d_sign)
        (5, 5, 5, 0),    # Aligned, no motion: d = 2(5) - 5 - 5 = 0
        (6, 5, 5, +1),   # Moving right (vx=+1), lead target = 7, paddle = 5 -> d = +2
        (4, 5, 5, -1),   # Moving left (vx=-1), lead target = 3, paddle = 5 -> d = -2
        (3, 4, 2, 0),    # Moving left (vx=-1), lead target = 2, paddle = 2 -> d = 0
        (7, 6, 8, 0),    # Moving right (vx=+1), lead target = 8, paddle = 8 -> d = 0
        (8, 7, 3, +1),   # Lead target = 9, paddle = 3 -> d = +6 (paddle far left, must move right)
        (2, 3, 7, -1),   # Lead target = 1, paddle = 7 -> d = -6 (paddle far right, must move left)
    ]

    for bx, tx, px, expected_dir in test_cases:
        obs = np.zeros((10, 10, 6), dtype=np.float32)
        obs[5, bx, 3] = 1.0  # Ball at row 5, col bx
        obs[4, tx, 4] = 1.0  # Trail at row 4, col tx
        obs[9, px, 0] = 1.0  # Paddle at row 9, col px

        flat_obs = obs.reshape(-1)
        logits = net.forward(flat_obs)
        probs = softmax(logits)

        # Inert actions (Up=2, Down=4, Fire=5) must be heavily suppressed
        for act in (2, 4, 5):
            assert logits[act] == -6.0, f"Action {act} logit should be -6.0, got {logits[act]}"
            assert probs[act] < 0.001, f"Action {act} prob should be < 0.1%, got {probs[act]}"

        if expected_dir == 0:
            # When aligned, No-Op logit is +1.5, Left and Right are ~0
            assert logits[0] > logits[1], f"No-op logit ({logits[0]}) should exceed Left ({logits[1]})"
            assert logits[0] > logits[3], f"No-op logit ({logits[0]}) should exceed Right ({logits[3]})"
            assert probs[0] > 0.60, f"No-op prob should dominate (~69%), got {probs[0]}"
        elif expected_dir > 0:
            # Need to move right: Right logit must exceed Left and No-op
            assert logits[3] > logits[1], f"Right logit ({logits[3]}) should exceed Left ({logits[1]})"
            assert probs[3] > probs[1], f"Right prob ({probs[3]}) should exceed Left ({probs[1]})"
        else:
            # Need to move left: Left logit must exceed Right and No-op
            assert logits[1] > logits[3], f"Left logit ({logits[1]}) should exceed Right ({logits[3]})"
            assert probs[1] > probs[3], f"Left prob ({probs[1]}) should exceed Right ({probs[3]})"


def test_gpt6_luna_visualizer_state_7way():
    """Verify VisualizerState operates correctly with all seven primary models in lockstep."""
    argon_path = "experiments/strict_replication/generation_001/output/argon_weights.npz"
    trained_path = "experiments/trained_baseline/trained_actor_weights.npz"

    state = VisualizerState(
        argon_weights_path=argon_path,
        trained_weights_path=trained_path,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        gpt6_luna_weights_path=GPT6_LUNA_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_DEFAULT_PATH,
        deepseek_weights_path=DEEPSEEK_DEFAULT_PATH,
        initial_seed=42,
    )

    assert state.has_gpt6_luna is True
    assert state.has_deepseek is True
    assert state.has_sonnet55 is True
    assert state.has_gpt6 is True
    assert state.has_astra is True
    assert len(state.gpt6_luna_probs) == NUM_ACTIONS
    assert np.isclose(np.sum(state.gpt6_luna_probs), 1.0, atol=1e-6)

    # Step simulation
    dones = state.step()
    assert len(dones) == 7
    assert state.env_gpt6_luna.step_count == 1
    assert state.env_deepseek.step_count == 1
    assert state.env_sonnet55.step_count == 1
    assert state.env_gpt6.step_count == 1
    assert state.env_astra.step_count == 1
    assert state.env_argon.step_count == 1
    assert state.env_trained.step_count == 1
