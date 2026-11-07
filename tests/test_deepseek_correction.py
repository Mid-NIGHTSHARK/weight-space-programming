"""Automated test suite for Phase 3D: DeepSeek Controlled Weight Repair and Evaluation."""

import os
import json
import numpy as np
import pytest

from environment.breakout import MinAtarBreakoutEnv
from environment.actions import MinAtarAction, ACTION_NAMES, NUM_ACTIONS
from network.architecture import MinAtarNetworkNumPy, MinAtarNetworkPyTorch
from network.inference import PolicyAgent, softmax, sample_action
from network.weight_loader import load_weights_npz, compute_file_sha256
from visualize_comparison import (
    VisualizerState,
    load_verified_benchmark_registry,
    ASTRA_REFERENCE_PATH,
    GPT6_DEFAULT_PATH,
    SONNET55_DEFAULT_PATH,
    DEEPSEEK_DEFAULT_PATH,
    DEEPSEEK_CORRECTED_DEFAULT_PATH,
)

ORIGINAL_RAW_PATH = "experiments/strict_replication/generation_004_deepseek/input/deepseek_raw_response.txt"
ORIGINAL_WEIGHTS_PATH = "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz"
CORRECTED_WEIGHTS_PATH = "experiments/controlled_interventions/deepseek_corrected/output/deepseek_corrected_weights.npz"
MANIFEST_PATH = "experiments/controlled_interventions/deepseek_corrected/intervention_manifest.json"

EXPECTED_ORIGINAL_RAW_SHA256 = "e16d5a6fc4cfff6aa0184211d983bf53274c34b04af6f0501d2f91ed0d1e9de0"
EXPECTED_ORIGINAL_WEIGHTS_SHA256 = "5ea84b9b5d04292aef19f5b4e12bf8c553145714017753b9a843625f19d969d9"
EXPECTED_CORRECTED_WEIGHTS_SHA256 = "21078c485156850253744ed0c2e1efd54361cf06f603a5ad2fd32ffea3c1c8bc"


@pytest.fixture(scope="module")
def original_weights():
    assert os.path.exists(ORIGINAL_WEIGHTS_PATH), f"Missing original weights: {ORIGINAL_WEIGHTS_PATH}"
    return load_weights_npz(ORIGINAL_WEIGHTS_PATH)


@pytest.fixture(scope="module")
def corrected_weights():
    assert os.path.exists(CORRECTED_WEIGHTS_PATH), f"Missing corrected weights: {CORRECTED_WEIGHTS_PATH}"
    return load_weights_npz(CORRECTED_WEIGHTS_PATH)


def test_deepseek_original_artifacts_unmodified():
    """Verify that original DeepSeek artifacts remain cryptographically pristine and unmodified."""
    assert compute_file_sha256(ORIGINAL_RAW_PATH) == EXPECTED_ORIGINAL_RAW_SHA256
    assert compute_file_sha256(ORIGINAL_WEIGHTS_PATH) == EXPECTED_ORIGINAL_WEIGHTS_SHA256


def test_deepseek_corrected_weights_sha256():
    """Verify cryptographic hash of the controlled intervention weights artifact."""
    assert compute_file_sha256(CORRECTED_WEIGHTS_PATH) == EXPECTED_CORRECTED_WEIGHTS_SHA256


def test_intervention_manifest_integrity():
    """Verify intervention manifest accurately documents the exact 60-parameter modification."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    assert manifest["base_artifacts"]["weights_sha256"] == EXPECTED_ORIGINAL_WEIGHTS_SHA256
    assert manifest["output_artifacts"]["weights_sha256"] == EXPECTED_CORRECTED_WEIGHTS_SHA256
    assert manifest["intervention_specification"]["total_changed_scalars"] == 60
    assert manifest["parameter_metrics"]["base_nonzero_parameters"] == 786
    assert manifest["parameter_metrics"]["corrected_nonzero_parameters"] == 726


def test_exact_sixty_parameter_diff(original_weights, corrected_weights):
    """Verify that ONLY W3[0:20, 2], W3[0:20, 4], and W3[0:20, 5] were altered (exactly 60 scalars)."""
    # W1, b1, W2, b2, b3 must be bitwise identical
    for key in ["W1", "b1", "W2", "b2", "b3"]:
        np.testing.assert_array_equal(
            original_weights[key],
            corrected_weights[key],
            err_msg=f"Tensor {key} was unexpectedly modified during intervention!",
        )

    # W3 diff must be exactly the 60 specified elements
    w3_orig = original_weights["W3"]
    w3_corr = corrected_weights["W3"]
    diff_mask = w3_orig != w3_corr
    diff_count = int(np.count_nonzero(diff_mask))
    assert diff_count == 60, f"Expected exactly 60 altered parameters in W3, found {diff_count}"

    # Verify which elements were changed
    diff_indices = np.argwhere(diff_mask)
    for r, c in diff_indices:
        assert 0 <= r < 20, f"Unexpected row {r} in diff"
        assert c in [2, 4, 5], f"Unexpected column {c} in diff"
        assert w3_orig[r, c] == -100.0, f"Original value was not -100.0 at ({r}, {c})"
        assert w3_corr[r, c] == 0.0, f"Corrected value is not 0.0 at ({r}, {c})"

    # Verify columns 0, 1, 3 are identical
    np.testing.assert_array_equal(w3_orig[:, 0], w3_corr[:, 0])
    np.testing.assert_array_equal(w3_orig[:, 1], w3_corr[:, 1])
    np.testing.assert_array_equal(w3_orig[:, 3], w3_corr[:, 3])


def test_corrected_parameter_sparsity(corrected_weights):
    """Verify 726 nonzeros out of 20,486 parameters (96.4561% sparsity)."""
    total = sum(v.size for v in corrected_weights.values())
    nonzeros = sum(int(np.count_nonzero(v)) for v in corrected_weights.values())

    assert total == 20486
    assert nonzeros == 726
    sparsity = 1.0 - (nonzeros / total)
    assert np.isclose(sparsity, 0.964561, atol=1e-5)


def test_inert_action_suppression(corrected_weights):
    """Verify Up, Down, Fire logits are strictly -100.0 and produce 0.000000% softmax probability."""
    net = MinAtarNetworkNumPy(corrected_weights)

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
        logits = net.forward(flat_obs)
        probs = softmax(logits)

        # Logits for actions 2 (Up), 4 (Down), 5 (Fire) must be exactly -100.0
        assert np.isclose(logits[2], -100.0, atol=1e-5)
        assert np.isclose(logits[4], -100.0, atol=1e-5)
        assert np.isclose(logits[5], -100.0, atol=1e-5)

        # Softmax probability for actions 2, 4, 5 must be 0.0
        assert probs[2] < 1e-12
        assert probs[4] < 1e-12
        assert probs[5] < 1e-12

        # Sum of active actions (0 No-op, 1 Left, 3 Right) must be 1.0
        active_sum = probs[0] + probs[1] + probs[3]
        assert np.isclose(active_sum, 1.0, atol=1e-6)


def test_synthetic_steering_behavior(corrected_weights):
    """Verify paddle steers toward ball trajectory in various synthetic board layouts."""
    net = MinAtarNetworkNumPy(corrected_weights)

    # Scenario A: Ball at col 7 moving right (trail at 6), Paddle at col 2
    # Paddle must strongly move RIGHT (Action 3)
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[5, 7, 3] = 1.0  # Ball at (5, 7)
    obs[4, 6, 4] = 1.0  # Trail at (4, 6) -> moving right (+1)
    obs[9, 2, 0] = 1.0  # Paddle at (9, 2)
    logits = net.forward(obs.reshape(-1))
    probs = softmax(logits)

    assert probs[MinAtarAction.RIGHT] > 0.95
    assert probs[MinAtarAction.LEFT] < 0.05
    assert probs[MinAtarAction.NOOP] < 0.05

    # Scenario B: Ball at col 2 moving left (trail at 3), Paddle at col 8
    # Paddle must strongly move LEFT (Action 1)
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[5, 2, 3] = 1.0  # Ball at (5, 2)
    obs[4, 3, 4] = 1.0  # Trail at (4, 3) -> moving left (-1)
    obs[9, 8, 0] = 1.0  # Paddle at (9, 8)
    logits = net.forward(obs.reshape(-1))
    probs = softmax(logits)

    assert probs[MinAtarAction.LEFT] > 0.95
    assert probs[MinAtarAction.RIGHT] < 0.05
    assert probs[MinAtarAction.NOOP] < 0.05

    # Scenario C: Ball at (8, 5) moving right (+1) from trail at (7, 4); Paddle already at landing column 6
    # Landing column aligned with paddle -> NO-OP should dominate at ~57.6%
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[8, 5, 3] = 1.0  # Ball at row 8, col 5
    obs[7, 4, 4] = 1.0  # Trail at row 7, col 4 -> dx = +1, landing column = 6
    obs[9, 6, 0] = 1.0  # Paddle at col 6 (aligned with landing target!)
    logits = net.forward(obs.reshape(-1))
    probs = softmax(logits)

    assert probs[MinAtarAction.NOOP] > probs[MinAtarAction.LEFT]
    assert probs[MinAtarAction.NOOP] > probs[MinAtarAction.RIGHT]
    assert probs[MinAtarAction.NOOP] > 0.50


def test_numpy_pytorch_equivalence(corrected_weights):
    """Verify NumPy and PyTorch implementations produce identical forward activations."""
    import torch

    net_np = MinAtarNetworkNumPy(corrected_weights)
    net_pt = MinAtarNetworkPyTorch()
    net_pt.load_from_numpy(corrected_weights)

    rng = np.random.RandomState(42)
    for _ in range(30):
        obs = rng.uniform(0.0, 1.0, size=(600,)).astype(np.float32)
        out_np = net_np.forward(obs)
        with torch.no_grad():
            out_pt = net_pt(torch.from_numpy(obs)).numpy()
        np.testing.assert_allclose(out_np, out_pt, atol=1e-4)


def test_visualizer_state_7way_simulation():
    """Verify VisualizerState correctly steps all 7 models with independent RNG and environments."""
    state = VisualizerState(
        argon_weights_path="experiments/strict_replication/generation_001/output/argon_weights.npz",
        trained_weights_path="experiments/trained_baseline/trained_actor_weights.npz",
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_DEFAULT_PATH,
        deepseek_weights_path=DEEPSEEK_DEFAULT_PATH,
        deepseek_corrected_weights_path=DEEPSEEK_CORRECTED_DEFAULT_PATH,
        initial_seed=42,
    )

    assert state.has_astra is True
    assert state.has_gpt6 is True
    assert state.has_sonnet55 is True
    assert state.has_deepseek is True
    assert state.has_deepseek_corrected is True
    assert state.env_argon is not None
    assert state.env_trained is not None

    # Step simulation 10 frames
    for _ in range(10):
        state.step()

    assert state.env_argon.step_count == 10
    assert state.env_trained.step_count == 10
    assert state.env_astra.step_count == 10
    assert state.env_gpt6.step_count == 10
    assert state.env_sonnet55.step_count == 10
    assert state.env_deepseek_corrected.step_count == 10


def test_benchmark_registry_sorting():
    """Corrected ranking comes from the complete, separated Phase 2 dataset."""
    registry = load_verified_benchmark_registry()

    assert len(registry) == 9

    # Extract all mean scores and verify strictly descending
    scores = [info["mean_score"] for info in registry.values()]
    assert scores == sorted(scores, reverse=True)

    # Check top ranks
    model_keys = list(registry.keys())
    assert model_keys[0] == "astra"
    assert model_keys[1] == "argon"
    assert model_keys[2] == "sonnet55"
    assert model_keys[3] == "trained"
    assert model_keys[4] == "gpt6"
    assert model_keys[5] == "gpt6_luna"
    assert model_keys[6:] == ["zero", "deepseek", "random"]
    assert "deepseek_corrected" not in registry


def test_original_dataset_remains_available_and_labeled():
    registry = load_verified_benchmark_registry(dataset="original")
    assert list(registry)[0] == "astra"
    assert registry["astra"]["episodes"] == 4000
    assert registry["random"]["episodes"] == 1000
    assert registry["zero"]["episodes"] == 1000
    assert all(item["dataset"] == "original_historical_csvs" for item in registry.values())
