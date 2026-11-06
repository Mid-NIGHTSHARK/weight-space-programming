"""Automated test suite for Gemini Argon 4 Medium generated neural network weights."""

import os
import sys
import numpy as np
import pytest

from environment.breakout import MinAtarBreakoutEnv
from network.architecture import MinAtarNetworkNumPy, MinAtarNetworkPyTorch
from network.inference import PolicyAgent, softmax
from network.weight_loader import load_weights_npz, compute_file_sha256
from network.weight_parser import SafeASTWeightParser
from visualize_comparison import VisualizerState

ARGON_WEIGHTS_PATH = "experiments/strict_replication/generation_001/output/argon_weights.npz"
ARGON_RAW_PATH = "experiments/strict_replication/generation_001/output/raw_response.txt"
EXPECTED_ARGON_WEIGHTS_SHA256 = "455722b09750deb693093c823c13dd37b99eb470c957ce472262b1419014abb5"


@pytest.fixture(scope="module")
def argon_weights():
    assert os.path.exists(ARGON_WEIGHTS_PATH), f"Weights not found: {ARGON_WEIGHTS_PATH}"
    return load_weights_npz(ARGON_WEIGHTS_PATH)


def test_argon_weight_reconstruction():
    """Verify SafeASTWeightParser reconstructs Argon weights with 0 errors."""
    assert os.path.exists(ARGON_RAW_PATH)
    with open(ARGON_RAW_PATH, "r") as f:
        content = f.read()

    parser = SafeASTWeightParser(strict=True)
    weights, report = parser.parse_text(content)

    assert report.block_found is True
    assert report.statements_applied == 86
    assert len(report.errors) == 0
    assert report.nonzero_total == 817


def test_argon_tensor_dimensions(argon_weights):
    """Verify all 6 tensors have exact required shapes and parameter count."""
    expected_shapes = {
        "W1": (600, 32),
        "b1": (32,),
        "W2": (32, 32),
        "b2": (32,),
        "W3": (32, 6),
        "b3": (6,),
    }
    total_params = 0
    for name, shape in expected_shapes.items():
        assert name in argon_weights, f"Missing tensor {name}"
        assert argon_weights[name].shape == shape, f"Shape mismatch for {name}: {argon_weights[name].shape}"
        assert argon_weights[name].dtype == np.float32
        total_params += argon_weights[name].size

    assert total_params == 20486


def test_argon_forward_pass_dimensions(argon_weights):
    """Verify forward-pass output dimensions for single and batched observations."""
    net = MinAtarNetworkNumPy(argon_weights)

    single_obs = np.zeros(600, dtype=np.float32)
    logits_single = net.forward(single_obs)
    assert logits_single.shape == (6,)

    batch_obs = np.zeros((10, 600), dtype=np.float32)
    logits_batch = net.forward(batch_obs)
    assert logits_batch.shape == (10, 6)


def test_argon_finite_logits_and_probabilities(argon_weights):
    """Verify all forward pass outputs are strictly finite without NaN or Inf."""
    net = MinAtarNetworkNumPy(argon_weights)
    agent = PolicyAgent(net)

    rng = np.random.RandomState(42)
    for _ in range(20):
        random_obs = rng.choice([0.0, 1.0], size=600).astype(np.float32)
        action, probs, logits = agent.get_action(random_obs)

        assert np.isfinite(logits).all(), f"Non-finite logits: {logits}"
        assert np.isfinite(probs).all(), f"Non-finite probs: {probs}"
        assert 0 <= action <= 5


def test_argon_probability_normalization(argon_weights):
    """Verify softmax probabilities sum to 1.0 and are non-negative."""
    net = MinAtarNetworkNumPy(argon_weights)
    obs = np.zeros(600, dtype=np.float32)
    logits = net.forward(obs)
    probs = softmax(logits)

    assert np.isclose(np.sum(probs), 1.0, atol=1e-6)
    assert np.all(probs >= 0.0)


def test_argon_action_selection_behavior(argon_weights):
    """Verify that action selection only chooses actions routed in Argon design."""
    net = MinAtarNetworkNumPy(argon_weights)
    agent = PolicyAgent(net)

    # In Argon's design, logits[2] (Up), logits[4] (Down), logits[5] (Fire) have bias -50
    # and no positive inputs, so action selection is strictly restricted to {0, 1, 3}
    env = MinAtarBreakoutEnv(random_seed=42)
    obs = env.reset(seed=42)

    selected_actions = set()
    for _ in range(100):
        action, _, _ = agent.get_action(obs)
        selected_actions.add(action)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    # Confirm only functional Breakout actions are selected
    assert selected_actions.issubset({0, 1, 3}), f"Unexpected action selected: {selected_actions}"


def test_argon_model_loading():
    """Verify weight archive loading and SHA-256 checksum."""
    sha256 = compute_file_sha256(ARGON_WEIGHTS_PATH)
    assert sha256 == EXPECTED_ARGON_WEIGHTS_SHA256

    weights = load_weights_npz(ARGON_WEIGHTS_PATH)
    assert len(weights) == 6


def test_argon_environment_stepping(argon_weights):
    """Verify stepping actual MinAtar environment with Argon policy."""
    net = MinAtarNetworkNumPy(argon_weights)
    agent = PolicyAgent(net)
    env = MinAtarBreakoutEnv(sticky_action_prob=0.1, random_seed=42)
    obs = env.reset(seed=42)

    for step in range(50):
        action, probs, _ = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (600,)
        assert reward >= 0.0
        assert isinstance(done, bool)
        if done:
            break


def test_visualizer_state_initialization(argon_weights):
    """Verify VisualizerState initialization and headless stepping."""
    # Use argon weights for both sides to test VisualizerState mechanics
    state = VisualizerState(
        argon_weights_path=ARGON_WEIGHTS_PATH,
        trained_weights_path=ARGON_WEIGHTS_PATH,
        initial_seed=42,
    )
    assert state.env_argon.step_count == 0
    assert state.env_trained.step_count == 0

    argon_done, trained_done = state.step()
    assert state.env_argon.step_count == 1
    assert state.env_trained.step_count == 1


def test_argon_evaluation_reproducibility(argon_weights):
    """Verify deterministic reproducibility of Argon evaluation with fixed seeds."""
    from evaluation.evaluate import PolicyEvaluator

    evaluator = PolicyEvaluator(num_episodes=5, base_seed=42)
    net1 = MinAtarNetworkNumPy(argon_weights)
    sum1 = evaluator.evaluate_network(net1, verbose=False)

    net2 = MinAtarNetworkNumPy(argon_weights)
    sum2 = evaluator.evaluate_network(net2, verbose=False)

    assert sum1.mean_score == sum2.mean_score
    assert sum1.mean_length == sum2.mean_length
    assert sum1.total_bricks_destroyed == sum2.total_bricks_destroyed
