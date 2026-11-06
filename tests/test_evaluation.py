"""Unit tests for the evaluation harness and metric computations."""

import os
import shutil
import numpy as np
import pytest

from evaluation.evaluate import PolicyEvaluator
from evaluation.metrics import compute_evaluation_metrics, EpisodeResult
from evaluation.seeds import generate_evaluation_seeds
from baselines.random_agent import RandomAgent
from baselines.zero_agent import ZeroWeightAgent
from network.architecture import MinAtarNetworkNumPy, create_zero_weights

TEST_OUTPUT_DIR = "tests/.tmp_eval_test"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


def test_seed_generation():
    seeds1 = generate_evaluation_seeds(num_episodes=50, base_seed=42)
    seeds2 = generate_evaluation_seeds(num_episodes=50, base_seed=42)
    assert len(seeds1) == 50
    assert seeds1 == seeds2
    assert len(set(seeds1)) == 50  # Unique seeds


def test_metric_computations():
    episodes = [
        EpisodeResult(0, 101, 10.0, 100, 10, True, False, {0: 50, 1: 50}),
        EpisodeResult(1, 102, 20.0, 200, 20, True, False, {0: 100, 3: 100}),
    ]
    summary = compute_evaluation_metrics(episodes)
    assert summary.episodes == 2
    assert summary.mean_score == 15.0
    assert summary.median_score == 15.0
    assert summary.min_score == 10.0
    assert summary.max_score == 20.0
    assert summary.total_bricks_destroyed == 30
    assert np.isclose(summary.std_score, np.std([10.0, 20.0], ddof=1))
    assert np.isclose(summary.se_score, summary.std_score / np.sqrt(2))


def test_evaluator_deterministic_reproducibility():
    """Verify that running the same agent on the same seeds yields bitwise identical scores."""
    evaluator = PolicyEvaluator(num_episodes=5, base_seed=123)
    random_agent1 = RandomAgent()
    random_agent2 = RandomAgent()

    res1 = evaluator.evaluate_agent(random_agent1, agent_id="repro_test", verbose=False)
    res2 = evaluator.evaluate_agent(random_agent2, agent_id="repro_test", verbose=False)

    assert res1.mean_score == res2.mean_score
    assert res1.mean_length == res2.mean_length


def test_evaluator_save_artifacts():
    evaluator = PolicyEvaluator(num_episodes=3, base_seed=42)
    zero_agent = ZeroWeightAgent()

    summary = evaluator.evaluate_agent(
        action_fn=zero_agent,
        agent_id="zero_agent_test",
        output_dir=TEST_OUTPUT_DIR,
        verbose=False,
    )

    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "episode_scores.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "summary.json"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "evaluation_report.md"))
    assert summary.episodes == 3


def test_evaluator_network_policy():
    net = MinAtarNetworkNumPy(create_zero_weights())
    evaluator = PolicyEvaluator(num_episodes=3, base_seed=999)
    summary = evaluator.evaluate_network(net, agent_id="net_test", verbose=False)
    assert summary.episodes == 3
    assert summary.mean_score >= 0.0
