"""Regression tests for independent corrected evaluation RNG streams."""

import numpy as np
import pytest

from baselines.random_agent import RandomAgent
from environment.breakout import MinAtarBreakoutEnv
from evaluation.corrected_evaluate import CorrectedPolicyEvaluator
from evaluation.rng_streams import RNG_PROTOCOL, derive_episode_stream_seeds, make_policy_rng
from network.inference import sample_action, softmax


def _random_action(obs, rng):
    return RandomAgent()(obs, rng)


def test_stream_seeds_are_spawned_from_distinct_seedsequence_children():
    streams = derive_episode_stream_seeds(42)
    root = np.random.SeedSequence(42)
    env_child, policy_child = root.spawn(2)
    assert streams.environment_seed == int(env_child.generate_state(1, dtype=np.uint32)[0])
    assert streams.policy_seed == int(policy_child.generate_state(1, dtype=np.uint32)[0])
    assert env_child.spawn_key != policy_child.spawn_key
    assert streams.environment_seed != streams.policy_seed
    assert RNG_PROTOCOL == "numpy-seedsequence-spawn-two-randomstate-v1"


def test_policy_sampling_does_not_advance_environment_random_state():
    env = MinAtarBreakoutEnv(sticky_action_prob=0.1)
    streams = derive_episode_stream_seeds(101)
    env.reset(seed=streams.environment_seed)
    before = env._minatar_env.random.get_state()
    rng = make_policy_rng(streams.policy_seed)
    sample_action(softmax(np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])), rng=rng)
    after = env._minatar_env.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_environment_randomness_does_not_advance_policy_random_state():
    env = MinAtarBreakoutEnv(sticky_action_prob=0.1)
    streams = derive_episode_stream_seeds(202)
    env.reset(seed=streams.environment_seed)
    policy_rng = make_policy_rng(streams.policy_seed)
    before = policy_rng.get_state()
    env.step(0)
    after = policy_rng.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_same_base_seed_reproduces_episode_and_replay_digest():
    first = derive_episode_stream_seeds(42)
    again = derive_episode_stream_seeds(42)
    other = derive_episode_stream_seeds(43)
    assert first == again
    assert (first.environment_seed, first.policy_seed) != (other.environment_seed, other.policy_seed)

    evaluator = CorrectedPolicyEvaluator(seed_group=42, base_seeds=[1001], max_steps=10_000)
    first_episode = evaluator.evaluate_episode(_random_action, 1001, 0)
    second_episode = evaluator.evaluate_episode(_random_action, 1001, 0)
    assert first_episode.to_record() == second_episode.to_record()
    record = first_episode.to_record()
    replay = evaluator.replay_record(_random_action, record)
    assert replay.to_record() == record


def test_corrected_evaluator_enforces_original_environment_parameters():
    with pytest.raises(ValueError, match="sticky_action_prob=0.1"):
        CorrectedPolicyEvaluator(42, [1], sticky_action_prob=0.2)
    with pytest.raises(ValueError, match="10,000-step cutoff"):
        CorrectedPolicyEvaluator(42, [1], max_steps=9999)


def test_categorical_action_sampling_and_temperature_remain_unchanged():
    probs = softmax(np.array([0.2, -0.1, 1.0, 0.4, -0.8, 0.3]), temperature=1.0)
    seed = 872
    expected_rng = np.random.RandomState(seed)
    policy_rng = np.random.RandomState(seed)
    expected = [int(expected_rng.choice(len(probs), p=probs)) for _ in range(100)]
    actual = [sample_action(probs, rng=policy_rng) for _ in range(100)]
    assert actual == expected
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)


def test_historical_evaluator_still_uses_legacy_same_integer_seed_path(monkeypatch):
    import evaluation.evaluate as historical_module
    from evaluation.evaluate import PolicyEvaluator

    actual_env_class = historical_module.MinAtarBreakoutEnv
    observed_env_seeds = []

    class SeedObservingEnv(actual_env_class):
        def reset(self, seed=None):
            observed_env_seeds.append(seed)
            return super().reset(seed)

    monkeypatch.setattr(historical_module, "MinAtarBreakoutEnv", SeedObservingEnv)
    historical = PolicyEvaluator(num_episodes=1, base_seed=42)
    base_episode_seed = historical.seeds[0]
    first_policy_draws = []

    def observe_policy_seed(obs, rng):
        first_policy_draws.append(rng.rand())
        return 0

    historical.evaluate_agent(observe_policy_seed, verbose=False)
    assert observed_env_seeds == [base_episode_seed]
    assert first_policy_draws[0] == np.random.RandomState(base_episode_seed).rand()
    # The historical implementation still seeds environment and policy from
    # the same episode integer; corrected behavior is a separate opt-in path.
    assert derive_episode_stream_seeds(base_episode_seed).environment_seed != base_episode_seed
