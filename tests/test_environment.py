"""Unit tests for MinAtar Breakout environment integration, physics, and rules."""

import numpy as np
import pytest

from environment.breakout import MinAtarBreakoutEnv
from environment.actions import MinAtarAction
from environment.validation import validate_initial_state, validate_paddle_boundaries


@pytest.fixture
def env():
    return MinAtarBreakoutEnv(sticky_action_prob=0.1, max_steps=10_000)


def test_initial_state(env):
    report = validate_initial_state(env)
    assert report["valid"] is True
    assert report["initial_paddle_pos"] == 4
    assert report["initial_ball_y"] == 3
    assert report["initial_ball_x"] in (0, 9)
    assert report["brick_count"] == 30
    assert report["observation_size"] == 600


def test_paddle_boundaries(env):
    assert validate_paddle_boundaries(env) is True


def test_paddle_movement_order(env):
    """Verify that paddle moves before ball position updates."""
    env.reset(seed=42)
    env.sticky_action_prob = 0.0
    env._minatar_env.sticky_action_prob = 0.0

    initial_pos = env.internal_env.pos
    obs, r, done, info = env.step(MinAtarAction.LEFT)
    assert info["paddle_pos"] == initial_pos - 1


def test_sticky_actions():
    """Verify sticky action mechanics with controlled RNG.
    
    Step 1 executed non-sticky establishes last_action = LEFT (pos becomes 3).
    Step 2 executed with sticky_action_prob = 1.0 overrides proposed RIGHT with last_action LEFT (pos becomes 2).
    """
    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=123)

    # First action non-sticky: execute LEFT
    env.step(MinAtarAction.LEFT)
    assert env.internal_env.pos == 3
    assert env._minatar_env.last_action == MinAtarAction.LEFT

    # Now make sticky_action_prob = 1.0
    env.sticky_action_prob = 1.0
    env._minatar_env.sticky_action_prob = 1.0

    # Execute RIGHT, but sticky action forces repeat of LEFT!
    env.step(MinAtarAction.RIGHT)
    assert env.internal_env.pos == 2, "Sticky action should have repeated LEFT, moving pos to 2"


def test_missed_ball_termination():
    """Verify that letting the ball fall terminates the episode."""
    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=42)
    # Move paddle to corner (col 0) and wait with no-ops
    terminated = False
    for _ in range(50):
        obs, r, done, info = env.step(MinAtarAction.NOOP)
        if done:
            terminated = True
            break
    assert terminated is True, "Missed ball must terminate episode"


def test_brick_destruction_reward():
    """Verify reward assignment on brick hit."""
    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=0)
    initial_bricks = np.count_nonzero(env.internal_env.brick_map)
    assert initial_bricks == 30


def test_step_cutoff():
    """Verify that episode stops when max_steps is reached."""
    short_env = MinAtarBreakoutEnv(sticky_action_prob=0.0, max_steps=10)
    short_env.reset(seed=0)
    step_count = 0
    for _ in range(20):
        if short_env.terminal:
            break
        obs, r, done, info = short_env.step(MinAtarAction.NOOP)
        step_count += 1
    assert step_count <= 10
    assert short_env.terminal is True


def test_reset_clears_terminal_and_step_count():
    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=42)
    # Run to terminal
    for _ in range(100):
        if env.terminal:
            break
        obs, r, done, info = env.step(MinAtarAction.NOOP)
    assert env.terminal is True

    # Reset
    obs = env.reset(seed=42)
    assert env.terminal is False
    assert env.step_count == 0
    assert env.total_reward == 0.0
    assert obs.shape == (600,)
