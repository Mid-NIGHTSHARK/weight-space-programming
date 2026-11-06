"""Automated environment validation utilities for MinAtar Breakout."""

from typing import Dict, Any
import numpy as np

from environment.breakout import MinAtarBreakoutEnv
from environment.actions import MinAtarAction
from environment.observations import (
    OBSERVATION_SIZE,
    CH_PLAYER,
    CH_BALL,
    CH_TRAIL,
    CH_REWARD,
    CH_DANGER,
    CH_GAUGE,
    coord_to_index,
)


def validate_initial_state(env: MinAtarBreakoutEnv) -> Dict[str, Any]:
    """Validate that the environment's initial state matches all canonical specifications."""
    obs = env.reset(seed=12345)
    inner = env.internal_env

    # 1. Paddle position must start at column 4, row 9
    assert inner.pos == 4, f"Initial paddle position must be 4, got {inner.pos}"
    paddle_idx = coord_to_index(9, 4, CH_PLAYER)
    assert obs[paddle_idx] == 1.0, "Paddle must be present in channel 0 at (row 9, col 4)"

    # 2. Ball y must start at row 3
    assert inner.ball_y == 3, f"Initial ball y must be 3, got {inner.ball_y}"
    assert inner.ball_x in (0, 9), f"Initial ball x must be 0 or 9, got {inner.ball_x}"

    # 3. Bricks must fill rows 1, 2, 3 across all 10 columns (30 bricks total)
    assert np.count_nonzero(inner.brick_map) == 30, (
        f"Initial brick count must be 30, got {np.count_nonzero(inner.brick_map)}"
    )
    for r in range(1, 4):
        for c in range(10):
            brick_idx = coord_to_index(r, c, CH_REWARD)
            assert obs[brick_idx] == 1.0, f"Brick must be active at (row {r}, col {c})"

    # 4. Channels 1 (danger) and 5 (gauge) must be completely empty
    for r in range(10):
        for c in range(10):
            danger_idx = coord_to_index(r, c, CH_DANGER)
            gauge_idx = coord_to_index(r, c, CH_GAUGE)
            assert obs[danger_idx] == 0.0, f"Danger channel must be empty at ({r}, {c})"
            assert obs[gauge_idx] == 0.0, f"Gauge channel must be empty at ({r}, {c})"

    # 5. Observation size must be exactly 600
    assert obs.shape == (OBSERVATION_SIZE,), f"Expected shape (600,), got {obs.shape}"

    return {
        "initial_paddle_pos": inner.pos,
        "initial_ball_x": inner.ball_x,
        "initial_ball_y": inner.ball_y,
        "initial_ball_dir": inner.ball_dir,
        "brick_count": 30,
        "observation_size": 600,
        "valid": True,
    }


def validate_paddle_boundaries(env: MinAtarBreakoutEnv) -> bool:
    """Validate that the paddle cannot move out of the 10-column bounds [0, 9]."""
    env.reset(seed=0)
    # Move left 20 times (sticky actions off for deterministic test)
    env.sticky_action_prob = 0.0
    env._minatar_env.sticky_action_prob = 0.0

    for _ in range(20):
        if env.terminal:
            break
        env.step(MinAtarAction.LEFT)
        assert 0 <= env.internal_env.pos <= 9, f"Paddle out of bounds: {env.internal_env.pos}"

    assert env.internal_env.pos == 0, f"Paddle should be clamped at 0, got {env.internal_env.pos}"

    # Move right 20 times
    for _ in range(20):
        if env.terminal:
            break
        env.step(MinAtarAction.RIGHT)
        assert 0 <= env.internal_env.pos <= 9, f"Paddle out of bounds: {env.internal_env.pos}"

    # Reset sticky action probability
    env.sticky_action_prob = 0.1
    env._minatar_env.sticky_action_prob = 0.1
    return True

