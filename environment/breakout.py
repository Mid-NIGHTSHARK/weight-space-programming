"""Authoritative MinAtar Breakout environment wrapper for Experiment 5 replication."""

import os
# Ensure headless matplotlib backend to prevent Cocoa / windowing delays
os.environ.setdefault("MPLBACKEND", "Agg")

from typing import Optional, Tuple
import numpy as np
from minatar import Environment

from environment.actions import NUM_ACTIONS, validate_action
from environment.observations import (
    OBSERVATION_SIZE,
    flatten_observation,
    get_observation,
    validate_observation,
)


class MinAtarBreakoutEnv:
    """Breakout environment conforming strictly to the Experiment 5 specifications.
    
    Wraps the official minatar==1.0.15 environment, enforces:
    - 6 discrete actions [0..5]
    - Sticky action probability of 0.1
    - 600-element binary observation representation
    - 10,000 maximum episode steps
    - Fully reproducible seeding
    """

    def __init__(
        self,
        sticky_action_prob: float = 0.1,
        max_steps: int = 10_000,
        random_seed: Optional[int] = None,
    ):
        self.sticky_action_prob = sticky_action_prob
        self.max_steps = max_steps
        self.random_seed = random_seed

        # Initialize the underlying MinAtar Environment
        self._minatar_env = Environment("breakout", sticky_action_prob=sticky_action_prob)
        if random_seed is not None:
            self.seed(random_seed)

        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = 0
        self.terminal = False

    def seed(self, seed: int) -> None:
        """Seed the environment and its underlying random number generators."""
        self.random_seed = seed
        self._minatar_env.seed(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to its initial state for a new episode.
        
        Returns:
            Flat 600-element float32 binary observation.
        """
        if seed is not None:
            self.seed(seed)
        self._minatar_env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = 0
        self._minatar_env.last_action = 0
        self.terminal = False
        return self.get_current_observation()

    def get_current_observation(self) -> np.ndarray:
        """Extract the current 6-channel 600-element observation vector."""
        native_state = self._minatar_env.state()
        obs_6ch = get_observation(native_state)
        obs_flat = flatten_observation(obs_6ch)
        return validate_observation(obs_flat)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            action: Discrete action in range [0, 5].
            
        Returns:
            observation: Flat 600-element float32 binary array.
            reward: Reward achieved this step (+1 per brick broken).
            done: Whether the episode has terminated (lost ball or step limit reached).
            info: Metadata dictionary containing step count, total reward, and terminal reason.
        """
        validate_action(action)
        if self.terminal:
            raise RuntimeError("Cannot step in a terminal environment; please call reset()")

        prev_action = self._minatar_env.last_action
        reward, done = self._minatar_env.act(action)
        executed_action = self._minatar_env.last_action
        sticky_override = (executed_action != action)

        self.step_count += 1
        self.total_reward += reward
        self.last_action = executed_action

        # Check maximum step cutoff
        step_cutoff = self.step_count >= self.max_steps
        self.terminal = bool(done or step_cutoff)

        obs = self.get_current_observation()
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "ball_lost": bool(done),
            "step_cutoff": step_cutoff,
            "paddle_pos": self._minatar_env.env.pos,
            "ball_x": self._minatar_env.env.ball_x,
            "ball_y": self._minatar_env.env.ball_y,
            "ball_dir": self._minatar_env.env.ball_dir,
            "bricks_remaining": int(np.count_nonzero(self._minatar_env.env.brick_map)),
            "selected_action": action,
            "executed_action": executed_action,
            "sticky_override": sticky_override,
        }
        return obs, float(reward), self.terminal, info

    @property
    def internal_env(self):
        """Access the inner MinAtar Env object for state verification."""
        return self._minatar_env.env

