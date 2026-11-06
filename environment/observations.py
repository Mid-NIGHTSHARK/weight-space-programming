"""Observation representations and channel mappings for MinAtar Breakout.

The network receives a 10 x 10 grid with 6 channels, flattened into 600 values:
  index = (row * 10 + column) * 6 + channel

The 6 channels are:
  0 (player): the paddle
  1 (danger): always empty in Breakout
  2 (reward): the bricks
  3 (projectile): the ball
  4 (trail): the ball's previous cell (indicates direction)
  5 (gauge): always empty in Breakout
"""

from typing import Tuple
import numpy as np

GRID_ROWS: int = 10
GRID_COLS: int = 10
NUM_CHANNELS: int = 6
OBSERVATION_SIZE: int = GRID_ROWS * GRID_COLS * NUM_CHANNELS  # 600

# Channel definitions
CH_PLAYER: int = 0
CH_DANGER: int = 1
CH_REWARD: int = 2
CH_BALL: int = 3
CH_TRAIL: int = 4
CH_GAUGE: int = 5

# MinAtar Breakout native channel indices
# self.channels = {'paddle': 0, 'ball': 1, 'trail': 2, 'brick': 3}
NATIVE_PADDLE: int = 0
NATIVE_BALL: int = 1
NATIVE_TRAIL: int = 2
NATIVE_BRICK: int = 3


def coord_to_index(row: int, column: int, channel: int) -> int:
    """Calculate the 1D flattened index from 2D coordinates and channel.
    
    Formula: (row * 10 + column) * 6 + channel
    """
    if not (0 <= row < GRID_ROWS):
        raise ValueError(f"row must be in [0, {GRID_ROWS - 1}], got {row}")
    if not (0 <= column < GRID_COLS):
        raise ValueError(f"column must be in [0, {GRID_COLS - 1}], got {column}")
    if not (0 <= channel < NUM_CHANNELS):
        raise ValueError(f"channel must be in [0, {NUM_CHANNELS - 1}], got {channel}")
    return (row * GRID_COLS + column) * NUM_CHANNELS + channel


def index_to_coord(index: int) -> Tuple[int, int, int]:
    """Calculate (row, column, channel) from a 1D flattened index."""
    if not (0 <= index < OBSERVATION_SIZE):
        raise ValueError(f"index must be in [0, {OBSERVATION_SIZE - 1}], got {index}")
    channel = index % NUM_CHANNELS
    temp = index // NUM_CHANNELS
    column = temp % GRID_COLS
    row = temp // GRID_COLS
    return row, column, channel


def get_observation(native_state: np.ndarray) -> np.ndarray:
    """Map MinAtar native (10, 10, 4) state to the 6-channel (10, 10, 6) observation.
    
    Mapping:
      channel 0 (player)     <- native channel 0 (paddle)
      channel 1 (danger)     <- empty (0)
      channel 2 (reward)     <- native channel 3 (brick)
      channel 3 (projectile) <- native channel 1 (ball)
      channel 4 (trail)      <- native channel 2 (trail)
      channel 5 (gauge)      <- empty (0)
    """
    if native_state.shape != (GRID_ROWS, GRID_COLS, 4):
        raise ValueError(
            f"Expected native state shape ({GRID_ROWS}, {GRID_COLS}, 4), got {native_state.shape}"
        )

    obs = np.zeros((GRID_ROWS, GRID_COLS, NUM_CHANNELS), dtype=np.float32)
    obs[:, :, CH_PLAYER] = native_state[:, :, NATIVE_PADDLE]
    obs[:, :, CH_DANGER] = 0.0
    obs[:, :, CH_REWARD] = native_state[:, :, NATIVE_BRICK]
    obs[:, :, CH_BALL] = native_state[:, :, NATIVE_BALL]
    obs[:, :, CH_TRAIL] = native_state[:, :, NATIVE_TRAIL]
    obs[:, :, CH_GAUGE] = 0.0
    return obs


def flatten_observation(obs_6ch: np.ndarray) -> np.ndarray:
    """Flatten (10, 10, 6) observation to a 600-element 1D array.
    
    Preserves C-contiguous row-major flattening matching index = (row * 10 + col) * 6 + ch.
    """
    if obs_6ch.shape != (GRID_ROWS, GRID_COLS, NUM_CHANNELS):
        raise ValueError(
            f"Expected shape ({GRID_ROWS}, {GRID_COLS}, {NUM_CHANNELS}), got {obs_6ch.shape}"
        )
    return obs_6ch.flatten(order='C').astype(np.float32)


def validate_observation(obs_flat: np.ndarray) -> np.ndarray:
    """Validate that a flattened observation matches all formal requirements."""
    if not isinstance(obs_flat, np.ndarray):
        raise TypeError(f"Observation must be a numpy ndarray, got {type(obs_flat)}")
    if obs_flat.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Observation must have shape ({OBSERVATION_SIZE},), got {obs_flat.shape}")
    if not np.all(np.isin(obs_flat, [0.0, 1.0])):
        raise ValueError("Observation values must be binary (0 or 1)")
    return obs_flat
