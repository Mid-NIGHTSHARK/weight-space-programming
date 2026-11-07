"""MinAtar Breakout environment package."""

from environment.breakout import MinAtarBreakoutEnv
from environment.actions import MinAtarAction, validate_action, action_to_char, NUM_ACTIONS
from environment.observations import (
    get_observation,
    flatten_observation,
    validate_observation,
    coord_to_index,
    index_to_coord,
    OBSERVATION_SIZE,
    GRID_ROWS,
    GRID_COLS,
    NUM_CHANNELS,
)

__all__ = [
    "MinAtarBreakoutEnv",
    "MinAtarAction",
    "validate_action",
    "action_to_char",
    "NUM_ACTIONS",
    "get_observation",
    "flatten_observation",
    "validate_observation",
    "coord_to_index",
    "index_to_coord",
    "OBSERVATION_SIZE",
    "GRID_ROWS",
    "GRID_COLS",
    "NUM_CHANNELS",
]

