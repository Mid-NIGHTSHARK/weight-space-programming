"""Action space specifications and mappings for MinAtar Breakout."""

from enum import IntEnum
from typing import Dict, List, Set


class MinAtarAction(IntEnum):
    NOOP = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4
    FIRE = 5


# MinAtar environment internal action characters
ACTION_CHARS: List[str] = ['n', 'l', 'u', 'r', 'd', 'f']

ACTION_NAMES: Dict[int, str] = {
    0: "no-op",
    1: "left",
    2: "up",
    3: "right",
    4: "down",
    5: "fire",
}

# In Breakout, only 'l' and 'r' alter paddle position; others act as no-op
FUNCTIONAL_ACTIONS: Set[int] = {MinAtarAction.LEFT, MinAtarAction.RIGHT}
NON_FUNCTIONAL_ACTIONS: Set[int] = {
    MinAtarAction.NOOP,
    MinAtarAction.UP,
    MinAtarAction.DOWN,
    MinAtarAction.FIRE,
}

NUM_ACTIONS: int = 6


def validate_action(action: int) -> int:
    """Validate that an action index is an integer in [0, 5]."""
    if not isinstance(action, (int, MinAtarAction)):
        raise TypeError(f"Action must be an integer, got {type(action)}")
    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"Action must be in range [0, {NUM_ACTIONS - 1}], got {action}")
    return int(action)


def action_to_char(action: int) -> str:
    """Convert an action index to its MinAtar character representation."""
    validate_action(action)
    return ACTION_CHARS[action]
