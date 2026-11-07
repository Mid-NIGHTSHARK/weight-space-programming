"""Unit tests for MinAtar action space definitions and mappings."""

import pytest
from environment.actions import (
    MinAtarAction,
    validate_action,
    action_to_char,
    ACTION_CHARS,
    ACTION_NAMES,
    NUM_ACTIONS,
    FUNCTIONAL_ACTIONS,
    NON_FUNCTIONAL_ACTIONS,
)


def test_action_constants():
    assert NUM_ACTIONS == 6
    assert len(ACTION_CHARS) == 6
    assert len(ACTION_NAMES) == 6
    assert ACTION_CHARS == ['n', 'l', 'u', 'r', 'd', 'f']


def test_action_enum_values():
    assert MinAtarAction.NOOP == 0
    assert MinAtarAction.LEFT == 1
    assert MinAtarAction.UP == 2
    assert MinAtarAction.RIGHT == 3
    assert MinAtarAction.DOWN == 4
    assert MinAtarAction.FIRE == 5


def test_functional_actions():
    assert MinAtarAction.LEFT in FUNCTIONAL_ACTIONS
    assert MinAtarAction.RIGHT in FUNCTIONAL_ACTIONS
    assert len(FUNCTIONAL_ACTIONS) == 2
    assert len(NON_FUNCTIONAL_ACTIONS) == 4
    assert FUNCTIONAL_ACTIONS.isdisjoint(NON_FUNCTIONAL_ACTIONS)


def test_action_validation_valid():
    for i in range(6):
        assert validate_action(i) == i
    assert validate_action(MinAtarAction.LEFT) == 1


def test_action_validation_invalid():
    with pytest.raises(ValueError):
        validate_action(-1)
    with pytest.raises(ValueError):
        validate_action(6)
    with pytest.raises(TypeError):
        validate_action("left")  # type: ignore


def test_action_to_char():
    expected = {0: 'n', 1: 'l', 2: 'u', 3: 'r', 4: 'd', 5: 'f'}
    for a, c in expected.items():
        assert action_to_char(a) == c

