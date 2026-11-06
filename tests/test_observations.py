"""Unit tests for observation representation, channel mapping, and flattening."""

import numpy as np
import pytest

from environment.observations import (
    coord_to_index,
    index_to_coord,
    get_observation,
    flatten_observation,
    validate_observation,
    OBSERVATION_SIZE,
    GRID_ROWS,
    GRID_COLS,
    NUM_CHANNELS,
    CH_PLAYER,
    CH_DANGER,
    CH_REWARD,
    CH_BALL,
    CH_TRAIL,
    CH_GAUGE,
    NATIVE_PADDLE,
    NATIVE_BALL,
    NATIVE_TRAIL,
    NATIVE_BRICK,
)


def test_observation_dimensions():
    assert GRID_ROWS == 10
    assert GRID_COLS == 10
    assert NUM_CHANNELS == 6
    assert OBSERVATION_SIZE == 600


def test_prompt_canonical_index_examples():
    """Verify specific examples explicitly given in the prompt."""
    # "input 0 is (row 0, column 0, channel 0)"
    assert coord_to_index(0, 0, 0) == 0
    assert index_to_coord(0) == (0, 0, 0)

    # "input 6 is (row 0, column 1, channel 0)"
    assert coord_to_index(0, 1, 0) == 6
    assert index_to_coord(6) == (0, 1, 0)

    # "input 599 is (row 9, column 9, channel 5)"
    assert coord_to_index(9, 9, 5) == 599
    assert index_to_coord(599) == (9, 9, 5)


def test_coord_index_bijective_roundtrip():
    """Verify that all 600 coordinate tuples map 1:1 with flat indices."""
    seen_indices = set()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            for ch in range(NUM_CHANNELS):
                idx = coord_to_index(r, c, ch)
                assert 0 <= idx < 600
                assert idx not in seen_indices
                seen_indices.add(idx)
                assert index_to_coord(idx) == (r, c, ch)
    assert len(seen_indices) == 600


def test_strided_ranges_from_prompt():
    """Verify the useful strided range formulas specified in line 171 of prompt:
    'all 10 cells of column c in channel ch are rows c*6+ch : 600 : 60 of W1;
     all 10 cells of row r in channel ch are rows r*60+ch : r*60+60 : 6.'
    """
    for ch in range(NUM_CHANNELS):
        # Column slice test
        for c in range(GRID_COLS):
            expected_indices = [coord_to_index(r, c, ch) for r in range(GRID_ROWS)]
            generated_slice = list(range(c * 6 + ch, 600, 60))
            assert expected_indices == generated_slice, f"Failed for col {c}, ch {ch}"

        # Row slice test
        for r in range(GRID_ROWS):
            expected_indices = [coord_to_index(r, c, ch) for c in range(GRID_COLS)]
            generated_slice = list(range(r * 60 + ch, r * 60 + 60, 6))
            assert expected_indices == generated_slice, f"Failed for row {r}, ch {ch}"


def test_channel_mapping_from_native():
    """Verify mapping from MinAtar native (10, 10, 4) to 6-channel observation."""
    native = np.zeros((10, 10, 4), dtype=bool)

    # Place elements at distinctive coordinates
    native[9, 4, NATIVE_PADDLE] = True
    native[3, 2, NATIVE_BALL] = True
    native[4, 1, NATIVE_TRAIL] = True
    native[2, 7, NATIVE_BRICK] = True

    obs_6ch = get_observation(native)
    assert obs_6ch.shape == (10, 10, 6)

    # Check mappings
    assert obs_6ch[9, 4, CH_PLAYER] == 1.0  # Paddle -> Channel 0
    assert obs_6ch[:, :, CH_DANGER].sum() == 0.0  # Danger is empty
    assert obs_6ch[2, 7, CH_REWARD] == 1.0  # Brick -> Channel 2
    assert obs_6ch[3, 2, CH_BALL] == 1.0  # Ball -> Channel 3
    assert obs_6ch[4, 1, CH_TRAIL] == 1.0  # Trail -> Channel 4
    assert obs_6ch[:, :, CH_GAUGE].sum() == 0.0  # Gauge is empty


def test_flatten_observation_order():
    """Verify that flatten preserves C-order without transposing."""
    obs_6ch = np.zeros((10, 10, 6), dtype=np.float32)
    # Set unique values at known coordinates
    obs_6ch[0, 0, 0] = 1.0
    obs_6ch[0, 1, 0] = 2.0
    obs_6ch[1, 0, 0] = 3.0
    obs_6ch[9, 9, 5] = 4.0

    flat = flatten_observation(obs_6ch)
    assert flat[0] == 1.0
    assert flat[6] == 2.0
    assert flat[60] == 3.0
    assert flat[599] == 4.0


def test_validate_observation():
    valid = np.zeros(600, dtype=np.float32)
    valid[10] = 1.0
    assert validate_observation(valid) is valid

    with pytest.raises(ValueError):
        validate_observation(np.zeros(599, dtype=np.float32))

    with pytest.raises(ValueError):
        invalid_vals = np.zeros(600, dtype=np.float32)
        invalid_vals[0] = 2.5
        validate_observation(invalid_vals)
