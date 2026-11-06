"""Unit and golden tests for the safe weight parser."""

import os
import numpy as np
import pytest

from network.weight_parser import (
    SafeASTWeightParser,
    extract_last_edits_block,
    WeightParserSecurityError,
    WeightParserSyntaxError,
)
from network.weight_loader import load_weights_npz


def test_individual_assignment():
    parser = SafeASTWeightParser(strict=True)
    text = "```edits\nW1[12, 0] = 2.5\n```"
    weights, report = parser.parse_text(text)
    assert report.statements_applied == 1
    assert weights["W1"][12, 0] == 2.5
    assert np.count_nonzero(weights["W1"]) == 1


def test_slice_list_assignment():
    parser = SafeASTWeightParser(strict=True)
    text = "```edits\nW1[0:6, 3] = [1, 0, -1, 0, 2, 0]\n```"
    weights, report = parser.parse_text(text)
    assert report.statements_applied == 1
    expected = np.array([1, 0, -1, 0, 2, 0], dtype=np.float32)
    assert np.array_equal(weights["W1"][0:6, 3], expected)


def test_scalar_fill_range():
    parser = SafeASTWeightParser(strict=True)
    text = "```edits\nb1[0:4] = -1\n```"
    weights, report = parser.parse_text(text)
    assert np.array_equal(weights["b1"][0:4], np.full(4, -1.0, dtype=np.float32))
    assert np.all(weights["b1"][4:] == 0.0)


def test_strided_slice():
    parser = SafeASTWeightParser(strict=True)
    text = "```edits\nW1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]\n```"
    weights, report = parser.parse_text(text)
    assert report.statements_applied == 1
    slice_vals = weights["W1"][3:600:60, 0]
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
    assert np.array_equal(slice_vals, expected)


def test_matrix_2d_assignment():
    parser = SafeASTWeightParser(strict=True)
    text = "```edits\nW2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]\n```"
    weights, report = parser.parse_text(text)
    assert report.statements_applied == 1
    expected = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 0.0]], dtype=np.float32)
    assert np.array_equal(weights["W2"][0:2, 0:3], expected)


def test_arithmetic_aug_assign():
    parser = SafeASTWeightParser(strict=True)
    text = """```edits
W3[:, 2] = 5.0
W3[:, 2] *= 2
W3[:, 2] += 3
W3[:, 2] -= 1
```"""
    weights, report = parser.parse_text(text)
    assert report.statements_applied == 4
    # 5.0 * 2 = 10; 10 + 3 = 13; 13 - 1 = 12
    assert np.all(weights["W3"][:, 2] == 12.0)


def test_sequential_overwrites():
    parser = SafeASTWeightParser(strict=True)
    text = """```edits
b3[:] = [1, 2, 3, 4, 5, 6]
b3[0] = 100
```"""
    weights, report = parser.parse_text(text)
    assert weights["b3"][0] == 100.0
    assert weights["b3"][1] == 2.0


def test_extract_only_last_edits_block():
    parser = SafeASTWeightParser(strict=True)
    text = """First try:
```edits
b1[0] = 10
```
Second and final try:
```edits
b1[0] = 42
```
"""
    weights, report = parser.parse_text(text)
    assert weights["b1"][0] == 42.0


def test_security_rejections():
    """Verify that dangerous constructs are strictly rejected."""
    dangerous_payloads = [
        "```edits\nimport os\n```",
        "```edits\n__import__('os').system('ls')\n```",
        "```edits\nopen('/etc/passwd')\n```",
        "```edits\nW1[0, 0] = eval('1 + 1')\n```",
        "```edits\nUnknownTensor[0] = 1\n```",
        "```edits\nwhile True: pass\n```",
        "```edits\ndef bad(): pass\n```",
        "```edits\nW1[0, 0] = 1e8\n```",  # Exceeds max abs limit 1e6
    ]
    for payload in dangerous_payloads:
        parser = SafeASTWeightParser(strict=True)
        with pytest.raises((WeightParserSecurityError, WeightParserSyntaxError)):
            parser.parse_text(payload)


def test_golden_astra_breakout_exact_reconstruction():
    """GOLDEN TEST: Parser must reconstruct Astra's original Breakout hand.npz with 0 diff."""
    # Load original answer text
    with open("research/original_response.md", "r") as f:
        response_text = f.read()

    # Load original released hand.npz
    ref_path = "_reference_repos/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz"
    ref_weights = load_weights_npz(ref_path)

    # Parse through safe parser
    parser = SafeASTWeightParser(strict=True)
    parsed_weights, report = parser.parse_text(response_text)

    # Verify report statistics match original apply.json
    assert report.block_found is True
    assert report.statements_total == 92
    assert report.statements_applied == 92
    assert len(report.errors) == 0
    assert report.nonzero_total == 815

    # Verify exact equality for all tensors
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        ref = ref_weights[name]
        parsed = parsed_weights[name]
        assert np.array_equal(parsed, ref), (
            f"Tensor {name} does not match reference: max diff = {np.max(np.abs(parsed - ref))}"
        )
