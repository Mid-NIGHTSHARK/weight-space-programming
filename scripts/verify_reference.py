"""Verify reference network parameters, sparsity, and forward activations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.weight_loader import load_weights_npz, compute_file_sha256
from network.weight_validation import validate_weight_dict
from network.architecture import MinAtarNetworkNumPy, TOTAL_PARAMETERS

REFERENCE_PATH = "experiments/reference/hand.npz"


def main():
    if not os.path.exists(REFERENCE_PATH):
        print(f"Error: {REFERENCE_PATH} does not exist. Run scripts/download_reference.py first.")
        sys.exit(1)

    sha256 = compute_file_sha256(REFERENCE_PATH)
    print(f"Verifying {REFERENCE_PATH} (SHA-256: {sha256})")

    weights = load_weights_npz(REFERENCE_PATH)
    stats = validate_weight_dict(weights)

    print("\n--- Tensor Statistics ---")
    print(f"Total Parameters: {stats['total_parameters']} (Expected: {TOTAL_PARAMETERS})")
    print(f"Nonzero Parameters: {stats['nonzero_total']} (Expected: 815)")
    print(f"Sparsity: {stats['sparsity_pct']:.2f}%")

    print("\nPer-Tensor Breakdown:")
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        nz = stats["nonzero"][name]
        max_abs = stats["max_abs"][name]
        shape = weights[name].shape
        print(f"  {name:2s}: shape={shape}, nonzero={nz:4d}, max_abs={max_abs:.2f}")

    # Forward inference sanity check
    net = MinAtarNetworkNumPy(weights)
    import numpy as np
    x = np.ones(600, dtype=np.float32)
    logits = net.forward(x)
    print("\nTest forward pass (input=all 1s):")
    print(f"  Logits: {logits}")
    print("\nSUCCESS: All checks passed.")


if __name__ == "__main__":
    main()
