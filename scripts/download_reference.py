"""Script to obtain and verify the original Astra Breakout reference network."""

import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.weight_loader import compute_file_sha256

EXPECTED_SHA256 = "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"
LOCAL_SOURCE = "_reference_repos/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz"
DESTINATION = "experiments/reference/hand.npz"


def main():
    os.makedirs(os.path.dirname(DESTINATION), exist_ok=True)
    if os.path.exists(LOCAL_SOURCE):
        print(f"Copying reference network from {LOCAL_SOURCE} to {DESTINATION}...")
        shutil.copy2(LOCAL_SOURCE, DESTINATION)
    else:
        print(f"Local reference not found at {LOCAL_SOURCE}; attempting curl...")
        import subprocess
        url = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp5_write_from_scratch/networks/breakout/hand.npz"
        subprocess.run(["curl", "-sL", url, "-o", DESTINATION], check=True)

    actual_hash = compute_file_sha256(DESTINATION)
    print(f"Destination: {DESTINATION}")
    print(f"Computed SHA-256: {actual_hash}")
    print(f"Expected SHA-256: {EXPECTED_SHA256}")

    if actual_hash != EXPECTED_SHA256:
        print("ERROR: SHA-256 hash mismatch!", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS: Reference network verified and ready.")


if __name__ == "__main__":
    main()

