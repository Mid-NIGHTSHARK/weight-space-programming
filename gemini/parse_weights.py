"""CLI parser to convert Gemini Argon's raw response into verified neural network weights."""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.weight_parser import SafeASTWeightParser
from network.weight_loader import save_weights_npz, compute_file_sha256
from network.weight_validation import validate_weight_dict

RESPONSE_RAW_PATH = "gemini/response_raw.txt"
OUTPUT_WEIGHTS_PATH = "gemini/generated_weights.npz"
METADATA_PATH = "gemini/metadata.json"


def main():
    if not os.path.exists(RESPONSE_RAW_PATH):
        print(f"Error: Raw response file not found at {RESPONSE_RAW_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(RESPONSE_RAW_PATH, "r") as f:
        content = f.read().strip()

    if not content or content.startswith("# PLACEHOLDER"):
        print("Error: gemini/response_raw.txt contains placeholder text. Paste Gemini Argon's response first.")
        sys.exit(1)

    response_sha256 = compute_file_sha256(RESPONSE_RAW_PATH)
    print(f"Processing raw response from {RESPONSE_RAW_PATH} (SHA-256: {response_sha256})")

    parser = SafeASTWeightParser(strict=False)
    weights, report = parser.parse_text(content)

    print("\n--- Parse Report ---")
    print(f"Edits Block Found: {report.block_found}")
    print(f"Statements Total: {report.statements_total}")
    print(f"Statements Applied: {report.statements_applied}")
    print(f"Errors Encountered: {len(report.errors)}")
    print(f"Nonzero Parameters: {report.nonzero_total} of 20,486")

    if report.errors:
        print("\nErrors Logged:")
        for err in report.errors:
            print(f"  Line {err.line_number} [{err.error_type}]: {err.message} (Stmt: '{err.raw_statement}')")

    # Validate weight dict
    stats = validate_weight_dict(weights)

    # Save to npz
    weights_sha256 = save_weights_npz(OUTPUT_WEIGHTS_PATH, weights, use_paper_keys=True)
    print(f"\nWeights saved to: {OUTPUT_WEIGHTS_PATH}")
    print(f"Weights SHA-256: {weights_sha256}")

    # Update metadata.json
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["generation"] = {
        "status": "completed",
        "response_raw_file": RESPONSE_RAW_PATH,
        "response_raw_sha256": response_sha256,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    metadata["parsed_weights"] = {
        "file": OUTPUT_WEIGHTS_PATH,
        "sha256": weights_sha256,
        "statements_total": report.statements_total,
        "statements_applied": report.statements_applied,
        "nonzero_parameters": report.nonzero_total,
        "max_abs": report.max_abs,
        "errors": [
            {
                "line": e.line_number,
                "statement": e.raw_statement,
                "error_type": e.error_type,
                "message": e.message,
            }
            for e in report.errors
        ],
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata updated in: {METADATA_PATH}")
    print("\nSUCCESS: Gemini weights parsed and ready for evaluation.")


if __name__ == "__main__":
    main()

