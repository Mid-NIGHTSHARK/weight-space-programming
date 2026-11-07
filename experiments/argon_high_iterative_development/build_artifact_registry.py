#!/usr/bin/env python3
"""Build a SHA-256 inventory for this additive experiment directory."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "artifact_registry.json"
PROTECTED = ROOT / "experiments/corrected_evaluation/preserved_artifacts_sha256.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protected_baseline():
    manifest = json.loads(PROTECTED.read_text())
    missing = []
    mismatched = []
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            missing.append(record["path"])
            continue
        if sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            mismatched.append(record["path"])
    return {
        "manifest_path": str(PROTECTED.relative_to(ROOT)),
        "manifest_file_count": len(manifest["files"]),
        "missing_count": len(missing),
        "mismatch_count": len(mismatched),
        "missing_paths": missing,
        "mismatched_paths": mismatched,
    }


def build():
    paths = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == OUTPUT or "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        paths.append({
            "path": str(path.relative_to(HERE)),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    baseline = verify_protected_baseline()
    if baseline["missing_count"] or baseline["mismatch_count"]:
        raise RuntimeError(f"Protected baseline integrity check failed: {baseline}")
    return {
        "schema_version": 1,
        "group_id": "gemini_argon_high_iterative_development",
        "scope": "All files in this additive experiment directory except artifact_registry.json itself.",
        "source_commit": "e148ea353c3cf0307c69de43f5e49201d2c4478f",
        "protected_baseline_integrity": baseline,
        "file_count": len(paths),
        "files": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Regenerate this derived hash inventory")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.refresh:
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}; pass --refresh to rebuild this derived registry")
    data = build()
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Registered {data['file_count']} files at {OUTPUT}")


if __name__ == "__main__":
    main()
