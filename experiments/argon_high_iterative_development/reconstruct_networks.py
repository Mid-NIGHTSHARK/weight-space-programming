#!/usr/bin/env python3
"""Reconstruct and validate the three user-supplied Argon High versions."""

import hashlib
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from network.architecture import (  # noqa: E402
    MinAtarNetworkNumPy,
    MinAtarNetworkPyTorch,
    TENSOR_SHAPES,
    count_nonzero_parameters,
)
from network.weight_loader import save_weights_npz  # noqa: E402
from network.weight_validation import validate_weight_dict  # noqa: E402


SOURCES = {
    "argon_high_original": HERE / "source/argon_high_original_inline_v1.py",
    "argon_high_followup_1": HERE / "source/argon_high_followup_1.py",
    "argon_high_followup_2": HERE / "source/argon_high_followup_2.py",
}
BUILDERS = {
    "argon_high_original": "build_network",
    "argon_high_followup_1": "build_network",
    "argon_high_followup_2": "build_refined_network",
}
WEIGHT_FILES = {
    "argon_high_original": "argon_high_original.npz",
    "argon_high_followup_1": "argon_high_followup_1.npz",
    "argon_high_followup_2": "argon_high_followup_2.npz",
}


def reconstruct_all():
    versions = {}
    for version_id, source_path in SOURCES.items():
        namespace = runpy.run_path(str(source_path), run_name=f"source_{version_id}")
        built = namespace[BUILDERS[version_id]]()
        if isinstance(built, tuple):
            weights = dict(zip(TENSOR_SHAPES, built))
        else:
            weights = built
        validate_weight_dict(weights)
        for tensor_name, expected_shape in TENSOR_SHAPES.items():
            tensor = weights[tensor_name]
            if tensor.shape != expected_shape:
                raise ValueError(f"{version_id}.{tensor_name}: {tensor.shape} != {expected_shape}")
            if not np.isfinite(tensor).all():
                raise ValueError(f"{version_id}.{tensor_name} contains a non-finite value")
        versions[version_id] = weights
    return versions


def verify_forward(weights):
    numpy_network = MinAtarNetworkNumPy(weights)
    torch_network = MinAtarNetworkPyTorch()
    torch_network.load_from_numpy(numpy_network.weights)
    probe_rng = np.random.RandomState(20261107)
    probe = probe_rng.randint(0, 2, size=(16, 600)).astype(np.float32)
    numpy_logits = numpy_network.forward(probe)
    import torch

    with torch.no_grad():
        torch_logits = torch_network(torch.from_numpy(probe)).cpu().numpy()
    maximum_error = float(np.max(np.abs(numpy_logits - torch_logits)))
    if not np.allclose(numpy_logits, torch_logits, rtol=2e-5, atol=2e-5):
        raise AssertionError(f"NumPy/PyTorch forward mismatch, max abs error={maximum_error}")
    return maximum_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="weights", help="New weights directory relative to this experiment directory")
    args = parser.parse_args()
    weights_dir = (HERE / args.output_dir).resolve()
    if HERE not in weights_dir.parents:
        raise ValueError("--output-dir must remain inside this experiment directory")
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / "reconstruction_manifest.json" if weights_dir == (HERE / "weights").resolve() else weights_dir / "reconstruction_manifest.json"
    expected_outputs = [weights_dir / value for value in WEIGHT_FILES.values()] + [manifest_path]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite reconstruction artifacts: " + ", ".join(existing))
    manifest = {
        "schema_version": 1,
        "reconstruction": "zero-initialized supplied assignments using project 600-32-32-6 architecture",
        "versions": {},
    }
    for version_id, weights in reconstruct_all().items():
        output_path = weights_dir / WEIGHT_FILES[version_id]
        weight_hash = save_weights_npz(str(output_path), weights, use_paper_keys=True)
        manifest["versions"][version_id] = {
            "source_file": str(SOURCES[version_id].relative_to(HERE)),
            "source_sha256": hashlib.sha256(SOURCES[version_id].read_bytes()).hexdigest(),
            "weight_file": str(output_path.relative_to(HERE)),
            "weight_sha256": weight_hash,
            "tensor_shapes": {name: list(tensor.shape) for name, tensor in weights.items()},
            "nonzero_by_tensor": count_nonzero_parameters(weights),
            "total_parameters": int(sum(tensor.size for tensor in weights.values())),
            "forward_max_abs_error_vs_pytorch": verify_forward(weights),
            "weight_max_abs": {name: float(np.max(np.abs(tensor))) for name, tensor in weights.items()},
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
