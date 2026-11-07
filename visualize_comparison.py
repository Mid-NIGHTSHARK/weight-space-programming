#!/usr/bin/env python3
"""Interactive Visual Comparison Application: Gemini Argon vs Astra Reference vs Trained Baseline.

Renders live, independent MinAtar Breakout environments for:
  1. Gemini Argon's generated neural-network weights (Zero-Shot)
  2. Astra's original reference weights (Original Experiment 5 hand.npz, if verified)
  3. PPO model trained from scratch (10M steps)

Features:
  - Full 6-action probability distributions for all models (summing to 100.0%)
  - Explicit visual distinction between Sampled Action and Executed Action (with sticky action alerts)
  - Full keyboard and button controls: Play, Pause, Step Frame, Reset, Prev/Next Episode
  - Boundary-safe episode navigation with dynamic button enable/disable states
  - Strict independent simulation stepping across all environments
  - Headless test and automated screenshot capture support
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.actions import ACTION_NAMES, NUM_ACTIONS
from environment.breakout import MinAtarBreakoutEnv
from network.architecture import MinAtarNetworkNumPy
from network.inference import PolicyAgent
from network.weight_loader import load_weights_npz, compute_file_sha256
from evaluation.corrected_analysis import ORIGINAL_EVALUATION_PATHS

# Standard verified reference path and hash
ASTRA_REFERENCE_PATH = "experiments/reference/hand.npz"
REPOSITORY_ROOT = Path(__file__).resolve().parent
CONDITION_REGISTRY_PATH = REPOSITORY_ROOT / "experiments/corrected_evaluation/condition_registry.json"
PRESERVATION_REGISTRY_PATH = REPOSITORY_ROOT / "experiments/corrected_evaluation/preserved_artifacts_sha256.json"
GPT6_DEFAULT_PATH = "experiments/strict_replication/generation_002_gpt6/output/gpt6_weights.npz"
SONNET55_DEFAULT_PATH = "experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz"
DEEPSEEK_DEFAULT_PATH = "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz"
GPT6_LUNA_DEFAULT_PATH = "experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz"
DEEPSEEK_CORRECTED_DEFAULT_PATH = "experiments/controlled_interventions/deepseek_corrected/output/deepseek_corrected_weights.npz"

CONDITION_TO_PANEL = {
    "astra_local_reference": "astra",
    "gemini_argon_4_medium": "argon",
    "claude_sonnet_55": "sonnet55",
    "deepseek_original": "deepseek",
    "gpt6_original": "gpt6",
    "gpt6_luna_extra_high": "gpt6_luna",
    "ppo_trained_baseline": "trained",
    "random_action_control": "random",
    "zero_weight_control": "zero",
}


def load_condition_registry(repository_root: Path = REPOSITORY_ROOT) -> dict:
    path = repository_root / "experiments/corrected_evaluation/condition_registry.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required condition registry is missing: {path}")
    return json.loads(path.read_text())


def verify_condition_weight(condition: dict, repository_root: Path = REPOSITORY_ROOT) -> Optional[str]:
    """Verify the canonical file for a condition against its recorded digest."""
    relative_path = condition.get("weight_path")
    expected_hash = condition.get("weight_sha256")
    if relative_path is None:
        if condition.get("id") == "zero_weight_control" and expected_hash == "in-memory-zero-tensors":
            return expected_hash
        if expected_hash is not None:
            raise ValueError(f"Condition {condition['id']} has a weight hash but no weight path")
        return None
    path = repository_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required benchmark weight file is missing: {path}")
    actual_hash = compute_file_sha256(str(path))
    if actual_hash != expected_hash:
        raise ValueError(
            f"Weight SHA-256 mismatch for {condition['id']}: expected {expected_hash}, observed {actual_hash}"
        )
    return actual_hash


def _verify_historical_csv(path: Path, relative_path: str, preserved_hashes: dict) -> str:
    expected = preserved_hashes.get(relative_path)
    if expected is None:
        raise ValueError(f"Historical result is absent from the Phase 2 preservation registry: {relative_path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Historical evaluation CSV SHA-256 mismatch for {relative_path}")
    return actual


def _load_original_benchmark(repository_root: Path) -> dict:
    registry = load_condition_registry(repository_root)
    preserved = json.loads((repository_root / "experiments/corrected_evaluation/preserved_artifacts_sha256.json").read_text())
    preserved_hashes = {item["path"]: item["sha256"] for item in preserved["files"]}
    original_metrics = {}
    for condition in registry["conditions"]:
        condition_id = condition["id"]
        verify_condition_weight(condition, repository_root)
        if condition_id not in ORIGINAL_EVALUATION_PATHS:
            continue
        groups = [42] if condition_id in {"random_action_control", "zero_weight_control"} else [42, 100, 200, 300] if condition_id == "astra_local_reference" else [42, 100, 200]
        scores = []
        source_hashes = {}
        for group in groups:
            relative = ORIGINAL_EVALUATION_PATHS[condition_id]
            relative = relative if "{group}" not in relative else relative.format(group=group)
            path = repository_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required historical benchmark CSV is missing: {path}")
            source_hashes[relative] = _verify_historical_csv(path, relative, preserved_hashes)
            with path.open(newline="") as fh:
                scores.extend(float(row["score"]) for row in csv.DictReader(fh))
        if not scores:
            raise ValueError(f"Historical dataset contains no scores for {condition_id}")
        values = np.asarray(scores, dtype=np.float64)
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        lengths = []
        selected_counts = {str(action): 0 for action in range(NUM_ACTIONS)}
        cutoff_count = 0
        score_rows = []
        for group in groups:
            relative = ORIGINAL_EVALUATION_PATHS[condition_id]
            relative = relative if "{group}" not in relative else relative.format(group=group)
            with (repository_root / relative).open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            score_rows.extend(rows)
        for row in score_rows:
            lengths.append(int(row["length"]))
            cutoff_count += int(str(row.get("step_cutoff", "false")).lower() in {"true", "1"})
            for action in range(NUM_ACTIONS):
                field = f"action_{action}_count"
                if field in row:
                    selected_counts[str(action)] += int(row[field])
        original_metrics[condition_id] = {
            "condition_id": condition_id,
            "model": condition["label"],
            "condition_class": condition["class"],
            "mean_score": float(np.mean(values)),
            "se_score": float(sd / np.sqrt(values.size)) if values.size > 1 else 0.0,
            "median_score": float(np.median(values)),
            "minimum_score": float(np.min(values)),
            "maximum_score": float(np.max(values)),
            "episodes": int(values.size),
            "mean_episode_length": float(np.mean(lengths)),
            "median_episode_length": float(np.median(lengths)),
            "maximum_episode_length": int(np.max(lengths)),
            "cutoff_count": cutoff_count,
            "cutoff_percentage": 100.0 * cutoff_count / len(score_rows),
            "selected_action_counts": selected_counts,
            "weight_sha256": verify_condition_weight(condition, repository_root),
            "dataset": "original_historical_csvs",
            "protocol": "legacy same-integer environment and policy seeds",
            "source_csv_sha256": source_hashes,
        }
    ordered = sorted(original_metrics.values(), key=lambda item: item["mean_score"], reverse=True)
    for rank, item in enumerate(ordered, 1):
        item["rank"] = rank
        if item["condition_id"] == "astra_local_reference":
            item["model"] = "Astra reference (local reevaluation, 4,000 episodes)"
    return {CONDITION_TO_PANEL[item["condition_id"]]: item for item in ordered}


def load_verified_benchmark_registry(
    dataset: str = "corrected",
    run_dir: Optional[Path] = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    """Load only complete, hash-verified metrics; never fall back to constants."""
    if dataset == "original":
        return _load_original_benchmark(repository_root)
    if dataset != "corrected":
        raise ValueError("dataset must be 'corrected' or 'original'")
    run_dir = Path(run_dir) if run_dir is not None else repository_root / "experiments/corrected_evaluation/full_run"
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corrected evaluation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("run_kind") != "full":
        raise ValueError(f"Corrected benchmark is not complete (status={manifest.get('status')!r})")
    condition_registry_bytes = (repository_root / "experiments/corrected_evaluation/condition_registry.json").read_bytes()
    config_bytes = (repository_root / "experiments/corrected_evaluation/evaluation_config.json").read_bytes()
    if hashlib.sha256(condition_registry_bytes).hexdigest() != manifest.get("condition_registry_sha256"):
        raise ValueError("Corrected benchmark condition registry hash mismatch")
    if hashlib.sha256(config_bytes).hexdigest() != manifest.get("evaluation_config_sha256"):
        raise ValueError("Corrected benchmark evaluation configuration hash mismatch")

    metrics_path = run_dir / "analysis/comparative_metrics.json"
    expected_metrics_hash = manifest.get("analysis_artifact_sha256", {}).get("comparative_metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Required corrected metrics are missing: {metrics_path}")
    actual_metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if actual_metrics_hash != expected_metrics_hash:
        raise ValueError("Corrected comparative metrics SHA-256 mismatch")
    records = json.loads(metrics_path.read_text())
    registry_by_id = {item["id"]: item for item in manifest["conditions"]}
    if "deepseek_corrected" in registry_by_id:
        raise ValueError("Researcher intervention cannot enter the primary corrected dataset")
    if {item["condition_id"] for item in records} != set(registry_by_id):
        raise ValueError("Corrected comparative metrics are missing or contain unexpected conditions")

    ordered = sorted(records, key=lambda item: item["mean_score"], reverse=True)
    output = {}
    for rank, item in enumerate(ordered, 1):
        condition = registry_by_id[item["condition_id"]]
        if item.get("dataset") != manifest.get("dataset") or item.get("weight_sha256") != condition.get("weight_sha256"):
            raise ValueError(f"Corrected metric provenance mismatch for {condition['id']}")
        verified_hash = verify_condition_weight(condition, repository_root)
        if verified_hash != item.get("weight_sha256"):
            raise ValueError(f"Corrected metric weight hash does not match source file for {condition['id']}")
        panel_key = CONDITION_TO_PANEL.get(condition["id"])
        if panel_key is None:
            raise ValueError(f"No visualizer panel mapping for corrected condition {condition['id']}")
        item["rank"] = rank
        item["condition_class"] = condition["class"]
        if condition["id"] == "astra_local_reference":
            item["model"] = "Astra reference (local reevaluation, 3,000 episodes)"
        output[panel_key] = item
    return output


def benchmark_for_loaded_weights(benchmark: Optional[dict], actual_weight_sha256: Optional[str]) -> Optional[dict]:
    """Attach a score only when the displayed tensor file is the evaluated file."""
    if benchmark is None or actual_weight_sha256 is None:
        return None
    if benchmark.get("weight_sha256") is None or benchmark["weight_sha256"] != actual_weight_sha256:
        return None
    return benchmark


def inspect_visualizer_weight(
    path: Optional[str],
    condition_id: Optional[str],
    label: str,
    required: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
) -> Optional[str]:
    """Hash every displayed file and distinguish canonical from custom weights."""
    if not path:
        if required:
            raise FileNotFoundError(f"Required {label} weight file path was not supplied")
        return None
    weight_path = Path(path)
    if not weight_path.is_absolute():
        weight_path = repository_root / weight_path
    if not weight_path.is_file():
        if required:
            raise FileNotFoundError(f"Required {label} weights are missing: {weight_path}")
        print(f"[Excluded] {label} weights are missing: {weight_path}")
        return None

    actual_hash = compute_file_sha256(str(weight_path))
    conditions = {item["id"]: item for item in load_condition_registry(repository_root)["conditions"]}
    condition = conditions.get(condition_id) if condition_id else None
    if condition is None or condition.get("weight_path") is None:
        print(f"[Unverified condition] {label} SHA-256 calculated ({actual_hash[:16]}…); no benchmark score will be attached.")
        return str(weight_path)
    canonical_path = repository_root / condition["weight_path"]
    expected_hash = condition["weight_sha256"]
    if actual_hash == expected_hash:
        print(f"[Verified hash match] {label} SHA-256 matches the condition registry ({actual_hash[:16]}…).")
    elif weight_path.resolve() == canonical_path.resolve():
        raise ValueError(
            f"Canonical {label} weight hash mismatch: expected {expected_hash}, observed {actual_hash}"
        )
    else:
        print(f"[Custom weights] {label} SHA-256 {actual_hash[:16]}… does not match the evaluated artifact; score withheld.")
    return str(weight_path)


class VisualizerState:
    """Manages independent game simulations and policies for all benchmark conditions."""

    def __init__(
        self,
        argon_weights_path: str,
        trained_weights_path: str,
        astra_weights_path: Optional[str] = None,
        gpt6_weights_path: Optional[str] = None,
        gpt6_luna_weights_path: Optional[str] = None,
        sonnet55_weights_path: Optional[str] = None,
        deepseek_weights_path: Optional[str] = None,
        deepseek_corrected_weights_path: Optional[str] = None,
        argon_high_weights_path: Optional[str] = None,
        initial_seed: int = 42,
        benchmark_dataset: str = "corrected",
    ):
        self.argon_weights_path = argon_weights_path
        self.trained_weights_path = trained_weights_path
        self.astra_weights_path = astra_weights_path
        self.gpt6_weights_path = gpt6_weights_path
        self.gpt6_luna_weights_path = gpt6_luna_weights_path
        self.sonnet55_weights_path = sonnet55_weights_path
        self.deepseek_weights_path = deepseek_weights_path
        self.deepseek_corrected_weights_path = deepseek_corrected_weights_path
        self.argon_high_weights_path = argon_high_weights_path
        self.current_seed = initial_seed
        self.benchmark_dataset = benchmark_dataset
        self.loaded_weight_hashes = {}

        # 1. Load Argon policy
        if not os.path.exists(argon_weights_path):
            raise FileNotFoundError(f"Argon weights not found at {argon_weights_path}")
        self.argon_weights = load_weights_npz(argon_weights_path)
        self.loaded_weight_hashes["argon"] = compute_file_sha256(argon_weights_path)
        self.argon_net = MinAtarNetworkNumPy(self.argon_weights)
        self.argon_agent = PolicyAgent(self.argon_net)

        # 2. Load Trained policy
        if not os.path.exists(trained_weights_path):
            raise FileNotFoundError(f"Trained model weights not found at {trained_weights_path}")
        self.trained_weights = load_weights_npz(trained_weights_path)
        self.loaded_weight_hashes["trained"] = compute_file_sha256(trained_weights_path)
        self.trained_net = MinAtarNetworkNumPy(self.trained_weights)
        self.trained_agent = PolicyAgent(self.trained_net)

        # 3. Optional Astra Reference policy
        self.has_astra = False
        if astra_weights_path is not None and os.path.exists(astra_weights_path):
            self.astra_weights = load_weights_npz(astra_weights_path)
            self.loaded_weight_hashes["astra"] = compute_file_sha256(astra_weights_path)
            self.astra_net = MinAtarNetworkNumPy(self.astra_weights)
            self.astra_agent = PolicyAgent(self.astra_net)
            self.env_astra = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_astra = True

        # 4. Optional GPT-6 Original policy
        self.has_gpt6 = False
        if gpt6_weights_path is not None and os.path.exists(gpt6_weights_path):
            self.gpt6_weights = load_weights_npz(gpt6_weights_path)
            self.loaded_weight_hashes["gpt6"] = compute_file_sha256(gpt6_weights_path)
            self.gpt6_net = MinAtarNetworkNumPy(self.gpt6_weights)
            self.gpt6_agent = PolicyAgent(self.gpt6_net)
            self.env_gpt6 = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_gpt6 = True

        # 5. Optional GPT-6 Luna Extra High policy
        self.has_gpt6_luna = False
        if gpt6_luna_weights_path is not None and os.path.exists(gpt6_luna_weights_path):
            self.gpt6_luna_weights = load_weights_npz(gpt6_luna_weights_path)
            self.loaded_weight_hashes["gpt6_luna"] = compute_file_sha256(gpt6_luna_weights_path)
            self.gpt6_luna_net = MinAtarNetworkNumPy(self.gpt6_luna_weights)
            self.gpt6_luna_agent = PolicyAgent(self.gpt6_luna_net)
            self.env_gpt6_luna = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_gpt6_luna = True

        # 6. Optional Claude Sonnet 5.5 policy
        self.has_sonnet55 = False
        if sonnet55_weights_path is not None and os.path.exists(sonnet55_weights_path):
            self.sonnet55_weights = load_weights_npz(sonnet55_weights_path)
            self.loaded_weight_hashes["sonnet55"] = compute_file_sha256(sonnet55_weights_path)
            self.sonnet55_net = MinAtarNetworkNumPy(self.sonnet55_weights)
            self.sonnet55_agent = PolicyAgent(self.sonnet55_net)
            self.env_sonnet55 = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_sonnet55 = True

        # 7. Optional DeepSeek policy (Original Zero-Shot)
        self.has_deepseek = False
        if deepseek_weights_path is not None and os.path.exists(deepseek_weights_path):
            self.deepseek_weights = load_weights_npz(deepseek_weights_path)
            self.loaded_weight_hashes["deepseek"] = compute_file_sha256(deepseek_weights_path)
            self.deepseek_net = MinAtarNetworkNumPy(self.deepseek_weights)
            self.deepseek_agent = PolicyAgent(self.deepseek_net)
            self.env_deepseek = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_deepseek = True

        # 7. Optional DeepSeek Corrected policy (Researcher-Modified Intervention)
        self.has_deepseek_corrected = False
        if deepseek_corrected_weights_path is not None and os.path.exists(deepseek_corrected_weights_path):
            self.deepseek_corrected_weights = load_weights_npz(deepseek_corrected_weights_path)
            self.loaded_weight_hashes["deepseek_corrected"] = compute_file_sha256(deepseek_corrected_weights_path)
            self.deepseek_corrected_net = MinAtarNetworkNumPy(self.deepseek_corrected_weights)
            self.deepseek_corrected_agent = PolicyAgent(self.deepseek_corrected_net)
            self.env_deepseek_corrected = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_deepseek_corrected = True

        # 7. Gemini Argon 4 High placeholder status
        self.has_argon_high = False
        if argon_high_weights_path is not None and os.path.exists(argon_high_weights_path):
            self.argon_high_weights = load_weights_npz(argon_high_weights_path)
            self.loaded_weight_hashes["argon_high"] = compute_file_sha256(argon_high_weights_path)
            self.argon_high_net = MinAtarNetworkNumPy(self.argon_high_weights)
            self.argon_high_agent = PolicyAgent(self.argon_high_net)
            self.env_argon_high = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_argon_high = True
        self.argon_high_available = self.has_argon_high

        # Environment instances
        self.env_argon = MinAtarBreakoutEnv(sticky_action_prob=0.1)
        self.env_trained = MinAtarBreakoutEnv(sticky_action_prob=0.1)

        self.reset_episode(self.current_seed)

    def reset_episode(self, seed: int):
        """Reset all environments and agents to the specified seed."""
        self.current_seed = max(1, seed)

        # Re-seed agents
        self.argon_agent.rng = np.random.RandomState(self.current_seed)
        self.trained_agent.rng = np.random.RandomState(self.current_seed)

        # Reset environments
        self.obs_argon = self.env_argon.reset(seed=self.current_seed)
        self.obs_trained = self.env_trained.reset(seed=self.current_seed)

        self.argon_done = False
        self.trained_done = False

        self.argon_info = {}
        self.trained_info = {}

        self.argon_executed_action: Optional[int] = None
        self.trained_executed_action: Optional[int] = None

        self.argon_sticky_override: bool = False
        self.trained_sticky_override: bool = False

        # Compute initial policy outputs
        self._update_argon_policy()
        self._update_trained_policy()

        # Handle Astra if enabled
        if self.has_astra:
            self.astra_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_astra = self.env_astra.reset(seed=self.current_seed)
            self.astra_done = False
            self.astra_info = {}
            self.astra_executed_action = None
            self.astra_sticky_override = False
            self._update_astra_policy()

        # Handle GPT-6 if enabled
        if self.has_gpt6:
            self.gpt6_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_gpt6 = self.env_gpt6.reset(seed=self.current_seed)
            self.gpt6_done = False
            self.gpt6_info = {}
            self.gpt6_executed_action = None
            self.gpt6_sticky_override = False
            self._update_gpt6_policy()

        # Handle GPT-6 Luna if enabled
        if self.has_gpt6_luna:
            self.gpt6_luna_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_gpt6_luna = self.env_gpt6_luna.reset(seed=self.current_seed)
            self.gpt6_luna_done = False
            self.gpt6_luna_info = {}
            self.gpt6_luna_executed_action = None
            self.gpt6_luna_sticky_override = False
            self._update_gpt6_luna_policy()

        # Handle Claude Sonnet 5.5 if enabled
        if self.has_sonnet55:
            self.sonnet55_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_sonnet55 = self.env_sonnet55.reset(seed=self.current_seed)
            self.sonnet55_done = False
            self.sonnet55_info = {}
            self.sonnet55_executed_action = None
            self.sonnet55_sticky_override = False
            self._update_sonnet55_policy()

        # Handle DeepSeek if enabled
        if self.has_deepseek:
            self.deepseek_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_deepseek = self.env_deepseek.reset(seed=self.current_seed)
            self.deepseek_done = False
            self.deepseek_info = {}
            self.deepseek_executed_action = None
            self.deepseek_sticky_override = False
            self._update_deepseek_policy()

        # Handle DeepSeek Corrected if enabled (Researcher Intervention)
        if self.has_deepseek_corrected:
            self.deepseek_corrected_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_deepseek_corrected = self.env_deepseek_corrected.reset(seed=self.current_seed)
            self.deepseek_corrected_done = False
            self.deepseek_corrected_info = {}
            self.deepseek_corrected_executed_action = None
            self.deepseek_corrected_sticky_override = False
            self._update_deepseek_corrected_policy()

        # Handle Argon High if enabled
        if self.has_argon_high:
            self.argon_high_agent.rng = np.random.RandomState(self.current_seed)
            self.obs_argon_high = self.env_argon_high.reset(seed=self.current_seed)
            self.argon_high_done = False
            self.argon_high_info = {}
            self.argon_high_executed_action = None
            self.argon_high_sticky_override = False
            self._update_argon_high_policy()

    def _update_argon_policy(self):
        if not self.argon_done:
            self.argon_selected_action, self.argon_probs, self.argon_logits = self.argon_agent.get_action(
                self.obs_argon
            )

    def _update_trained_policy(self):
        if not self.trained_done:
            self.trained_selected_action, self.trained_probs, self.trained_logits = self.trained_agent.get_action(
                self.obs_trained
            )

    def _update_astra_policy(self):
        if self.has_astra and not self.astra_done:
            self.astra_selected_action, self.astra_probs, self.astra_logits = self.astra_agent.get_action(
                self.obs_astra
            )

    def _update_gpt6_policy(self):
        if self.has_gpt6 and not self.gpt6_done:
            self.gpt6_selected_action, self.gpt6_probs, self.gpt6_logits = self.gpt6_agent.get_action(
                self.obs_gpt6
            )

    def _update_gpt6_luna_policy(self):
        if self.has_gpt6_luna and not self.gpt6_luna_done:
            self.gpt6_luna_selected_action, self.gpt6_luna_probs, self.gpt6_luna_logits = self.gpt6_luna_agent.get_action(
                self.obs_gpt6_luna
            )

    def _update_sonnet55_policy(self):
        if self.has_sonnet55 and not self.sonnet55_done:
            self.sonnet55_selected_action, self.sonnet55_probs, self.sonnet55_logits = self.sonnet55_agent.get_action(
                self.obs_sonnet55
            )

    def _update_deepseek_policy(self):
        if self.has_deepseek and not self.deepseek_done:
            self.deepseek_selected_action, self.deepseek_probs, self.deepseek_logits = self.deepseek_agent.get_action(
                self.obs_deepseek
            )

    def _update_deepseek_corrected_policy(self):
        if self.has_deepseek_corrected and not self.deepseek_corrected_done:
            (
                self.deepseek_corrected_selected_action,
                self.deepseek_corrected_probs,
                self.deepseek_corrected_logits,
            ) = self.deepseek_corrected_agent.get_action(self.obs_deepseek_corrected)

    def _update_argon_high_policy(self):
        if self.has_argon_high and not self.argon_high_done:
            self.argon_high_selected_action, self.argon_high_probs, self.argon_high_logits = self.argon_high_agent.get_action(
                self.obs_argon_high
            )

    # Backward compatibility aliases for selected actions
    @property
    def argon_action(self) -> int:
        return self.argon_selected_action

    @property
    def trained_action(self) -> int:
        return self.trained_selected_action

    @property
    def astra_action(self) -> int:
        return getattr(self, "astra_selected_action", 0)

    @property
    def gpt6_action(self) -> int:
        return getattr(self, "gpt6_selected_action", 0)

    @property
    def gpt6_luna_action(self) -> int:
        return getattr(self, "gpt6_luna_selected_action", 0)

    @property
    def sonnet55_action(self) -> int:
        return getattr(self, "sonnet55_selected_action", 0)

    @property
    def deepseek_action(self) -> int:
        return getattr(self, "deepseek_selected_action", 0)

    @property
    def deepseek_corrected_action(self) -> int:
        return getattr(self, "deepseek_corrected_selected_action", 0)

    def step(self):
        """Advance each active environment by 1 frame.
        
        Preserves individual terminal states and records executed actions and sticky overrides.
        """
        # Step Argon
        if not self.argon_done:
            action_to_step = self.argon_selected_action
            self.obs_argon, _, self.argon_done, self.argon_info = self.env_argon.step(action_to_step)
            self.argon_executed_action = self.argon_info.get("executed_action", action_to_step)
            self.argon_sticky_override = self.argon_info.get("sticky_override", False)
            if not self.argon_done:
                self._update_argon_policy()

        # Step Trained
        if not self.trained_done:
            action_to_step = self.trained_selected_action
            self.obs_trained, _, self.trained_done, self.trained_info = self.env_trained.step(action_to_step)
            self.trained_executed_action = self.trained_info.get("executed_action", action_to_step)
            self.trained_sticky_override = self.trained_info.get("sticky_override", False)
            if not self.trained_done:
                self._update_trained_policy()

        # Step Astra if enabled
        if self.has_astra:
            if not self.astra_done:
                action_to_step = self.astra_selected_action
                self.obs_astra, _, self.astra_done, self.astra_info = self.env_astra.step(action_to_step)
                self.astra_executed_action = self.astra_info.get("executed_action", action_to_step)
                self.astra_sticky_override = self.astra_info.get("sticky_override", False)
                if not self.astra_done:
                    self._update_astra_policy()

        # Step GPT-6 if enabled
        if self.has_gpt6:
            if not self.gpt6_done:
                action_to_step = self.gpt6_selected_action
                self.obs_gpt6, _, self.gpt6_done, self.gpt6_info = self.env_gpt6.step(action_to_step)
                self.gpt6_executed_action = self.gpt6_info.get("executed_action", action_to_step)
                self.gpt6_sticky_override = self.gpt6_info.get("sticky_override", False)
                if not self.gpt6_done:
                    self._update_gpt6_policy()

        # Step GPT-6 Luna if enabled
        if self.has_gpt6_luna:
            if not self.gpt6_luna_done:
                action_to_step = self.gpt6_luna_selected_action
                self.obs_gpt6_luna, _, self.gpt6_luna_done, self.gpt6_luna_info = self.env_gpt6_luna.step(action_to_step)
                self.gpt6_luna_executed_action = self.gpt6_luna_info.get("executed_action", action_to_step)
                self.gpt6_luna_sticky_override = self.gpt6_luna_info.get("sticky_override", False)
                if not self.gpt6_luna_done:
                    self._update_gpt6_luna_policy()

        # Step Claude Sonnet 5.5 if enabled
        if self.has_sonnet55:
            if not self.sonnet55_done:
                action_to_step = self.sonnet55_selected_action
                self.obs_sonnet55, _, self.sonnet55_done, self.sonnet55_info = self.env_sonnet55.step(action_to_step)
                self.sonnet55_executed_action = self.sonnet55_info.get("executed_action", action_to_step)
                self.sonnet55_sticky_override = self.sonnet55_info.get("sticky_override", False)
                if not self.sonnet55_done:
                    self._update_sonnet55_policy()

        # Step DeepSeek (Original) if enabled
        if self.has_deepseek:
            if not self.deepseek_done:
                action_to_step = self.deepseek_selected_action
                self.obs_deepseek, _, self.deepseek_done, self.deepseek_info = self.env_deepseek.step(action_to_step)
                self.deepseek_executed_action = self.deepseek_info.get("executed_action", action_to_step)
                self.deepseek_sticky_override = self.deepseek_info.get("sticky_override", False)
                if not self.deepseek_done:
                    self._update_deepseek_policy()

        # Step DeepSeek Corrected if enabled (Researcher Intervention)
        if self.has_deepseek_corrected:
            if not self.deepseek_corrected_done:
                action_to_step = self.deepseek_corrected_selected_action
                self.obs_deepseek_corrected, _, self.deepseek_corrected_done, self.deepseek_corrected_info = (
                    self.env_deepseek_corrected.step(action_to_step)
                )
                self.deepseek_corrected_executed_action = self.deepseek_corrected_info.get("executed_action", action_to_step)
                self.deepseek_corrected_sticky_override = self.deepseek_corrected_info.get("sticky_override", False)
                if not self.deepseek_corrected_done:
                    self._update_deepseek_corrected_policy()

        # Step Argon High if enabled
        if self.has_argon_high:
            if not self.argon_high_done:
                action_to_step = self.argon_high_selected_action
                self.obs_argon_high, _, self.argon_high_done, self.argon_high_info = self.env_argon_high.step(action_to_step)
                self.argon_high_executed_action = self.argon_high_info.get("executed_action", action_to_step)
                self.argon_high_sticky_override = self.argon_high_info.get("sticky_override", False)
                if not self.argon_high_done:
                    self._update_argon_high_policy()

        # Return backward-compatible tuple:
        if self.has_deepseek_corrected:
            return (
                self.argon_done,
                self.trained_done,
                self.astra_done,
                self.gpt6_done,
                self.sonnet55_done,
                self.deepseek_done,
                self.deepseek_corrected_done,
            )

        if self.has_deepseek:
            if self.has_gpt6_luna:
                return (
                    self.argon_done,
                    self.trained_done,
                    self.astra_done,
                    self.gpt6_done,
                    self.gpt6_luna_done,
                    self.sonnet55_done,
                    self.deepseek_done,
                )
            return self.argon_done, self.trained_done, self.astra_done, self.gpt6_done, self.sonnet55_done, self.deepseek_done

        if self.has_sonnet55:
            if self.has_gpt6 and self.has_astra:
                return self.argon_done, self.trained_done, self.astra_done, self.gpt6_done, self.sonnet55_done
            if self.has_gpt6:
                return self.argon_done, self.trained_done, self.gpt6_done, self.sonnet55_done
            if self.has_astra:
                return self.argon_done, self.trained_done, self.astra_done, self.sonnet55_done
            return self.argon_done, self.trained_done, self.sonnet55_done

        if self.has_gpt6:
            if self.has_astra:
                return self.argon_done, self.trained_done, self.astra_done, self.gpt6_done
            return self.argon_done, self.trained_done, self.gpt6_done
        if self.has_astra:
            return self.argon_done, self.trained_done, self.astra_done
        return self.argon_done, self.trained_done

    @property
    def all_done(self) -> bool:
        """Whether all active environments have completed their episodes."""
        done = self.argon_done and self.trained_done
        if self.has_astra:
            done = done and self.astra_done
        if self.has_gpt6:
            done = done and self.gpt6_done
        if self.has_gpt6_luna:
            done = done and self.gpt6_luna_done
        if self.has_sonnet55:
            done = done and self.sonnet55_done
        if self.has_deepseek:
            done = done and self.deepseek_done
        if self.has_deepseek_corrected:
            done = done and self.deepseek_corrected_done
        if self.has_argon_high:
            done = done and self.argon_high_done
        return done


class BreakoutVisualizerGUI:
    """Tkinter interactive visualizer application for Breakout policies."""

    def __init__(self, state: VisualizerState, root, initial_fps: int = 15):
        import tkinter as tk

        self.state = state
        self.root = root

        # Count active models
        active_count = (
            2
            + (1 if self.state.has_astra else 0)
            + (1 if self.state.has_gpt6 else 0)
            + (1 if self.state.has_gpt6_luna else 0)
            + (1 if self.state.has_sonnet55 else 0)
            + (1 if self.state.has_deepseek else 0)
            + (1 if self.state.has_deepseek_corrected else 0)
        )

        # Window configuration
        self.root.configure(bg="#141419")
        if active_count >= 7:
            self.root.title("MinAtar Breakout: Seven-Way Performance-Ranked Policy Lab (Astra, Argon, Sonnet 5.5, PPO, GPT-6, Luna, DeepSeek)")
            self.root.geometry("1820x870")
            self.root.minsize(1540, 750)
            self.cell_size = 14
        elif active_count == 6:
            self.root.title("MinAtar Breakout: Six-Way Policy Comparison Lab (Argon, Astra, GPT-6, Sonnet 5.5, DeepSeek, PPO)")
            self.root.geometry("1800x870")
            self.root.minsize(1520, 750)
            self.cell_size = 16
        elif active_count == 5:
            self.root.title("MinAtar Breakout: Five-Way Policy Comparison Lab (Argon, Astra, GPT-6, Sonnet 5.5, PPO)")
            self.root.geometry("1620x860")
            self.root.minsize(1380, 750)
            self.cell_size = 19
        elif active_count == 4:
            self.root.title("MinAtar Breakout: Four-Way Policy Comparison Lab (Argon, Astra, GPT-6, PPO)")
            self.root.geometry("1480x860")
            self.root.minsize(1280, 750)
            self.cell_size = 21
        elif active_count == 3:
            self.root.title("MinAtar Breakout: Gemini Argon vs. Astra Reference vs. Trained PPO")
            self.root.geometry("1300x820")
            self.root.minsize(1150, 750)
            self.cell_size = 26
        else:
            self.root.title("MinAtar Breakout: Gemini Argon vs. Trained Model (PPO)")
            self.root.geometry("980x820")
            self.root.minsize(850, 750)
            self.cell_size = 28

        self.board_pixels = 10 * self.cell_size

        # Colors
        self.COLOR_BG = "#141419"
        self.COLOR_PANEL = "#1e1e26"
        self.COLOR_BOARD_BG = "#0a0a0f"
        self.COLOR_GRID = "#1a1a24"
        self.COLOR_PADDLE_ARGON = "#00e5ff"             # Cyan
        self.COLOR_PADDLE_ASTRA = "#ffd700"             # Gold
        self.COLOR_PADDLE_GPT6 = "#d070ff"              # Orchid / Violet
        self.COLOR_PADDLE_GPT6_LUNA = "#ff4081"         # Rose / Pink
        self.COLOR_PADDLE_SONNET55 = "#ff7043"          # Coral / Warm Orange-Red
        self.COLOR_PADDLE_DEEPSEEK = "#3949ab"          # Deep Indigo (Zero-shot)
        self.COLOR_PADDLE_DEEPSEEK_CORRECTED = "#00b0ff" # Electric Sky Blue (Researcher Intervention)
        self.COLOR_PADDLE_TRAINED = "#00e676"           # Bright Green
        self.COLOR_BALL = "#ffff00"                     # Bright Yellow
        self.COLOR_TRAIL = "#ff9100"                    # Orange
        self.COLOR_BRICK_ROW1 = "#ff5252"               # Red
        self.COLOR_BRICK_ROW2 = "#ff9800"               # Amber
        self.COLOR_BRICK_ROW3 = "#4caf50"               # Green
        self.COLOR_TEXT = "#e0e0e0"
        self.COLOR_TEXT_MUTED = "#888899"
        self.COLOR_STICKY = "#ff9800"                   # Amber alert
        self.COLOR_TERMINAL = "#ff5252"

        # State Variables
        self.is_running = tk.BooleanVar(value=True)
        self.fps_var = tk.IntVar(value=initial_fps)
        self.seed_var = tk.IntVar(value=self.state.current_seed)
        self.auto_advance_job = None

        self._build_ui()
        self._bind_events()
        self.redraw()
        self.root.after(100, self._game_loop)

    def _build_ui(self):
        import tkinter as tk

        # --- HEADER ---
        header = tk.Frame(self.root, bg=self.COLOR_PANEL, pady=8, padx=15)
        header.pack(fill=tk.X, side=tk.TOP)

        active_count = (
            2
            + (1 if self.state.has_astra else 0)
            + (1 if self.state.has_gpt6 else 0)
            + (1 if self.state.has_gpt6_luna else 0)
            + (1 if self.state.has_sonnet55 else 0)
            + (1 if self.state.has_deepseek else 0)
            + (1 if self.state.has_deepseek_corrected else 0)
        )
        title_map = {
            8: "BREAKOUT EIGHT-WAY PERFORMANCE-RANKED POLICY LAB (*DeepSeek Corrected is an intervention)",
            7: "BREAKOUT SEVEN-WAY ZERO-SHOT & BASELINE BENCHMARK LAB",
            6: "BREAKOUT SIX-WAY POLICY COMPARISON LAB (Argon, Astra, GPT-6, Sonnet 5.5, DeepSeek, PPO)",
            5: "BREAKOUT FIVE-WAY POLICY COMPARISON LAB (Argon, Astra, GPT-6, Sonnet 5.5, PPO)",
            4: "BREAKOUT FOUR-WAY POLICY COMPARISON LAB (Argon, Astra, GPT-6, PPO)",
            3: "BREAKOUT THREE-WAY POLICY COMPARISON LAB",
            2: "BREAKOUT TWO-WAY POLICY COMPARISON LAB",
        }
        title_text = title_map.get(active_count, "BREAKOUT POLICY LAB")
        protocol_title = (
            "CORRECTED EVALUATION · INDEPENDENT RNG"
            if self.state.benchmark_dataset == "corrected"
            else "HISTORICAL EVALUATION · LEGACY SAME-SEED RNG"
        )

        self.title_label = tk.Label(
            header,
            text=f"{protocol_title} · {title_text}",
            font=("Helvetica", 14, "bold"),
            fg="#ffffff",
            bg=self.COLOR_PANEL,
        )
        self.title_label.pack(side=tk.LEFT)

        # Right header controls: Argon High badge & Seed
        right_header = tk.Frame(header, bg=self.COLOR_PANEL)
        right_header.pack(side=tk.RIGHT)

        # Argon 4 High status badge
        btn_argon_high = tk.Button(
            right_header,
            text="⚡ Argon 4 High: Pending Experiment",
            command=self.show_argon_high_info,
            font=("Helvetica", 9, "bold"),
            bg="#382e18",
            fg="#ffb74d",
            activebackground="#4e3e20",
            activeforeground="#ffe082",
            relief=tk.FLAT,
            bd=1,
            padx=8,
            pady=2,
            cursor="hand2",
        )
        btn_argon_high.pack(side=tk.LEFT, padx=(0, 15))

        self.seed_label = tk.Label(
            right_header,
            text=f"Seed: {self.state.current_seed}",
            font=("Helvetica", 13, "bold"),
            fg="#00e5ff",
            bg=self.COLOR_PANEL,
        )
        self.seed_label.pack(side=tk.RIGHT)

        # --- MAIN ARENA (MODELS SORTED BY PERFORMANCE RANK) ---
        self.arena = tk.Frame(self.root, bg=self.COLOR_BG, padx=8, pady=8)
        self.arena.pack(fill=tk.BOTH, expand=True)

        benchmarks = load_verified_benchmark_registry(dataset=self.state.benchmark_dataset)

        model_meta = {
            "astra": {
                "title": "ASTRA REFERENCE",
                "tag": "Local Reference",
                "cond_type": "Reference",
                "subtitle": "815 nz | 96.0% sparse",
                "mechanism": "Hand-coded reference policy",
                "color": self.COLOR_PADDLE_ASTRA,
                "is_active": self.state.has_astra,
                "env_attr": "env_astra",
                "done_attr": "astra_done",
                "sel_attr": "astra_selected_action",
                "exec_attr": "astra_executed_action",
                "sticky_attr": "astra_sticky_override",
                "probs_attr": "astra_probs",
            },
            "argon": {
                "title": "GEMINI ARGON 4M",
                "tag": "Zero-Shot Generation",
                "cond_type": "Zero-Shot",
                "subtitle": "817 nz | 96.0% sparse",
                "mechanism": "Channel-4 one-step target features",
                "color": self.COLOR_PADDLE_ARGON,
                "is_active": True,
                "env_attr": "env_argon",
                "done_attr": "argon_done",
                "sel_attr": "argon_selected_action",
                "exec_attr": "argon_executed_action",
                "sticky_attr": "argon_sticky_override",
                "probs_attr": "argon_probs",
            },
            "sonnet55": {
                "title": "CLAUDE SONNET 5.5",
                "tag": "Zero-Shot Generation",
                "cond_type": "Zero-Shot",
                "subtitle": "551 nz | 97.3% sparse",
                "mechanism": "Ball, trail, and paddle features",
                "color": self.COLOR_PADDLE_SONNET55,
                "is_active": self.state.has_sonnet55,
                "env_attr": "env_sonnet55",
                "done_attr": "sonnet55_done",
                "sel_attr": "sonnet55_selected_action",
                "exec_attr": "sonnet55_executed_action",
                "sticky_attr": "sonnet55_sticky_override",
                "probs_attr": "sonnet55_probs",
            },
            "deepseek_corrected": {
                "title": "DEEPSEEK (CORRECTED)*",
                "tag": "Researcher Intervention",
                "cond_type": "Intervention",
                "subtitle": "726 nz | 96.5% sparse",
                "mechanism": "W3[0:20,[2,4,5]]=0 (Corrected)",
                "color": self.COLOR_PADDLE_DEEPSEEK_CORRECTED,
                "is_active": self.state.has_deepseek_corrected,
                "env_attr": "env_deepseek_corrected",
                "done_attr": "deepseek_corrected_done",
                "sel_attr": "deepseek_corrected_selected_action",
                "exec_attr": "deepseek_corrected_executed_action",
                "sticky_attr": "deepseek_corrected_sticky_override",
                "probs_attr": "deepseek_corrected_probs",
            },
            "trained": {
                "title": "TRAINED BASELINE",
                "tag": "PPO Deep RL (10M)",
                "cond_type": "Trained RL",
                "subtitle": "20,486 params | Dense",
                "mechanism": "Trained PPO policy",
                "color": self.COLOR_PADDLE_TRAINED,
                "is_active": True,
                "env_attr": "env_trained",
                "done_attr": "trained_done",
                "sel_attr": "trained_selected_action",
                "exec_attr": "trained_executed_action",
                "sticky_attr": "trained_sticky_override",
                "probs_attr": "trained_probs",
            },
            "gpt6": {
                "title": "GPT-6 (CHATGPT)",
                "tag": "Zero-Shot Generation",
                "cond_type": "Zero-Shot",
                "subtitle": "398 nz | 98.1% sparse",
                "mechanism": "Position and row features",
                "color": self.COLOR_PADDLE_GPT6,
                "is_active": self.state.has_gpt6,
                "env_attr": "env_gpt6",
                "done_attr": "gpt6_done",
                "sel_attr": "gpt6_selected_action",
                "exec_attr": "gpt6_executed_action",
                "sticky_attr": "gpt6_sticky_override",
                "probs_attr": "gpt6_probs",
            },
            "gpt6_luna": {
                "title": "GPT-6 LUNA",
                "tag": "Zero-Shot Generation",
                "cond_type": "Zero-Shot",
                "subtitle": "309 nz | 98.5% sparse",
                "mechanism": "Nonlinear tanh column tracking",
                "color": self.COLOR_PADDLE_GPT6_LUNA,
                "is_active": self.state.has_gpt6_luna,
                "env_attr": "env_gpt6_luna",
                "done_attr": "gpt6_luna_done",
                "sel_attr": "gpt6_luna_selected_action",
                "exec_attr": "gpt6_luna_executed_action",
                "sticky_attr": "gpt6_luna_sticky_override",
                "probs_attr": "gpt6_luna_probs",
            },
            "deepseek": {
                "title": "DEEPSEEK (ORIGINAL)",
                "tag": "Zero-Shot Generation",
                "cond_type": "Zero-Shot",
                "subtitle": "786 nz | 96.2% sparse",
                "mechanism": "Saturated logits favor inert actions",
                "color": self.COLOR_PADDLE_DEEPSEEK,
                "is_active": self.state.has_deepseek,
                "env_attr": "env_deepseek",
                "done_attr": "deepseek_done",
                "sel_attr": "deepseek_selected_action",
                "exec_attr": "deepseek_executed_action",
                "sticky_attr": "deepseek_sticky_override",
                "probs_attr": "deepseek_probs",
            },
        }

        # Filter active models
        active_keys = [k for k, v in model_meta.items() if v["is_active"]]

        def panel_sort_key(panel_key):
            matched = benchmark_for_loaded_weights(
                benchmarks.get(panel_key), self.state.loaded_weight_hashes.get(panel_key)
            )
            return (matched is None, -matched["mean_score"] if matched else 0.0)

        active_keys.sort(key=panel_sort_key)

        self.panels = {}
        for k in active_keys:
            meta = model_meta[k]
            loaded_hash = self.state.loaded_weight_hashes.get(k)
            bench = benchmark_for_loaded_weights(benchmarks.get(k), loaded_hash)
            if bench and "mean_score" in bench:
                rank_str = f"#{bench['rank']}"
                score_str = f"Mean: {bench['mean_score']:.2f} ± {bench.get('se_score', 0.0):.2f}"
                subtitle = f"[{meta['cond_type']}] {score_str}\n{meta['subtitle']}\nWeight hash matches dataset: {loaded_hash[:12]}…"
            else:
                rank_str = "[Intervention]" if meta["cond_type"] == "Intervention" else "[Unranked]"
                hash_status = f"Loaded weight SHA-256 {loaded_hash[:12]}…; no matching benchmark record" if loaded_hash else "No weight file loaded"
                subtitle = f"[{meta['cond_type']}] Score withheld\n{meta['subtitle']}\n{hash_status}"

            panel = self._create_model_panel(
                parent=self.arena,
                title=f"{rank_str} {meta['title']}",
                tag=meta["tag"],
                subtitle=subtitle,
                mechanism=meta["mechanism"],
                color=meta["color"],
            )
            self.panels[k] = (panel, meta)
            # Store backward-compatible attribute on app (e.g. self.argon_panel)
            setattr(self, f"{k}_panel", panel)

        # --- CONTROL BAR ---
        self._build_control_bar()

    def _create_model_panel(
        self,
        parent,
        title: str,
        tag: str,
        subtitle: str,
        mechanism: str,
        color: str,
    ):
        import tkinter as tk

        frame = tk.Frame(parent, bg=self.COLOR_PANEL, padx=8, pady=8, relief=tk.RIDGE, bd=2)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # Header Frame: Title and Tag
        hdr_frame = tk.Frame(frame, bg=self.COLOR_PANEL)
        hdr_frame.pack(fill=tk.X)

        lbl_title = tk.Label(hdr_frame, text=title, font=("Helvetica", 11, "bold"), fg=color, bg=self.COLOR_PANEL)
        lbl_title.pack(side=tk.LEFT)

        lbl_tag = tk.Label(hdr_frame, text=tag, font=("Helvetica", 8), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_PANEL)
        lbl_tag.pack(side=tk.RIGHT)

        # Subtitle: Sparsity and parameter stats
        lbl_sub = tk.Label(
            frame,
            text=subtitle,
            font=("Helvetica", 8, "italic"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_PANEL,
            anchor=tk.W,
        )
        lbl_sub.pack(fill=tk.X, pady=(1, 2))

        # Stats label (Score, Steps, Bricks, Status)
        lbl_stats = tk.Label(
            frame,
            text="Score: 0   Steps: 0   Bricks: 30/30\nStatus: PLAYING",
            font=("Helvetica", 9),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_PANEL,
            justify=tk.LEFT,
        )
        lbl_stats.pack(anchor=tk.W, pady=(1, 3))

        # Game Canvas
        canvas_board = tk.Canvas(
            frame,
            width=self.board_pixels,
            height=self.board_pixels,
            bg=self.COLOR_BOARD_BG,
            highlightthickness=1,
            highlightbackground="#333344",
        )
        canvas_board.pack(pady=2)

        # Action Box Frame
        action_frame = tk.Frame(frame, bg="#16161e", padx=5, pady=3, relief=tk.GROOVE, bd=1)
        action_frame.pack(fill=tk.X, pady=(3, 3))

        lbl_selected_action = tk.Label(
            action_frame,
            text="Selected Action: None",
            font=("Helvetica", 9, "bold"),
            fg=color,
            bg="#16161e",
            anchor=tk.W,
        )
        lbl_selected_action.pack(fill=tk.X)

        lbl_executed_action = tk.Label(
            action_frame,
            text="Executed Action: None",
            font=("Helvetica", 8),
            fg=self.COLOR_TEXT_MUTED,
            bg="#16161e",
            anchor=tk.W,
        )
        lbl_executed_action.pack(fill=tk.X)

        # Probability Header
        lbl_probs_title = tk.Label(
            frame,
            text="ACTION PROBABILITIES (100%):",
            font=("Helvetica", 8, "bold"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_PANEL,
            anchor=tk.W,
        )
        lbl_probs_title.pack(anchor=tk.W, pady=(2, 1))

        # Probability Canvas (height=135 fits all 6 actions + footer)
        canvas_probs = tk.Canvas(frame, width=self.board_pixels, height=135, bg=self.COLOR_PANEL, highlightthickness=0)
        canvas_probs.pack(fill=tk.X, pady=1)

        # Mechanism footer
        lbl_mech = tk.Label(
            frame,
            text=f"Policy: {mechanism}",
            font=("Helvetica", 7, "italic"),
            fg="#707085",
            bg=self.COLOR_PANEL,
            anchor=tk.W,
        )
        lbl_mech.pack(fill=tk.X, pady=(2, 0))

        return {
            "frame": frame,
            "title_label": lbl_title,
            "stats_label": lbl_stats,
            "canvas_board": canvas_board,
            "selected_action_label": lbl_selected_action,
            "executed_action_label": lbl_executed_action,
            "canvas_probs": canvas_probs,
            "mechanism_label": lbl_mech,
            "color": color,
        }

    def _build_control_bar(self):
        import tkinter as tk

        ctrl_frame = tk.Frame(self.root, bg=self.COLOR_PANEL, pady=8, padx=15)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Button Style
        btn_font = ("Helvetica", 10, "bold")
        btn_bg = "#2b2b38"
        btn_fg = "#ffffff"

        self.btn_play = tk.Button(
            ctrl_frame,
            text="Pause (Space)",
            command=self.toggle_play,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=7,
            pady=3,
        )
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.btn_step = tk.Button(
            ctrl_frame,
            text="Step Frame (→)",
            command=self.step_single,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=7,
            pady=3,
        )
        self.btn_step.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tk.Button(
            ctrl_frame,
            text="Reset (R)",
            command=self.reset_current,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=7,
            pady=3,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        self.btn_prev = tk.Button(
            ctrl_frame,
            text="Prev (P)",
            command=self.prev_episode,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=7,
            pady=3,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = tk.Button(
            ctrl_frame,
            text="Next (N)",
            command=self.next_episode,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=7,
            pady=3,
        )
        self.btn_next.pack(side=tk.LEFT, padx=3)

        # Benchmark Summary Button
        self.btn_benchmark = tk.Button(
            ctrl_frame,
            text="Benchmark Table (M)",
            command=self.show_benchmark_summary,
            font=btn_font,
            bg="#302b48",
            fg="#d070ff",
            padx=7,
            pady=3,
        )
        self.btn_benchmark.pack(side=tk.LEFT, padx=(10, 3))

        # Network Inspector Button
        self.btn_inspector = tk.Button(
            ctrl_frame,
            text="Network Inspector (I)",
            command=self.show_network_inspector,
            font=btn_font,
            bg="#3d2a20",
            fg="#ff7043",
            padx=7,
            pady=3,
        )
        self.btn_inspector.pack(side=tk.LEFT, padx=3)

        # FPS Slider
        tk.Label(ctrl_frame, text="Speed:", font=("Helvetica", 9), fg=self.COLOR_TEXT, bg=self.COLOR_PANEL).pack(
            side=tk.LEFT, padx=(12, 2)
        )
        self.slider_speed = tk.Scale(
            ctrl_frame,
            from_=1,
            to=60,
            orient=tk.HORIZONTAL,
            variable=self.fps_var,
            bg=self.COLOR_PANEL,
            fg="#ffffff",
            troughcolor="#333344",
            highlightthickness=0,
            length=75,
        )
        self.slider_speed.pack(side=tk.LEFT, padx=2)

        # Shortcuts Legend
        lbl_shortcuts = tk.Label(
            ctrl_frame,
            text="[Space] Play/Pause   [→] Step   [R] Reset   [P/N] Prev/Next   [M] Benchmark   [I] Inspector",
            font=("Helvetica", 9),
            fg="#9999aa",
            bg=self.COLOR_PANEL,
        )
        lbl_shortcuts.pack(side=tk.RIGHT, padx=5)

    def _bind_events(self):
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step_single())
        self.root.bind("r", lambda e: self.reset_current())
        self.root.bind("R", lambda e: self.reset_current())
        self.root.bind("n", lambda e: self.next_episode())
        self.root.bind("N", lambda e: self.next_episode())
        self.root.bind("p", lambda e: self.prev_episode())
        self.root.bind("P", lambda e: self.prev_episode())
        self.root.bind("m", lambda e: self.show_benchmark_summary())
        self.root.bind("M", lambda e: self.show_benchmark_summary())
        self.root.bind("i", lambda e: self.show_network_inspector())
        self.root.bind("I", lambda e: self.show_network_inspector())
        self.root.bind("h", lambda e: self.show_argon_high_info())
        self.root.bind("H", lambda e: self.show_argon_high_info())

    def show_benchmark_summary(self):
        """Display comprehensive comparative benchmark table across all models."""
        import tkinter as tk

        benchmarks = load_verified_benchmark_registry(dataset=self.state.benchmark_dataset)
        ranked = sorted(benchmarks.values(), key=lambda item: item["rank"])
        lines = [
            f"Dataset: {self.state.benchmark_dataset}",
            "Protocol: " + (ranked[0].get("protocol", "corrected protocol") if ranked else "unavailable"),
            "Scores are loaded from hash-verified benchmark records. Rankings include controls and baselines.",
            "",
            "Rank  Condition                                 N     Mean   SE   Median   Max  MeanLen  Cutoff%",
            "----  ----------------------------------------  ----  ------  ----  ------  ----  -------  -------",
        ]
        for item in ranked:
            name = item.get("model", item["condition_id"])
            lines.append(
                f"{item['rank']:>4}  {name[:40]:<40}  {item['episodes']:>4}  "
                f"{item['mean_score']:>6.2f}  {item.get('se_score', 0.0):>4.2f}  "
                f"{item.get('median_score', float('nan')):>6.1f}  "
                f"{item.get('maximum_score', float('nan')):>4.0f}  "
                f"{item.get('mean_episode_length', float('nan')):>7.1f}  "
                f"{item.get('cutoff_percentage', float('nan')):>6.2f}%"
            )
        lines.extend([
            "",
            "Action counts in evaluation artifacts are selected actions; environment-executed actions can differ under sticky actions.",
            "DeepSeek Corrected is a researcher intervention and is excluded from the primary ranking.",
            "Astra local evaluation is distinct from the paper's published 1,000-episode result.",
            "",
            "For matched inference and paired differences, see experiments/corrected_evaluation/full_run/analysis/paired_comparisons.json.",
        ])

        win = tk.Toplevel(self.root)
        win.title("MinAtar Breakout: Comparative Benchmark Summary")
        win.geometry("980x640")
        win.configure(bg="#181822")

        lbl_hdr = tk.Label(
            win,
            text=f"MINATAR BREAKOUT · {self.state.benchmark_dataset.upper()} DATASET",
            font=("Helvetica", 13, "bold"),
            fg="#ffffff",
            bg="#181822",
            pady=10,
        )
        lbl_hdr.pack()

        txt = tk.Text(win, font=("Courier", 10), bg="#0f0f16", fg="#d0d0e0", padx=12, pady=10, relief=tk.FLAT)
        txt.insert(tk.END, "\n".join(lines))
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

    def show_network_inspector(self):
        """Display deep inspection modal of neural network weights, sparsity, and mathematical policies."""
        import tkinter as tk

        win = tk.Toplevel(self.root)
        win.title("Neural Network Architecture & Policy Inspector")
        win.geometry("980x680")
        win.configure(bg="#181822")

        lbl_hdr = tk.Label(
            win,
            text="WEIGHT-SPACE PROGRAMMING: NEURAL POLICY DEEP INSPECTOR",
            font=("Helvetica", 13, "bold"),
            fg="#ff7043",
            bg="#181822",
            pady=10,
        )
        lbl_hdr.pack()

        inspector_text = """
NETWORK AND POLICY NOTES

The benchmark summary loads episode metrics from the selected dataset. The corrected
dataset is the default. The historical dataset remains available through --dataset original.
Scores are attached to panels only when the displayed weight file's SHA-256 matches the
weight artifact recorded for that benchmark condition. A custom or intervention file has
no primary benchmark score. DeepSeek Corrected is a researcher intervention.

GPT-6 LUNA EXTRA HIGH — EXACT NONLINEAR COMPUTATION
For column c, the active first-layer coordinate readouts are u(c) = 0.08 * (c - 4.5).
The relevant hidden activations are h_ball = tanh(u(ball_x)), h_trail = tanh(u(trail_x)),
and h_paddle = tanh(u(paddle_x)). The active second-layer unit is
h = tanh(12*h_ball - 6*h_trail - 6*h_paddle).
Its steering logits are Left = -4*h and Right = 4*h; No-op has logit 1.5, and Up, Down,
and Fire each have logit -6. The expression 2*ball_x - trail_x - paddle_x is a linearized
approximation to the hidden preactivation, not the exact policy. Softmax sampling remains
stochastic, so these logits alone do not establish an episode-level failure cause.

TRAJECTORY EVIDENCE
The archived historical trace replay records one episode each for GPT-6 Original, Gemini
Argon, Claude Sonnet, DeepSeek Original, and Luna. It verifies the states and actions for
those episodes against the historical CSV, but one episode per model cannot establish an
aggregate failure mechanism. In the Sonnet replay, ball and trail coincide only in the
terminal state; the last policy selection was Right. This does not support the prior claim
that a wall bounce caused a zero-velocity No-op.

MINATAR PADDLE COLLISION
The environment moves the paddle, computes and wall-clamps the ball's next horizontal
coordinate, then checks both the ball's current x and clamped next x against the updated
paddle position. A single target coordinate approximation can therefore miss the collision rule.

DeepSeek's logits and selected-action distribution support a saturated, inert-action
explanation for this particular policy in this environment. They do not establish a broader
claim about the model family. See the saved corrected analysis for paired comparisons and
episode-level action distributions.
"""
        txt = tk.Text(win, font=("Courier", 10), bg="#0f0f16", fg="#d0d0e0", padx=12, pady=10, relief=tk.FLAT)
        txt.insert(tk.END, inspector_text.strip())
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

    def show_argon_high_info(self):
        """Display status for pending Gemini Argon 4 High experiment."""
        import tkinter as tk

        win = tk.Toplevel(self.root)
        win.title("Gemini Argon 4 High — Experiment Status")
        win.geometry("640x300")
        win.configure(bg="#181822")

        info_text = """
GEMINI ARGON 4 HIGH (ULTRA-HIGH REASONING)
Status: PENDING INDEPENDENT EXPERIMENT

• Experimental Input Package:
  experiments/strict_replication/generation_001/input/argon_experiment_001.zip
  SHA-256: 976c17cd21bed3febb81e5f4f5b7c43467a7cf87230bb646c0135406d86bec64

• Prompt Integrity:
  Verified identical to original Astra prompt (SHA-256: 766ee251...)

• Experimental Protocol:
  Awaiting official generation output via Gemini Web Interface with
  Argon 4 High reasoning profile. No mock data or surrogate weights
  are substituted to maintain scientific integrity.
"""
        txt = tk.Text(win, font=("Courier", 10), bg="#0f0f16", fg="#ffb74d", padx=12, pady=10, relief=tk.FLAT)
        txt.insert(tk.END, info_text.strip())
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def update_control_states(self):
        """Update button enabled/disabled states based on simulation status and boundaries."""
        import tkinter as tk

        # Play / Pause text
        self.btn_play.config(text="Pause (Space)" if self.is_running.get() else "Play (Space)")

        # Prev button disabled at seed 1
        if self.seed_var.get() <= 1:
            self.btn_prev.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=tk.NORMAL)

        # Step button state
        if self.state.all_done and not self.is_running.get():
            self.btn_step.config(state=tk.DISABLED)
        else:
            self.btn_step.config(state=tk.NORMAL)

    def toggle_play(self):
        """Toggle simulation playback between running and paused."""
        self._cancel_auto_advance()
        self.is_running.set(not self.is_running.get())
        self.update_control_states()

    def step_single(self):
        """Pause playback and execute exactly one simulation frame across all active models."""
        self._cancel_auto_advance()
        self.is_running.set(False)
        self.state.step()
        self.redraw()
        self.update_control_states()

    def reset_current(self):
        """Reset the current episode seed to frame 0."""
        self._cancel_auto_advance()
        self.state.reset_episode(self.seed_var.get())
        self.seed_label.config(text=f"Seed: {self.seed_var.get()}")
        self.redraw()
        self.update_control_states()

    def next_episode(self):
        """Advance to the next episode (seed + 1)."""
        self._cancel_auto_advance()
        new_seed = self.seed_var.get() + 1
        self.seed_var.set(new_seed)
        self.state.reset_episode(new_seed)
        self.seed_label.config(text=f"Seed: {new_seed}")
        self.redraw()
        self.update_control_states()

    def prev_episode(self):
        """Navigate to previous episode (max(1, seed - 1))."""
        self._cancel_auto_advance()
        if self.seed_var.get() <= 1:
            return
        new_seed = self.seed_var.get() - 1
        self.seed_var.set(new_seed)
        self.state.reset_episode(new_seed)
        self.seed_label.config(text=f"Seed: {new_seed}")
        self.redraw()
        self.update_control_states()

    def _cancel_auto_advance(self):
        if self.auto_advance_job is not None:
            self.root.after_cancel(self.auto_advance_job)
            self.auto_advance_job = None

    def draw_board(self, canvas, env: MinAtarBreakoutEnv, paddle_color: str):
        """Draw the 10x10 MinAtar Breakout board on the specified Tkinter Canvas."""
        canvas.delete("all")
        internal = env.internal_env
        cs = self.cell_size

        # Grid cells
        for r in range(10):
            for c in range(10):
                x0, y0 = c * cs, r * cs
                x1, y1 = x0 + cs, y0 + cs
                canvas.create_rectangle(x0, y0, x1, y1, outline=self.COLOR_GRID, fill=self.COLOR_BOARD_BG)

        # Bricks (Rows 1..3)
        brick_map = internal.brick_map
        row_colors = [self.COLOR_BRICK_ROW1, self.COLOR_BRICK_ROW2, self.COLOR_BRICK_ROW3]
        for r in range(1, 4):
            for c in range(10):
                if brick_map[r, c] == 1:
                    x0, y0 = c * cs + 2, r * cs + 2
                    x1, y1 = (c + 1) * cs - 2, (r + 1) * cs - 2
                    canvas.create_rectangle(x0, y0, x1, y1, fill=row_colors[r - 1], outline="")

        # Trail
        lx, ly = internal.last_x, internal.last_y
        if 0 <= lx < 10 and 0 <= ly < 10 and not (lx == internal.ball_x and ly == internal.ball_y):
            cx, cy = lx * cs + cs // 2, ly * cs + cs // 2
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=self.COLOR_TRAIL, outline="")

        # Ball
        bx, by = internal.ball_x, internal.ball_y
        if 0 <= bx < 10 and 0 <= by < 10:
            cx, cy = bx * cs + cs // 2, by * cs + cs // 2
            canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=self.COLOR_BALL, outline="#ffffff")

        # Paddle (Row 9)
        pos = internal.pos
        x0 = pos * cs + 2
        y0 = 9 * cs + 3
        x1 = (pos + 1) * cs - 2
        y1 = 10 * cs - 3
        canvas.create_rectangle(x0, y0, x1, y1, fill=paddle_color, outline="#ffffff", width=1)

    def draw_probs(
        self,
        canvas,
        probs: np.ndarray,
        selected_action: int,
        primary_color: str,
    ):
        """Draw full 6-action softmax probability distribution bars summing to 100%."""
        canvas.delete("all")
        import tkinter as tk

        # All 6 MinAtar actions strictly defined
        action_defs = [
            (0, "0 No-Op"),
            (1, "1 Left"),
            (2, "2 Up"),
            (3, "3 Right"),
            (4, "4 Down"),
            (5, "5 Fire"),
        ]

        bar_h = 13
        start_y = 4
        gap_y = 18
        track_x = 60
        track_w = self.board_pixels - track_x - 52  # Leaves room for 45px text on right

        # Draw each action row
        for idx, (act_id, label) in enumerate(action_defs):
            p = float(probs[act_id]) if act_id < len(probs) else 0.0
            is_chosen = (act_id == selected_action)
            y = start_y + idx * gap_y

            # Action label with indicator
            prefix = "▶ " if is_chosen else "  "
            lbl_text = prefix + label
            lbl_color = primary_color if is_chosen else self.COLOR_TEXT
            lbl_font = ("Helvetica", 8, "bold" if is_chosen else "normal")

            canvas.create_text(
                track_x - 4,
                y + bar_h // 2,
                text=lbl_text,
                fill=lbl_color,
                font=lbl_font,
                anchor=tk.E,
            )

            # Background track
            canvas.create_rectangle(
                track_x,
                y,
                track_x + track_w,
                y + bar_h,
                fill="#20202c",
                outline="#2e2e3e",
            )

            # Filled bar
            fill_w = max(0, min(track_w, int(track_w * p)))
            bar_color = primary_color if is_chosen else "#444458"
            if fill_w > 0:
                canvas.create_rectangle(
                    track_x,
                    y,
                    track_x + fill_w,
                    y + bar_h,
                    fill=bar_color,
                    outline="",
                )

            # Percentage text
            pct_text = f"{p * 100:.1f}%"
            canvas.create_text(
                track_x + track_w + 6,
                y + bar_h // 2,
                text=pct_text,
                fill="#ffffff" if is_chosen else self.COLOR_TEXT_MUTED,
                font=("Helvetica", 8, "bold" if is_chosen else "normal"),
                anchor=tk.W,
            )

        # Footer row: explicit distribution sum verification
        total_sum = float(np.sum(probs[:6])) * 100.0
        footer_y = start_y + 6 * gap_y + 3
        canvas.create_text(
            track_x + track_w + 40,
            footer_y,
            text=f"Total: {total_sum:.1f}%",
            fill="#888899",
            font=("Helvetica", 8, "italic"),
            anchor=tk.E,
        )

    def _update_panel_display(
        self,
        panel: dict,
        env: MinAtarBreakoutEnv,
        is_done: bool,
        selected_act: int,
        executed_act: Optional[int],
        sticky_override: bool,
        probs: np.ndarray,
    ):
        """Update widgets for a single model panel."""
        import tkinter as tk

        # Draw board
        self.draw_board(panel["canvas_board"], env, panel["color"])

        # Determine status text
        bricks_left = int(np.count_nonzero(env.internal_env.brick_map))
        if is_done:
            if env.step_count >= env.max_steps:
                status_str = f"STEP CUTOFF ({env.max_steps})"
                status_color = self.COLOR_STICKY
            else:
                status_str = f"LOST BALL (Step {env.step_count})"
                status_color = self.COLOR_TERMINAL
        else:
            status_str = "PLAYING"
            status_color = "#00e676"

        panel["stats_label"].config(
            text=f"Score: {int(env.total_reward)}   Steps: {env.step_count}   Bricks: {bricks_left}/30\nStatus: {status_str}",
            fg=self.COLOR_TEXT,
        )

        # Action: Selected (sampled by policy)
        sel_name = ACTION_NAMES.get(selected_act, "Unknown")
        panel["selected_action_label"].config(
            text=f"Selected: {sel_name.capitalize()} ({selected_act})",
            fg=panel["color"],
        )

        # Action: Executed (after environment sticky check)
        if executed_act is None:
            exec_text = "Executed: - (Initial Frame)"
            exec_color = self.COLOR_TEXT_MUTED
        elif sticky_override:
            exec_name = ACTION_NAMES.get(executed_act, "Unknown")
            exec_text = f"Executed: {exec_name.capitalize()} ({executed_act})  ⚠ [STICKY OVERRIDE]"
            exec_color = self.COLOR_STICKY
        else:
            exec_name = ACTION_NAMES.get(executed_act, "Unknown")
            exec_text = f"Executed: {exec_name.capitalize()} ({executed_act})"
            exec_color = self.COLOR_TEXT_MUTED

        panel["executed_action_label"].config(text=exec_text, fg=exec_color)

        # Draw probability bars
        self.draw_probs(panel["canvas_probs"], probs, selected_act, panel["color"])

    def redraw(self):
        """Redraw all model panels in their performance-ranked order."""
        for k, (panel, meta) in self.panels.items():
            env = getattr(self.state, meta["env_attr"])
            is_done = getattr(self.state, meta["done_attr"])
            sel_act = getattr(self.state, meta["sel_attr"])
            exec_act = getattr(self.state, meta["exec_attr"])
            sticky = getattr(self.state, meta["sticky_attr"])
            probs = getattr(self.state, meta["probs_attr"])
            self._update_panel_display(
                panel=panel,
                env=env,
                is_done=is_done,
                selected_act=sel_act,
                executed_act=exec_act,
                sticky_override=sticky,
                probs=probs,
            )
        self.update_control_states()

    def _game_loop(self):
        """Simulation tick handler."""
        if self.is_running.get():
            if self.state.all_done:
                # All models finished current episode: wait 800ms smoothly then auto-advance
                if self.auto_advance_job is None:
                    self.auto_advance_job = self.root.after(800, self._on_auto_advance)
            else:
                self.state.step()
                self.redraw()

        delay = max(10, int(1000 / max(1, self.fps_var.get())))
        self.root.after(delay, self._game_loop)

    def _on_auto_advance(self):
        self.auto_advance_job = None
        if self.is_running.get() and self.state.all_done:
            self.next_episode()


def run_gui(state: VisualizerState, initial_fps: int = 15):
    """Launch interactive Tkinter visualizer."""
    import tkinter as tk

    root = tk.Tk()
    app = BreakoutVisualizerGUI(state, root, initial_fps=initial_fps)
    root.mainloop()


def run_headless_verification(state: VisualizerState, steps: int = 50) -> bool:
    """Run non-GUI simulation to verify all active models step in lockstep without errors."""
    print(f"Running headless visualizer verification for {steps} steps...")
    for s in range(steps):
        state.step()
        if (s + 1) % 10 == 0:
            msg = (
                f"  [Step {s + 1}/{steps}] "
                f"Argon score={state.env_argon.total_reward} (done={state.argon_done}) | "
                f"Trained score={state.env_trained.total_reward} (done={state.trained_done})"
            )
            if state.has_astra:
                msg += f" | Astra score={state.env_astra.total_reward} (done={state.astra_done})"
            if state.has_gpt6:
                msg += f" | GPT-6 score={state.env_gpt6.total_reward} (done={state.gpt6_done})"
            if state.has_gpt6_luna:
                msg += f" | GPT-6 Luna score={state.env_gpt6_luna.total_reward} (done={state.gpt6_luna_done})"
            if state.has_sonnet55:
                msg += f" | Sonnet 5.5 score={state.env_sonnet55.total_reward} (done={state.sonnet55_done})"
            if state.has_deepseek:
                msg += f" | DeepSeek score={state.env_deepseek.total_reward} (done={state.deepseek_done})"
            if state.has_deepseek_corrected:
                msg += f" | DeepSeek (Corr) score={state.env_deepseek_corrected.total_reward} (done={state.deepseek_corrected_done})"
            print(msg)
    print("Headless verification PASSED cleanly.")
    return True


def capture_screenshot(
    output_path: str,
    argon_path: str,
    trained_path: str,
    astra_path: Optional[str] = None,
    gpt6_path: Optional[str] = None,
    gpt6_luna_path: Optional[str] = None,
    sonnet55_path: Optional[str] = None,
    deepseek_path: Optional[str] = None,
    deepseek_corrected_path: Optional[str] = None,
    argon_high_path: Optional[str] = None,
    seed: int = 42,
    steps_to_run: int = 25,
    benchmark_dataset: str = "corrected",
) -> bool:
    """Launch GUI, step through simulation, capture window screenshot, and exit."""
    import subprocess
    import tempfile
    import tkinter as tk
    from PIL import Image

    root = tk.Tk()
    state = VisualizerState(
        argon_weights_path=argon_path,
        trained_weights_path=trained_path,
        astra_weights_path=astra_path,
        gpt6_weights_path=gpt6_path,
        gpt6_luna_weights_path=gpt6_luna_path,
        sonnet55_weights_path=sonnet55_path,
        deepseek_weights_path=deepseek_path,
        deepseek_corrected_weights_path=deepseek_corrected_path,
        argon_high_weights_path=argon_high_path,
        initial_seed=seed,
        benchmark_dataset=benchmark_dataset,
    )
    app = BreakoutVisualizerGUI(state, root, initial_fps=30)
    app.is_running.set(False)

    geo_w = (
        1820
        if (state.has_gpt6_luna or state.has_deepseek_corrected or state.has_deepseek)
        else (1620 if state.has_sonnet55 else (1480 if state.has_gpt6 else (1300 if state.has_astra else 980)))
    )
    geo_h = 870 if (state.has_gpt6_luna or state.has_deepseek or state.has_deepseek_corrected) else (860 if (state.has_gpt6 or state.has_sonnet55) else 820)
    root.geometry(f"{geo_w}x{geo_h}+0+28")
    root.deiconify()
    root.lift()
    root.focus_force()
    root.attributes("-topmost", True)
    root.update()

    for _ in range(steps_to_run):
        app.step_single()

    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    root.update()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        res = subprocess.run(["screencapture", "-o", "-m", tmp.name], capture_output=True)
        root.destroy()
        if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
            im = Image.open(tmp.name)
            crop_h = min(im.height, geo_h + 45)
            crop_w = min(im.width, geo_w)
            cropped = im.crop((0, 0, crop_w, crop_h))
            cropped.save(output_path)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="MinAtar Breakout Policy Visualizer (Argon, Astra, GPT-6, Claude Sonnet 5.5, DeepSeek, Trained)"
    )
    parser.add_argument(
        "--argon",
        type=str,
        default="experiments/strict_replication/generation_001/output/argon_weights.npz",
        help="Path to Argon weights .npz",
    )
    parser.add_argument(
        "--astra",
        type=str,
        default=ASTRA_REFERENCE_PATH if os.path.exists(ASTRA_REFERENCE_PATH) else None,
        help="Path to Astra reference weights .npz (optional, default: experiments/reference/hand.npz)",
    )
    parser.add_argument(
        "--gpt6",
        type=str,
        default=GPT6_DEFAULT_PATH if os.path.exists(GPT6_DEFAULT_PATH) else None,
        help="Path to GPT-6 Original weights .npz (optional, default: experiments/strict_replication/generation_002_gpt6/output/gpt6_weights.npz)",
    )
    parser.add_argument(
        "--gpt6-luna",
        "--luna",
        type=str,
        default=GPT6_LUNA_DEFAULT_PATH if os.path.exists(GPT6_LUNA_DEFAULT_PATH) else None,
        help="Path to GPT-6 Luna Extra High weights .npz (optional, default: experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz)",
    )
    parser.add_argument(
        "--sonnet55",
        "--sonnet",
        type=str,
        default=SONNET55_DEFAULT_PATH if os.path.exists(SONNET55_DEFAULT_PATH) else None,
        help="Path to Claude Sonnet 5.5 weights .npz (optional, default: experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz)",
    )
    parser.add_argument(
        "--deepseek",
        type=str,
        default=DEEPSEEK_DEFAULT_PATH if os.path.exists(DEEPSEEK_DEFAULT_PATH) else None,
        help="Path to DeepSeek Original weights .npz (optional, default: experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz)",
    )
    parser.add_argument(
        "--deepseek-corrected",
        "--corrected",
        type=str,
        default=None,
        help="Path to DeepSeek Corrected weights .npz (optional; post-hoc researcher intervention)",
    )
    parser.add_argument(
        "--argon-high",
        type=str,
        default=None,
        help="Path to Gemini Argon 4 High weights .npz (optional; pending experiment)",
    )
    parser.add_argument(
        "--trained",
        type=str,
        default="experiments/trained_baseline/trained_actor_weights.npz",
        help="Path to Trained model weights .npz",
    )
    parser.add_argument("--seed", type=int, default=42, help="Initial episode random seed")
    parser.add_argument(
        "--dataset",
        choices=["corrected", "original"],
        default="corrected",
        help="Benchmark ranking dataset (default: complete Phase 2 corrected evaluation; use original for historical results).",
    )
    parser.add_argument("--fps", type=int, default=15, help="Simulation speed in FPS")
    parser.add_argument(
        "--headless-test",
        action="store_true",
        help="Run non-GUI verification test instead of opening window",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Capture a screenshot to the specified path and exit",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of simulation steps to advance when capturing screenshot",
    )
    args = parser.parse_args()

    # Every loaded file is hashed. Canonical files must match the condition registry;
    # custom files can be inspected but cannot inherit a canonical benchmark score.
    weight_paths = {
        "argon": inspect_visualizer_weight(args.argon, "gemini_argon_4_medium", "Gemini Argon", required=True),
        "trained": inspect_visualizer_weight(args.trained, "ppo_trained_baseline", "PPO baseline", required=True),
        "astra": inspect_visualizer_weight(args.astra, "astra_local_reference", "Astra reference"),
        "gpt6": inspect_visualizer_weight(args.gpt6, "gpt6_original", "GPT-6 Original"),
        "gpt6_luna": inspect_visualizer_weight(args.gpt6_luna, "gpt6_luna_extra_high", "GPT-6 Luna Extra High"),
        "sonnet55": inspect_visualizer_weight(args.sonnet55, "claude_sonnet_55", "Claude Sonnet 5.5"),
        "deepseek": inspect_visualizer_weight(args.deepseek, "deepseek_original", "DeepSeek Original"),
        "deepseek_corrected": inspect_visualizer_weight(args.deepseek_corrected, None, "DeepSeek Corrected intervention"),
        "argon_high": inspect_visualizer_weight(args.argon_high, None, "Gemini Argon 4 High"),
    }

    if args.screenshot:
        success = capture_screenshot(
            output_path=args.screenshot,
            argon_path=weight_paths["argon"],
            trained_path=weight_paths["trained"],
            astra_path=weight_paths["astra"],
            gpt6_path=weight_paths["gpt6"],
            gpt6_luna_path=weight_paths["gpt6_luna"],
            sonnet55_path=weight_paths["sonnet55"],
            deepseek_path=weight_paths["deepseek"],
            deepseek_corrected_path=weight_paths["deepseek_corrected"],
            argon_high_path=weight_paths["argon_high"],
            seed=args.seed,
            steps_to_run=args.steps,
            benchmark_dataset=args.dataset,
        )
        print(f"Screenshot captured: {success} -> {args.screenshot}")
        sys.exit(0 if success else 1)

    state = VisualizerState(
        argon_weights_path=weight_paths["argon"],
        trained_weights_path=weight_paths["trained"],
        astra_weights_path=weight_paths["astra"],
        gpt6_weights_path=weight_paths["gpt6"],
        gpt6_luna_weights_path=weight_paths["gpt6_luna"],
        sonnet55_weights_path=weight_paths["sonnet55"],
        deepseek_weights_path=weight_paths["deepseek"],
        deepseek_corrected_weights_path=weight_paths["deepseek_corrected"],
        argon_high_weights_path=weight_paths["argon_high"],
        initial_seed=args.seed,
        benchmark_dataset=args.dataset,
    )

    if args.headless_test:
        success = run_headless_verification(state)
        sys.exit(0 if success else 1)
    else:
        run_gui(state, initial_fps=args.fps)


if __name__ == "__main__":
    main()
