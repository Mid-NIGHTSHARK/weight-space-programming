"""Tests that the corrected protocol config encodes the requested comparison."""

from pathlib import Path
import json

from scripts.run_corrected_evaluation import build_schedule, load_evaluation_config, record_failure, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_corrected_config_has_exact_episode_and_seed_design():
    config = load_evaluation_config()
    assert config["seed_groups"] == [42, 100, 200]
    assert config["episodes_per_seed_group"] == 1000
    assert config["pilot_episodes_per_condition"] in range(10, 21)
    assert config["environment"]["sticky_action_probability"] == 0.1
    assert config["environment"]["maximum_episode_steps"] == 10_000
    assert config["policy"]["temperature"] == 1.0

    full = build_schedule("full", config)
    pilot = build_schedule("pilot", config)
    assert len(full) == 3000
    assert len(pilot) == 10
    assert {group: sum(row["seed_group"] == group for row in full) for group in config["seed_groups"]} == {
        42: 1000, 100: 1000, 200: 1000,
    }
    assert len({(row["seed_group"], row["episode_index"]) for row in full}) == 3000
    assert all(row["environment_seed"] != row["policy_seed"] for row in full)


def test_condition_registry_excludes_intervention_from_primary_comparison():
    registry = json.loads((ROOT / "experiments/corrected_evaluation/condition_registry.json").read_text())
    conditions = registry["conditions"]
    primary = {condition["id"] for condition in conditions if condition["class"] == "primary_zero_shot"}
    assert primary == {
        "gemini_argon_4_medium", "gpt6_original", "claude_sonnet_55",
        "deepseek_original", "gpt6_luna_extra_high",
    }
    assert "deepseek_corrected" not in {condition["id"] for condition in conditions}


def test_failed_run_recording_preserves_error_and_attempt_status(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest_path = tmp_path / "run_manifest.json"
    manifest = {"status": "running", "run_attempt": "fresh_start", "resumed": False}
    write_json(manifest_path, manifest)
    try:
        raise RuntimeError("synthetic pilot failure")
    except RuntimeError as exc:
        record_failure(tmp_path, logs, manifest, "pilot", exc)

    observed = json.loads(manifest_path.read_text())
    assert observed["status"] == "failed"
    assert observed["run_attempt"] == "fresh_start"
    assert observed["resumed"] is False
    assert observed["error"]["message"] == "synthetic pilot failure"
    assert "RuntimeError: synthetic pilot failure" in (logs / "error.txt").read_text()
