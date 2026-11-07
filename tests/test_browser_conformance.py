"""Compare the static browser port with the repository's Python reference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from environment.breakout import MinAtarBreakoutEnv
from environment.observations import flatten_observation, get_observation
from network.architecture import MinAtarNetworkNumPy
from network.weight_loader import load_weights_npz


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/site_conformance_runner.mjs"
PRIMARY_WEIGHT_FILES = {
    "gemini_argon_4_medium": "experiments/strict_replication/generation_001/output/argon_weights.npz",
    "claude_sonnet_55": "experiments/strict_replication/generation_003_sonnet55/output/sonnet55_weights.npz",
    "gpt6_original": "experiments/strict_replication/generation_002_gpt6/output/gpt6_weights.npz",
    "gpt6_luna_extra_high": "experiments/strict_replication/generation_005_gpt6_luna_extra_high/output/gpt6_luna_weights.npz",
    "deepseek_original": "experiments/strict_replication/generation_004_deepseek/output/deepseek_weights.npz",
    "argon_high_original": "experiments/argon_high_iterative_development/weights/argon_high_original.npz",
    "argon_high_followup_1": "experiments/argon_high_iterative_development/weights/argon_high_followup_1.npz",
    "argon_high_followup_2": "experiments/argon_high_iterative_development/weights/argon_high_followup_2.npz",
}


def run_browser(operation: str, payload: dict) -> list:
    input_text = json.dumps({"operation": operation, **payload}, separators=(",", ":"))
    result = subprocess.run(
        ["node", str(RUNNER)],
        input=input_text,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def build_reference_case(spec: dict) -> tuple[dict, dict]:
    sticky_probability = spec.get("sticky_probability", 0.1)
    max_steps = spec.get("max_steps", 10000)
    env = MinAtarBreakoutEnv(sticky_action_prob=sticky_probability, max_steps=max_steps)
    env.reset(seed=spec.get("seed", 42))
    game = env.internal_env
    game.ball_x = spec["ball_x"]
    game.ball_y = spec["ball_y"]
    game.ball_dir = spec["ball_dir"]
    game.pos = spec["paddle"]
    game.last_x = spec.get("last_x", spec["ball_x"])
    game.last_y = spec.get("last_y", spec["ball_y"])
    game.strike = spec.get("strike", False)
    game.terminal = False
    game.brick_map[:] = 0
    if spec.get("bricks", "standard") == "standard":
        game.brick_map[1:4, :] = 1
    elif spec.get("bricks") == "empty":
        pass
    for row, column in spec.get("extra_bricks", []):
        game.brick_map[row, column] = 1
    initial_bricks = np.asarray(game.brick_map, dtype=np.uint8).reshape(-1).tolist()
    env._minatar_env.last_action = spec.get("last_action", 0)
    env.last_action = spec.get("last_action", 0)
    env.step_count = spec.get("step", 0)
    env.total_reward = spec.get("score", 0.0)

    random_state = env._minatar_env.random.get_state()
    sticky_draw = float(env._minatar_env.random.rand())
    env._minatar_env.random.set_state(random_state)
    _, reward, done, info = env.step(spec["action"])
    after = env.internal_env
    reason = "ball_lost" if info["ball_lost"] else "step_cutoff" if info["step_cutoff"] else None
    expected_state = {
        "ballX": int(after.ball_x),
        "ballY": int(after.ball_y),
        "ballDir": int(after.ball_dir),
        "paddle": int(after.pos),
        "lastX": int(after.last_x),
        "lastY": int(after.last_y),
        "bricks": np.asarray(after.brick_map, dtype=np.uint8).reshape(-1).tolist(),
        "strike": bool(after.strike),
        "terminal": bool(env.terminal),
        "step": int(env.step_count),
        "score": float(env.total_reward),
        "lastAction": int(env._minatar_env.last_action),
        "terminalReason": reason,
    }
    expected = {
        "state": expected_state,
        "reward": float(reward),
        "done": bool(done),
        "info": {
            "step": int(info["step"]),
            "totalReward": float(info["total_reward"]),
            "ballLost": bool(info["ball_lost"]),
            "stepCutoff": bool(info["step_cutoff"]),
            "paddle": int(info["paddle_pos"]),
            "ballX": int(info["ball_x"]),
            "ballY": int(info["ball_y"]),
            "ballDir": int(info["ball_dir"]),
            "bricksRemaining": int(info["bricks_remaining"]),
            "selectedAction": int(info["selected_action"]),
            "executedAction": int(info["executed_action"]),
            "stickyOverride": bool(info["sticky_override"]),
        },
        "observation": flatten_observation(get_observation(env.internal_env_state if hasattr(env, "internal_env_state") else env._minatar_env.state())).astype(int).tolist(),
    }
    initial = {
        "state": {
            "ballX": int(spec["ball_x"]),
            "ballY": int(spec["ball_y"]),
            "ballDir": int(spec["ball_dir"]),
            "paddle": int(spec["paddle"]),
            "lastX": int(spec.get("last_x", spec["ball_x"])),
            "lastY": int(spec.get("last_y", spec["ball_y"])),
            "bricks": initial_bricks,
            "strike": bool(spec.get("strike", False)),
            "terminal": False,
            "step": int(spec.get("step", 0)),
            "score": float(spec.get("score", 0.0)),
            "lastAction": int(spec.get("last_action", 0)),
            "terminalReason": None,
        },
        "action": spec["action"],
        "sticky_draw": sticky_draw,
        "sticky_probability": sticky_probability,
        "max_steps": max_steps,
    }
    return initial, expected


def test_browser_transitions_match_minatar_reference():
    specs = [
        {"ball_x": 9, "ball_y": 5, "ball_dir": 2, "paddle": 4, "action": 3},  # side wall + paddle
        {"ball_x": 5, "ball_y": 0, "ball_dir": 0, "paddle": 4, "action": 0},  # top wall
        {"ball_x": 4, "ball_y": 2, "ball_dir": 2, "paddle": 4, "action": 0},  # brick + score
        {"ball_x": 5, "ball_y": 8, "ball_dir": 3, "paddle": 6, "action": 1},  # current-x catch after paddle moves
        {"ball_x": 5, "ball_y": 8, "ball_dir": 2, "paddle": 5, "action": 3},  # clamped/next-x catch
        {"ball_x": 5, "ball_y": 8, "ball_dir": 2, "paddle": 4, "action": 0},  # miss
        {"ball_x": 5, "ball_y": 5, "ball_dir": 2, "paddle": 4, "action": 3, "last_action": 1, "sticky_probability": 1.0},  # sticky override
        {"ball_x": 5, "ball_y": 5, "ball_dir": 2, "paddle": 4, "action": 0, "step": 9, "max_steps": 10},  # step cutoff
        {"ball_x": 4, "ball_y": 2, "ball_dir": 2, "paddle": 4, "action": 0, "strike": True},  # strike latch
        {"ball_x": 4, "ball_y": 8, "ball_dir": 2, "paddle": 4, "action": 0, "bricks": "empty"},  # new bricks then miss
        {"ball_x": 5, "ball_y": 5, "ball_dir": 2, "paddle": 0, "action": 1},  # paddle left clamp
        {"ball_x": 5, "ball_y": 5, "ball_dir": 2, "paddle": 9, "action": 3},  # paddle right clamp
    ]
    cases = []
    expected = []
    for spec in specs:
        case, result = build_reference_case(spec)
        cases.append(case)
        expected.append(result)
    actual = run_browser("transitions", {"cases": cases})
    assert len(actual) == len(expected)
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        assert observed["state"] == reference["state"], f"state mismatch in fixture {index}"
        assert observed["reward"] == reference["reward"], f"reward mismatch in fixture {index}"
        assert observed["done"] == reference["done"], f"termination mismatch in fixture {index}"
        assert observed["info"] == reference["info"], f"step metadata mismatch in fixture {index}"
        assert observed["observation"] == reference["observation"], f"observation mismatch in fixture {index}"


def test_browser_observation_mapping_matches_python():
    states = []
    expected = []
    for seed in (0, 42, 100):
        env = MinAtarBreakoutEnv(sticky_action_prob=0.1)
        env.reset(seed=seed)
        native = env.internal_env.state()
        native[9, 4, 0] = True
        native[6, 7, 1] = True
        native[5, 3, 2] = True
        native[2, 8, 3] = True
        states.append(native.astype(int).reshape(100, 4).tolist())
        expected.append(flatten_observation(get_observation(native)).astype(int).tolist())
    actual = run_browser("observations", {"states": states})
    assert actual == expected


def test_browser_network_outputs_match_numpy_for_all_primary_weights():
    rng = np.random.RandomState(631)
    observations = [
        np.zeros(600, dtype=np.float32),
        rng.randint(0, 2, size=600).astype(np.float32),
    ]
    cases = []
    expected = []
    for model_id, relative_path in PRIMARY_WEIGHT_FILES.items():
        weights = load_weights_npz(str(ROOT / relative_path))
        network = MinAtarNetworkNumPy(weights)
        weight_payload = {
            name: {"shape": list(value.shape), "values": value.reshape(-1).astype(float).tolist()}
            for name, value in network.weights.items()
        }
        for observation in observations:
            result = network.forward(observation)
            shifted = result - np.max(result)
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            cases.append({"model": model_id, "weights": weight_payload, "observation": observation.astype(float).tolist()})
            expected.append({"logits": result.tolist(), "probabilities": probabilities.tolist()})
    actual = run_browser("networks", {"cases": cases})
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(observed["logits"], reference["logits"], rtol=2e-4, atol=2e-4, err_msg=f"logits case {index}")
        np.testing.assert_allclose(observed["probabilities"], reference["probabilities"], rtol=2e-4, atol=2e-4, err_msg=f"softmax case {index}")
