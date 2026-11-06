"""Automated test suite for the Breakout Visualizer UI, event handling, and rendering."""

import os
import subprocess
import sys
import tkinter as tk
import numpy as np
import pytest

from environment.actions import MinAtarAction, ACTION_NAMES, NUM_ACTIONS
from environment.breakout import MinAtarBreakoutEnv
from network.inference import sample_action, softmax
from visualize_comparison import (
    VisualizerState,
    BreakoutVisualizerGUI,
    ASTRA_REFERENCE_PATH,
)

ARGON_WEIGHTS = "experiments/strict_replication/generation_001/output/argon_weights.npz"
TRAINED_WEIGHTS = "experiments/trained_baseline/trained_actor_weights.npz"


@pytest.fixture(scope="module")
def visualizer_state():
    return VisualizerState(
        argon_weights_path=ARGON_WEIGHTS,
        trained_weights_path=TRAINED_WEIGHTS,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        initial_seed=42,
    )


@pytest.fixture(scope="module")
def tk_app(visualizer_state):
    """Module-scoped Tkinter GUI instance to prevent Cocoa window manager conflicts."""
    root = tk.Tk()
    root.withdraw()  # Keep window hidden during test suite
    app = BreakoutVisualizerGUI(visualizer_state, root, initial_fps=30)
    app.is_running.set(False)
    yield root, app
    root.destroy()


def test_visualizer_state_all_six_action_probabilities(visualizer_state):
    """Verify that all models produce a 6-element probability distribution summing to 100%."""
    visualizer_state.reset_episode(42)

    for name, probs in [
        ("Argon", visualizer_state.argon_probs),
        ("Trained", visualizer_state.trained_probs),
        ("Astra", visualizer_state.astra_probs),
    ]:
        assert len(probs) == NUM_ACTIONS, f"{name} probs must have length 6"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-6), f"{name} probs must sum to 1.0"
        assert (probs >= 0.0).all(), f"{name} probabilities must be non-negative"
        # Action 4 (Down) must be present and finite
        assert np.isfinite(probs[4]), f"{name} Action 4 (Down) probability must be finite"


def test_action_down_mapping_and_execution():
    """Verify Action ID 4 is Down, mapped correctly, and executes without paddle displacement."""
    assert MinAtarAction.DOWN == 4
    assert ACTION_NAMES[4] == "down"

    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=42)
    initial_pos = env.internal_env.pos

    obs, reward, done, info = env.step(4)
    assert info["selected_action"] == 4
    assert info["executed_action"] == 4
    assert info["sticky_override"] is False
    # In Breakout, action 4 (Down) is inert and does not move the paddle
    assert env.internal_env.pos == initial_pos


def test_sampled_action_vs_argmax():
    """Verify action selection reflects categorical sampling rather than naive argmax."""
    probs = np.array([0.1, 0.45, 0.0, 0.45, 0.0, 0.0], dtype=np.float32)
    # With two equal highest probs, stochastic sampling will choose both across trials
    sampled_actions = set()
    for seed in range(50):
        rng = np.random.RandomState(seed)
        action = sample_action(probs, rng=rng)
        sampled_actions.add(action)

    assert 1 in sampled_actions, "Sampled actions should include Action 1"
    assert 3 in sampled_actions, "Sampled actions should include Action 3"


def test_sticky_action_distinction():
    """Verify that sticky action override is clearly distinguished from selected action."""
    env = MinAtarBreakoutEnv(sticky_action_prob=0.0)
    env.reset(seed=42)

    # Step 1: select LEFT non-sticky
    obs, r, done, info = env.step(MinAtarAction.LEFT)
    assert info["selected_action"] == MinAtarAction.LEFT
    assert info["executed_action"] == MinAtarAction.LEFT
    assert info["sticky_override"] is False

    # Force sticky action probability to 1.0
    env.sticky_action_prob = 1.0
    env._minatar_env.sticky_action_prob = 1.0

    # Step 2: select RIGHT; sticky action must force repetition of LEFT
    obs, r, done, info = env.step(MinAtarAction.RIGHT)
    assert info["selected_action"] == MinAtarAction.RIGHT
    assert info["executed_action"] == MinAtarAction.LEFT
    assert info["sticky_override"] is True


def test_gui_button_controls_and_state_transitions(tk_app):
    """Verify all GUI control buttons: Play/Pause, Step, Reset, Prev, Next."""
    root, app = tk_app

    # 1. Play / Pause toggle
    assert app.is_running.get() is False
    app.btn_play.invoke()
    assert app.is_running.get() is True
    assert "Pause" in app.btn_play.cget("text")
    app.btn_play.invoke()
    assert app.is_running.get() is False
    assert "Play" in app.btn_play.cget("text")

    # 2. Step single frame
    initial_step = app.state.env_argon.step_count
    app.btn_step.invoke()
    assert app.state.env_argon.step_count == initial_step + 1
    assert app.is_running.get() is False

    # 3. Reset episode
    app.btn_reset.invoke()
    assert app.state.env_argon.step_count == 0
    assert app.state.env_trained.step_count == 0
    if app.state.has_astra:
        assert app.state.env_astra.step_count == 0

    # 4. Episode Navigation & Boundary Control
    app.seed_var.set(1)
    app.state.reset_episode(1)
    app.update_control_states()
    assert str(app.btn_prev.cget("state")) == str(tk.DISABLED)

    # Next Episode -> seed 2 -> Prev enabled
    app.btn_next.invoke()
    assert app.seed_var.get() == 2
    assert str(app.btn_prev.cget("state")) == str(tk.NORMAL)

    # Prev Episode -> seed 1 -> Prev disabled
    app.btn_prev.invoke()
    assert app.seed_var.get() == 1
    assert str(app.btn_prev.cget("state")) == str(tk.DISABLED)


def test_gui_keyboard_bindings(tk_app):
    """Verify that keyboard shortcuts are properly registered to visualizer handlers."""
    root, app = tk_app

    # Check key bindings exist on root window
    expected_bindings = ["<space>", "<Right>", "r", "R", "n", "N", "p", "P"]
    for key in expected_bindings:
        binding_cmd = root.bind(key)
        assert binding_cmd != "", f"Missing binding for key {key}"


def test_multi_model_environment_isolation(visualizer_state):
    """Verify that all active environments advance completely independently."""
    visualizer_state.reset_episode(42)

    # Force trained model to terminate quickly
    visualizer_state.trained_done = True

    # Step simulation 5 times
    for _ in range(5):
        visualizer_state.step()

    # Trained environment should NOT have stepped after done
    assert visualizer_state.env_trained.step_count == 0
    # Argon and Astra should have stepped 5 times
    assert visualizer_state.env_argon.step_count == 5
    if visualizer_state.has_astra:
        assert visualizer_state.env_astra.step_count == 5


def test_all_six_action_probability_bars_drawn(tk_app):
    """Verify that draw_probs renders all 6 action items and summary row on the canvas."""
    root, app = tk_app
    canvas = app.argon_panel["canvas_probs"]

    probs = np.array([0.1, 0.4, 0.05, 0.25, 0.15, 0.05], dtype=np.float32)
    app.draw_probs(canvas, probs, selected_action=3, primary_color="#00e5ff")

    # Inspect drawn canvas text items
    items = canvas.find_all()
    texts = [canvas.itemcget(i, "text") for i in items if canvas.type(i) == "text"]

    expected_labels = ["0 No-Op", "1 Left", "2 Up", "3 Right", "4 Down", "5 Fire"]
    for label in expected_labels:
        assert any(label in t for t in texts), f"Missing label {label} in canvas texts: {texts}"

    # Verify total sum text is rendered
    assert any("Total: 100.0%" in t for t in texts), f"Missing total sum in canvas texts: {texts}"


def test_screenshot_capture_utility(tmp_path):
    """Verify headless screenshot capture creates a valid image file via CLI subprocess."""
    output_shot = str(tmp_path / "test_screenshot.png")
    res = subprocess.run([
        sys.executable,
        "visualize_comparison.py",
        "--screenshot",
        output_shot,
        "--seed",
        "42",
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"Screenshot subprocess failed with code {res.returncode}: {res.stderr}"
    assert os.path.exists(output_shot), f"Screenshot file was not created: {output_shot}"
    assert os.path.getsize(output_shot) > 1000, "Screenshot file is empty"


def test_visualizer_state_with_gpt6():
    """Verify VisualizerState correctly integrates GPT-6 alongside existing policies."""
    from visualize_comparison import GPT6_DEFAULT_PATH

    state = VisualizerState(
        argon_weights_path=ARGON_WEIGHTS,
        trained_weights_path=TRAINED_WEIGHTS,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        initial_seed=42,
    )
    assert state.has_gpt6 is True
    assert state.has_astra is True
    assert len(state.gpt6_probs) == NUM_ACTIONS
    assert np.isclose(np.sum(state.gpt6_probs), 1.0, atol=1e-6)
    assert (state.gpt6_probs >= 0.0).all()

    # Step simulation
    argon_done, trained_done, astra_done, gpt6_done = state.step()
    assert state.env_gpt6.step_count == 1
    assert state.env_argon.step_count == 1
    assert state.env_astra.step_count == 1
    assert state.env_trained.step_count == 1


def test_visualizer_gui_4way():
    """Verify that BreakoutVisualizerGUI constructs and renders the GPT-6 panel in 4-way mode."""
    from visualize_comparison import GPT6_DEFAULT_PATH

    state = VisualizerState(
        argon_weights_path=ARGON_WEIGHTS,
        trained_weights_path=TRAINED_WEIGHTS,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        initial_seed=42,
    )
    root = tk.Tk()
    root.withdraw()
    app = BreakoutVisualizerGUI(state, root, initial_fps=30)
    app.is_running.set(False)

    try:
        assert hasattr(app, "gpt6_panel")
        assert app.gpt6_panel["canvas_board"] is not None
        assert app.gpt6_panel["canvas_probs"] is not None
        assert "GPT-6" in app.gpt6_panel["title_label"].cget("text")

        # Verify redraw executes cleanly with 4 panels
        app.redraw()

        # Step single frame via app
        initial_steps = app.state.env_gpt6.step_count
        app.btn_step.invoke()
        assert app.state.env_gpt6.step_count == initial_steps + 1

        # Verify benchmark window can open without errors
        app.show_benchmark_summary()
        app.show_argon_high_info()
    finally:
        root.destroy()


def test_visualizer_gui_5way():
    """Verify that BreakoutVisualizerGUI constructs and renders the Sonnet 5.5 panel in 5-way mode."""
    from visualize_comparison import GPT6_DEFAULT_PATH, SONNET55_DEFAULT_PATH

    state = VisualizerState(
        argon_weights_path=ARGON_WEIGHTS,
        trained_weights_path=TRAINED_WEIGHTS,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_DEFAULT_PATH,
        initial_seed=42,
    )
    root = tk.Tk()
    root.withdraw()
    app = BreakoutVisualizerGUI(state, root, initial_fps=30)
    app.is_running.set(False)

    try:
        assert hasattr(app, "sonnet55_panel")
        assert app.sonnet55_panel["canvas_board"] is not None
        assert app.sonnet55_panel["canvas_probs"] is not None
        assert "SONNET 5.5" in app.sonnet55_panel["title_label"].cget("text")

        # Verify redraw executes cleanly with 5 panels
        app.redraw()

        # Step single frame via app
        initial_steps = app.state.env_sonnet55.step_count
        app.btn_step.invoke()
        assert app.state.env_sonnet55.step_count == initial_steps + 1

        # Verify modals can open without errors
        app.show_benchmark_summary()
        app.show_network_inspector()
        app.show_argon_high_info()
    finally:
        root.destroy()


def test_visualizer_gui_6way():
    """Verify that BreakoutVisualizerGUI constructs and renders the DeepSeek panel in 6-way mode."""
    from visualize_comparison import GPT6_DEFAULT_PATH, SONNET55_DEFAULT_PATH, DEEPSEEK_DEFAULT_PATH

    state = VisualizerState(
        argon_weights_path=ARGON_WEIGHTS,
        trained_weights_path=TRAINED_WEIGHTS,
        astra_weights_path=ASTRA_REFERENCE_PATH,
        gpt6_weights_path=GPT6_DEFAULT_PATH,
        sonnet55_weights_path=SONNET55_DEFAULT_PATH,
        deepseek_weights_path=DEEPSEEK_DEFAULT_PATH,
        initial_seed=42,
    )
    root = tk.Tk()
    root.withdraw()
    app = BreakoutVisualizerGUI(state, root, initial_fps=30)
    app.is_running.set(False)

    try:
        assert hasattr(app, "deepseek_panel")
        assert app.deepseek_panel["canvas_board"] is not None
        assert app.deepseek_panel["canvas_probs"] is not None
        assert "DEEPSEEK" in app.deepseek_panel["title_label"].cget("text")

        # Verify redraw executes cleanly with 6 panels
        app.redraw()

        # Step single frame via app
        initial_steps = app.state.env_deepseek.step_count
        app.btn_step.invoke()
        assert app.state.env_deepseek.step_count == initial_steps + 1

        # Verify all modals open cleanly in 6-way mode
        app.show_benchmark_summary()
        app.show_network_inspector()
        app.show_argon_high_info()
    finally:
        root.destroy()



