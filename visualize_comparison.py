#!/usr/bin/env python3
"""Side-by-side Visual Comparison Application: Gemini Argon vs Astra Reference vs Trained Baseline.

Renders live, independent MinAtar Breakout environments for:
  1. Gemini Argon's generated neural-network weights (Zero-Shot)
  2. Astra's original reference weights (Original Experiment 5 hand.npz, if verified)
  3. PPO model trained from scratch (10M steps)

Usage:
  python visualize_comparison.py
  python visualize_comparison.py --argon path/to/argon.npz --astra path/to/hand.npz --trained path/to/trained.npz
  python visualize_comparison.py --headless-test
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple, Union

import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.actions import ACTION_NAMES
from environment.breakout import MinAtarBreakoutEnv
from network.architecture import MinAtarNetworkNumPy
from network.inference import PolicyAgent
from network.weight_loader import load_weights_npz, compute_file_sha256

# Standard verified reference path and hash
ASTRA_REFERENCE_PATH = "experiments/reference/hand.npz"
ASTRA_EXPECTED_SHA256 = "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"


class VisualizerState:
    """Manages the independent game simulations and policies for Argon, Trained, and optional Astra."""

    def __init__(
        self,
        argon_weights_path: str,
        trained_weights_path: str,
        astra_weights_path: Optional[str] = None,
        initial_seed: int = 42,
    ):
        self.argon_weights_path = argon_weights_path
        self.trained_weights_path = trained_weights_path
        self.astra_weights_path = astra_weights_path
        self.current_seed = initial_seed

        # 1. Load Argon policy
        if not os.path.exists(argon_weights_path):
            raise FileNotFoundError(f"Argon weights not found at {argon_weights_path}")
        argon_weights = load_weights_npz(argon_weights_path)
        self.argon_net = MinAtarNetworkNumPy(argon_weights)
        self.argon_agent = PolicyAgent(self.argon_net)

        # 2. Load Trained policy
        if not os.path.exists(trained_weights_path):
            raise FileNotFoundError(f"Trained model weights not found at {trained_weights_path}")
        trained_weights = load_weights_npz(trained_weights_path)
        self.trained_net = MinAtarNetworkNumPy(trained_weights)
        self.trained_agent = PolicyAgent(self.trained_net)

        # 3. Optional Astra Reference policy
        self.has_astra = False
        if astra_weights_path is not None and os.path.exists(astra_weights_path):
            astra_weights = load_weights_npz(astra_weights_path)
            self.astra_net = MinAtarNetworkNumPy(astra_weights)
            self.astra_agent = PolicyAgent(self.astra_net)
            self.env_astra = MinAtarBreakoutEnv(sticky_action_prob=0.1)
            self.has_astra = True

        # Environment instances
        self.env_argon = MinAtarBreakoutEnv(sticky_action_prob=0.1)
        self.env_trained = MinAtarBreakoutEnv(sticky_action_prob=0.1)

        self.reset_episode(self.current_seed)

    def reset_episode(self, seed: int):
        self.current_seed = seed
        self.argon_agent.rng = np.random.RandomState(seed)
        self.trained_agent.rng = np.random.RandomState(seed)

        self.obs_argon = self.env_argon.reset(seed=seed)
        self.obs_trained = self.env_trained.reset(seed=seed)

        self.argon_done = False
        self.trained_done = False

        self.argon_info = {}
        self.trained_info = {}

        self.argon_action = 0
        self.argon_probs = np.zeros(6, dtype=np.float32)
        self.trained_action = 0
        self.trained_probs = np.zeros(6, dtype=np.float32)

        # Compute initial action distributions
        self._update_argon_policy()
        self._update_trained_policy()

        if self.has_astra:
            self.astra_agent.rng = np.random.RandomState(seed)
            self.obs_astra = self.env_astra.reset(seed=seed)
            self.astra_done = False
            self.astra_info = {}
            self.astra_action = 0
            self.astra_probs = np.zeros(6, dtype=np.float32)
            self._update_astra_policy()

    def _update_argon_policy(self):
        if not self.argon_done:
            self.argon_action, self.argon_probs, _ = self.argon_agent.get_action(self.obs_argon)

    def _update_trained_policy(self):
        if not self.trained_done:
            self.trained_action, self.trained_probs, _ = self.trained_agent.get_action(self.obs_trained)

    def _update_astra_policy(self):
        if self.has_astra and not self.astra_done:
            self.astra_action, self.astra_probs, _ = self.astra_agent.get_action(self.obs_astra)

    def step(self) -> Union[Tuple[bool, bool], Tuple[bool, bool, bool]]:
        """Advance environments by 1 frame if not terminal."""
        # Step Argon
        if not self.argon_done:
            self.obs_argon, _, self.argon_done, self.argon_info = self.env_argon.step(self.argon_action)
            self._update_argon_policy()

        # Step Trained
        if not self.trained_done:
            self.obs_trained, _, self.trained_done, self.trained_info = self.env_trained.step(self.trained_action)
            self._update_trained_policy()

        # Step Astra if enabled
        if self.has_astra:
            if not self.astra_done:
                self.obs_astra, _, self.astra_done, self.astra_info = self.env_astra.step(self.astra_action)
                self._update_astra_policy()
            return self.argon_done, self.trained_done, self.astra_done

        return self.argon_done, self.trained_done

    @property
    def all_done(self) -> bool:
        if self.has_astra:
            return self.argon_done and self.trained_done and self.astra_done
        return self.argon_done and self.trained_done


def run_gui(state: VisualizerState, initial_fps: int = 15):
    """Launch interactive Tkinter visualizer."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    if state.has_astra:
        root.title("MinAtar Breakout: Gemini Argon vs. Astra Reference vs. Trained PPO")
        root.geometry("1240x760")
        CELL_SIZE = 26
    else:
        root.title("MinAtar Breakout: Gemini Argon vs. Trained Model (PPO)")
        root.geometry("980x760")
        CELL_SIZE = 30

    root.configure(bg="#141419")

    # Styling colors
    COLOR_BG = "#141419"
    COLOR_PANEL = "#1e1e26"
    COLOR_BOARD_BG = "#0a0a0f"
    COLOR_PADDLE_ARGON = "#00e5ff"    # Cyan
    COLOR_PADDLE_ASTRA = "#ffd700"    # Gold / Yellow-Amber
    COLOR_PADDLE_TRAINED = "#00e676"  # Bright Green
    COLOR_BALL = "#ffff00"            # Bright Yellow
    COLOR_TRAIL = "#ff9100"           # Orange
    COLOR_BRICK_ROW1 = "#ff5252"      # Red
    COLOR_BRICK_ROW2 = "#ff9800"      # Amber
    COLOR_BRICK_ROW3 = "#4caf50"      # Green
    COLOR_TEXT = "#e0e0e0"

    BOARD_PIXELS = 10 * CELL_SIZE

    is_running = tk.BooleanVar(value=True)
    fps_var = tk.IntVar(value=initial_fps)
    seed_var = tk.IntVar(value=state.current_seed)

    # Header Frame
    header = tk.Frame(root, bg=COLOR_PANEL, pady=10, padx=20)
    header.pack(fill=tk.X, side=tk.TOP)

    header_title = (
        "BREAKOUT THREE-WAY POLICY COMPARISON LAB"
        if state.has_astra
        else "BREAKOUT TWO-WAY POLICY COMPARISON LAB"
    )
    title_label = tk.Label(
        header,
        text=header_title,
        font=("Helvetica", 15, "bold"),
        fg="#ffffff",
        bg=COLOR_PANEL,
    )
    title_label.pack(side=tk.LEFT)

    seed_label = tk.Label(
        header,
        text=f"Seed: {state.current_seed}",
        font=("Helvetica", 13, "bold"),
        fg="#00e5ff",
        bg=COLOR_PANEL,
    )
    seed_label.pack(side=tk.RIGHT)

    # Main Arena Frame (Side by Side)
    arena = tk.Frame(root, bg=COLOR_BG, padx=15, pady=12)
    arena.pack(fill=tk.BOTH, expand=True)

    # --- 1. LEFT SIDE: GEMINI ARGON ---
    frame_argon = tk.Frame(arena, bg=COLOR_PANEL, padx=12, pady=12, relief=tk.RIDGE, bd=2)
    frame_argon.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

    label_argon_title = tk.Label(
        frame_argon,
        text="GEMINI ARGON (Zero-Shot)",
        font=("Helvetica", 12, "bold"),
        fg=COLOR_PADDLE_ARGON,
        bg=COLOR_PANEL,
    )
    label_argon_title.pack(anchor=tk.W)

    argon_stats_label = tk.Label(
        frame_argon,
        text="Score: 0   Steps: 0   Status: Running",
        font=("Helvetica", 10),
        fg=COLOR_TEXT,
        bg=COLOR_PANEL,
    )
    argon_stats_label.pack(anchor=tk.W, pady=(4, 6))

    canvas_argon = tk.Canvas(
        frame_argon,
        width=BOARD_PIXELS,
        height=BOARD_PIXELS,
        bg=COLOR_BOARD_BG,
        highlightthickness=1,
        highlightbackground="#333344",
    )
    canvas_argon.pack(pady=4)

    argon_action_label = tk.Label(
        frame_argon,
        text="Action: None",
        font=("Helvetica", 11, "bold"),
        fg=COLOR_PADDLE_ARGON,
        bg=COLOR_PANEL,
    )
    argon_action_label.pack(anchor=tk.W, pady=(6, 2))

    canvas_probs_argon = tk.Canvas(frame_argon, width=BOARD_PIXELS, height=65, bg=COLOR_PANEL, highlightthickness=0)
    canvas_probs_argon.pack(fill=tk.X, pady=2)

    # --- 2. CENTER: ASTRA REFERENCE (if enabled) ---
    if state.has_astra:
        frame_astra = tk.Frame(arena, bg=COLOR_PANEL, padx=12, pady=12, relief=tk.RIDGE, bd=2)
        frame_astra.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        label_astra_title = tk.Label(
            frame_astra,
            text="ASTRA REFERENCE (Original)",
            font=("Helvetica", 12, "bold"),
            fg=COLOR_PADDLE_ASTRA,
            bg=COLOR_PANEL,
        )
        label_astra_title.pack(anchor=tk.W)

        astra_stats_label = tk.Label(
            frame_astra,
            text="Score: 0   Steps: 0   Status: Running",
            font=("Helvetica", 10),
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
        )
        astra_stats_label.pack(anchor=tk.W, pady=(4, 6))

        canvas_astra = tk.Canvas(
            frame_astra,
            width=BOARD_PIXELS,
            height=BOARD_PIXELS,
            bg=COLOR_BOARD_BG,
            highlightthickness=1,
            highlightbackground="#333344",
        )
        canvas_astra.pack(pady=4)

        astra_action_label = tk.Label(
            frame_astra,
            text="Action: None",
            font=("Helvetica", 11, "bold"),
            fg=COLOR_PADDLE_ASTRA,
            bg=COLOR_PANEL,
        )
        astra_action_label.pack(anchor=tk.W, pady=(6, 2))

        canvas_probs_astra = tk.Canvas(frame_astra, width=BOARD_PIXELS, height=65, bg=COLOR_PANEL, highlightthickness=0)
        canvas_probs_astra.pack(fill=tk.X, pady=2)

    # --- 3. RIGHT SIDE: TRAINED MODEL ---
    frame_trained = tk.Frame(arena, bg=COLOR_PANEL, padx=12, pady=12, relief=tk.RIDGE, bd=2)
    frame_trained.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

    label_trained_title = tk.Label(
        frame_trained,
        text="TRAINED BASELINE (PPO)",
        font=("Helvetica", 12, "bold"),
        fg=COLOR_PADDLE_TRAINED,
        bg=COLOR_PANEL,
    )
    label_trained_title.pack(anchor=tk.W)

    trained_stats_label = tk.Label(
        frame_trained,
        text="Score: 0   Steps: 0   Status: Running",
        font=("Helvetica", 10),
        fg=COLOR_TEXT,
        bg=COLOR_PANEL,
    )
    trained_stats_label.pack(anchor=tk.W, pady=(4, 6))

    canvas_trained = tk.Canvas(
        frame_trained,
        width=BOARD_PIXELS,
        height=BOARD_PIXELS,
        bg=COLOR_BOARD_BG,
        highlightthickness=1,
        highlightbackground="#333344",
    )
    canvas_trained.pack(pady=4)

    trained_action_label = tk.Label(
        frame_trained,
        text="Action: None",
        font=("Helvetica", 11, "bold"),
        fg=COLOR_PADDLE_TRAINED,
        bg=COLOR_PANEL,
    )
    trained_action_label.pack(anchor=tk.W, pady=(6, 2))

    canvas_probs_trained = tk.Canvas(frame_trained, width=BOARD_PIXELS, height=65, bg=COLOR_PANEL, highlightthickness=0)
    canvas_probs_trained.pack(fill=tk.X, pady=2)

    # --- CONTROL BAR ---
    ctrl_frame = tk.Frame(root, bg=COLOR_PANEL, pady=10, padx=20)
    ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

    def toggle_play():
        is_running.set(not is_running.get())
        btn_play.config(text="Pause (Space)" if is_running.get() else "Play (Space)")

    def step_single():
        is_running.set(False)
        btn_play.config(text="Play (Space)")
        state.step()
        redraw()

    def reset_current():
        state.reset_episode(seed_var.get())
        seed_label.config(text=f"Seed: {seed_var.get()}")
        redraw()

    def next_episode():
        new_seed = seed_var.get() + 1
        seed_var.set(new_seed)
        state.reset_episode(new_seed)
        seed_label.config(text=f"Seed: {new_seed}")
        redraw()

    def prev_episode():
        new_seed = max(1, seed_var.get() - 1)
        seed_var.set(new_seed)
        state.reset_episode(new_seed)
        seed_label.config(text=f"Seed: {new_seed}")
        redraw()

    btn_play = tk.Button(
        ctrl_frame,
        text="Pause (Space)",
        command=toggle_play,
        font=("Helvetica", 11, "bold"),
        bg="#333344",
        fg="#ffffff",
        padx=10,
        pady=3,
    )
    btn_play.pack(side=tk.LEFT, padx=5)

    btn_step = tk.Button(
        ctrl_frame,
        text="Step 1 Frame (→)",
        command=step_single,
        font=("Helvetica", 11),
        bg="#333344",
        fg="#ffffff",
        padx=8,
        pady=3,
    )
    btn_step.pack(side=tk.LEFT, padx=5)

    btn_reset = tk.Button(
        ctrl_frame,
        text="Reset (R)",
        command=reset_current,
        font=("Helvetica", 11),
        bg="#333344",
        fg="#ffffff",
        padx=8,
        pady=3,
    )
    btn_reset.pack(side=tk.LEFT, padx=5)

    btn_prev = tk.Button(
        ctrl_frame,
        text="Prev Episode (P)",
        command=prev_episode,
        font=("Helvetica", 11),
        bg="#333344",
        fg="#ffffff",
        padx=8,
        pady=3,
    )
    btn_prev.pack(side=tk.LEFT, padx=5)

    btn_next = tk.Button(
        ctrl_frame,
        text="Next Episode (N)",
        command=next_episode,
        font=("Helvetica", 11),
        bg="#333344",
        fg="#ffffff",
        padx=8,
        pady=3,
    )
    btn_next.pack(side=tk.LEFT, padx=5)

    # Speed Control Slider
    tk.Label(ctrl_frame, text="Speed (FPS):", font=("Helvetica", 11), fg=COLOR_TEXT, bg=COLOR_PANEL).pack(
        side=tk.LEFT, padx=(20, 5)
    )
    slider_speed = tk.Scale(
        ctrl_frame,
        from_=1,
        to=60,
        orient=tk.HORIZONTAL,
        variable=fps_var,
        bg=COLOR_PANEL,
        fg="#ffffff",
        troughcolor="#333344",
        highlightthickness=0,
        length=110,
    )
    slider_speed.pack(side=tk.LEFT, padx=5)

    # Keybindings
    root.bind("<space>", lambda e: toggle_play())
    root.bind("<Right>", lambda e: step_single())
    root.bind("r", lambda e: reset_current())
    root.bind("n", lambda e: next_episode())
    root.bind("p", lambda e: prev_episode())

    def draw_board(canvas: tk.Canvas, env: MinAtarBreakoutEnv, paddle_color: str):
        canvas.delete("all")
        internal = env.internal_env

        # Grid lines
        for r in range(10):
            for c in range(10):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                canvas.create_rectangle(x0, y0, x1, y1, outline="#1a1a24", fill=COLOR_BOARD_BG)

        # Bricks (rows 1..3)
        brick_map = internal.brick_map
        row_colors = [COLOR_BRICK_ROW1, COLOR_BRICK_ROW2, COLOR_BRICK_ROW3]
        for r in range(1, 4):
            for c in range(10):
                if brick_map[r, c] == 1:
                    x0, y0 = c * CELL_SIZE + 2, r * CELL_SIZE + 2
                    x1, y1 = (c + 1) * CELL_SIZE - 2, (r + 1) * CELL_SIZE - 2
                    canvas.create_rectangle(x0, y0, x1, y1, fill=row_colors[r - 1], outline="")

        # Trail
        lx, ly = internal.last_x, internal.last_y
        if 0 <= lx < 10 and 0 <= ly < 10 and not (lx == internal.ball_x and ly == internal.ball_y):
            cx, cy = lx * CELL_SIZE + CELL_SIZE // 2, ly * CELL_SIZE + CELL_SIZE // 2
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=COLOR_TRAIL, outline="")

        # Ball
        bx, by = internal.ball_x, internal.ball_y
        if 0 <= bx < 10 and 0 <= by < 10:
            cx, cy = bx * CELL_SIZE + CELL_SIZE // 2, by * CELL_SIZE + CELL_SIZE // 2
            canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=COLOR_BALL, outline="#ffffff")

        # Paddle (Row 9)
        pos = internal.pos
        x0 = pos * CELL_SIZE + 2
        y0 = 9 * CELL_SIZE + 4
        x1 = (pos + 1) * CELL_SIZE - 2
        y1 = 10 * CELL_SIZE - 4
        canvas.create_rectangle(x0, y0, x1, y1, fill=paddle_color, outline="#ffffff", width=1)

    def draw_probs(canvas: tk.Canvas, probs: np.ndarray, chosen_act: int, primary_color: str):
        canvas.delete("all")
        actions_to_show = [(0, "No-Op"), (1, "Left"), (3, "Right")]
        bar_height = 14
        start_y = 4
        track_w = BOARD_PIXELS - 100

        for idx, (a, label) in enumerate(actions_to_show):
            prob = probs[a]
            y = start_y + idx * (bar_height + 4)
            # Label
            canvas.create_text(
                35,
                y + bar_height // 2,
                text=label,
                fill=primary_color if a == chosen_act else COLOR_TEXT,
                font=("Helvetica", 9, "bold" if a == chosen_act else "normal"),
                anchor=tk.E,
            )
            # Background track
            canvas.create_rectangle(42, y, 42 + track_w, y + bar_height, fill="#252530", outline="")
            # Filled bar
            bar_w = int(track_w * prob)
            canvas.create_rectangle(
                42,
                y,
                42 + bar_w,
                y + bar_height,
                fill=primary_color if a == chosen_act else "#555566",
                outline="",
            )
            # Percent text
            canvas.create_text(
                46 + track_w,
                y + bar_height // 2,
                text=f"{prob * 100:.1f}%",
                fill=COLOR_TEXT,
                font=("Helvetica", 9),
                anchor=tk.W,
            )

    def redraw():
        # Draw Argon
        draw_board(canvas_argon, state.env_argon, COLOR_PADDLE_ARGON)
        argon_status = "LOST BALL" if state.argon_done else "PLAYING"
        argon_stats_label.config(
            text=f"Score: {int(state.env_argon.total_reward)}   "
            f"Steps: {state.env_argon.step_count}   "
            f"Status: {argon_status}"
        )
        argon_action_label.config(
            text=f"Action: {ACTION_NAMES.get(state.argon_action, 'Unknown')} ({state.argon_action})"
        )
        draw_probs(canvas_probs_argon, state.argon_probs, state.argon_action, COLOR_PADDLE_ARGON)

        # Draw Astra (if enabled)
        if state.has_astra:
            draw_board(canvas_astra, state.env_astra, COLOR_PADDLE_ASTRA)
            astra_status = "LOST BALL" if state.astra_done else "PLAYING"
            astra_stats_label.config(
                text=f"Score: {int(state.env_astra.total_reward)}   "
                f"Steps: {state.env_astra.step_count}   "
                f"Status: {astra_status}"
            )
            astra_action_label.config(
                text=f"Action: {ACTION_NAMES.get(state.astra_action, 'Unknown')} ({state.astra_action})"
            )
            draw_probs(canvas_probs_astra, state.astra_probs, state.astra_action, COLOR_PADDLE_ASTRA)

        # Draw Trained
        draw_board(canvas_trained, state.env_trained, COLOR_PADDLE_TRAINED)
        trained_status = "LOST BALL" if state.trained_done else "PLAYING"
        trained_stats_label.config(
            text=f"Score: {int(state.env_trained.total_reward)}   "
            f"Steps: {state.env_trained.step_count}   "
            f"Status: {trained_status}"
        )
        trained_action_label.config(
            text=f"Action: {ACTION_NAMES.get(state.trained_action, 'Unknown')} ({state.trained_action})"
        )
        draw_probs(canvas_probs_trained, state.trained_probs, state.trained_action, COLOR_PADDLE_TRAINED)

    def game_loop():
        if is_running.get():
            if state.all_done:
                time.sleep(0.5)
                next_episode()
            else:
                state.step()
                redraw()

        delay = max(10, int(1000 / max(1, fps_var.get())))
        root.after(delay, game_loop)

    redraw()
    root.after(100, game_loop)
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
            print(msg)
    print("Headless verification PASSED cleanly.")
    return True


def main():
    parser = argparse.ArgumentParser(description="MinAtar Breakout Policy Visualizer (Argon, Astra, Trained)")
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
        "--trained",
        type=str,
        default="experiments/trained_baseline/trained_actor_weights.npz",
        help="Path to Trained model weights .npz",
    )
    parser.add_argument("--seed", type=int, default=42, help="Initial episode random seed")
    parser.add_argument("--fps", type=int, default=15, help="Simulation speed in FPS")
    parser.add_argument(
        "--headless-test",
        action="store_true",
        help="Run non-GUI verification test instead of opening window",
    )
    args = parser.parse_args()

    # Provenance check for Astra if supplied
    astra_path = args.astra
    if astra_path is not None:
        if not os.path.exists(astra_path):
            print(f"[Warning] Astra weights file not found at {astra_path}; falling back to 2-model visualizer.")
            astra_path = None
        else:
            actual_hash = compute_file_sha256(astra_path)
            if actual_hash != ASTRA_EXPECTED_SHA256:
                print(f"[Warning] Astra SHA-256 mismatch ({actual_hash} != {ASTRA_EXPECTED_SHA256}); excluding Astra.")
                astra_path = None
            else:
                print(f"[Verified] Astra reference weights verified ({actual_hash[:16]}...). Enabling 3-way visualizer.")

    state = VisualizerState(
        argon_weights_path=args.argon,
        trained_weights_path=args.trained,
        astra_weights_path=astra_path,
        initial_seed=args.seed,
    )

    if args.headless_test:
        success = run_headless_verification(state)
        sys.exit(0 if success else 1)
    else:
        run_gui(state, initial_fps=args.fps)


if __name__ == "__main__":
    main()
