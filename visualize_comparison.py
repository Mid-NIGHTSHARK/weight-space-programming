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
import os
import sys
import time
from typing import Optional, Tuple, Union

import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.actions import ACTION_NAMES, NUM_ACTIONS
from environment.breakout import MinAtarBreakoutEnv
from network.architecture import MinAtarNetworkNumPy
from network.inference import PolicyAgent
from network.weight_loader import load_weights_npz, compute_file_sha256

# Standard verified reference path and hash
ASTRA_REFERENCE_PATH = "experiments/reference/hand.npz"
ASTRA_EXPECTED_SHA256 = "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"


class VisualizerState:
    """Manages independent game simulations and policies for Argon, Trained, and optional Astra."""

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

    def step(self) -> Union[Tuple[bool, bool], Tuple[bool, bool, bool]]:
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
            return self.argon_done, self.trained_done, self.astra_done

        return self.argon_done, self.trained_done

    @property
    def all_done(self) -> bool:
        """Whether all active environments have completed their episodes."""
        if self.has_astra:
            return self.argon_done and self.trained_done and self.astra_done
        return self.argon_done and self.trained_done


class BreakoutVisualizerGUI:
    """Tkinter interactive visualizer application for Breakout policies."""

    def __init__(self, state: VisualizerState, root, initial_fps: int = 15):
        import tkinter as tk

        self.state = state
        self.root = root

        # Window configuration
        self.root.configure(bg="#141419")
        if self.state.has_astra:
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
        self.COLOR_PADDLE_ARGON = "#00e5ff"    # Cyan
        self.COLOR_PADDLE_ASTRA = "#ffd700"    # Gold
        self.COLOR_PADDLE_TRAINED = "#00e676"  # Bright Green
        self.COLOR_BALL = "#ffff00"            # Bright Yellow
        self.COLOR_TRAIL = "#ff9100"           # Orange
        self.COLOR_BRICK_ROW1 = "#ff5252"      # Red
        self.COLOR_BRICK_ROW2 = "#ff9800"      # Amber
        self.COLOR_BRICK_ROW3 = "#4caf50"      # Green
        self.COLOR_TEXT = "#e0e0e0"
        self.COLOR_TEXT_MUTED = "#888899"
        self.COLOR_STICKY = "#ff9800"          # Amber alert
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
        header = tk.Frame(self.root, bg=self.COLOR_PANEL, pady=10, padx=20)
        header.pack(fill=tk.X, side=tk.TOP)

        title_text = (
            "BREAKOUT THREE-WAY POLICY COMPARISON LAB"
            if self.state.has_astra
            else "BREAKOUT TWO-WAY POLICY COMPARISON LAB"
        )
        self.title_label = tk.Label(
            header,
            text=title_text,
            font=("Helvetica", 15, "bold"),
            fg="#ffffff",
            bg=self.COLOR_PANEL,
        )
        self.title_label.pack(side=tk.LEFT)

        self.seed_label = tk.Label(
            header,
            text=f"Seed: {self.state.current_seed}",
            font=("Helvetica", 13, "bold"),
            fg="#00e5ff",
            bg=self.COLOR_PANEL,
        )
        self.seed_label.pack(side=tk.RIGHT)

        # --- MAIN ARENA (MODELS) ---
        self.arena = tk.Frame(self.root, bg=self.COLOR_BG, padx=12, pady=10)
        self.arena.pack(fill=tk.BOTH, expand=True)

        # 1. ARGON PANEL
        self.argon_panel = self._create_model_panel(
            parent=self.arena,
            title="GEMINI ARGON (Zero-Shot)",
            color=self.COLOR_PADDLE_ARGON,
        )

        # 2. ASTRA PANEL (optional)
        if self.state.has_astra:
            self.astra_panel = self._create_model_panel(
                parent=self.arena,
                title="ASTRA REFERENCE (Original)",
                color=self.COLOR_PADDLE_ASTRA,
            )

        # 3. TRAINED BASELINE PANEL
        self.trained_panel = self._create_model_panel(
            parent=self.arena,
            title="TRAINED BASELINE (PPO 10M)",
            color=self.COLOR_PADDLE_TRAINED,
        )

        # --- CONTROL BAR ---
        self._build_control_bar()

    def _create_model_panel(self, parent, title: str, color: str):
        import tkinter as tk

        frame = tk.Frame(parent, bg=self.COLOR_PANEL, padx=10, pady=10, relief=tk.RIDGE, bd=2)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Title
        lbl_title = tk.Label(frame, text=title, font=("Helvetica", 12, "bold"), fg=color, bg=self.COLOR_PANEL)
        lbl_title.pack(anchor=tk.W)

        # Stats label (Score, Steps, Bricks, Status)
        lbl_stats = tk.Label(
            frame,
            text="Score: 0   Steps: 0   Bricks: 30/30\nStatus: PLAYING",
            font=("Helvetica", 9),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_PANEL,
            justify=tk.LEFT,
        )
        lbl_stats.pack(anchor=tk.W, pady=(2, 4))

        # Game Canvas
        canvas_board = tk.Canvas(
            frame,
            width=self.board_pixels,
            height=self.board_pixels,
            bg=self.COLOR_BOARD_BG,
            highlightthickness=1,
            highlightbackground="#333344",
        )
        canvas_board.pack(pady=4)

        # Action Box Frame
        action_frame = tk.Frame(frame, bg="#16161e", padx=6, pady=4, relief=tk.GROOVE, bd=1)
        action_frame.pack(fill=tk.X, pady=(4, 4))

        lbl_selected_action = tk.Label(
            action_frame,
            text="Selected Action: None",
            font=("Helvetica", 10, "bold"),
            fg=color,
            bg="#16161e",
            anchor=tk.W,
        )
        lbl_selected_action.pack(fill=tk.X)

        lbl_executed_action = tk.Label(
            action_frame,
            text="Executed Action: None",
            font=("Helvetica", 9),
            fg=self.COLOR_TEXT_MUTED,
            bg="#16161e",
            anchor=tk.W,
        )
        lbl_executed_action.pack(fill=tk.X)

        # Probability Header
        lbl_probs_title = tk.Label(
            frame,
            text="ACTION PROBABILITIES (6 Actions, 100%):",
            font=("Helvetica", 8, "bold"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_PANEL,
            anchor=tk.W,
        )
        lbl_probs_title.pack(anchor=tk.W, pady=(4, 1))

        # Probability Canvas (height=145 fits all 6 actions + footer)
        canvas_probs = tk.Canvas(frame, width=self.board_pixels, height=140, bg=self.COLOR_PANEL, highlightthickness=0)
        canvas_probs.pack(fill=tk.X, pady=2)

        return {
            "frame": frame,
            "title_label": lbl_title,
            "stats_label": lbl_stats,
            "canvas_board": canvas_board,
            "selected_action_label": lbl_selected_action,
            "executed_action_label": lbl_executed_action,
            "canvas_probs": canvas_probs,
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
            padx=8,
            pady=3,
        )
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_step = tk.Button(
            ctrl_frame,
            text="Step Frame (→)",
            command=self.step_single,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=8,
            pady=3,
        )
        self.btn_step.pack(side=tk.LEFT, padx=4)

        self.btn_reset = tk.Button(
            ctrl_frame,
            text="Reset (R)",
            command=self.reset_current,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=8,
            pady=3,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        self.btn_prev = tk.Button(
            ctrl_frame,
            text="Prev Episode (P)",
            command=self.prev_episode,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=8,
            pady=3,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=4)

        self.btn_next = tk.Button(
            ctrl_frame,
            text="Next Episode (N)",
            command=self.next_episode,
            font=btn_font,
            bg=btn_bg,
            fg=btn_fg,
            padx=8,
            pady=3,
        )
        self.btn_next.pack(side=tk.LEFT, padx=4)

        # FPS Slider
        tk.Label(ctrl_frame, text="Speed (FPS):", font=("Helvetica", 10), fg=self.COLOR_TEXT, bg=self.COLOR_PANEL).pack(
            side=tk.LEFT, padx=(15, 3)
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
            length=80,
        )
        self.slider_speed.pack(side=tk.LEFT, padx=3)

        # Shortcuts Legend
        lbl_shortcuts = tk.Label(
            ctrl_frame,
            text="[Space] Play/Pause   [→] Step   [R] Reset   [P/N] Prev/Next",
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
        """Redraw all model panels and refresh controls."""
        # Update Argon
        self._update_panel_display(
            panel=self.argon_panel,
            env=self.state.env_argon,
            is_done=self.state.argon_done,
            selected_act=self.state.argon_selected_action,
            executed_act=self.state.argon_executed_action,
            sticky_override=self.state.argon_sticky_override,
            probs=self.state.argon_probs,
        )

        # Update Astra if enabled
        if self.state.has_astra:
            self._update_panel_display(
                panel=self.astra_panel,
                env=self.state.env_astra,
                is_done=self.state.astra_done,
                selected_act=self.state.astra_selected_action,
                executed_act=self.state.astra_executed_action,
                sticky_override=self.state.astra_sticky_override,
                probs=self.state.astra_probs,
            )

        # Update Trained Baseline
        self._update_panel_display(
            panel=self.trained_panel,
            env=self.state.env_trained,
            is_done=self.state.trained_done,
            selected_act=self.state.trained_selected_action,
            executed_act=self.state.trained_executed_action,
            sticky_override=self.state.trained_sticky_override,
            probs=self.state.trained_probs,
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
            print(msg)
    print("Headless verification PASSED cleanly.")
    return True


def capture_screenshot(
    output_path: str,
    argon_path: str,
    trained_path: str,
    astra_path: Optional[str] = None,
    seed: int = 42,
    steps_to_run: int = 25,
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
        initial_seed=seed,
    )
    app = BreakoutVisualizerGUI(state, root, initial_fps=30)
    app.is_running.set(False)

    root.geometry("1300x820+0+28")
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
            crop_h = min(im.height, 865)
            crop_w = min(im.width, 1300)
            cropped = im.crop((0, 0, crop_w, crop_h))
            cropped.save(output_path)
            return True
    return False


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

    # Provenance check for Astra if supplied
    astra_path = args.astra
    if astra_path is not None and astra_path != "":
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
    else:
        astra_path = None

    if args.screenshot:
        success = capture_screenshot(
            output_path=args.screenshot,
            argon_path=args.argon,
            trained_path=args.trained,
            astra_path=astra_path,
            seed=args.seed,
            steps_to_run=args.steps,
        )
        print(f"Screenshot captured: {success} -> {args.screenshot}")
        sys.exit(0 if success else 1)

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
