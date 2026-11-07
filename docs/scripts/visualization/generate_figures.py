#!/usr/bin/env python3
"""Generate publication-style figures from preserved episode CSVs.

The script reads existing Phase 2 and Argon iterative-development outputs and
the corresponding analysis JSON. It never modifies experimental source data.
Run from any directory with the repository's Python environment:

    .venv/bin/python docs/scripts/visualization/generate_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "wsp-matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "experiments/corrected_evaluation/full_run"
ITERATIVE = ROOT / "experiments/argon_high_iterative_development/evaluation/full_run"
PHASE2_ANALYSIS = PHASE2 / "analysis"
ITER_ANALYSIS = ITERATIVE / "analysis"
TRACE_FILE = ROOT / "experiments/corrected_evaluation/failure_trace_replays.json"
OUT = ROOT / "docs/assets/figures"
MANIFEST = ROOT / "docs/assets/figures/media_manifest.json"
SCRIPT_PATH = "docs/scripts/visualization/generate_figures.py"

PHASE2_ORDER = [
    "gemini_argon_4_medium",
    "claude_sonnet_55",
    "ppo_trained_baseline",
    "gpt6_original",
    "gpt6_luna_extra_high",
    "zero_weight_control",
    "deepseek_original",
    "random_action_control",
    "astra_local_reference",
]
PRIMARY = [
    "gemini_argon_4_medium",
    "claude_sonnet_55",
    "gpt6_original",
    "gpt6_luna_extra_high",
    "deepseek_original",
]
ITER_ORDER = ["argon_high_original", "argon_high_followup_1", "argon_high_followup_2"]
LABELS = {
    "astra_local_reference": "Astra reference",
    "gemini_argon_4_medium": "Gemini Argon 4 Medium",
    "claude_sonnet_55": "Claude Sonnet 5.5",
    "ppo_trained_baseline": "Trained PPO baseline",
    "gpt6_original": "GPT-6 Original",
    "gpt6_luna_extra_high": "GPT-6 Luna Extra High",
    "zero_weight_control": "Zero-weight control",
    "deepseek_original": "DeepSeek Original",
    "random_action_control": "Random-action control",
    "argon_high_original": "Original first-prompt assignments",
    "argon_high_followup_1": "Follow-up 1",
    "argon_high_followup_2": "Follow-up 2",
}
CLASSES = {
    "gemini_argon_4_medium": "Primary zero-shot",
    "claude_sonnet_55": "Primary zero-shot",
    "gpt6_original": "Primary zero-shot",
    "gpt6_luna_extra_high": "Primary zero-shot",
    "deepseek_original": "Primary zero-shot",
    "astra_local_reference": "Handcrafted/reference policy",
    "ppo_trained_baseline": "Trained baseline",
    "zero_weight_control": "Control",
    "random_action_control": "Control",
}
COLORS = {
    "gemini_argon_4_medium": "#205fbd",
    "claude_sonnet_55": "#0f8a78",
    "gpt6_original": "#7857a6",
    "gpt6_luna_extra_high": "#b26a1c",
    "deepseek_original": "#b34f68",
    "astra_local_reference": "#503f96",
    "ppo_trained_baseline": "#36805a",
    "zero_weight_control": "#818b99",
    "random_action_control": "#4b5563",
    "argon_high_original": "#9a5b1e",
    "argon_high_followup_1": "#c57b2b",
    "argon_high_followup_2": "#e0a15b",
}
ACTION_NAMES = ["No-op", "Left", "Up", "Right", "Down", "Fire"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#64748b",
        "axes.labelcolor": "#253247",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#172033",
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "grid.color": "#dbe2ea",
        "grid.linewidth": 0.65,
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase2_path(condition: str) -> Path:
    return PHASE2 / "conditions" / condition / "episode_scores.csv"


def iterative_path(condition: str) -> Path:
    return ITERATIVE / "conditions" / condition / "episode_scores.csv"


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    cutoffs = np.asarray([row["step_cutoff"].strip().lower() == "true" for row in rows])
    return {
        "n": int(scores.size),
        "mean": float(np.mean(scores)),
        "se": float(np.std(scores, ddof=1) / math.sqrt(scores.size)),
        "median": float(np.median(scores)),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "lengths": lengths,
        "scores": scores,
        "cutoff_count": int(cutoffs.sum()),
        "cutoff_rate": float(cutoffs.mean()),
    }


def verified_inputs() -> tuple[dict[str, list[dict[str, str]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_rows = {condition: load_csv(phase2_path(condition)) for condition in PHASE2_ORDER}
    iterative_rows = {condition: load_csv(iterative_path(condition)) for condition in ITER_ORDER}
    phase2_analysis = load_json(PHASE2_ANALYSIS / "condition_metrics.json")["conditions"]
    iterative_analysis = load_json(ITER_ANALYSIS / "condition_metrics.json")
    phase2_raw = {key: summarize(rows) for key, rows in source_rows.items()}
    iterative_raw = {key: summarize(rows) for key, rows in iterative_rows.items()}

    for key, computed in phase2_raw.items():
        reported = phase2_analysis[key]
        assert computed["n"] == int(reported["episodes"]) == 3000, f"episode count mismatch: {key}"
        for actual, field in [("mean", "mean_score"), ("se", "standard_error_score"), ("median", "median_score"), ("minimum", "minimum_score"), ("maximum", "maximum_score"), ("mean_length", "mean_episode_length"), ("median_length", "median_episode_length")]:
            assert math.isclose(computed[actual], float(reported[field]), rel_tol=0.0, abs_tol=1e-10), f"{key} {field} does not match raw CSV"
        assert computed["cutoff_count"] == int(reported["cutoff_count"]), f"cutoff mismatch: {key}"
    for key, computed in iterative_raw.items():
        reported = iterative_analysis[key]
        assert computed["n"] == int(reported["episodes"]) == 3000, f"episode count mismatch: {key}"
        for actual, field in [("mean", "mean_score"), ("se", "standard_error_score"), ("median", "median_score"), ("minimum", "minimum_score"), ("maximum", "maximum_score"), ("mean_length", "mean_episode_length"), ("median_length", "median_episode_length")]:
            assert math.isclose(computed[actual], float(reported[field]), rel_tol=0.0, abs_tol=1e-10), f"{key} {field} does not match raw CSV"
        assert computed["cutoff_count"] == int(reported["cutoff_count"]), f"cutoff mismatch: {key}"

    for condition, rows in source_rows.items():
        assert len({(r["seed_group"], r["episode_index"]) for r in rows}) == 3000, f"duplicate episode key: {condition}"
        counts = [sum(int(row.get(f"action_{a}_selected_count", "0")) for row in rows) for a in range(6)]
        assert sum(counts) == int(np.sum(phase2_raw[condition]["lengths"])), f"action-count/length mismatch: {condition}"
        phase2_raw[condition]["action_counts"] = counts
    for condition, rows in iterative_rows.items():
        assert len({(r["seed_group"], r["episode_index"]) for r in rows}) == 3000, f"duplicate episode key: {condition}"
        assert all(row["step_cutoff"].lower() == "false" and row["ball_lost"].lower() == "true" for row in rows), f"unexpected outcome: {condition}"

    paired = load_json(PHASE2_ANALYSIS / "paired_comparisons.json")
    assert paired["matched_episode_count"] == 3000
    assert len(paired["pairwise_comparisons"]) == 36
    indexed = {
        condition: {
            (int(row["seed_group"]), int(row["episode_index"])): row
            for row in condition_rows
        }
        for condition, condition_rows in source_rows.items()
    }
    for item in paired["pairwise_comparisons"]:
        a, b = item["condition_a_id"], item["condition_b_id"]
        a_rows, b_rows = indexed[a], indexed[b]
        assert a_rows.keys() == b_rows.keys(), f"pairing mismatch: {a} vs {b}"
        keys = sorted(a_rows)
        assert all(
            (a_rows[key]["base_seed"], a_rows[key]["environment_seed"], a_rows[key]["policy_seed"])
            == (b_rows[key]["base_seed"], b_rows[key]["environment_seed"], b_rows[key]["policy_seed"])
            for key in keys
        ), f"seed-triplet mismatch: {a} vs {b}"
        differences = np.asarray([float(a_rows[key]["score"]) - float(b_rows[key]["score"]) for key in keys])
        paired_mean = float(differences.mean())
        paired_sd = float(differences.std(ddof=1))
        paired_dz = paired_mean / paired_sd if paired_sd > 0 else None
        assert int(item["matched_episode_pairs"]) == len(keys) == 3000
        assert math.isclose(paired_mean, float(item["mean_paired_difference"]), rel_tol=0.0, abs_tol=1e-10), f"paired mean mismatch: {a} vs {b}"
        assert math.isclose(paired_sd, float(item["paired_difference_sample_std"]), rel_tol=0.0, abs_tol=1e-10), f"paired SD mismatch: {a} vs {b}"
        if paired_dz is not None:
            assert math.isclose(paired_dz, float(item["effect_size_cohen_dz"]), rel_tol=0.0, abs_tol=1e-10), f"paired dz mismatch: {a} vs {b}"
    return source_rows, phase2_raw, iterative_raw, {"paired": paired, "iterative_rows": iterative_rows, "phase2_rows": source_rows, "iterative_analysis": iterative_analysis}


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa7b7")
    ax.spines["bottom"].set_color("#9aa7b7")
    ax.grid(axis="x", alpha=0.75)
    ax.grid(axis="y", visible=False)


def finish(
    fig: plt.Figure,
    stem: str,
    description: str,
    sources: list[Path],
    assets: list[dict[str, Any]],
    layout_rect: tuple[float, float, float, float] = (0.0, 0.05, 1.0, 0.91),
    manual_layout: dict[str, float] | None = None,
) -> None:
    if manual_layout:
        fig.subplots_adjust(**manual_layout)
    else:
        fig.tight_layout(pad=1.3, rect=layout_rect)
    paths = [OUT / f"{stem}.svg", OUT / f"{stem}.png", OUT / f"{stem}.pdf"]
    fig.savefig(paths[0], format="svg", bbox_inches="tight", metadata={"Creator": "Weight-Space Programming figure generator", "Date": None})
    # Matplotlib writes trailing spaces on multiline SVG path commands. Strip
    # those semantically irrelevant spaces so generated vector assets pass
    # repository whitespace checks reproducibly.
    svg_text = paths[0].read_text(encoding="utf-8")
    paths[0].write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(paths[1], format="png", dpi=300, bbox_inches="tight")
    fig.savefig(paths[2], format="pdf", bbox_inches="tight", metadata={"Creator": "Weight-Space Programming figure generator", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    for path in paths:
        item: dict[str, Any] = {
            "asset_name": path.name,
            "description": description,
            "source_data": [p.relative_to(ROOT).as_posix() for p in sources],
            "generation_script": SCRIPT_PATH,
            "format": path.suffix.lstrip("."),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix == ".png":
            with path.open("rb") as png:
                header = png.read(24)
            if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError(f"PNG header is invalid: {path}")
            width, height = struct.unpack(">II", header[16:24])
            item["resolution_px"] = [width, height]
            item["dpi"] = 300
        else:
            item["resolution"] = "vector"
            item["figure_size_inches"] = [round(float(v), 3) for v in fig.get_size_inches()]
        assets.append(item)


def panel_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=12)


def chart_main(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    display = [key for key in PHASE2_ORDER if key != "astra_local_reference"]
    fig, (ax, ref) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [3.6, 1.15]})
    y = np.arange(len(display))
    for i, key in enumerate(display):
        m = raw[key]
        ax.barh(i, m["mean"], color=COLORS[key], height=0.62, alpha=0.9)
        ax.errorbar(m["mean"], i, xerr=m["se"], color="#162033", capsize=3, lw=1.15, zorder=3)
        ax.text(m["mean"] + m["se"] + 1.2, i, f"{m['mean']:.3f} ± {m['se']:.3f}   n={m['n']:,}", va="center", fontsize=8)
    ax.set_yticks(y, [f"{LABELS[k]}  ·  {CLASSES[k]}" for k in display])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Mean bricks destroyed per episode (± 1 SE)")
    panel_title(ax, "Corrected benchmark conditions", "Eight conditions shown on a shared linear scale; 3,000 episodes each")
    style_axis(ax)

    m = raw["astra_local_reference"]
    ref.barh([0], [m["mean"]], color=COLORS["astra_local_reference"], height=0.62)
    ref.errorbar(m["mean"], 0, xerr=m["se"], color="#162033", capsize=3, lw=1.15)
    ref.set_yticks([0], ["Astra\nreference"])
    ref.set_xlim(0, 590)
    ref.set_ylim(-0.65, 0.65)
    ref.set_xlabel("Mean score, separate linear scale")
    ref.text(8, 0.48, f"Mean {m['mean']:.3f} ± {m['se']:.3f}\nn={m['n']:,} · handcrafted/reference policy", va="top", fontsize=8)
    panel_title(ref, "Reference policy", "Separate panel; no broken axis")
    style_axis(ref)
    fig.suptitle("MinAtar Breakout — Phase 2 corrected local evaluation", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.01, "Primary zero-shot policies use blue, teal, purple, orange and rose; PPO is a trained baseline; remaining non-reference conditions are controls. Scores are bricks destroyed.", fontsize=8, color="#526174")
    finish(fig, "benchmark_conditions", "Mean score and standard error across all nine corrected benchmark conditions; Astra is separated on its own linear panel rather than a broken axis.", sources, assets, manual_layout={"left": 0.23, "bottom": 0.15, "right": 0.98, "top": 0.87, "wspace": 0.18})


def chart_primary_ranking(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    ranked = sorted(PRIMARY, key=lambda key: raw[key]["mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    y = np.arange(len(ranked))
    for rank, (i, key) in enumerate(zip(y, ranked), start=1):
        m = raw[key]
        ax.barh(i, m["mean"], color=COLORS[key], height=0.58)
        ax.errorbar(m["mean"], i, xerr=m["se"], color="#172033", capsize=3, lw=1.3, zorder=3)
        ax.text(m["mean"] + m["se"] + 1.1, i, f"{m['mean']:.3f} ± {m['se']:.3f}    n={m['n']:,}", va="center", fontsize=9)
        ax.text(-0.015, i, f"{rank}.", transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold", color="#39475c")
    ax.set_yticks(y, [LABELS[key] for key in ranked])
    ax.invert_yaxis()
    ax.set_xlim(0, 79)
    ax.set_xlabel("Mean bricks destroyed per episode (± 1 SE)")
    panel_title(ax, "Primary zero-shot model ranking", "Five original model-generated policies only · 3,000 episodes each · corrected local protocol")
    style_axis(ax)
    fig.text(0.12, 0.015, "Source: corrected Phase 2 episode CSVs · 3,000 episodes each. Ranking is by raw mean score; provider identity/settings were not independently authenticated.", fontsize=8, color="#526174")
    finish(fig, "primary_zero_shot_ranking", "Five original zero-shot conditions ranked by raw mean score with standard error; excludes Astra, PPO, controls, and DeepSeek Corrected.", sources, assets)


def chart_score_distributions(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.65, 1]})
    groups = [PRIMARY, ["ppo_trained_baseline", "zero_weight_control", "random_action_control", "astra_local_reference"]]
    subtitles = ["Five primary zero-shot policies", "Trained baseline, controls, and reference"]
    for ax, keys, subtitle in zip(axes, groups, subtitles):
        for key in keys:
            s = np.sort(raw[key]["scores"])
            x = np.log1p(s)
            y = np.arange(1, len(s) + 1, dtype=np.float64) / len(s)
            ax.step(x, y, where="post", color=COLORS[key], lw=1.5, label=LABELS[key], alpha=0.96)
        ax.set_xlabel("Score per episode (log1p scale; tick labels are raw scores)")
        ax.set_xlim(left=0)
        ticks = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 850]
        ax.set_xticks(np.log1p(ticks), [str(v) for v in ticks])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Cumulative fraction of episodes")
        panel_title(ax, subtitle, "")
        style_axis(ax)
    fig.suptitle("Episode-level score distributions", x=0.02, ha="left", fontsize=15, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    other_handles, other_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles + other_handles, labels + other_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035), fontsize=7.6)
    fig.text(0.02, 0.005, "Source: corrected Phase 2 raw episode CSVs · n=3,000 per condition. Each line is an ECDF; larger scores lie farther right. A labeled log1p scale keeps low scores legible while retaining Astra's full range. DeepSeek Original appears only in the primary-model panel.", fontsize=8, color="#526174")
    finish(fig, "score_distributions_ecdf", "Empirical cumulative distributions from all 27,000 corrected Phase 2 episode scores, shown in separate role-aware panels with a clearly labeled log1p score axis.", sources, assets, manual_layout={"left": 0.06, "bottom": 0.23, "right": 0.99, "top": 0.87, "wspace": 0.10})


def chart_per_seed(rows_by_condition: dict[str, list[dict[str, str]]], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(len(PRIMARY), 1, figsize=(9.6, 10.2), sharex=True)
    seed_order = [42, 100, 200]
    for ax, key in zip(axes, PRIMARY):
        rows = rows_by_condition[key]
        means, ses = [], []
        for seed in seed_order:
            vals = np.asarray([float(r["score"]) for r in rows if int(r["seed_group"]) == seed], dtype=np.float64)
            assert vals.size == 1000, f"expected 1,000 episodes per seed group for {key}, {seed}"
            means.append(float(vals.mean()))
            ses.append(float(vals.std(ddof=1) / math.sqrt(vals.size)))
        ax.errorbar(seed_order, means, yerr=ses, marker="o", markersize=5, lw=1.8, capsize=3, color=COLORS[key])
        ax.set_ylabel("Mean score\n(± 1 SE)")
        ax.text(0.012, 0.77, LABELS[key], transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.set_xticks(seed_order, ["42", "100", "200"])
        ax.set_xlim(25, 217)
        style_axis(ax)
    axes[-1].set_xlabel("Seed group (1,000 episodes per model and group)")
    fig.suptitle("Primary zero-shot performance by seed group", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.005, "Source: corrected Phase 2 raw episode CSVs. Each point uses 1,000 episodes in the recorded 42, 100, or 200 seed group; this descriptive split does not replace matched-pair analysis.", fontsize=8, color="#526174")
    finish(fig, "primary_per_seed", "Mean scores and standard errors for each of the five primary zero-shot conditions split across seed groups 42, 100, and 200; 1,000 raw episodes per point.", sources, assets)


def chart_iterative(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    x = np.arange(len(ITER_ORDER))
    for i, key in enumerate(ITER_ORDER):
        m = raw[key]
        ax.bar(i, m["mean"], color=COLORS[key], width=0.58)
        ax.errorbar(i, m["mean"], yerr=m["se"], color="#172033", capsize=4, lw=1.25)
        ax.text(i, m["mean"] + m["se"] + 1.05, f"{m['mean']:.3f} ± {m['se']:.3f}\nmedian {m['median']:.0f} · n={m['n']:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [LABELS[k] for k in ITER_ORDER])
    ax.set_ylim(0, 76)
    ax.set_ylabel("Mean bricks destroyed per episode (± 1 SE)")
    panel_title(ax, "Gemini Argon High iterative-development versions", "Separate post-generation development experiment · paired on seed group and episode index · 3,000 episodes each")
    style_axis(ax)
    fig.text(0.05, 0.015, "Source: three iterative raw episode CSVs · 3,000 episodes per version. All 9,000 ended by ball loss; zero time cutoffs. Follow-ups are interventions; nonsignificant Follow-up 2 does not establish equivalence.", fontsize=8, color="#526174")
    finish(fig, "iterative_development_comparison", "Three distinct Argon High iterative-development versions with mean, standard error, median, and 3,000 episode count; not interchangeable with zero-shot conditions.", sources, assets)


def chart_paired_statistics(stats: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    rows = [p for p in stats["pairwise_comparisons"] if p["condition_a_id"] in PRIMARY and p["condition_b_id"] in PRIMARY]
    rows.sort(key=lambda p: p["mean_paired_difference"])
    fig, ax = plt.subplots(figsize=(15.2, 7.0))
    for i, item in enumerate(rows):
        d = float(item["mean_paired_difference"])
        lo, hi = map(float, item["paired_difference_ci_95_percentile_bootstrap"])
        color = COLORS[item["condition_a_id"]]
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ecolor=color, capsize=3, lw=1.35, markersize=5)
    ax.axvline(0, color="#182338", lw=1.0, linestyle="--")
    labels = [f"{p['condition_a_label']} − {p['condition_b_label']}" for p in rows]
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired mean score difference (A − B), bricks (95% percentile paired-bootstrap CI)")
    ax.set_xlim(-12, 68)
    panel_title(ax, "Matched comparisons among five primary zero-shot conditions", "10 selected rows from the 36-comparison Holm family · 3,000 matched episode pairs per comparison")
    style_axis(ax)
    for i, p in enumerate(rows):
        dz = float(p["effect_size_cohen_dz"])
        p_raw = float(p["paired_sign_flip_p_value"])
        p_holm = float(p["p_value_holm_adjusted_all_pairwise_conditions"])
        ax.text(1.035, i, f"dz {dz:+.3f}   p {p_raw:.3g}   Holm {p_holm:.3g}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.7, color="#334155", clip_on=False)
    fig.text(0.02, 0.008, "Source: corrected Phase 2 paired-analysis JSON · 3,000 matched pairs per row. P-values are two-sided; Holm adjustment uses the full 36-pair family. Nonsignificance is not evidence of equivalence.", fontsize=8, color="#526174")
    fig.subplots_adjust(left=0.34, bottom=0.12, top=0.88, right=0.68)
    finish(fig, "primary_paired_comparisons", "Paired-bootstrap 95% intervals for the ten primary zero-shot pairwise mean differences, with exact paired Cohen's dz, two-sided sign-flip p-values, and Holm p-values from the full 36-comparison family.", sources, assets, manual_layout={"left": 0.39, "bottom": 0.13, "right": 0.72, "top": 0.88})


def chart_actions(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    y = np.arange(len(PRIMARY))
    bottom = np.zeros(len(PRIMARY))
    cmap = ["#9aa5b1", "#2775c7", "#b05e85", "#138a78", "#cb8c32", "#7b6fb0"]
    for action, (name, color) in enumerate(zip(ACTION_NAMES, cmap)):
        vals = np.asarray([raw[k]["action_counts"][action] for k in PRIMARY], dtype=np.float64)
        totals = np.asarray([sum(raw[k]["action_counts"]) for k in PRIMARY], dtype=np.float64)
        frac = vals / totals
        ax.barh(y, frac, left=bottom, color=color, height=0.63, label=f"{action} {name}")
        for i, value in enumerate(frac):
            if value >= 0.065:
                ax.text(bottom[i] + value / 2, i, f"{value*100:.1f}%", ha="center", va="center", color="white" if action in (1, 3, 5) else "#1d2939", fontsize=7.5, fontweight="bold")
        bottom += frac
    ax.set_yticks(y, [LABELS[k] for k in PRIMARY])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Fraction of policy-selected actions across recorded episode steps")
    handles, labels = ax.get_legend_handles_labels()
    panel_title(ax, "Selected action frequencies", "Aggregated from raw per-episode action counts; 3,000 episodes per primary condition")
    style_axis(ax)
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.075), fontsize=8)
    fig.text(0.02, 0.012, "Source: corrected Phase 2 raw episode CSVs · 3,000 episodes per primary condition. IDs: 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire. Only left/right move the paddle; sticky handling can change execution.", fontsize=8, color="#526174")
    finish(fig, "primary_action_distributions", "Normalized action-selection frequencies computed by summing actual action counts from raw episode CSVs for the five primary zero-shot policies; does not conflate selected with executed actions.", sources, assets, layout_rect=(0.19, 0.19, 1.0, 0.87))


def chart_episode_lengths(raw: dict[str, Any], sources: list[Path], assets: list[dict[str, Any]]) -> None:
    keys = [k for k in PHASE2_ORDER if k != "astra_local_reference"]
    fig, (ax, cut, summary) = plt.subplots(1, 3, figsize=(16, 5.7), gridspec_kw={"width_ratios": [2.0, 1.25, 1.55]})
    for key in keys:
        values = np.sort(raw[key]["lengths"])
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(np.log1p(values), y, where="post", color=COLORS[key], lw=1.35, label=LABELS[key])
    ax.set_xticks(np.log1p([0, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000]), ["0", "5", "10", "20", "50", "100", "500", "1k", "5k", "10k"])
    ax.set_xlabel("Episode length, steps (log1p scale)")
    ax.set_ylabel("Cumulative fraction of episodes")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.2)
    panel_title(ax, "Episode length distributions", "Eight non-reference conditions")
    style_axis(ax)
    rates = [raw[k]["cutoff_rate"] * 100 for k in PHASE2_ORDER]
    y = np.arange(len(PHASE2_ORDER))
    for i, key in enumerate(PHASE2_ORDER):
        cut.barh(i, rates[i], color=COLORS[key], height=0.58)
        cut.text(rates[i] + 0.8, i, f"{raw[key]['cutoff_count']:,} / 3,000 ({rates[i]:.2f}%)", va="center", fontsize=7.3)
    cut.set_yticks(y, [LABELS[k] for k in PHASE2_ORDER], fontsize=7.3)
    cut.invert_yaxis()
    cut.set_xlim(0, 72)
    cut.set_xlabel("10,000-step cutoff rate")
    panel_title(cut, "Time-limit cutoffs", "Episode reaches 10,000 steps")
    style_axis(cut)
    all_y = np.arange(len(PHASE2_ORDER))
    for i, key in enumerate(PHASE2_ORDER):
        mean_x = np.log1p(raw[key]["mean_length"])
        median_x = np.log1p(raw[key]["median_length"])
        summary.scatter(mean_x, i - 0.13, s=25, color=COLORS[key], marker="o", zorder=3)
        summary.scatter(median_x, i + 0.13, s=25, color=COLORS[key], marker="s", facecolors="white", linewidths=1.3, zorder=3)
    summary.set_yticks(all_y, [""] * len(all_y))
    summary.invert_yaxis()
    summary.set_ylim(len(all_y) - 0.5, -0.5)
    summary.set_xlim(0, np.log1p(15000))
    length_ticks = [0, 10, 50, 100, 500, 1000, 5000, 10000]
    summary.set_xticks(np.log1p(length_ticks), ["0", "10", "50", "100", "500", "1k", "5k", "10k"])
    summary.set_xlabel("Episode length, steps (log1p scale)")
    panel_title(summary, "Mean and median lengths", "Circle = mean · square = median")
    style_axis(summary)
    fig.suptitle("Episode duration and time-limit outcomes", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.01, "Source: nine corrected Phase 2 raw episode CSVs · 3,000 episodes per condition. ECDF omits Astra because it is longer; cutoff and mean/median panels include all conditions. Astra had 1,484 cutoffs; others had zero. Filled circles = mean; open squares = median.", fontsize=8, color="#526174")
    finish(fig, "episode_lengths_and_cutoffs", "Episode-length ECDFs, mean/median lengths, and 10,000-step cutoff counts/rates computed from all nine corrected raw episode CSVs; the ECDF panel omits Astra's much longer durations, while the other panels include all conditions.", sources, assets, manual_layout={"left": 0.07, "bottom": 0.16, "right": 0.99, "top": 0.87, "wspace": 0.14})


def chart_trace_diagnostics(sources: list[Path], assets: list[dict[str, Any]]) -> None:
    trace = load_json(TRACE_FILE)
    fig, axes = plt.subplots(len(PRIMARY), 1, figsize=(11, 12.8), sharex=False)
    action_colors = ["#8a94a3", "#2c75c4", "#b05e85", "#138a78", "#cb8c32", "#7664a5"]
    for ax, key in zip(axes, PRIMARY):
        detail = trace["conditions"][key]
        steps = detail["trajectory"]
        t = np.asarray([int(row["step"]) for row in steps])
        ball_x = np.asarray([int(row["after"]["ball_x"]) for row in steps])
        ball_y = np.asarray([int(row["after"]["ball_y"]) for row in steps])
        paddle = np.asarray([int(row["after"]["paddle_x"]) for row in steps])
        bricks = np.asarray([int(row["after"]["bricks_remaining"]) for row in steps])
        ax.plot(t, ball_x, color="#236bb6", lw=0.85, alpha=0.83)
        ax.plot(t, ball_y, color="#11a087", lw=0.85, alpha=0.83)
        ax.plot(t, paddle, color="#182235", lw=0.9, alpha=0.8)
        selected_action = np.asarray([int(row["selected_action"]) for row in steps])
        executed_action = np.asarray([int(row["executed_action"]) for row in steps])
        for aid, color in enumerate(action_colors):
            selected_mask = selected_action == aid
            executed_mask = executed_action == aid
            if selected_mask.any():
                ax.scatter(t[selected_mask], np.full(int(selected_mask.sum()), -0.25), s=7, color=color, marker="s", linewidths=0)
            if executed_mask.any():
                ax.scatter(t[executed_mask], np.full(int(executed_mask.sum()), -0.75), s=7, color=color, marker="^", linewidths=0)
        bricks_axis = ax.twinx()
        bricks_axis.plot(t, bricks, color="#775b9c", lw=0.8, ls="--", alpha=0.75)
        bricks_axis.set_ylim(0, 30)
        bricks_axis.set_yticks([0, 15, 30])
        bricks_axis.tick_params(axis="y", labelsize=6.5, colors="#775b9c", length=2)
        bricks_axis.spines["right"].set_color("#b8a9ca")
        bricks_axis.grid(False)
        ax.set_ylim(-1.05, 10.1)
        ax.set_yticks([0, 2, 4, 6, 8, 9])
        ax.set_ylabel("Board col/row")
        ax.set_title(f"{LABELS[key]} · seed group {detail['seed_group']} · diagnostic score {detail['historical_csv_score']} · {len(steps)} recorded steps", loc="left", fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlim(0, max(1, len(steps)))
        style_axis(ax)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("Step in one preserved historical diagnostic trace")
    handles = [Line2D([0], [0], color="#236bb6", lw=1.5, label="Ball column"), Line2D([0], [0], color="#11a087", lw=1.5, label="Ball row"), Line2D([0], [0], color="#182235", lw=1.5, label="Paddle column"), Line2D([0], [0], color="#775b9c", lw=1.2, ls="--", label="Bricks remaining (right axis)")]
    handles.extend(Patch(facecolor=c, label=f"Action {i}: {ACTION_NAMES[i]}") for i, c in enumerate(action_colors))
    handles.extend([Line2D([0], [0], color="#516072", marker="s", lw=0, label="Selected action"), Line2D([0], [0], color="#516072", marker="^", lw=0, label="Executed action")])
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.065), fontsize=7.5)
    fig.suptitle("Observed state/action timelines from preserved diagnostic replays", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.008, "Source: preserved historical diagnostic trace JSON · one episode per primary model, not a representative population trace. Lines show ball x/y, paddle x, and bricks remaining; action color gives ID and marker shape distinguishes selected from executed.", fontsize=7.7, color="#526174")
    finish(fig, "diagnostic_trajectory_traces", "Step-by-step state positions and selected/executed actions and bricks remaining from five preserved single-episode historical diagnostic traces; not representative aggregate behavior.", sources, assets, manual_layout={"left": 0.10, "bottom": 0.22, "top": 0.94, "right": 0.98, "hspace": 0.38})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, raw, iterative_raw, extra = verified_inputs()
    phase_sources = [phase2_path(k) for k in PHASE2_ORDER]
    iterative_sources = [iterative_path(k) for k in ITER_ORDER]
    stats_path = PHASE2_ANALYSIS / "paired_comparisons.json"
    trace_path = TRACE_FILE
    assets: list[dict[str, Any]] = []
    chart_main(raw, phase_sources, assets)
    chart_primary_ranking(raw, phase_sources, assets)
    chart_score_distributions(raw, phase_sources, assets)
    chart_per_seed(rows, phase_sources, assets)
    chart_iterative(iterative_raw, iterative_sources, assets)
    chart_paired_statistics(extra["paired"], phase_sources + [stats_path], assets)
    chart_actions(raw, phase_sources, assets)
    chart_episode_lengths(raw, phase_sources, assets)
    chart_trace_diagnostics([trace_path] + phase_sources, assets)

    manifest = {
        "manifest_version": 1,
        "generated_from": "Preserved raw episode CSVs and checked analysis/trace records; no benchmark data were written.",
        "protocols": {
            "phase2": "Corrected local evaluation; nine conditions × 3,000 episodes; 27,000 total.",
            "iterative_development": "Separate Argon High iterative experiment; three versions × 3,000 episodes; 9,000 total.",
            "diagnostic_traces": "Single preserved historical diagnostic replay per primary model; not representative aggregate traces.",
        },
        "sources": [
            {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)}
            for p in sorted(set(phase_sources + iterative_sources + [stats_path, PHASE2_ANALYSIS / "condition_metrics.json", ITER_ANALYSIS / "condition_metrics.json", trace_path]))
        ],
        "assets": assets,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified raw CSV summaries: {len(raw)} Phase 2 conditions, {len(iterative_raw)} iterative versions; 36 statistical pairs.")
    print(f"Generated {len(assets)} figure exports in {OUT.relative_to(ROOT)}.")
    print(f"Media manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
