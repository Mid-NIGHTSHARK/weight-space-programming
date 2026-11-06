"""Generate statistical comparison and distribution plots between Gemini Argon, Trained PPO, and Astra."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "experiments/strict_replication/generation_001/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
argon_files = [
    "experiments/strict_replication/generation_001/eval_seed42/episode_scores.csv",
    "experiments/strict_replication/generation_001/eval_seed100/episode_scores.csv",
    "experiments/strict_replication/generation_001/eval_seed200/episode_scores.csv",
]

trained_files = [
    "experiments/trained_baseline/eval_seed42/episode_scores.csv",
    "experiments/trained_baseline/eval_seed100/episode_scores.csv",
    "experiments/trained_baseline/eval_seed200/episode_scores.csv",
]

astra_files = [
    "experiments/reference_eval_runs/run_seed42/episode_scores.csv",
    "experiments/reference_eval_runs/run_seed100/episode_scores.csv",
    "experiments/reference_eval_runs/run_seed200/episode_scores.csv",
    "experiments/reference_eval_runs/run_seed300/episode_scores.csv",
]


def load_dataset(files: list, name: str) -> pd.DataFrame:
    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["model"] = name
    return combined


df_argon = load_dataset(argon_files, "Gemini Argon (Zero-Shot)")
df_trained = load_dataset(trained_files, "PPO 10M Steps (Trained)")
df_astra = load_dataset(astra_files, "GPT-6 Astra (Zero-Shot)")


def compute_metrics(df: pd.DataFrame) -> dict:
    scores = df["score"].values
    lengths = df["length"].values
    n = len(scores)
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores, ddof=1))
    se_s = float(std_s / np.sqrt(n))
    med_s = float(np.median(scores))
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))

    mean_l = float(np.mean(lengths))
    med_l = float(np.median(lengths))
    cutoffs = int(np.sum(lengths >= 10000))
    cutoff_pct = float(cutoffs / n * 100)

    # 95% Confidence Interval for mean score
    ci_low = float(mean_s - 1.96 * se_s)
    ci_high = float(mean_s + 1.96 * se_s)

    # Action counts
    act_cols = [c for c in df.columns if c.startswith("action_") or c in ["0", "1", "2", "3", "4", "5"]]
    act_dist = {}
    if act_cols:
        total_acts = df[act_cols].values.sum()
        for c in act_cols:
            act_dist[c] = float(df[c].values.sum() / total_acts)

    return {
        "episodes": n,
        "mean_score": mean_s,
        "std_score": std_s,
        "se_score": se_s,
        "ci_95": [ci_low, ci_high],
        "median_score": med_s,
        "min_score": min_s,
        "max_score": max_s,
        "mean_length": mean_l,
        "median_length": med_l,
        "cutoff_count": cutoffs,
        "cutoff_pct": cutoff_pct,
        "action_distribution": act_dist,
    }


metrics_argon = compute_metrics(df_argon)
metrics_trained = compute_metrics(df_trained)
metrics_astra = compute_metrics(df_astra)

# Statistical Significance Tests
t_stat, p_val_ttest = st.ttest_ind(df_argon["score"], df_trained["score"], equal_var=False)
u_stat, p_val_mwu = st.mannwhitneyu(df_argon["score"], df_trained["score"], alternative="two-sided")

# Cohen's d effect size
n1, n2 = len(df_argon), len(df_trained)
s1, s2 = metrics_argon["std_score"], metrics_trained["std_score"]
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
cohens_d = (metrics_argon["mean_score"] - metrics_trained["mean_score"]) / pooled_std

stat_comparison = {
    "argon": metrics_argon,
    "trained": metrics_trained,
    "astra_reference": metrics_astra,
    "comparison": {
        "score_diff_argon_minus_trained": metrics_argon["mean_score"] - metrics_trained["mean_score"],
        "cohens_d_effect_size": float(cohens_d),
        "welch_t_test": {"t_statistic": float(t_stat), "p_value": float(p_val_ttest)},
        "mann_whitney_u_test": {"u_statistic": float(u_stat), "p_value": float(p_val_mwu)},
    },
}

with open(os.path.join(OUTPUT_DIR, "statistical_comparison.json"), "w") as f:
    json.dump(stat_comparison, f, indent=2)

print("Saved statistical comparison to", os.path.join(OUTPUT_DIR, "statistical_comparison.json"))

# --- PLOT 1: Score Distribution Comparison ---
plt.figure(figsize=(12, 6))

# Panel 1: Histogram / KDE comparison
plt.subplot(1, 2, 1)
bins = np.linspace(0, 500, 50)
plt.hist(
    df_argon["score"],
    bins=bins,
    alpha=0.6,
    color="#00bcd4",
    label=f"Gemini Argon (Mean: {metrics_argon['mean_score']:.1f})",
    density=True,
)
plt.hist(
    df_trained["score"],
    bins=bins,
    alpha=0.6,
    color="#4caf50",
    label=f"PPO Trained (Mean: {metrics_trained['mean_score']:.1f})",
    density=True,
)
plt.axvline(metrics_argon["mean_score"], color="#00838f", linestyle="--", linewidth=2)
plt.axvline(metrics_trained["mean_score"], color="#2e7d32", linestyle="--", linewidth=2)
plt.xlabel("Episode Score (Bricks Destroyed)", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.title("Score Density Distribution", fontsize=12, fontweight="bold")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

# Panel 2: Boxplot Comparison (including Astra)
plt.subplot(1, 2, 2)
plot_data = [df_trained["score"], df_argon["score"], df_astra["score"]]
labels = ["PPO (10M)", "Gemini Argon", "GPT-6 Astra"]
colors = ["#4caf50", "#00bcd4", "#9c27b0"]
box = plt.boxplot(plot_data, tick_labels=labels, patch_artist=True, showmeans=True)
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel("Episode Score", fontsize=11)
plt.title("Policy Performance Comparison", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "score_distributions_comparison.png")
plt.savefig(plot1_path, dpi=180)
plt.close()
print("Saved score distribution plot to", plot1_path)

# --- PLOT 2: Episode Length Survival Curves ---
plt.figure(figsize=(9, 5))
for df, label, color in [
    (df_trained, "PPO Trained Baseline", "#4caf50"),
    (df_argon, "Gemini Argon (Zero-Shot)", "#00bcd4"),
    (df_astra, "GPT-6 Astra Reference", "#9c27b0"),
]:
    sorted_len = np.sort(df["length"].values)
    survival_prob = 1.0 - np.arange(len(sorted_len)) / len(sorted_len)
    plt.plot(sorted_len, survival_prob, label=label, color=color, linewidth=2)

plt.xlabel("Episode Length (Steps)", fontsize=11)
plt.ylabel("Fraction of Episodes Surviving", fontsize=11)
plt.title("Episode Survival Curves (10,000 Step Cutoff)", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(loc="center right")
plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, "cumulative_survival_curves.png")
plt.savefig(plot2_path, dpi=180)
plt.close()
print("Saved survival curve plot to", plot2_path)
