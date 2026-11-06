"""Statistical analysis comparing DeepSeek, Claude Sonnet 5.5, GPT-6, Gemini Argon, Astra, and PPO baseline."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

models = {}

# 1. DeepSeek (Zero-Shot)
deepseek_scores, deepseek_lengths = [], []
deepseek_cutoffs = 0
for s in [42, 100, 200]:
    f = f"experiments/strict_replication/generation_004_deepseek/evaluation/eval_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        deepseek_scores.extend(df["score"].values)
        deepseek_lengths.extend(df["length"].values)
        deepseek_cutoffs += int((df["step_cutoff"] == True).sum())
models["DeepSeek (Zero-Shot)"] = {
    "scores": np.array(deepseek_scores),
    "lengths": np.array(deepseek_lengths),
    "cutoffs": deepseek_cutoffs,
    "color": "#2979ff", # Deep Electric Blue
}

# 2. Claude Sonnet 5.5
sonnet_scores, sonnet_lengths = [], []
sonnet_cutoffs = 0
for s in [42, 100, 200]:
    f = f"experiments/strict_replication/generation_003_sonnet55/evaluation/eval_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        sonnet_scores.extend(df["score"].values)
        sonnet_lengths.extend(df["length"].values)
        sonnet_cutoffs += int((df["step_cutoff"] == True).sum())
models["Claude Sonnet 5.5 (Zero-Shot)"] = {
    "scores": np.array(sonnet_scores),
    "lengths": np.array(sonnet_lengths),
    "cutoffs": sonnet_cutoffs,
    "color": "#ff7043",
}

# 3. GPT-6
gpt6_scores, gpt6_lengths = [], []
gpt6_cutoffs = 0
for s in [42, 100, 200]:
    f = f"experiments/strict_replication/generation_002_gpt6/evaluation/eval_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        gpt6_scores.extend(df["score"].values)
        gpt6_lengths.extend(df["length"].values)
        gpt6_cutoffs += int((df["step_cutoff"] == True).sum())
models["GPT-6 (Zero-Shot)"] = {
    "scores": np.array(gpt6_scores),
    "lengths": np.array(gpt6_lengths),
    "cutoffs": gpt6_cutoffs,
    "color": "#ab47bc",
}

# 4. Gemini Argon 4M
argon_scores, argon_lengths = [], []
argon_cutoffs = 0
for s in [42, 100, 200]:
    f = f"experiments/strict_replication/generation_001/eval_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        argon_scores.extend(df["score"].values)
        argon_lengths.extend(df["length"].values)
        argon_cutoffs += int((df["step_cutoff"] == True).sum())
models["Gemini Argon 4M (Zero-Shot)"] = {
    "scores": np.array(argon_scores),
    "lengths": np.array(argon_lengths),
    "cutoffs": argon_cutoffs,
    "color": "#00e5ff",
}

# 5. Astra Reference
astra_scores, astra_lengths = [], []
astra_cutoffs = 0
for s in [42, 100, 200, 300]:
    f = f"experiments/reference_eval_runs/run_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        astra_scores.extend(df["score"].values)
        astra_lengths.extend(df["length"].values)
        astra_cutoffs += int((df["step_cutoff"] == True).sum())
models["Astra (Reference)"] = {
    "scores": np.array(astra_scores),
    "lengths": np.array(astra_lengths),
    "cutoffs": astra_cutoffs,
    "color": "#ffd700",
}

# 6. Trained PPO
ppo_scores, ppo_lengths = [], []
ppo_cutoffs = 0
for s in [42, 100, 200]:
    f = f"experiments/trained_baseline/eval_seed{s}/episode_scores.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        ppo_scores.extend(df["score"].values)
        ppo_lengths.extend(df["length"].values)
        ppo_cutoffs += int((df["step_cutoff"] == True).sum())
models["PPO Baseline (10M)"] = {
    "scores": np.array(ppo_scores),
    "lengths": np.array(ppo_lengths),
    "cutoffs": ppo_cutoffs,
    "color": "#00e676",
}

# 7. Random Baseline
random_scores = pd.read_csv("results/baselines/random/episode_scores.csv")["score"].values
models["Random Baseline"] = {
    "scores": random_scores,
    "lengths": pd.read_csv("results/baselines/random/episode_scores.csv")["length"].values,
    "cutoffs": 0,
    "color": "#78909c",
}

summary_table = []
for name, data in models.items():
    sc = data["scores"]
    lg = data["lengths"]
    cut = data["cutoffs"]
    if len(sc) > 0:
        mean_s = float(np.mean(sc))
        se_s = float(np.std(sc, ddof=1) / np.sqrt(len(sc)))
        std_s = float(np.std(sc, ddof=1))
        med_s = float(np.median(sc))
        min_s = float(np.min(sc))
        max_s = float(np.max(sc))
        mean_l = float(np.mean(lg))
        med_l = float(np.median(lg))
        cutoff_rate = float(cut / len(sc) * 100.0)
        summary_table.append({
            "model": name,
            "episodes": len(sc),
            "mean_score": mean_s,
            "se_score": se_s,
            "std_score": std_s,
            "median_score": med_s,
            "min_score": min_s,
            "max_score": max_s,
            "mean_length": mean_l,
            "median_length": med_l,
            "cutoffs": cut,
            "cutoff_rate_pct": cutoff_rate,
        })

print(f"{'Model':<30} | {'N':<5} | {'Mean':<8} | {'SE':<6} | {'Median':<6} | {'Max':<6} | {'Mean Len':<8} | {'Cutoff%'}")
print("-" * 92)
for row in summary_table:
    print(f"{row['model']:<30} | {row['episodes']:<5} | {row['mean_score']:<8.3f} | {row['se_score']:<6.3f} | {row['median_score']:<6.1f} | {row['max_score']:<6.1f} | {row['mean_length']:<8.1f} | {row['cutoff_rate_pct']:<6.2f}%")

# Save summary to JSON
out_dir = "experiments/strict_replication/generation_004_deepseek/analysis"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "comparative_metrics.json"), "w") as f:
    json.dump(summary_table, f, indent=2)

# Generate comparative figures
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

labels = [r["model"].replace(" (Zero-Shot)", "").replace(" (Reference)", " (Ref)") for r in summary_table]
means = [r["mean_score"] for r in summary_table]
ses = [r["se_score"] for r in summary_table]
colors = [models[m]["color"] for m in models]

axes[0].bar(labels, means, yerr=ses, capsize=4, color=colors, alpha=0.85, edgecolor="black")
axes[0].set_ylabel("Mean Score (Bricks Destroyed)", fontsize=11)
axes[0].set_title("Mean Breakout Score Comparison (with SE)", fontsize=12, fontweight="bold")
axes[0].set_yscale("symlog", linthresh=1.0)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(means):
    axes[0].text(i, v * 1.15 if v > 1 else v + 0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

score_lists = [models[m]["scores"] for m in models]
axes[1].boxplot(score_lists, tick_labels=labels, showmeans=True)
axes[1].set_yscale("symlog", linthresh=1.0)
axes[1].set_ylabel("Score (symlog scale)", fontsize=11)
axes[1].set_title("Score Distribution Across Models (Symlog Boxplot)", fontsize=12, fontweight="bold")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plot_path = os.path.join(out_dir, "deepseek_comparative_score_distributions.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"Plot saved to {plot_path}")
