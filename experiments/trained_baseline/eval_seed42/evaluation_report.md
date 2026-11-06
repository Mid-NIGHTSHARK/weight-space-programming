# MinAtar Breakout Evaluation: ppo_trained_seed42

**Timestamp:** 2026-11-06T13:43:50.792587
**Agent / Model Identifier:** `ppo_trained_seed42`
**Weight Hash (SHA-256):** `9ca0c46d59cf41179b3db3cfe7b31bd48cd57ce54785e2b407b524c82450ea86`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **5.718** | 517.411 | 25.344 |
| **Standard Error** | ±0.035 | ±7.417 | ±0.121 |
| **Median Score** | 6.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 7.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 66.9 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 5,718 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 17.57% |
| 1 | Left | 10.85% |
| 2 | Up | 17.35% |
| 3 | Right | 11.66% |
| 4 | Down | 19.96% |
| 5 | Fire | 22.62% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 5.72).
