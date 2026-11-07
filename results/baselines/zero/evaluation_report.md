# MinAtar Breakout Evaluation: zero_weight_baseline

**Timestamp:** 2026-11-06T10:20:56.197311
**Agent / Model Identifier:** `zero_weight_baseline`
**Weight Hash (SHA-256):** `N/A`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **0.517** | 517.411 | 25.344 |
| **Standard Error** | ±0.023 | ±7.417 | ±0.121 |
| **Median Score** | 0.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 4.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 11.3 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 517 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 16.99% |
| 1 | Left | 16.94% |
| 2 | Up | 17.19% |
| 3 | Right | 15.83% |
| 4 | Down | 16.47% |
| 5 | Fire | 16.58% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 0.52).
