# MinAtar Breakout Evaluation: random_baseline

**Timestamp:** 2026-11-06T10:20:55.779508
**Agent / Model Identifier:** `random_baseline`
**Weight Hash (SHA-256):** `N/A`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **0.485** | 517.411 | 25.344 |
| **Standard Error** | ±0.023 | ±7.417 | ±0.121 |
| **Median Score** | 0.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 4.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 11.0 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 485 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 16.33% |
| 1 | Left | 16.87% |
| 2 | Up | 16.71% |
| 3 | Right | 16.72% |
| 4 | Down | 17.18% |
| 5 | Fire | 16.19% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 0.48).
