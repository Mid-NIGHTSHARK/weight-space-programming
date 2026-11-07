# MinAtar Breakout Evaluation: deepseek_zero_shot

**Timestamp:** 2026-11-06T17:23:27.010257
**Agent / Model Identifier:** `deepseek_zero_shot`
**Weight Hash (SHA-256):** `5ea84b9b5d04292aef19f5b4e12bf8c553145714017753b9a843625f19d969d9`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **0.473** | 517.411 | 25.344 |
| **Standard Error** | ±0.016 | ±7.417 | ±0.121 |
| **Median Score** | 0.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 1.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 10.7 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 473 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 0.00% |
| 1 | Left | 0.00% |
| 2 | Up | 33.62% |
| 3 | Right | 0.00% |
| 4 | Down | 33.15% |
| 5 | Fire | 33.23% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 0.47).
