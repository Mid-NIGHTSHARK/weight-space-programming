# MinAtar Breakout Evaluation: astra_ref_seed100

**Timestamp:** 2026-11-06T10:34:36.792657
**Agent / Model Identifier:** `astra_ref_seed100`
**Weight Hash (SHA-256):** `6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **500.422** | 517.411 | 25.344 |
| **Standard Error** | ±7.882 | ±7.417 | ±0.121 |
| **Median Score** | 644.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 814.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 7127.5 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 500,422 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 506 (50.6%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 50.65% |
| 1 | Left | 24.67% |
| 2 | Up | 0.00% |
| 3 | Right | 24.68% |
| 4 | Down | 0.00% |
| 5 | Fire | 0.00% |

## Scientific Verdict

> [!NOTE]
> **High-Competence Policy:** The evaluated network exhibits strategic, highly competent play (mean score: 500.42), far exceeding conventional reinforcement learning baselines (PPO 500M: 25.344) and demonstrating non-trivial weight-space programming capability.
