# MinAtar Breakout Evaluation: gpt6_luna_extra_high

**Timestamp:** 2026-11-06T18:44:24.127795
**Agent / Model Identifier:** `gpt6_luna_extra_high`
**Weight Hash (SHA-256):** `1f275bbbecaf05bc16cc3d6c791768235ccc65c3c91285fe442836c2c27adce9`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **1.223** | 517.411 | 25.344 |
| **Standard Error** | ±0.037 | ±7.417 | ±0.121 |
| **Median Score** | 1.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 8.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 18.6 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 1,223 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 25.73% |
| 1 | Left | 35.81% |
| 2 | Up | 0.02% |
| 3 | Right | 38.42% |
| 4 | Down | 0.01% |
| 5 | Fire | 0.01% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 1.22).
