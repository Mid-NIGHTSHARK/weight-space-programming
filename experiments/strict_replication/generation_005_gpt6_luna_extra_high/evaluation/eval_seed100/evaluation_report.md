# MinAtar Breakout Evaluation: gpt6_luna_extra_high

**Timestamp:** 2026-11-06T18:44:23.509997
**Agent / Model Identifier:** `gpt6_luna_extra_high`
**Weight Hash (SHA-256):** `1f275bbbecaf05bc16cc3d6c791768235ccc65c3c91285fe442836c2c27adce9`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **1.323** | 517.411 | 25.344 |
| **Standard Error** | ±0.042 | ±7.417 | ±0.121 |
| **Median Score** | 1.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 10.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 19.7 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 1,323 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 25.12% |
| 1 | Left | 36.16% |
| 2 | Up | 0.01% |
| 3 | Right | 38.70% |
| 4 | Down | 0.01% |
| 5 | Fire | 0.01% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 1.32).
