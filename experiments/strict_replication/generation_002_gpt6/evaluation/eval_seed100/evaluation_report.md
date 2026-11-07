# MinAtar Breakout Evaluation: gpt6_seed100

**Timestamp:** 2026-11-06T16:09:20.962534
**Agent / Model Identifier:** `gpt6_seed100`
**Weight Hash (SHA-256):** `f15aa7d37cfefb76e25b7b8576751d4f155b960baf7eddc701c8f79af9bd3c3b`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **2.565** | 517.411 | 25.344 |
| **Standard Error** | ±0.062 | ±7.417 | ±0.121 |
| **Median Score** | 2.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 9.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 33.3 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 2,565 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 59.88% |
| 1 | Left | 19.45% |
| 2 | Up | 0.01% |
| 3 | Right | 20.66% |
| 4 | Down | 0.00% |
| 5 | Fire | 0.00% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 2.56).
