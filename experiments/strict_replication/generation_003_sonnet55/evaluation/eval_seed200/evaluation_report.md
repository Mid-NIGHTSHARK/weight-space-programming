# MinAtar Breakout Evaluation: sonnet55_seed200

**Timestamp:** 2026-11-06T16:24:19.739338
**Agent / Model Identifier:** `sonnet55_seed200`
**Weight Hash (SHA-256):** `2fb65f893087fa93eff087a3d8af2605c5c834b069c83bc8c8e08bf2fb60993b`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **11.105** | 517.411 | 25.344 |
| **Standard Error** | ±0.398 | ±7.417 | ±0.121 |
| **Median Score** | 7.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 111.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 130.7 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 11,105 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 8.91% |
| 1 | Left | 45.29% |
| 2 | Up | 0.00% |
| 3 | Right | 45.80% |
| 4 | Down | 0.00% |
| 5 | Fire | 0.00% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 11.11).
