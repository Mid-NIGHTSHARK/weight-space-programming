# MinAtar Breakout Evaluation: deepseek_corrected

**Timestamp:** 2026-11-06T17:49:35.782732
**Agent / Model Identifier:** `deepseek_corrected`
**Weight Hash (SHA-256):** `21078c485156850253744ed0c2e1efd54361cf06f603a5ad2fd32ffea3c1c8bc`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **6.364** | 517.411 | 25.344 |
| **Standard Error** | ±0.269 | ±7.417 | ±0.121 |
| **Median Score** | 3.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 88.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 90.3 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 6,364 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 5.18% |
| 1 | Left | 47.22% |
| 2 | Up | 0.00% |
| 3 | Right | 47.60% |
| 4 | Down | 0.00% |
| 5 | Fire | 0.00% |

## Scientific Verdict

> [!WARNING]
> **Low Competence / Baseline Policy:** The evaluated network achieves scores consistent with untrained or random behavior (mean score: 6.36).
