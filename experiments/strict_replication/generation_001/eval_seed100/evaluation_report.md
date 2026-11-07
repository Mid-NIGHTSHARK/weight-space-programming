# MinAtar Breakout Evaluation: gemini_argon_seed100

**Timestamp:** 2026-11-06T13:35:27.253796
**Agent / Model Identifier:** `gemini_argon_seed100`
**Weight Hash (SHA-256):** `455722b09750deb693093c823c13dd37b99eb470c957ce472262b1419014abb5`
**Environment:** MinAtar Breakout (`minatar==1.0.15`)
**Evaluation Protocol:** 1000 episodes, sticky actions 0.1, max 10,000 steps

## Executive Summary

| Metric | Observed Result | Paper Astra Hand Reference | Paper PPO (500M) Reference |
|---|---|---|---|
| **Mean Score** | **58.898** | 517.411 | 25.344 |
| **Standard Error** | ±2.009 | ±7.417 | ±0.121 |
| **Median Score** | 38.0 | N/A | N/A |
| **Score Range [Min, Max]** | [0.0, 463.0] | [N/A, 800.0] | [N/A, 29.0] |
| **Mean Episode Length** | 882.8 steps | 7,417.2 steps | 281.9 steps |
| **Total Bricks Destroyed** | 58,898 | N/A | N/A |
| **Step Cutoffs (>=10,000)** | 0 (0.0%) | N/A | N/A |

## Action Selection Distribution

| Action Index | Action Name | Proportion |
|---|---|---|
| 0 | No-op | 10.27% |
| 1 | Left | 44.85% |
| 2 | Up | 0.00% |
| 3 | Right | 44.88% |
| 4 | Down | 0.00% |
| 5 | Fire | 0.00% |

## Scientific Verdict

> [!NOTE]
> **Moderate Policy:** The evaluated network plays above random/zero baselines and achieves performance comparable to multi-million step PPO training (mean score: 58.90).
