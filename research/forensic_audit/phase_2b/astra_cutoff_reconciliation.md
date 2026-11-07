# Forensic Reconciliation: Astra 10,000-Step Cutoff Rate Discrepancy

**Audit Phase:** Phase 2B Supplementary Forensic Audit  
**Date:** November 6, 2026  
**Auditor:** Senior Research Software Engineer & Reproducibility Specialist  
**Target:** Reconciling conflicting historical reports of the Astra reference cutoff rate (51.10% vs. 50.6%)  
**Finding Classification:** `VERIFIED CORRECT` (Empirical Truth: 2,044 / 4,000 = 51.10%; Phase 2 Table Contained a Transcription Typo)

---

## 1. The Discrepancy

Two conflicting figures were recorded across historical repository reports:
1. **Earlier Evaluation Report (`research/final_verification_audit.md`):**
   - Evaluated Episodes: 4,000
   - Episodes Reaching Cutoff: 2,044
   - Step Cutoff Rate: **51.10%**
2. **Phase 2 Forensic Report (`research/forensic_audit/FORENSIC_AUDIT_REPORT.md`):**
   - Evaluated Episodes: 4,000
   - Step Cutoff Rate: **50.6% (reported as 2,024 / 4,000)**

This supplementary audit inspected the raw, immutable episode-level CSV records in [`experiments/reference_eval_runs/`](file:///Users/zero/Documents/Area/testing_llms/experiments/reference_eval_runs/) to resolve which figure is factually correct.

---

## 2. Raw Episode Record Inspection

Every evaluation run records individual episode outcomes in `episode_scores.csv`. The schema contains:
- `score`: Total game score (bricks destroyed).
- `length`: Total simulation steps.
- `step_cutoff`: Boolean flag (`True` if `length >= 10000`, `False` otherwise).
- `ball_lost`: Boolean flag (`True` if terminated due to ball falling below paddle, `False` otherwise).

A complete programmatic re-audit of all 4,000 rows across all four independent seeds produced the following exact results:

### Detailed Seed Breakdown

| Evaluation Run | CSV File Path | Episodes ($N$) | Step Cutoffs (`step_cutoff == True`) | Ball Losses (`ball_lost == True`) | Cutoff Percentage | Mean Score | SE Score | Mean Episode Length |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`run_seed42`** | [`experiments/reference_eval_runs/run_seed42/episode_scores.csv`](file:///Users/zero/Documents/Area/testing_llms/experiments/reference_eval_runs/run_seed42/episode_scores.csv) | 1,000 | 498 | 502 | 49.80% | 503.021 | $\pm 7.672$ | 7,178.392 |
| **`run_seed100`** | [`experiments/reference_eval_runs/run_seed100/episode_scores.csv`](file:///Users/zero/Documents/Area/testing_llms/experiments/reference_eval_runs/run_seed100/episode_scores.csv) | 1,000 | 506 | 494 | 50.60% | 500.422 | $\pm 7.882$ | 7,127.524 |
| **`run_seed200`** | [`experiments/reference_eval_runs/run_seed200/episode_scores.csv`](file:///Users/zero/Documents/Area/testing_llms/experiments/reference_eval_runs/run_seed200/episode_scores.csv) | 1,000 | 531 | 469 | 53.10% | 518.933 | $\pm 7.620$ | 7,377.034 |
| **`run_seed300`** | [`experiments/reference_eval_runs/run_seed300/episode_scores.csv`](file:///Users/zero/Documents/Area/testing_llms/experiments/reference_eval_runs/run_seed300/episode_scores.csv) | 1,000 | 509 | 491 | 50.90% | 509.873 | $\pm 7.676$ | 7,239.296 |
| **POOLED TOTAL** | — | **4,000** | **2,044** | **1,956** | **51.1000%** | **508.062** | $\mathbf{\pm 3.857}$ | **7,230.561** |

---

## 3. Forensic Identification of Error Origin

Summing the exact cutoff counts across the four runs:
$$498 + 506 + 531 + 509 = \mathbf{2,044}$$
$$\text{Exact Cutoff Rate} = \frac{2,044}{4,000} = \mathbf{0.511000} \implies \mathbf{51.10\%}$$

Summing the exact ball loss terminations:
$$502 + 494 + 469 + 491 = \mathbf{1,956}$$
$$\text{Exact Ball Loss Rate} = \frac{1,956}{4,000} = \mathbf{0.489000} \implies \mathbf{48.90\%}$$

### Why did the Phase 2 report write "50.6% (2,024 / 4,000)"?
Forensic inspection reveals the exact mechanism of the error:
1. In `run_seed100`, the individual run cutoff rate was exactly **$50.60\%$** ($506 / 1000$).
2. During preparation of the Phase 2 master summary table, the author inadvertently transcribed the individual `run_seed100` rate ($50.6\%$) into the pooled row, and back-computed $0.506 \times 4000 = 2,024$, or mistyped $2,044$ as $2,024$ ($2024 / 4000 = 50.6\%$).
3. The raw CSV files and `summary.json` files were never modified or corrupted.

---

## 4. Audit Reconciliation Verdict

- **The earlier evaluation report was 100% CORRECT:** Exactly 2,044 episodes out of 4,000 reached the 10,000-step ceiling ($51.10\%$).
- **The Phase 2 forensic report contained a transcription typo:** The figure "50.6% (2,024/4000)" is an error of reporting, not an error of simulation or raw data.
- **Auditor Action:** In accordance with audit protocols, historical records remain uncorrupted. The correction is officially documented in this report, and the Master Audit Report will be updated to reflect the true verified figure of **$2,044 / 4,000$ ($51.10\%$)**.

**Status:** `VERIFIED CORRECT`

