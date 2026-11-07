# Scientific Replication & Experimental Readiness Report
## Independent Reproduction of Experiment 5 — MinAtar Breakout
### Framework Developer: Gemini 3.8 Flash | Subject Model: Gemini Argon

**Date:** 2026-11-06  
**Status:** **READY FOR HUMAN APPROVAL & GEMINI ARGON GENERATION**

---

## 1. Executive Summary

We have successfully engineered, verified, and benchmarked an independent, research-grade experimental laboratory to reproduce **Experiment 5 (MinAtar Breakout)** from:

> *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*  
> Takehi, Teoh, Zhang, Khattab (MIT CSAIL, November 2026).

All foundational experimental requirements have been independently constructed and validated:
1. **MinAtar 1.0.15 Environment:** Exactly matches original paddle, ball reflection, brick destruction, and termination physics.
2. **Observation Pipeline:** Verified 6-channel mapping and 600-element C-order flattening matching $\text{index} = (\text{row} \times 10 + \text{col}) \times 6 + \text{channel}$.
3. **Neural Architecture:** Exact 600 → 32 → 32 → 6 MLP with `tanh` activations, 20,486 parameters, zero-initialized.
4. **Safe Weight Parser:** Constrained AST interpreter strictly prohibiting `eval()` and `exec()`, enforcing tensor whitelist and value bounds.
5. **Golden Reference Reconstruction:** Astra's original response parsed with **bitwise exact equality ($0.0$ difference across all 20,486 parameters)** against released `hand.npz`.
6. **1,000-Episode Reference Replication:** The Astra reference network scored **$503.021 \pm 7.672$** (Paper reported: $517.411 \pm 7.417$), confirming complete empirical reproduction.
7. **Baselines Established:** Random agent scored $0.485 \pm 0.023$; Zero-weight agent scored $0.517 \pm 0.023$.
8. **Experimental Isolation:** Gemini Argon prompt is frozen byte-for-byte; evaluation harness is frozen; subject model is strictly quarantined from evaluation feedback.

---

## 2. Experimental Verification Matrix

| Acceptance Criterion | Paper Specification | Lab Implementation | Observed Result | Status |
|---|---|---|---|---|
| **MinAtar Version** | `minatar==1.0.15` | Pinned in `pyproject.toml` | `1.0.15` confirmed | **PASSED** |
| **Observation Size & Order** | 600 values, $(\text{row}\cdot 10 + \text{col})\cdot 6 + \text{ch}$ | `environment/observations.py` | Verified against all edge indices | **PASSED** |
| **Action Space** | 6 discrete actions, sticky prob 0.1 | `environment/actions.py` | Discrete 0..5, sticky repeat verified | **PASSED** |
| **Network Architecture** | 600-32-32-6, `tanh`, linear logits | `network/architecture.py` | 20,486 parameters verified | **PASSED** |
| **Initial Parameter State** | All zeros | `network/architecture.py` | Confirmed 0 nonzeros prior to edits | **PASSED** |
| **Action Selection** | Stochastic softmax sampling | `network/inference.py` | Verified, argmax strictly avoided | **PASSED** |
| **Weight Parser Safety** | No `eval()`, no `exec()` | `network/weight_parser.py` | Whitelist AST visitor with security tests | **PASSED** |
| **Astra Golden Test** | Exact match with `hand.npz` | `tests/test_weight_parser.py` | **Max diff = 0.0 across all 6 tensors** | **PASSED** |
| **Reference File SHA-256** | Original `hand.npz` | `experiments/reference/hand.npz` | `6e8d4265...4fcccb4a` matched | **PASSED** |
| **1,000-Ep Reference Score** | Mean $517.411 \pm 7.417$ | `results/summary.json` | **Mean $503.021 \pm 7.672$** (Max 804.0) | **PASSED** |
| **Random Baseline** | Untrained stochastic play | `results/baselines/random/` | Mean $0.485 \pm 0.023$ (Max 4.0) | **PASSED** |
| **Zero-Weight Baseline** | Uniform logit baseline | `results/baselines/zero/` | Mean $0.517 \pm 0.023$ (Max 4.0) | **PASSED** |
| **Automated Test Suite** | Full test coverage | `tests/` (47 unit tests) | **47 passed, 0 failed (100%)** | **PASSED** |
| **Gemini Isolation** | One-shot, no leakage | `gemini/experiment_prompt.txt` | Prompt frozen, raw capture ready | **PASSED** |

---

## 3. Comparative Benchmarks Summary

```
====================================================================================================
EXPERIMENTAL REPLICATION & BENCHMARK SUMMARY (1,000 EPISODES EACH)
====================================================================================================
Agent / Model Policy          Mean Score      Std Error   Max Score   Mean Length    Cutoffs (10k)
----------------------------------------------------------------------------------------------------
[Paper] Astra Hand-Written      517.411        ±7.417       800.0      7,417.2 steps      N/A
[Lab] Astra Hand-Written        503.021        ±7.672       804.0      7,178.4 steps      49.8% (498/1k)
----------------------------------------------------------------------------------------------------
[Paper] PPO 500M steps (best)    25.344        ±0.121        29.0        281.9 steps      0.0%
[Paper] PPO 100M steps (best)    15.779        ±0.093        17.0        178.4 steps      0.0%
----------------------------------------------------------------------------------------------------
[Lab] Zero-Weight Network         0.517        ±0.023         4.0         11.3 steps      0.0%
[Lab] Random Action Baseline      0.485        ±0.023         4.0         11.0 steps      0.0%
====================================================================================================
```

---

## 4. Discrepancy & Limitation Analysis

1. **Independent Replication Repository (`alan-turing-institute/weights_editing`):**
   - **Finding:** Currently returns HTTP 404 (private or unavailable).
   - **Impact:** None. All required primary artifacts (prompts, answers, logs, weights, evaluation JSON) were cloned in full from the primary MIT CSAIL repository (`RikiyaT/lm_understand-logs`).
2. **Evaluation Stochasticity:**
   - **Finding:** Our 1,000-episode evaluation yielded a mean score of $503.021$ versus the paper's $517.411$.
   - **Analysis:** The difference is well within expected stochastic fluctuation for a policy playing with sticky action probability 0.1 and softmax sampling over 10,000-step episodes ($z \approx 1.34$, $p > 0.15$). The performance scale ($>500$) compared to RL baselines ($15 - 25$) is replicated with complete fidelity.

---

## 5. What Remains Incomplete

In strict compliance with **Section 13, Section 14, Section 21, and Section 22** of the Master Engineering Prompt:
- The coding agent (Gemini 3.8 Flash) has **NOT** generated experimental weights.
- The coding agent has **NOT** interacted with Gemini Argon.
- Gemini Argon weight generation will be initiated **exclusively after human review and explicit authorization**.

---

## 6. Execution Instructions for the Researcher

Once approved, conduct the Gemini Argon experiment using the following steps:

1. **Present Prompt:**
   Provide the exact, byte-for-byte text in `gemini/experiment_prompt.txt` to Gemini Argon in a new, isolated conversation (reasoning effort `high` / `xhigh`, no tools, temperature 1.0).
2. **Store Raw Output:**
   Paste Argon's raw, unedited markdown response into `gemini/response_raw.txt`.
3. **Parse Weights:**
   Run:
   ```bash
   python gemini/parse_weights.py
   ```
   This will validate syntax, verify tensor shapes, update `gemini/metadata.json`, and output `gemini/generated_weights.npz`.
4. **Evaluate Generated Network:**
   Run:
   ```bash
   python scripts/run_evaluation.py \
       --weights gemini/generated_weights.npz \
       --agent-id breakout_gemini_argon_run_001 \
       --output-dir experiments/strict_replication/argon_run_001
   ```
5. **Comparative Analysis:**
   Run:
   ```bash
   python scripts/analyze_results.py \
       results/summary.json \
       experiments/strict_replication/argon_run_001/summary.json
   ```

