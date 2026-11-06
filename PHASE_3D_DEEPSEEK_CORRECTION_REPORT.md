# Phase 3D Research Audit & Empirical Evaluation Report: DeepSeek Controlled Weight Repair and Performance-Ranked Visualizer

**Date:** November 6, 2026  
**Repository:** `/Users/zero/Documents/Area/testing_llms`  
**Baseline Git Commit:** `450e0eb293ce7816b306ffcf2edd352e34608f75`  
**Experimental Condition:** `deepseek_corrected` (Targeted Post-Hoc Researcher Intervention)  
**Base Condition:** `deepseek_zero_shot` (`experiments/strict_replication/generation_004_deepseek`)

---

## 1. Executive Summary

In Phase 3C, zero-shot evaluation of DeepSeek's directly generated neural-network weights revealed an empirical score of **0.4843 ± 0.0091**, matching random action selection (0.485) and the zero-weight baseline (0.517). Forensic mathematical analysis established that this catastrophic failure was not caused by a failure of spatial reasoning or trajectory prediction, but by an **inverted $\tanh$ suppression error**: DeepSeek assigned large negative weights ($W_3[0:20, [2,4,5]] = -100.0$) to suppress inert actions (Up, Down, Fire), erroneously assuming inactive Layer 2 neurons evaluate to $0.0$. In $\tanh$ architectures, inactive units saturate to $-1.0$, producing an unintended positive logit spike of $+1700$ to $+1724$ on inert actions, reducing movement actions (Left, Right) to $0.000000\%$ probability.

In Phase 3D, we conducted a strictly controlled post-hoc experimental intervention:
1. **Preserved Pristine Zero-Shot Artifacts:** The original DeepSeek condition (`deepseek_zero_shot`) remains completely untouched in `experiments/strict_replication/generation_004_deepseek/` with verified cryptographic integrity.
2. **Minimal 60-Parameter Intervention:** We created `experiments/controlled_interventions/deepseek_corrected/`, zeroing out exactly 60 scalar parameters ($W_3[0:20, 2] = 0.0, W_3[0:20, 4] = 0.0, W_3[0:20, 5] = 0.0$). No weights were trained, fine-tuned, optimized, or altered elsewhere.
3. **Restored Numerical Dynamics:** Logits for Up, Down, and Fire dropped to $-100.0$ ($0.000000\%$ softmax probability), unmasking DeepSeek's underlying trajectory calculation.
4. **Empirical Benchmark (3,000 Episodes):** Over 3,000 benchmark episodes across seeds 42, 100, and 200, the corrected policy achieved a mean score of **6.5750 ± 0.1605** (SEM) with a maximum score of **90.0** bricks and mean episode length of **93.20 steps**.
5. **Surpassing the 10M PPO Baseline:** The restored policy **surpasses the 10,000,000-step trained PPO reinforcement learning baseline** ($5.757 \pm 0.019$) and exceeds GPT-6 ($2.493 \pm 0.036$) by **2.6×**.
6. **Performance-Ranked 7-Way Visualizer:** We upgraded `visualize_comparison.py` to sort all active models dynamically from highest to lowest verified mean benchmark score (#1 Astra through #7 DeepSeek Original), displaying rank badges, scores $\pm$ SEM, and distinct visual themes.

---

## 2. Mandatory Conceptual Distinction Statement

> [!IMPORTANT]
> **The original DeepSeek result measures the model's unmodified zero-shot weight generation. The corrected DeepSeek result measures a targeted researcher intervention informed by the original model's observed failure. The two results must not be conflated.**

This post-hoc intervention serves as an **ablative diagnostic proof**: it isolates whether the zero-shot model possessed functional internal representations of game mechanics and trajectory dynamics, demonstrating that the failure was strictly localized to output-layer logit suppression logic.

---

## 3. Controlled Intervention Specification

The intervention is formally specified in `experiments/controlled_interventions/deepseek_corrected/intervention_manifest.json`.

### Minimal Parameter Modifications

DeepSeek's original prompt response assigned:
```python
W3[0:20, 2] = -100; b3[2] = -100   # Up action
W3[0:20, 4] = -100; b3[4] = -100   # Down action
W3[0:20, 5] = -100; b3[5] = -100   # Fire action
```

The controlled intervention replaces these assignments with exactly:
```python
W3[0:20, 2] = 0.0   # Clear coupling from inactive Layer 2 units to Up
W3[0:20, 4] = 0.0   # Clear coupling from inactive Layer 2 units to Down
W3[0:20, 5] = 0.0   # Clear coupling from inactive Layer 2 units to Fire
```

### Parameter Differential Audit

| Tensor | Shape | Total Elements | Altered Elements | Parameter Values Changed | Bitwise Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $W_1$ | $(600, 32)$ | 19,200 | 0 | None | **Identical** |
| $b_1$ | $(32,)$ | 32 | 0 | None | **Identical** |
| $W_2$ | $(32, 32)$ | 1,024 | 0 | None | **Identical** |
| $b_2$ | $(32,)$ | 32 | 0 | None | **Identical** |
| $W_3$ | $(32, 6)$ | 192 | **60** | Rows 0..19 of Cols 2, 4, 5: $-100.0 \to 0.0$ | **Modified** |
| $b_3$ | $(6,)$ | 6 | 0 | None (Kept $-100.0$ on bias) | **Identical** |
| **Total** | — | **20,486** | **60** | **Exactly 60 Scalars Altered** | — |

- **Nonzero Parameters:** Reduced from 786 to **726** nonzeros.
- **Sparsity Rate:** Increased from 96.1632% to **96.4561%** (19,760 zero parameters).
- **Prohibited Actions:** No tuning of $W_1, b_1, W_2, b_2, b_3$, no training steps, and no heuristic overrides.

---

## 4. Mathematical and Numerical Audit

### Output Layer Logit Derivation

Let $h_2 \in [-1, 1]^{32}$ denote the activation of hidden Layer 2. In DeepSeek's architecture:
- Units $0 \dots 19$ correspond to 20 gated condition detectors ($10 \text{ rows} \times 2 \text{ horizontal directions}$).
- Exactly 1 unit activates when a valid ball position and direction are detected ($h_{2,\text{active}} \in [-1, 1]$).
- The remaining 19 units are inactive, driven by unactivated inputs with negative biases:
  $$\forall j \ne \text{active}, \quad h_{2,j} \approx \tanh(-20) = -1.0$$

#### In Original DeepSeek (Zero-Shot):
For inert actions $a \in \{2, 4, 5\}$ (Up, Down, Fire):
$$\text{Logit}[a] = \sum_{j=0}^{19} W_3[j, a] \cdot h_{2,j} + b_3[a] = (-100.0) \cdot h_{2,\text{active}} + 19 \cdot (-100.0) \cdot (-1.0) + (-100.0)$$
$$\text{Logit}[a] = -100 \cdot h_{2,\text{active}} + 1900 - 100 \approx +1700 \text{ to } +1724$$

Meanwhile, for movement actions:
$$\text{Logit}[\text{Left}] = -8.0 \cdot h_{2,\text{active}} - 1.0 \in [-9.0, +7.0]$$
$$\text{Logit}[\text{Right}] = +8.0 \cdot h_{2,\text{active}} - 1.0 \in [-9.0, +7.0]$$
$$\text{Logit}[\text{No-op}] = 0.0$$

Because $\text{Logit}[2, 4, 5] \approx +1700 \gg +7$, the softmax denominator is dominated entirely by the three inert actions:
$$P(\text{Up}) = P(\text{Down}) = P(\text{Fire}) \approx \frac{1}{3} = 33.33\%, \quad P(\text{Left}) = P(\text{Right}) = 0.000000\%$$

#### In Corrected DeepSeek (Researcher Intervention):
With $W_3[0:20, a] = 0.0$ for $a \in \{2, 4, 5\}$:
$$\text{Logit}[a] = 0.0 + b_3[a] = -100.0$$

Under standard IEEE 754 float32 computation:
$$e^{-100.0} \approx 3.72 \times 10^{-44} \approx 0.0$$
$$\sum_{a \in \{0, 1, 3\}} P(a) = 1.000000, \quad P(\text{Up}) = P(\text{Down}) = P(\text{Fire}) = 0.000000\%$$

### Active Steering Behavior

With the inert actions suppressed, DeepSeek's steering equations operate unimpeded:
- When the predicted landing column $x_{\text{land}} > x_{\text{paddle}}$, unit $h_{2,\text{active}} > 0 \implies \text{Logit}[\text{Right}] \approx +7.0 \implies P(\text{Right}) \approx 99.8\%$.
- When $x_{\text{land}} < x_{\text{paddle}}$, unit $h_{2,\text{active}} < 0 \implies \text{Logit}[\text{Left}] \approx +7.0 \implies P(\text{Left}) \approx 99.8\%$.
- When $x_{\text{land}} = x_{\text{paddle}}$, unit $h_{2,\text{active}} \approx 0.0 \implies \text{Logit}[\text{Left}] \approx -1.0, \text{Logit}[\text{Right}] \approx -1.0, \text{Logit}[\text{No-op}] = 0.0 \implies P(\text{No-op}) = 57.61\%$.

---

## 5. Empirical Benchmark Results (3,000 Episodes)

Evaluation was conducted under strict MinAtar Breakout 1.0.15 benchmark conditions (10,000-step cutoff, sticky action probability 0.10, deterministic episode seeds).

### Seed-by-Seed Replication

| Seed Set | Episode Range | Episodes | Mean Score ± SEM | 95% Confidence Interval | Median | Max | Mean Length (steps) | Cutoffs (10k steps) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 1 – 1,000 | 1,000 | $6.3640 \pm 0.2691$ | $[5.836, 6.892]$ | 3.0 | 88.0 | $90.26 \pm 4.96$ | 0 (0.00%) |
| **Seed 100** | 1,001 – 2,000 | 1,000 | $6.2240 \pm 0.2679$ | $[5.698, 6.750]$ | 3.0 | 90.0 | $87.60 \pm 4.88$ | 0 (0.00%) |
| **Seed 200** | 2,001 – 3,000 | 1,000 | $7.1370 \pm 0.2954$ | $[6.557, 7.717]$ | 3.0 | 58.0 | $101.74 \pm 5.76$ | 0 (0.00%) |
| **Combined** | **1 – 3,000** | **3,000** | **$6.5750 \pm 0.1605$** | **$[6.261, 6.890]$** | **3.0** | **90.0** | **$93.20 \pm 3.06$** | **0 (0.00%)** |

### Action Execution Distribution Across 279,592 Steps

| Action ID | Action Name | Selected Count | Selected % | Executed Count | Executed % | Role in Policy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | No-Op | 14,120 | 5.05% | 14,034 | 5.02% | Paddle aligned with trajectory |
| 1 | Left | 132,070 | 47.24% | 131,885 | 47.17% | Steer left toward landing column |
| 2 | Up | 0 | 0.00% | 76 | 0.03% | Suppressed (0.00%); 76 sticky repeats from init |
| 3 | Right | 133,402 | 47.71% | 133,522 | 47.76% | Steer right toward landing column |
| 4 | Down | 0 | 0.00% | 40 | 0.01% | Suppressed (0.00%); 40 sticky repeats from init |
| 5 | Fire | 0 | 0.00% | 35 | 0.01% | Suppressed (0.00%); 35 sticky repeats from init |
| **Total** | — | **279,592** | **100.00%** | **279,592** | **100.00%** | — |

*Note: The 151 non-zero executed actions on Up, Down, and Fire resulted exclusively from initial-frame environment sticky action inheritance before the first policy step.*

---

## 6. Official Comparative Performance Ranking

All conditions evaluated under the identical MinAtar Breakout benchmark harness:

| Rank | Model / Condition | Evaluated Episodes | Mean Score ± SEM | Median | Max Score | Mean Length (Steps) | Cutoff Rate (10k) | Condition Nature |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | **Astra Reference** | 4,000 | **$508.062 \pm 3.857$** | 645.0 | 814.0 | 7,230.6 | 51.10% | Zero-Shot Reference Policy |
| **#2** | **Gemini Argon 4 Medium** | 3,000 | **$57.866 \pm 1.136$** | 38.0 | 493.0 | 871.6 | 0.00% | Zero-Shot Model Generation |
| **#3** | **Claude Sonnet 5.5** | 3,000 | **$10.957 \pm 0.219$** | 7.0 | 111.0 | 128.9 | 0.00% | Zero-Shot Model Generation |
| **#4** | **DeepSeek (Corrected)** | 3,000 | **$6.575 \pm 0.161$** | 3.0 | 90.0 | 93.2 | 0.00% | **Post-Hoc Intervention** |
| **#5** | **Trained PPO Baseline** | 3,000 | **$5.757 \pm 0.019$** | 6.0 | 8.0 | 67.4 | 0.00% | 10M-Step Deep RL Baseline |
| **#6** | **GPT-6 (ChatGPT)** | 3,000 | **$2.493 \pm 0.036$** | 2.0 | 10.0 | 32.5 | 0.00% | Zero-Shot Model Generation |
| **#7** | **DeepSeek (Original)** | 3,000 | **$0.484 \pm 0.009$** | 0.0 | 1.0 | 10.8 | 0.00% | Zero-Shot Model Generation |
| — | *Zero-Weight Control* | 1,000 | $0.517 \pm 0.023$ | 0.0 | 4.0 | 11.3 | 0.00% | Untrained Null Control |
| — | *Random Action Control* | 1,000 | $0.485 \pm 0.023$ | 0.0 | 4.0 | 11.0 | 0.00% | Stochastic Noise Control |
| — | *Gemini Argon 4 High* | 0 | *[Pending]* | — | — | — | — | Planned Independent Exp. |

```
SCORE COMPARISON (Mean Score ± SEM):
Astra Reference      [================================================================] 508.062 ± 3.857
Gemini Argon 4M      [=======] 57.866 ± 1.136
Claude Sonnet 5.5    [=] 10.957 ± 0.219
DeepSeek Corrected   [.] 6.575 ± 0.161  <-- SURPASSES 10M PPO & GPT-6
Trained PPO (10M)    [.] 5.757 ± 0.019
GPT-6                [.] 2.493 ± 0.036
Zero-Weight Baseline [.] 0.517 ± 0.023
Random Baseline      [.] 0.485 ± 0.023
DeepSeek Original    [.] 0.484 ± 0.009
```

---

## 7. Forensic Root-Cause Analysis: Corrected vs Superior Policies

While DeepSeek Corrected scores $6.575$ and occasionally achieves runs up to **90 bricks**, it lags behind Claude Sonnet 5.5 ($10.957$), Gemini Argon 4M ($57.866$), and Astra ($508.062$). A mathematical trace of failure trajectories reveals two fundamental mechanical causes:

### 1. Absence of Side-Wall Bounce Folding
DeepSeek uses linear intercept extrapolation:
$$x_{\text{landing}} = x_{\text{ball}} \pm (9 - y_{\text{ball}})$$
This formula is strictly valid only when the trajectory reaches row 9 without striking a side wall ($0 \le x_{\text{landing}} \le 9$).
When a ball bounces off a wall (e.g., $x_{\text{ball}} = 1$ moving left with $9 - y = 4$), the actual physical bounce reflects the ball back inward to column 3 ($|1 - 4| = 3$). DeepSeek's linear unit calculates $1 - 4 = -3$. Because no bounce reflection folds $-3$ back into $[0, 9]$, the paddle saturates at column 0 and fails to catch the rebound.

### 2. Sticky Action Lag Compensation
Unlike Claude Sonnet 5.5, which purposefully offsets its aim by $+1$ lead step to compensate for the 10% sticky action probability, DeepSeek aims directly at the unbuffered landing column. When sticky action repeats occur during high-velocity descent, the paddle lags behind by one step, causing terminal ball losses on sharp angled trajectories.

---

## 8. Visualizer Reorganization & Performance Ranking

The interactive visualizer [`visualize_comparison.py`](file:///Users/zero/Documents/Area/testing_llms/visualize_comparison.py) was enhanced to support dynamic data-driven performance sorting:

1. **Dynamic Rank Sorting:** The layout loads [`comparative_metrics.json`](file:///Users/zero/Documents/Area/testing_llms/experiments/controlled_interventions/deepseek_corrected/analysis/comparative_metrics.json) at initialization via `load_verified_benchmark_registry()`, ordering panels strictly descending by verified mean score.
2. **Visual Differentiation:**
   - **DeepSeek (Original):** Deep Indigo (`#3949ab`), labelled `#7 DeepSeek [Zero-Shot]` with score `0.484 ± 0.009`.
   - **DeepSeek (Corrected):** Sky Blue (`#00b0ff`), labelled `#4 DeepSeek [Intervention]` with score `6.575 ± 0.161`.
3. **Geometry and Sizing:** Window size auto-configures to $1820 \times 870$ with $14\text{px}$ cell sizing across 7 panels, allowing clean side-by-side rendering of all models.
4. **Modal Updates:**
   - **Benchmark Modal (`M`):** Fully populated with all 7 conditions and baselines, highlighting the mandatory conceptual distinction.
   - **Inspector Modal (`I`):** Updated with Section 5 detailing the exact 60-parameter intervention, $\tanh$ saturation dynamics, and restored performance.

---

## 9. Cryptographic Artifact Hashes

| Artifact Description | File Path | SHA-256 Digest | Size (bytes) |
| :--- | :--- | :--- | :---: |
| **DeepSeek Corrected Weights** | `experiments/controlled_interventions/deepseek_corrected/output/deepseek_corrected_weights.npz` | `21078c485156850253744ed0c2e1efd54361cf06f603a5ad2fd32ffea3c1c8bc` | 83,382 |
| **Intervention Manifest** | `experiments/controlled_interventions/deepseek_corrected/intervention_manifest.json` | `fca7c1af5f4413e70dd7b3745e6b03aa1d7de701eee2ecb98c801fae221d210d` | 1,918 |
| **Seed 42 Summary** | `experiments/controlled_interventions/deepseek_corrected/evaluation/eval_seed42/summary.json` | `aca670f0a69e3d7545887506aad76e324d5f1cbfebcf2e6b4a3f20449e7b4d3d` | 811 |
| **Seed 100 Summary** | `experiments/controlled_interventions/deepseek_corrected/evaluation/eval_seed100/summary.json` | `22cb1901650abaf64fa951f060e3581574c43d8b0ae363af554750671c6e850f` | 810 |
| **Seed 200 Summary** | `experiments/controlled_interventions/deepseek_corrected/evaluation/eval_seed200/summary.json` | `01fc326d05d9d85ba4a550bbcb2a26d1737be59b33c1f97e5f70bbf4d12e5987` | 813 |
| **Comparative Metrics** | `experiments/controlled_interventions/deepseek_corrected/analysis/comparative_metrics.json` | `911ff47da44b527c4c0de5315da58435d0e2d1a04b535cb470f967f9a8031c88` | 3,538 |
| **Comparative Distribution Plot** | `experiments/controlled_interventions/deepseek_corrected/analysis/deepseek_corrected_comparative_score_distributions.png` | `f8b59d28e41de1d0ecd22c20b54556b790d9fc7488fe7f81e8e6174fb10cfebb` | 153,303 |
| **7-Way Visualizer Screenshot** | `experiments/controlled_interventions/deepseek_corrected/analysis/visualizer_7way_screenshot.png` | `7573c3b5bfe143c5725d1f7ca5c54d8eb7d299d0590d743db72028636cf9f8cc` | 30,675 |
| **Visualizer Code** | `visualize_comparison.py` | `675453f66683d40e3900fe419a1d750b1598fbb8ffb0bde13d1213c82c787d13` | 76,941 |

---

## 10. Automated Verification & Quality Assurance

A dedicated test suite was implemented in [`tests/test_deepseek_correction.py`](file:///Users/zero/Documents/Area/testing_llms/tests/test_deepseek_correction.py) along with enhancements to [`tests/test_visualizer_gui.py`](file:///Users/zero/Documents/Area/testing_llms/tests/test_visualizer_gui.py).

The entire project test suite was executed:
```bash
.venv/bin/pytest -v
============================= 100 passed in 3.31s ==============================
```
- **Total Test Cases:** 100
- **Passing Test Cases:** 100 (100% Pass Rate)
- **Regressions Introduced:** 0

---

## 11. Methodological Implications for Weight-Space Programming

The results of Phase 3D deliver three crucial methodological takeaways for the broader study of zero-shot neural weight generation:

1. **Non-Trivial Representational Capability in LLMs:** DeepSeek synthesized a sparse 3-layer neural network whose internal representations correctly perform multi-feature extraction, velocity direction detection, and spatial intercept extrapolation, beating millions of environment steps of PPO when unmasked.
2. **Global Numerical Sensitivity as a Distinct Failure Mode:** In software programming, a small logic flaw produces a localized bug. In weight-space programming, an activation sign mismatch in a single weight layer cascades through the softmax distribution, entirely drowning out otherwise functional subnetworks.
3. **Rigorous Experimental Demarcation:** Post-hoc researcher repairs must be strictly recorded and clearly demarcated from pure zero-shot outputs. By tracking both `deepseek_zero_shot` (Rank #7) and `deepseek_corrected` (Rank #4) side-by-side, we preserve scientific fidelity while achieving mechanistic transparency.
