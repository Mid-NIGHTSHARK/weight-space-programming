# Gemini Argon Generation 001: Weight-Space Programming Replication

Independent reproduction and empirical evaluation of neural-network weights synthesized zero-shot by **Gemini Argon 4 Medium** (`gemini-argon-4-medium`) for MinAtar Breakout (Experiment 5 replication).

---

## Directory Overview

```
experiments/strict_replication/generation_001/
├── ARGON_HANDOFF.md                  # Protocol and environment configuration for human researcher
├── ARGON_AUDIT_REPORT.md             # Complete scientific audit answering the 10 mandated questions
├── README.md                         # This file: replication guide and execution instructions
├── input/
│   ├── argon_experiment_001.zip     # Verified handoff package (contains experiment_5_prompt.txt)
│   └── experiment_5_prompt.txt       # Immutable byte-for-byte original Experiment 5 prompt
├── output/
│   ├── argon_response_original.docx  # Raw response exported directly from Gemini web UI
│   ├── raw_response.txt              # Extracted plain text of Argon's full response
│   ├── argon_weights.npz             # Reconstructed neural network weights (817 nonzero parameters)
│   ├── generation_metadata.json      # Complete experimental provenance and tensor verification log
│   └── generation_metadata_template.json # Template for recording model generation runs
├── eval_seed42/                      # 1,000-episode evaluation run (Seed 42)
│   ├── episode_scores.csv
│   ├── evaluation_report.md
│   └── summary.json
├── eval_seed100/                     # 1,000-episode evaluation run (Seed 100)
│   ├── episode_scores.csv
│   ├── evaluation_report.md
│   └── summary.json
├── eval_seed200/                     # 1,000-episode evaluation run (Seed 200)
│   ├── episode_scores.csv
│   ├── evaluation_report.md
│   └── summary.json
└── analysis/
    ├── statistical_comparison.json   # Pooled statistical tests (Welch t-test, Mann-Whitney U, Cohen's d)
    ├── score_distributions_comparison.png  # Comparative score distribution plots
    └── cumulative_survival_curves.png     # Step survival curves comparing Argon, PPO, and Astra
```

---

## Artifact Integrity & Checksums

| Artifact | Relative Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Original Prompt** | `input/experiment_5_prompt.txt` | `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876` |
| **Input Archive** | `input/argon_experiment_001.zip` | `1e2338c03c58be1619aa6e6c276b509ee44a80ce6c3e986252b47ea56b26eb2a` |
| **Raw DOCX** | `output/argon_response_original.docx` | `2e40dafa935ed1b21290a1dcba0ae67f88a54b79c1f924fa493d37c49f80a2e7` |
| **Raw Response Text** | `output/raw_response.txt` | `6e593d588c827afd4f1ab2d81ae1a5c65585210e0345920fb887a76b24e248ac` |
| **Argon Weights NPZ** | `output/argon_weights.npz` | `455722b09750deb693093c823c13dd37b99eb470c957ce472262b1419014abb5` |

---

## Quick Replication Commands

All commands should be executed from the project root (`/Users/zero/Documents/Area/testing_llms`) within the virtual environment (`.venv`).

### 1. Run Automated Test Suite
To verify environment correctness, parser behavior, tensor shapes, Argon forward passes, and visualizer initialization:
```bash
.venv/bin/pytest -v
```
*(All 57 tests must pass).*

### 2. Parse & Reconstruct Weights from Raw Response
To re-extract the weights from `raw_response.txt` using the safe AST parser:
```bash
.venv/bin/python -c "
from network.weight_parser import SafeASTWeightParser
from network.weight_loader import save_weights_npz, compute_file_sha256

with open('experiments/strict_replication/generation_001/output/raw_response.txt') as f:
    text = f.read()

parser = SafeASTWeightParser(strict=True)
weights, report = parser.parse_text(text)
print(f'Parsed {report.statements_applied} statements, Nonzero: {report.nonzero_total}, Errors: {len(report.errors)}')
save_weights_npz(weights, 'experiments/strict_replication/generation_001/output/argon_weights.npz')
print('SHA-256:', compute_file_sha256('experiments/strict_replication/generation_001/output/argon_weights.npz'))
"
```

### 3. Run Multi-Seed Empirical Evaluation
To evaluate Argon's reconstructed weights over 1,000 episodes on Seed 42:
```bash
.venv/bin/python evaluation/evaluate.py \
    --weights experiments/strict_replication/generation_001/output/argon_weights.npz \
    --episodes 1000 \
    --seed 42 \
    --output experiments/strict_replication/generation_001/eval_seed42 \
    --model-id gemini_argon_seed42
```

### 4. Generate Comparative Analysis & Visualizations
To recalculate statistical comparisons and generate publication plots:
```bash
.venv/bin/python scripts/generate_argon_comparison.py
```
Outputs:
- `experiments/strict_replication/generation_001/analysis/statistical_comparison.json`
- `experiments/strict_replication/generation_001/analysis/score_distributions_comparison.png`
- `experiments/strict_replication/generation_001/analysis/cumulative_survival_curves.png`

### 5. Launch Interactive Visual Comparison Application
To launch the side-by-side Tkinter visualizer rendering Gemini Argon vs. the 10M-step Trained PPO model:
```bash
python visualize_comparison.py
```
*(Or specify custom weights, seed, and FPS)*:
```bash
python visualize_comparison.py \
    --argon experiments/strict_replication/generation_001/output/argon_weights.npz \
    --trained experiments/trained_baseline/trained_actor_weights.npz \
    --seed 42 \
    --fps 15
```

#### Visualizer Controls:
- **`Space`**: Play / Pause simulation.
- **`Right Arrow`**: Step forward one frame (when paused).
- **`R`**: Reset current game seed.
- **`N`**: Next seed (+1).
- **`P`**: Previous seed (-1).
- **`Speed Slider`**: Dynamically adjust frame rate from 1 to 60 FPS.

---

## Summary of Empirical Results

| Metric | Gemini Argon (Zero-Shot) | Trained PPO (10M Steps) | Astra Reference (Published) |
| :--- | :--- | :--- | :--- |
| **Episodes Evaluated** | 3,000 | 3,000 | 4,000 |
| **Mean Score** | **57.87 ± 1.14** | **5.76 ± 0.02** | **508.06 ± 3.86** |
| **Median Score** | 38.0 | 6.0 | 645.0 |
| **Max Score** | **493.0** | **8.0** | **814.0** |
| **Mean Episode Length** | 871.62 steps | 67.35 steps | 7,230.56 steps |
| **10,000-Step Cutoffs** | 0 (0.0%) | 0 (0.0%) | 2,044 (51.1%) |
| **Welch's t-Test vs Trained**| $t = 45.87, p < 10^{-300}$ | — | — |
| **Cohen's d Effect Size** | **1.184** (Large effect) | — | — |

For complete mechanistic analysis of the policy and root-cause tracing of failure modes, see [ARGON_AUDIT_REPORT.md](ARGON_AUDIT_REPORT.md).
