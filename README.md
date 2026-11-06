# Weight-Space Programming Replication Lab
## Independent Reproduction of Experiment 5 — MinAtar Breakout
### Experimental Subject: Gemini Argon | Infrastructure Engineering: Gemini 3.8 Flash

---

## 1. Overview & Research Objective

This repository contains a complete, research-grade experimental framework to independently reproduce Experiment 5 from the seminal paper:

> **"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"**  
> *Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL, November 2026)*  
> *Original Publication:* [https://rikiyat.github.io/blog/weight-space-programming/](https://rikiyat.github.io/blog/weight-space-programming/)  
> *Original Logs & Code:* [https://github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)  

### Core Research Question:
**Can Gemini Argon independently construct a functional neural-network policy for MinAtar Breakout by directly generating the network's weights, without reinforcement learning, gradient descent, backpropagation, or feedback from the environment?**

---

## 2. System Architecture & Components

```
weight-space-gemini/
│
├── README.md                      # Complete system documentation
├── pyproject.toml                 # Pinned dependencies & metadata
├── .gitignore                     # Git configuration
│
├── research/                      # Research audit and protocols
│   ├── research_audit.md          # Comprehensive audit of claims, evidence, and decisions
│   ├── experiment_5_protocol.md   # Exact reproducible evaluation protocol
│   ├── original_prompt.txt        # Byte-for-byte original Experiment 5 prompt
│   ├── original_response.md       # Astra's exact raw answer and edits block
│   ├── references.md              # Academic and source citations
│   └── decisions.md               # Architecture and engineering rationales
│
├── environment/                   # MinAtar Breakout integration
│   ├── __init__.py
│   ├── breakout.py                # Authoritative environment wrapper
│   ├── observations.py            # 6-channel mapping & 600-element C-order flattening
│   ├── actions.py                 # Discrete action space [0..5] & character mappings
│   └── validation.py              # Environment sanity and boundary validators
│
├── network/                       # Neural network architecture & parsing
│   ├── __init__.py
│   ├── architecture.py            # 600-32-32-6 MLP (NumPy authoritative & PyTorch reference)
│   ├── inference.py               # Softmax action sampling engine
│   ├── weight_loader.py           # NPZ archive loading/saving & SHA-256 validation
│   ├── weight_parser.py           # Safe AST weight parser (no eval/exec)
│   └── weight_validation.py       # Sparsity analysis & parameter validation
│
├── baselines/                     # Comparison agents
│   ├── random_agent.py            # Uniform random action sampler
│   ├── zero_agent.py              # Zero-weight policy baseline
│   ├── train_ppo.py               # CleanRL-style PPO reference implementation
│   └── evaluate_baselines.py      # Automated baseline evaluation script
│
├── gemini/                        # Gemini Argon experiment interface
│   ├── experiment_prompt.txt      # Frozen prompt for Argon
│   ├── response_raw.txt           # Raw response storage (human researcher input)
│   ├── parse_weights.py           # Isolated weight parser CLI
│   └── metadata.json              # Experiment tracking & model metadata
│
├── evaluation/                    # Rigorous evaluation harness
│   ├── evaluate.py                # 1,000-episode evaluation harness
│   ├── metrics.py                 # Aggregate statistics (mean, se, median, etc.)
│   ├── seeds.py                   # Deterministic seed generator
│   └── statistical_analysis.py    # Report generator & reference comparisons
│
├── experiments/                   # Experiment artifacts
│   ├── reference/hand.npz         # GPT-6 Astra reference weights (verified SHA-256)
│   ├── strict_replication/        # Isolated Gemini Argon runs
│   └── extended_experiments/      # Extended / ablation studies
│
├── tests/                         # Comprehensive automated test suite
│   ├── test_actions.py            # Action validation tests
│   ├── test_observations.py       # Observation indexing & channel tests
│   ├── test_environment.py       # MinAtar physics & sticky action tests
│   ├── test_architecture.py      # Architecture, shapes, & PyTorch equivalence
│   ├── test_weight_parser.py      # AST parser unit & golden tests
│   ├── test_reference_network.py  # Reference network property & hash tests
│   └── test_evaluation.py        # Evaluation harness & metrics tests
│
├── scripts/                       # Executable CLI tools
│   ├── download_reference.py      # Download & verify original hand.npz
│   ├── verify_reference.py        # Parameter & sparsity audit script
│   ├── run_evaluation.py          # CLI evaluation runner
│   └── analyze_results.py         # Comparative analysis tool
│
└── results/                       # Evaluation outputs (CSV, JSON, Markdown)
    ├── episode_scores.csv         # Individual episode logs
    ├── summary.json               # Structured metrics summary
    └── evaluation_report.md       # Full Markdown research report
```

---

## 3. Specifications & Standards

### Environment & Observation:
- **Game:** MinAtar Breakout (`minatar==1.0.15`).
- **Board:** 10 × 10 grid.
- **Input Channels (6):** `player` (0), `danger` (1), `reward` (2), `projectile` (3), `trail` (4), `gauge` (5).
- **Flattening Convention:** `index = (row * 10 + column) * 6 + channel` (standard C-order flattening into 600 float32 values).
- **Actions:** 0: `no-op`, 1: `left`, 2: `up`, 3: `right`, 4: `down`, 5: `fire`.
- **Sticky Actions:** Probability 0.1 of repeating `last_action`.

### Neural Network:
- **Shape:** 600 → 32 → 32 → 6.
- **Activations:** Hidden layers: `tanh`. Output: Linear (logits).
- **Parameters:** 20,486 float32 values. Initialized to zero.
- **Action Sampling:** Softmax with stochastic sampling: $p = \text{softmax}(\text{logits})$, $a \sim \text{Categorical}(p)$.

### Safe Weight Parser:
- Parses model-generated Python-like assignments (`W1[i, j] = ...`, slices, strides, lists, `+=`, `*=`) using `ast.parse`.
- **Strictly prohibits `eval()` and `exec()`**. Whitelists only approved tensor identifiers (`W1, b1, W2, b2, W3, b3`) and numerical bounds ($|v| \le 10^6$).
- **Golden Verification:** Successfully reconstructs GPT-6 Astra's released `hand.npz` weights with **exact bitwise equality (0.0 difference)**.

---

## 4. Evaluation Protocol

- **Episodes:** 1,000 independent runs.
- **Cutoff:** Maximum 10,000 steps per episode.
- **Seeds:** Deterministically managed via PRNG.
- **Published Reference Scores:**
  - **GPT-6 Astra (Hand-written one-shot):** Mean **517.411** ± 7.417 (Max 800.0, Mean Length 7,417 steps).
  - **PPO 500M steps (Best seed):** Mean **25.344** ± 0.121 (Max 29.0).
  - **PPO 100M steps (Best seed):** Mean **15.779** ± 0.093 (Max 17.0).
  - **Random Agent Baseline:** Mean **0.485** ± 0.023 (Max 4.0).
  - **Zero-Weight Baseline:** Mean **0.517** ± 0.023 (Max 4.0).

---

## 5. Quick Start & Execution

### Setup Environment:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Run All Automated Tests:
```bash
pytest
```

### Verify Astra Reference Network:
```bash
python scripts/verify_reference.py
```

### Evaluate Baselines:
```bash
python baselines/evaluate_baselines.py 1000
```

### Evaluate Reference Weights:
```bash
python scripts/run_evaluation.py --weights experiments/reference/hand.npz --episodes 1000
```

---

## 6. Gemini Argon Experiment Workflow (Strict One-Shot)

1. Deliver `gemini/experiment_prompt.txt` to Gemini Argon in a clean conversation.
2. Save Argon's complete raw response to `gemini/response_raw.txt`.
3. Parse and validate the weights:
   ```bash
   python gemini/parse_weights.py
   ```
4. Evaluate Argon's generated policy:
   ```bash
   python scripts/run_evaluation.py --weights gemini/generated_weights.npz --agent-id gemini_argon_run_001 --output-dir experiments/strict_replication/argon_run_001
   ```
5. Compare results:
   ```bash
   python scripts/analyze_results.py results/summary.json experiments/strict_replication/argon_run_001/summary.json
   ```
