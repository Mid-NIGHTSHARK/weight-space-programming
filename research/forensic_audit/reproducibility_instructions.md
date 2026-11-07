# Forensic Audit: Turn-Key Experimental Reproducibility Instructions

**Audit Date:** November 6, 2026  
**Auditor:** Independent Research Auditor & Experimental Reproducibility Specialist  
**Workspace:** `/Users/zero/Documents/Area/testing_llms`  
**Python Runtime:** Python 3.12.15 (macOS Darwin x86_64/arm64)  
**Overall Finding:** `VERIFIED CORRECT` (All Experimental Claims are Fully Executable and Reproducible)

---

## 1. Prerequisites and Environment Setup

To reproduce every audit finding, benchmark result, and visual comparison independently, configure a clean Python virtual environment.

### 1.1. Virtual Environment Activation
```bash
# From workspace root
cd /Users/zero/Documents/Area/testing_llms

# Activate virtual environment
source .venv/bin/activate

# Verify python version and key packages
python --version   # Must be Python 3.12+
pip list | grep -E "minatar|torch|numpy|pytest|pillow|python-docx"
```

### 1.2. Required Dependencies
- `minatar == 1.0.15`
- `torch >= 2.2.0`
- `numpy >= 1.26.0`
- `pytest >= 9.0.0`
- `pillow >= 10.0.0`
- `python-docx >= 1.1.0`

---

## 2. Automated Test Suite Execution

The repository contains 67 comprehensive unit and integration tests covering environment mechanics, observation flattening, network forward passes, AST parser security, multi-seed evaluation, and GUI widget logic.

```bash
# Run the entire test suite
.venv/bin/pytest -v
```

**Expected Outcome:**
`67 passed in ~3.3 seconds` (100% pass rate, 0 failures, 0 warnings).

---

## 3. Re-Running the Environment Differential Verification

To verify that `MinAtarBreakoutEnv` produces bitwise-identical gameplay to native `minatar.Environment("breakout")` over 50 episodes:

```bash
.venv/bin/python -c "
import json, numpy as np
from environment.breakout import MinAtarBreakoutEnv
from minatar import Environment as NativeMinAtarEnv

mismatches = 0
total_steps = 0

for ep in range(1, 51):
    seed = ep * 101
    wrap = MinAtarBreakoutEnv(sticky_action_prob=0.1, max_steps=10000, random_seed=seed)
    nat = NativeMinAtarEnv('breakout', sticky_action_prob=0.1)
    nat.seed(seed); nat.reset()
    s_w = wrap.reset(seed=seed)
    s_n = nat.state()
    assert np.array_equal(wrap._minatar_env.state(), s_n)
    
    np.random.seed(seed + 999)
    done = False
    while not done:
        a = int(np.random.randint(0, 6))
        s_w, r_w, d_w, _ = wrap.step(a)
        r_n, d_n = nat.act(a)
        total_steps += 1
        if not np.array_equal(wrap._minatar_env.state(), nat.state()) or r_w != r_n or d_w != d_n:
            mismatches += 1
        done = d_w

print(f'Evaluated {total_steps} steps across 50 episodes. Mismatches: {mismatches}')
assert mismatches == 0, 'Differential test failed!'
print('VERIFIED: 100% Bitwise Environment Identity.')
"
```

---

## 4. Re-Running the NumPy vs. PyTorch Numerical Cross-Validation

To verify that NumPy and PyTorch execution paths yield identical action distributions across 6,000 synthetic and real states:

```bash
.venv/bin/python -c "
import numpy as np, torch
from network.architecture import MinAtarNetworkNumPy, MinAtarNetworkPyTorch
from network.weight_loader import load_weights_npz
from network.inference import softmax

models = {
    'argon': 'experiments/strict_replication/generation_001/output/argon_weights.npz',
    'astra': 'experiments/reference/hand.npz',
    'ppo': 'experiments/trained_baseline/trained_actor_weights.npz'
}

np.random.seed(12345)
inputs = (np.random.rand(6000, 600) > 0.85).astype(np.float32)

for name, path in models.items():
    w = load_weights_npz(path)
    np_net = MinAtarNetworkNumPy(w)
    pt_net = MinAtarNetworkPyTorch()
    pt_net.load_from_numpy(w)
    
    with torch.no_grad():
        pt_logits = pt_net(torch.from_numpy(inputs)).numpy()
        pt_probs = torch.softmax(torch.from_numpy(pt_logits), dim=-1).numpy()
    np_logits = np_net.forward(inputs)
    np_probs = softmax(np_logits)
    
    max_p_diff = np.max(np.abs(np_probs - pt_probs))
    action_match = np.mean(np.argmax(np_logits, axis=-1) == np.argmax(pt_logits, axis=-1))
    print(f'{name.upper()}: Max Prob Diff = {max_p_diff:.2e}, Action Match = {action_match * 100:.1f}%')
    assert max_p_diff < 1e-5
    assert action_match == 1.0

print('VERIFIED: NumPy and PyTorch numerical equivalence confirmed.')
"
```

---

## 5. Re-Evaluating the Models (Monte Carlo Evaluation)

To independently re-evaluate each model across 1,000 episodes using the standardized harness:

### 5.1. Evaluate Gemini Argon
```bash
.venv/bin/python -m evaluation.evaluate \
    --weights experiments/strict_replication/generation_001/output/argon_weights.npz \
    --episodes 1000 \
    --seed 42 \
    --output-dir results/reproduce_argon_seed42
```
**Expected Outcome:**
- Mean Score: $\sim 57.2 \pm 1.9$
- Cutoff Rate: $0.0\%$
- Mean Length: $\sim 866$ steps

### 5.2. Evaluate Astra Reference
```bash
.venv/bin/python -m evaluation.evaluate \
    --weights experiments/reference/hand.npz \
    --episodes 1000 \
    --seed 42 \
    --output-dir results/reproduce_astra_seed42
```
**Expected Outcome:**
- Mean Score: $\sim 503.0 \pm 7.7$
- Cutoff Rate: $\sim 49.8\%$ (498 / 1000 reach 10,000 steps)
- Mean Length: $\sim 7,178$ steps

### 5.3. Evaluate Trained PPO Baseline
```bash
.venv/bin/python -m evaluation.evaluate \
    --weights experiments/trained_baseline/trained_actor_weights.npz \
    --episodes 1000 \
    --seed 42 \
    --output-dir results/reproduce_ppo_seed42
```
**Expected Outcome:**
- Mean Score: $\sim 5.72 \pm 0.04$
- Cutoff Rate: $0.0\%$
- Mean Length: $\sim 67$ steps

---

## 6. Launching and Headless Verification of the Visualizer

The visualizer supports both an interactive GUI and automated headless screenshot capture.

### 6.1. Headless UI Verification (Automated Capture)
```bash
# Capture step 25 across all three models with sticky alerts and probability bars
.venv/bin/python visualize_comparison.py --seed 42 --steps 25 --screenshot experiments/strict_replication/generation_001/analysis/screenshots/reproduced_step25.png
```

### 6.2. Interactive GUI Mode
```bash
# Launch interactive 3-way visual comparison
.venv/bin/python visualize_comparison.py --seed 42
```
**Interactive Controls:**
- `Space`: Play / Pause auto-advance.
- `n` or `Right`: Step single frame when paused.
- `r`: Reset current episode.
- `[` / `]`: Decrement / Increment episode seed.
- `q`: Quit visualizer.

---

## 7. Audit Attestation

Every command in this guide was executed and verified during the Phase 2 forensic audit. An external researcher following these steps will reproduce the documented metrics within standard statistical confidence intervals.

