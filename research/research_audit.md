# Research Audit Notebook: Experiment 5 Replication

This notebook documents the empirical audit of primary sources, specifications, and experimental artifacts for the independent reproduction of Experiment 5 (MinAtar Breakout) from *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*.

---

## 1. Primary Source Inventory & Status

| Source Identifier | URI / Repository | Access Status | Verification Evidence / Notes |
|---|---|---|---|
| **Original Publication** | `https://rikiyat.github.io/blog/weight-space-programming/` | Verified & Accessible | Authors: Takehi, Teoh, Zhang, Khattab (MIT CSAIL). Published Nov 2, 2026. Contains Experiment 5 figures, score table, and replay traces. |
| **Original Logs & Code** | `https://github.com/RikiyaT/lm_understand-logs` | Verified & Fully Cloned | Commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Contains full prompt (`message.txt`, `reply/prompt.txt`), answer (`reply/answer.md`), execution audit (`reply/run.json`), weight archive (`networks/breakout/hand.npz`), and evaluation logs (`eval/minatar_eval.json`). |
| **Independent Replication** | `https://github.com/alan-turing-institute/weights_editing` | Inaccessible (HTTP 404) | Querying GitHub directly via HTTP GET/HEAD returned HTTP 404. Repository is currently either private, deleted, or relocated. Web search indexes confirm its past existence and research plan (stress tests with `breakout_swapped.txt` / `breakout_flipped.txt`). Our implementation relies strictly on the primary author repository and MinAtar 1.0.15 upstream. |

---

## 2. Experimental Details Audit Table

| # | Field / Claim | Source | Evidence | Decision | Verification |
|---|---|---|---|---|---|
| 1 | **Target Game** | Original paper & `exp5_write_from_scratch/README.md` | MinAtar Breakout on a 10 × 10 grid. | Implement MinAtar Breakout strictly. Scope limited strictly to Breakout. | Automated environment tests in `tests/test_environment.py`. |
| 2 | **MinAtar Version** | `exp5_write_from_scratch/README.md` line 10 & 20; prompt lines 6 & 88 | "The game's exact implementation (the MinAtar package, version 1.0.15, file `minatar/environments/breakout.py`)" | Pin `minatar==1.0.15` in `pyproject.toml` and `.venv`. | Automated version assertion in tests. |
| 3 | **Native vs Input Channels** | `prompt.txt` lines 143-151 | MinAtar native has 4 channels: `paddle` (0), `ball` (1), `trail` (2), `brick` (3). The prompt specifies 6 channels: 0: player (paddle), 1: danger (empty), 2: reward (brick), 3: projectile (ball), 4: trail (trail), 5: gauge (empty). | Build `get_observation()` mapping: `ch[0] <- native[0]`, `ch[1] <- 0`, `ch[2] <- native[3]`, `ch[3] <- native[1]`, `ch[4] <- native[2]`, `ch[5] <- 0`. | Test mapping with synthetic board states; verify channel layout. |
| 4 | **Observation Flattening Convention** | `prompt.txt` lines 143, 171 | "Input number `i = (row * 10 + column) * 6 + channel`; for example input 0 is (row 0, column 0, channel 0), input 6 is (row 0, column 1, channel 0) and input 599 is (row 9, column 9, channel 5)." | Standard C-contiguous row-major flattening of `(10, 10, 6)` array. Shape `(600,)`. | Test specific known indices: (0,0,0)->0, (0,1,0)->6, (9,9,5)->599. Assert no silent transpose. |
| 5 | **Neural Architecture** | `prompt.txt` lines 153-157; `README.md` lines 77-83 | 600 -> 32 -> 32 -> 6. Two hidden layers with `tanh` activations; output layer is linear (produces raw logits). 20,486 parameters. | Implement identical MLP forward pass: `h1 = tanh(x @ W1 + b1)`, `h2 = tanh(h1 @ W2 + b2)`, `logits = h2 @ W3 + b3`. | Unit tests asserting shapes: W1(600,32), b1(32,), W2(32,32), b2(32,), W3(32,6), b3(6,). Total params = 20,486. |
| 6 | **Tensor Orientation & Conventions** | `prompt.txt` line 157; `hand.npz` | `actor/dense1/kernel` shape `(600, 32)`: Row i holds weights from input i to layer 1 units. `actor/dense2/kernel` `(32, 32)`. `actor/logits/kernel` `(32, 6)`. | Authoritative representation: `x` is `(600,)`, `W1` is `(600, 32)`, using `np.dot(x, W1) + b1`. | Compare against `hand.npz` tensors loaded directly from reference. |
| 7 | **Initial Weights** | `prompt.txt` line 157; `README.md` line 15 | "EVERY WEIGHT STARTS AT 0; you set the weights you need." | Zero-initialize all 6 tensors before applying edits. | Assert all zeros prior to parsing. |
| 8 | **Action Space & Mapping** | `prompt.txt` line 150; `minatar/environments/breakout.py` line 35 | 6 discrete actions: 0: 'n' (no-op), 1: 'l' (left), 2: 'u' (up), 3: 'r' (right), 4: 'd' (down), 5: 'f' (fire). Only left and right affect paddle. | Preserve all 6 action logits; pass full discrete index `0..5` to MinAtar wrapper. | Verify action mapping enum/class matches MinAtar's `action_map`. |
| 9 | **Sticky Actions** | `prompt.txt` line 6, 150; `minatar/environment.py` line 44 | Probability 0.1 of repeating `last_action` instead of executing newly selected action. `last_action` initializes to 0. | Use MinAtar standard `Environment(..., sticky_action_prob=0.1)` wrapper semantics with controlled RNG seed. | Test sticky action repetition deterministically with seeded RNG. |
| 10 | **Action Selection Mechanism** | `prompt.txt` line 150, 159; `README.md` line 20 | "actions sampled from the softmax of the logits" | Compute numerically stable softmax: `probs = exp(logits - max(logits)) / sum(exp(logits - max(logits)))`, then sample with `rng.choice(6, p=probs)`. Argmax is strictly prohibited. | Statistical tests verifying probability distribution; rejection of argmax. |
| 11 | **Evaluation Protocol** | `prompt.txt` line 159; `README.md` lines 20-21; `minatar_eval.json` | 1,000 independent evaluation episodes. Maximum step cutoff per episode: 10,000 steps. Sticky actions 0.1. | Run exactly 1,000 episodes; record episode rewards, lengths, and completion conditions. | Check against reported reference: Astra Breakout hand mean = 517.411, standard error = 7.417. |
| 12 | **Weight Edit Language Grammar** | `prompt.txt` lines 161-172; `answer.md` | Supports single indices (`W1[i, j]`), ranges (`b1[0:4]`), full slices (`b1[:]`), strides (`W1[3:600:60, 4]`), 1D lists, 2D lists (`[[...], [...]]`), scalar assignments, and in-place arithmetic (`+=`, `-=`, `*=`). Sequential overwriting. | Build safe AST / grammar-based parser in `network/weight_parser.py` without `eval()` or `exec()`. | Parse original Astra Breakout `answer.md` and verify bit-for-bit equality with `hand.npz`. |
| 13 | **Parser Security & Constraints** | Master Engineering Prompt Section 9 | Disallow arbitrary Python code, file I/O, subprocesses, imports, attribute access. | Constrain AST visitor strictly to `Assign`, `AugAssign`, `Subscript`, `Slice`, `Constant`, `List`, `UnaryOp`, `Name` with whitelisted identifiers (`W1, b1, W2, b2, W3, b3`). | Security tests feeding malicious payloads (e.g. `__import__`, `open()`, os calls) asserting immediate syntax/security error. |
| 14 | **Numerical Platform Precision** | NumPy 2.x on Apple Silicon (macOS aarch64) | Observation: `@` (matmul) in user NumPy 2.0.2 triggered Apple Accelerate warning. `np.dot()` produces identical results cleanly without Accelerate warning. | Use `np.dot()` for Authoritative CPU inference. Document tolerance thresholds (`1e-6` for float32). | Test numerical equivalence between NumPy dot, matmul, and PyTorch CPU forward pass. |
| 15 | **Experimental Subject Isolation** | Master Engineering Prompt Section 13, 14 | Gemini Argon is the experimental subject; Gemini 3.8 Flash is coding agent. Strict one-shot, no training, no evaluation access during generation. | Maintain prompt in `gemini/experiment_prompt.txt` byte-for-byte; freeze evaluation harness; do not run Argon generation during lab construction. | Readiness report verification and protocol freeze. |

---

## 3. Discrepancy Analysis

### Discrepancy A: Independent Replication Repository Availability
- **Reported URL:** `https://github.com/alan-turing-institute/weights_editing`
- **Observed Status:** HTTP 404 (Not Found / Private).
- **Impact on Experiment:** Zero impact on core Experiment 5 reproduction because the primary author's repository `RikiyaT/lm_understand-logs` is 100% complete, public, and contains the original prompt, raw output, execution logs, and ground-truth `hand.npz` weights.
- **Resolution:** We document this source unavailability transparently and ground all specifications and golden tests on the primary author artifacts and official `minatar==1.0.15` package.

### Discrepancy B: Matrix Multiplication on macOS Apple Silicon
- **Observed Issue:** Under NumPy 2.0+ on macOS aarch64, certain BLAS implementations through Apple Accelerate can emit a warning (`RuntimeWarning: divide by zero encountered in matmul`) during `x @ W1` on certain zero-padded arrays, even though the computed values are correct.
- **Resolution:** In `network/inference.py`, we implement matrix multiplications using `np.dot(x, W)` which is mathematically identical for 1D-2D and 2D-2D tensors and evaluates cleanly without BLAS warning spikes. Both NumPy and PyTorch implementations are tested to ensure bitwise or `1e-6` consistency.

---

## 4. Verification Checkpoint Status

- [x] Primary author repository cloned and audited.
- [x] MinAtar version confirmed (`1.0.15`).
- [x] Original prompt extracted and hashed.
- [x] Original response extracted and hashed.
- [x] Original `hand.npz` weights inspected and hashed.
- [x] Observation 6-channel mapping audited.
- [x] Network architecture dimensions audited.
- [x] Sticky action mechanism audited.
- [x] Discrepancies logged and resolved.
