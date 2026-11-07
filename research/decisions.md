# Architectural and Experimental Decisions

This document records the foundational technical and experimental decisions made for the independent replication of MinAtar Breakout Weight-Space Programming (Experiment 5).

---

## 1. Environment & Dependency Management
- **Decision:** Use Python 3.12 managed via `uv`, pinning `minatar==1.0.15` and core scientific libraries (`numpy`, `torch`, `scipy`, `pandas`, `matplotlib`, `pytest`).
- **Rationale:** MinAtar 1.0.15 is the exact version used in the original paper. Using Python 3.12 with `uv` guarantees reproducible virtual environment builds and fast dependency resolution on macOS Apple Silicon.
- **Verification:** Automated tests verify environment versions upon execution.

## 2. MinAtar Integration Architecture
- **Decision:** Wrap the native `minatar.Environment('breakout')` class in an authoritative `MinAtarBreakoutEnv` class located in `environment/breakout.py`.
- **Rationale:** The native Breakout environment returns a `(10, 10, 4)` state with native channels `[paddle, ball, trail, brick]`. The Experiment 5 prompt specifies a 6-channel observation `[paddle, danger, reward, ball, trail, gauge]` flattened into a 600-dimensional vector according to the convention `index = (row * 10 + col) * 6 + channel`. Wrapping this cleanly isolates the native game physics from the observation transformation.
- **Verification:** Unit tests confirm that brick collisions, wall bounces, paddle movements, rewards, and terminations match MinAtar native behavior exactly.

## 3. Observation Flattening and Indexing Convention
- **Decision:** Implement authoritative observation generation in `environment/observations.py` with the formula:
  $$\text{index} = (\text{row} \times 10 + \text{column}) \times 6 + \text{channel}$$
  where rows are $0..9$ (top to bottom), columns are $0..9$ (left to right), and channels are $0..5$.
- **Rationale:** This corresponds to the standard C-contiguous row-major flattening of a NumPy array of shape `(10, 10, 6)`. Preserving this exact mapping is critical because Astra's generated weights explicitly hardcode indices matching this formula (e.g., `W1[540:600:6, 0]` targeting paddle positions at row 9, channel 0).
- **Verification:** Tests in `tests/test_observations.py` assert explicit mappings for all edge positions and channels.

## 4. Authoritative Neural Network Engine
- **Decision:** Implement the primary forward pass in NumPy (`network/inference.py`), with a secondary PyTorch model (`network/architecture.py`) used strictly for cross-validation and reference consistency tests.
- **Rationale:** In Experiment 5, evaluation is purely forward inference (no backward pass, no gradients, no autograd). Pure NumPy inference provides minimal overhead, zero dependencies on heavyweight framework runtimes, and exact numerical transparency.
- **Computation Formula:**
  $$h_1 = \tanh(x \cdot W_1 + b_1)$$
  $$h_2 = \tanh(h_1 \cdot W_2 + b_2)$$
  $$\text{logits} = h_2 \cdot W_3 + b_3$$
  $$p = \text{softmax}(\text{logits})$$
  $$\text{action} \sim \text{Categorical}(p)$$

## 5. Weight Parser Security & AST Interpretation
- **Decision:** Build a safe Abstract Syntax Tree (AST) interpreter in `network/weight_parser.py` that parses the markdown `edits` block without calling Python's `eval()` or `exec()`.
- **Rationale:** Model-generated text cannot be assumed to be safe or trusted. A restricted AST interpreter that only permits targeted tensor assignments, slices, list literals, numerical constants, and basic arithmetic (`=`, `+=`, `-=`, `*=`) provides complete security against malicious or accidental arbitrary code execution.
- **Verification:** Golden test parsing Astra's `answer.md` to replicate `hand.npz` with zero discrepancies. Security tests asserting rejection of unauthorized syntax and builtins.

## 6. Evaluation Protocol Fidelity
- **Decision:** Implement the evaluation runner in `evaluation/evaluate.py` configured for exactly 1,000 episodes, sticky action probability 0.1, and 10,000 maximum steps per episode.
- **Rationale:** Matches the original paper's protocol byte-for-byte. Every evaluation run outputs detailed metrics (mean, median, standard error, max, episode length distribution) into immutable CSV and JSON records.
- **Verification:** Running the harness on the original Astra reference weights reproduces the high-performance Breakout policy.

## 7. Experimental Isolation for Gemini Argon
- **Decision:** Prepare the exact prompt in `gemini/experiment_prompt.txt` and execution tooling, but do NOT execute the generation of Gemini Argon weights during lab construction.
- **Rationale:** Strict adherence to research protocol: the infrastructure engineer (Gemini 3.8 Flash) builds and verifies the testbed; the experimental subject (Gemini Argon) will be evaluated in a clean, isolated session without leakage of Astra's responses or prior evaluation feedback.

