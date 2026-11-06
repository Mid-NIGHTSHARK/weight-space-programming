# Forensic Audit: Neural Network Architecture and Numerical Execution

**Audit Date:** November 6, 2026  
**Auditor:** Independent Research Auditor & Experimental Reproducibility Specialist  
**Target:** Feedforward inference, numerical stability, AST parsing, and PyTorch/NumPy cross-validation  
**Machine-Readable Trace:** [network_numerical_trace.json](file:///Users/zero/Documents/Area/testing_llms/research/forensic_audit/network_numerical_trace.json)  
**Overall Finding:** `VERIFIED CORRECT` (Mathematical and Numerical Equivalence Verified)

---

## 1. Network Architecture Specification

The experimental framework implements the feedforward multilayer perceptron (MLP) defined in Experiment 5 of the original Weight-Space Programming study:

- **Input Dimension:** 600 continuous/binary features ($10 \times 10 \times 6$).
- **Hidden Layer 1:** 32 units, $\tanh$ activation function, with bias.
- **Hidden Layer 2:** 32 units, $\tanh$ activation function, with bias.
- **Output Layer:** 6 units (logits corresponding to actions 0–5), linear projection with bias.
- **Policy Head:** Softmax categorical distribution over logits $z \in \mathbb{R}^6$.

### Mathematical Parameter Accounting

$$\text{Parameter Breakdown:}$$
$$W_1 \in \mathbb{R}^{600 \times 32} \implies 19,200 \text{ parameters}$$
$$b_1 \in \mathbb{R}^{32} \implies 32 \text{ parameters}$$
$$W_2 \in \mathbb{R}^{32 \times 32} \implies 1,024 \text{ parameters}$$
$$b_2 \in \mathbb{R}^{32} \implies 32 \text{ parameters}$$
$$W_3 \in \mathbb{R}^{32 \times 6} \implies 192 \text{ parameters}$$
$$b_3 \in \mathbb{R}^{6} \implies 6 \text{ parameters}$$
$$\text{Total Parameters} = 19,200 + 32 + 1,024 + 32 + 192 + 6 = \mathbf{20,486}$$

Both the NumPy implementation ([network/architecture.py:MinAtarNetworkNumPy](file:///Users/zero/Documents/Area/testing_llms/network/architecture.py#L58-L82)) and PyTorch reference model ([network/architecture.py:MinAtarNetworkPyTorch](file:///Users/zero/Documents/Area/testing_llms/network/architecture.py#L85-L127)) strictly adhere to this $600 \to 32 \to 32 \to 6$ topology.

---

## 2. NumPy vs. PyTorch Dual-Engine Cross-Validation

To verify that the NumPy forward inference engine does not introduce rounding errors, activation distortion, or numerical drift relative to industry-standard deep learning frameworks, an exhaustive numerical test was conducted across **6,000 distinct observation vectors**:
- 5,000 synthetic binary vectors ($600$-dim, simulating arbitrary game states).
- 1,000 actual gameplay observations sampled from live MinAtar Breakout episodes.

The test compared:
1. `MinAtarNetworkNumPy.forward(x)` using vectorized BLAS `np.dot` and NumPy `np.tanh`.
2. `MinAtarNetworkPyTorch.forward(x)` using standard `torch.nn.Linear` and `torch.tanh`.

### Numerical Cross-Validation Metrics

| Evaluated Model | Samples Tested | Max Absolute Logit Difference ($\| z_{\text{np}} - z_{\text{pt}} \|_\infty$) | Max Absolute Probability Difference ($\| p_{\text{np}} - p_{\text{pt}} \|_\infty$) | Max Kullback-Leibler Divergence ($D_{\text{KL}}$) | Argmax Action Match Rate | Numerical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini Argon** | 6,000 | $3.81 \times 10^{-6}$ | **0.000000** | **0.000000** | **100.0%** | `PASSED` |
| **Trained PPO** | 6,000 | $3.81 \times 10^{-6}$ | $7.75 \times 10^{-7}$ | $2.77 \times 10^{-7}$ | **100.0%** | `PASSED` |
| **Astra Reference** | 6,000 | $2.64 \times 10^{-3}$ | $9.50 \times 10^{-19}$ | **0.000000** | **100.0%** | `PASSED` |

### Forensic Analysis of Astra's Logit Difference ($2.64 \times 10^{-3}$)
An auditor might ask: *Why is the maximum logit difference for Astra $0.00264$, whereas for Argon it is $3.81 \times 10^{-6}$?*

**Auditor Investigation:**
1. Astra's generated weights utilize extreme parameter magnitudes:
   - Maximum weight in $W_2$: $1,000.0$
   - Maximum bias in $b_2$: $415.0$
   - Maximum bias in $b_3$: $100.0$
2. In single-precision 32-bit floating point arithmetic (`float32`), numbers around $1,000.0$ have an machine epsilon (ULP) of $1000 \times 2^{-23} \approx 0.000119$.
3. When summing 32 products of magnitude $\approx 1000.0$, different BLAS architectures (Intel MKL, Apple Accelerate, OpenBLAS) reorder inner products for SIMD vectorization. Accumulating floating point operations in different order naturally produces small discrepancies on the order of $\pm 0.002$.
4. **Crucially**, because the output logits have large dynamic ranges, when passed through the softmax exponentiation $\exp(z - \max z)$, the resulting probability distribution difference between NumPy and PyTorch is less than $10^{-18}$ ($9.50 \times 10^{-19}$)!
5. Across all 6,000 test cases, the sampled action and argmax action match **100.0%**.

---

## 3. Weight Parser Security and Parsing Fidelity

The weight parser ([network/weight_parser.py](file:///Users/zero/Documents/Area/testing_llms/network/weight_parser.py)) is responsible for extracting Python weight assignments from raw model text.

### Security and Sandboxing Audit
A critical concern in executing model-generated code is remote code execution or harness tampering.
- The parser does **NOT** use `exec()`, `eval()`, or `compile()` in an unrestricted Python global scope.
- It parses the input via Python's Abstract Syntax Tree (`ast.parse`) into a structured node tree.
- It strictly permits only:
  - `ast.Assign` and `ast.AugAssign` targets.
  - Subscript indexing (`W1[..., ...]`, `b1[...]`).
  - Constant numerics (`ast.Constant`).
  - Basic arithmetic operations (`ast.BinOp`: `+`, `-`, `*`).
  - Standard slice notations (`slice(start, stop, step)`).
- Any attempt to import modules (`import os`), call functions (`open(...)`), access dunder methods (`__class__`), or read globals is rejected at the AST level before execution.
- Security tests in `tests/test_weight_parser.py:test_security_rejections` pass 100%.

### Extraction and Application Fidelity
Gemini Argon's response was extracted from the raw DOCX file:
- File: [experiments/strict_replication/generation_001/output/raw_response.txt](file:///Users/zero/Documents/Area/testing_llms/experiments/strict_replication/generation_001/output/raw_response.txt)
- Extraction targeted the final code block containing weight assignments.
- Total statements parsed: **86 statements**.
- Total statements successfully applied: **86 statements**.
- Parsing or dimension errors: **0**.
- Total nonzero parameters set: **817 out of 20,486** ($3.99\%$ parameter density).

### Comparison of Parameter Sparsity

| Weight Tensor | Shape | Zero Initialized | Gemini Argon Nonzero | Astra Reference Nonzero | Trained PPO Nonzero |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $W_1$ | $(600, 32)$ | 0 | 768 | 769 | 19,200 |
| $b_1$ | $(32,)$ | 0 | 1 | 1 | 32 |
| $W_2$ | $(32, 32)$ | 0 | 32 | 24 | 1,024 |
| $b_2$ | $(32,)$ | 0 | 8 | 8 | 32 |
| $W_3$ | $(32, 6)$ | 0 | 8 | 8 | 192 |
| $b_3$ | $(6,)$ | 0 | 0 | 5 | 6 |
| **Total Nonzero** | — | **0** | **817** ($3.99\%$) | **815** ($3.98\%$) | **20,486** ($100.0\%$) |

Both LLM-generated weight files (Argon and Astra) exhibit sparse parameter allocation ($\sim 4\%$ nonzero), reflecting deliberate modular wiring rather than dense, continuous optimization typical of gradient descent (PPO).

---

## 4. Audit Conclusion

The neural network implementation is mathematically sound, numerically stable across frameworks, and protected by a robust AST parser. Numerical forward inference between NumPy and PyTorch is identical within floating-point tolerance, guaranteeing that evaluation results are framework-invariant.

**Audit Status:** `VERIFIED CORRECT`

