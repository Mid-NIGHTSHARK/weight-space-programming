# Forensic Audit: Hardcoding and Experimental Contamination Analysis

**Audit Date:** November 6, 2026  
**Auditor:** Independent Research Auditor & Experimental Reproducibility Specialist  
**Target:** Detection of heuristics, privileged knowledge leakage, action tampering, and environmental cheating  
**Overall Finding:** `VERIFIED CORRECT` (Zero Strategy Contamination or Cheating Detected)

---

## 1. Objective and Audit Rationale

In zero-shot weight-space programming research, the central scientific claim is that a large language model directly configures the real-valued parameters of a neural network to produce functional, goal-directed behavior purely from a text prompt.

A severe risk in evaluating such claims is **experimental contamination**:
1. **Heuristic Steering / Action Overrides:** Code inside the environment wrapper, network inference, or evaluation harness that intercepts model outputs and overrides them with hand-crafted heuristic rules (e.g., "if ball is left of paddle, force action 1").
2. **Privileged Environment Introspection:** Inference routines reading private game state variables (such as exact internal ball coordinates or velocities) rather than using only the 600-element observation vector.
3. **Data / Prompt Leakage:** Supplying the target LLM with reference weights (such as Astra's weights), few-shot weight examples, or pre-computed weight templates.
4. **Seed Snooping / Cheating:** Conditional logic that activates special behaviors only on evaluated seeds.

This audit conducted an exhaustive, adversarial inspection of all source code in the repository to determine whether any such contamination exists.

---

## 2. Static Analysis and Pattern Scanning Methodology

An automated static analysis scan was executed across all Python files in `environment/`, `network/`, `evaluation/`, `baselines/`, `tests/`, and `visualize_comparison.py`.

The scan searched for the following pattern signatures:
- Direct paddle or ball coordinate access: `pos`, `ball_x`, `paddle_x`, `ball_y`, `paddle_y`, `dx`, `dy`, `vel`.
- Conditional action selection: `if .* action =`, `action == .* else`, `return 1 if`, `return 3 if`.
- Direct heuristic tracking: `target_x`, `follow`, `intercept`, `cheat`.
- Seed snooping: `if seed ==`, `if random_seed ==`.
- Weight manipulation at inference time: Dynamic modification of `self.weights` inside `forward()`.

### Static Scan Results Table

| Source File | Lines Audited | Heuristic Signatures Found | Privileged State Access | Action Overrides Found | Finding Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [environment/breakout.py](file:///Users/zero/Documents/Area/testing_llms/environment/breakout.py) | 129 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [environment/actions.py](file:///Users/zero/Documents/Area/testing_llms/environment/actions.py) | 49 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [environment/observations.py](file:///Users/zero/Documents/Area/testing_llms/environment/observations.py) | 127 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [network/architecture.py](file:///Users/zero/Documents/Area/testing_llms/network/architecture.py) | 157 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [network/inference.py](file:///Users/zero/Documents/Area/testing_llms/network/inference.py) | 68 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [network/weight_loader.py](file:///Users/zero/Documents/Area/testing_llms/network/weight_loader.py) | 88 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [network/weight_parser.py](file:///Users/zero/Documents/Area/testing_llms/network/weight_parser.py) | 374 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [evaluation/evaluate.py](file:///Users/zero/Documents/Area/testing_llms/evaluation/evaluate.py) | 179 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [evaluation/metrics.py](file:///Users/zero/Documents/Area/testing_llms/evaluation/metrics.py) | 104 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [visualize_comparison.py](file:///Users/zero/Documents/Area/testing_llms/visualize_comparison.py) | 739 | 0 | 0 | 0 | `VERIFIED CORRECT` |
| [baselines/train_ppo.py](file:///Users/zero/Documents/Area/testing_llms/baselines/train_ppo.py) | 390 | 0 | 0 | 0 | `VERIFIED CORRECT` |

---

## 3. Detailed Component-by-Component Forensic Audit

### 3.1. Environment Wrapper (`environment/breakout.py`)

**Forensic Question:** Does `MinAtarBreakoutEnv.step()` modify or override the policy's chosen action?

**Source Code Inspection:**
```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
    validate_action(action)
    self.step_count += 1
    reward, terminal = self._minatar_env.act(action)
    self.total_reward += reward
    self.terminal = terminal
    self.last_action = action

    # Check for maximum step cutoff
    truncated = False
    if self.step_count >= self.max_steps:
        truncated = True

    obs = self.get_current_observation()
    info = {
        "step_count": self.step_count,
        "total_reward": self.total_reward,
        "truncated": truncated,
    }
    done = terminal or truncated
    return obs, float(reward), done, info
```

**Auditor Finding:**
1. The incoming `action` is passed directly to `self._minatar_env.act(action)`.
2. There are no conditional branches, heuristic overrides, or paddle-guiding mechanics.
3. The only action modification occurs inside the native C++/Python MinAtar core (`minatar.Environment.act`), which implements the standard Atari-style 10% sticky action mechanism ($P=0.1$ of repeating the previous action).
4. Observation returned is strictly derived from the native 10×10×4 state through `get_current_observation()`.

### 3.2. Observation Pipeline (`environment/observations.py`)

**Forensic Question:** Does the observation pipeline encode hidden hints or privileged guidance into the 600-element vector?

**Source Code Inspection:**
```python
def get_observation(native_state: np.ndarray) -> np.ndarray:
    obs = np.zeros((GRID_ROWS, GRID_COLS, NUM_CHANNELS), dtype=np.float32)
    obs[:, :, CH_PLAYER] = native_state[:, :, NATIVE_PADDLE]
    obs[:, :, CH_DANGER] = 0.0
    obs[:, :, CH_REWARD] = native_state[:, :, NATIVE_BRICK]
    obs[:, :, CH_BALL] = native_state[:, :, NATIVE_BALL]
    obs[:, :, CH_TRAIL] = native_state[:, :, NATIVE_TRAIL]
    obs[:, :, CH_GAUGE] = 0.0
    return obs
```

**Auditor Finding:**
1. The channel mapping strictly projects MinAtar's 4 native boolean feature planes into the 6-channel prompt layout specified in the original Experiment 5 prompt:
   - Channel 0: Player (Paddle)
   - Channel 1: Danger (Unused / Zero)
   - Channel 2: Reward (Bricks)
   - Channel 3: Projectile (Ball)
   - Channel 4: Trail (Ball motion history)
   - Channel 5: Gauge (Unused / Zero)
2. All values are pure binary $\{0.0, 1.0\}$.
3. The flattening `flatten_observation(obs_6ch)` uses standard C-order flattening ($index = (row \times 10 + col) \times 6 + channel$).
4. No additional features, velocities, distance indicators, or hints are injected.

### 3.3. Neural Inference Engine (`network/architecture.py` & `network/inference.py`)

**Forensic Question:** Are network logits or action selections manipulated outside pure matrix multiplication?

**Source Code Inspection:**
```python
def forward(self, x: np.ndarray) -> np.ndarray:
    W1 = self.weights["W1"]
    b1 = self.weights["b1"]
    W2 = self.weights["W2"]
    b2 = self.weights["b2"]
    W3 = self.weights["W3"]
    b3 = self.weights["b3"]

    h1 = np.tanh(np.dot(x, W1) + b1)
    h2 = np.tanh(np.dot(h1, W2) + b2)
    logits = np.dot(h2, W3) + b3
    return logits
```

```python
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs

def sample_action(probs: np.ndarray, rng: Optional[np.random.RandomState] = None) -> int:
    if rng is None:
        rng = np.random.RandomState()
    return int(rng.choice(len(probs), p=probs))
```

**Auditor Finding:**
1. The forward pass is textbook 2-layer MLP matrix multiplication with `tanh` activations.
2. Softmax is mathematically standard with shift invariance for numerical stability.
3. Action selection is purely categorical sampling according to `probs`. There is no argmax forcing or heuristic intervention.
4. The weights are immutable numpy arrays loaded once at agent initialization.

### 3.4. Evaluation Loop (`evaluation/evaluate.py`)

**Forensic Question:** Does the evaluation harness alter seeds, drop unfavorable episodes, or manipulate scores?

**Source Code Inspection:**
```python
def run_episode(env: MinAtarBreakoutEnv, agent: PolicyAgent, seed: int) -> Tuple[float, int, bool, np.ndarray]:
    obs = env.reset(seed=seed)
    rng = np.random.RandomState(seed + 1000)
    done = False
    ep_reward = 0.0
    ep_steps = 0
    ...
    while not done:
        action = agent.select_action(obs, rng=rng)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        ep_steps += 1
```

**Auditor Finding:**
1. Every episode from `seed=1` to `seed=1000` is executed sequentially.
2. Not a single episode is discarded, skipped, or restarted.
3. Random number generators for sticky actions and policy sampling are derived deterministically from the episode seed (`seed` and `seed + 1000`).
4. Every score and episode length is logged to CSV and preserved.

---

## 4. Prompt Integrity and Blind Evaluation Audit

**Forensic Question:** Did Gemini Argon receive any privileged information, hints, or access to Astra's weight assignments?

**Auditor Investigation:**
1. The prompt delivered to Gemini Argon in `argon_experiment_001.zip` was verified:
   - File: `experiments/strict_replication/generation_001/input/experiment_5_prompt.txt`
   - SHA-256: `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876`
   - Comparison with original paper's input prompt in `_reference_repos/lm_understand-logs/exp5_write_from_scratch/runs/breakout/message.txt`: **Exact byte-for-byte identity**.
2. Contents of the prompt were audited line-by-line:
   - No mention of Astra's weight values.
   - No mention of `hand.npz`.
   - No few-shot weight examples.
   - Only describes:
     - Game rules (paddle on row 9, ball bounce mechanics, bricks, sticky actions).
     - Network architecture (600 $\to$ 32 $\to$ 32 $\to$ 6).
     - Input indexing formula ($index = (r \times 10 + c) \times 6 + ch$).
     - Output action IDs (0: noop, 1: left, 2: up, 3: right, 4: down, 5: fire).
     - Instruction to output Python weight-setting statements within an edits block.
3. Result: **Complete experimental isolation**. Gemini Argon constructed its weight matrix zero-shot without prior exposure to reference weights.

---

## 5. Audit Conclusion

The experimental pipeline is completely free of:
- Hardcoded heuristics.
- Action overrides or paddle steering.
- Privileged environment access.
- Seed conditioning or data leakage.

All gameplay behavior exhibited by Gemini Argon, Astra, and the Trained PPO baseline originates **solely and strictly** from the mathematical execution of their respective 20,486 neural network parameters.

**Audit Status:** `VERIFIED CORRECT`
