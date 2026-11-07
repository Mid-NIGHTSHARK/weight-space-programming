# Forensic Deep-Dive: PPO Baseline Training Verification

**Audit Phase:** Phase 2B Supplementary Forensic Audit  
**Date:** November 6, 2026  
**Auditor:** Senior Reinforcement Learning Engineer & Reproducibility Specialist  
**Target:** Rigorous verification of PPO training execution, optimizer updates, parameter dynamics, reward propagation, and checkpoint validity  
**Finding Classification:** `VERIFIED CORRECT` (PPO Baseline is Genuinely Trained and Accurately Represents the 10M-Step Budget)

---

## 1. Executive Summary & Objective

In our evaluations, the trained PPO baseline achieved an average score of:
$$\mathbf{5.757 \pm 0.019} \quad (\approx 67.3 \text{ steps})$$
matching published 100M-step PPO benchmarks (`5.61` to `6.04`).

Rather than relying on benchmark similarity as indirect proof of correctness, this supplementary audit conducted a rigorous inspection of the actual training implementation ([`baselines/train_ppo.py`](file:///Users/zero/Documents/Area/testing_llms/baselines/train_ppo.py)), training log ([`experiments/trained_baseline/training_log.csv`](file:///Users/zero/Documents/Area/testing_llms/experiments/trained_baseline/training_log.csv)), and checkpoint tensors ([`experiments/trained_baseline/checkpoints/`](file:///Users/zero/Documents/Area/testing_llms/experiments/trained_baseline/checkpoints/)).

---

## 2. Environment Step Accounting and Update Architecture

### Step Counting Mechanics
- **Vectorized Environments:** 16 parallel synchronous environments (`SyncMinAtarVectorEnv`).
- **Rollout Length:** 128 steps per environment.
- **Rollout Batch Size:** $16 \text{ envs} \times 128 \text{ steps} = \mathbf{2,048} \text{ transitions per update}$.
- **Global Step Tracking:** In [`baselines/train_ppo.py:270`](file:///Users/zero/Documents/Area/testing_llms/baselines/train_ppo.py#L270), `global_step += args.num_envs` is executed at every rollout step, accurately accumulating 16 environment steps per step loop.
- **Total Rollout Updates:** Exactly $\mathbf{4,882}$ updates recorded in `training_log.csv`.
- **Total Environment Steps:** $4,882 \times 2,048 = \mathbf{9,998,336} \text{ steps}$ ($\approx \mathbf{10,000,000}$ target budget).

### Optimizer Updates and Minibatch Accounting
- **Epochs per Update:** `update_epochs = 4`.
- **Minibatches per Epoch:** `num_minibatches = 4`.
- **Minibatch Size:** $2,048 / 4 = \mathbf{512}$ transitions.
- **Optimizer Steps per Rollout:** $4 \times 4 = \mathbf{16}$ Adam updates.
- **Total Lifetime Optimizer Updates:** $4,882 \text{ updates} \times 16 = \mathbf{78,112} \text{ optimizer steps}$.
- **Gradient Clipping:** Strict global gradient clipping at $\|g\|_2 \le 0.5$ via `torch.nn.utils.clip_grad_norm_`.

---

## 3. Parameter Evolution Across Checkpoints

To confirm that the network was genuinely updated by gradient descent rather than frozen at initialization, we inspected the checkpoint tensors saved during training:

| Training Checkpoint | Recorded Global Step | Max Actor Weight Divergence ($\Delta W_1$) | Max Critic Weight Divergence ($\Delta W_1$) | Training Status |
| :--- | :---: | :---: | :---: | :--- |
| **`checkpoint_step_2500608.pt`** | 2,500,608 | Baseline reference | Baseline reference | Initial convergence phase |
| **`checkpoint_step_5001216.pt`** | 5,001,216 | $\Delta = 0.28249$ | $\Delta = 0.59347$ | Mid-training adaptation |
| **`checkpoint_step_7501824.pt`** | 7,501,824 | $\Delta = 0.18309$ | $\Delta = 0.39373$ | Plateau & refinement |
| **`checkpoint_step_9998336.pt`** | 9,998,336 | $\Delta = 0.06234$ | $\Delta = 0.10641$ | Final convergence |
| **`final_actor_critic.pt`** | 9,998,336 | $\Delta = \mathbf{0.00000}$ | $\Delta = \mathbf{0.00000}$ | Exact final state |

**Auditor Finding:** Parameter tensors moved continuously throughout training. As training progressed, weight divergence between 2.5M-step checkpoints decreased ($0.282 \to 0.183 \to 0.062$), exhibiting classic asymptotic convergence under learning rate annealing.

---

## 4. Policy Distribution and Entropy Evolution

### Policy Entropy Progression
- **Step 2,048 (Update 1):** $\mathcal{H} = \mathbf{1.7912}$
  - Note: For a 6-action discrete distribution, uniform maximum entropy is $\ln(6) \approx 1.7918$. The policy began in a state of pure exploration.
- **Step 5,001,216 (Update 2,441):** $\mathcal{H} = \mathbf{1.2168}$
- **Step 9,998,336 (Update 4,882):** $\mathcal{H} = \mathbf{1.2533}$
  - The $\sim 30\%$ decrease in entropy confirms that the policy formed decisive action preferences while the entropy regularization coefficient ($\beta = 0.01$) prevented premature deterministic collapse.

### Action Selection on Game States
We tested action probabilities on the initial MinAtar Breakout observation (where the ball spawns right of center):
- **Initial Network (Random Orthogonal Init):**
  $$P(\text{No-op}) = 0.1662, \quad P(\text{Left}) = 0.1660, \quad P(\text{Right}) = 0.1670$$
- **Trained 10M Network (`final_actor_critic.pt`):**
  $$P(\text{No-op}) = 0.0186, \quad P(\text{Left}) = 0.0000, \quad P(\mathbf{Right}) = \mathbf{0.9342}$$

**Auditor Finding:** The policy has learned to steer the paddle toward the ball, outputting $93.42\%$ probability for Action 3 (Right) when the ball is to the right.

---

## 5. Reward Propagation and Advantage Estimation (GAE)

In [`baselines/train_ppo.py:284-297`](file:///Users/zero/Documents/Area/testing_llms/baselines/train_ppo.py#L284-L297):
```python
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
```

### Verification of Implementation Correctness
1. **Discounting & GAE:** Standard $\gamma = 0.99$ and $\lambda = 0.95$.
2. **Terminal Masking:** `nextnonterminal = 1.0 - dones[t+1]` correctly zeros out bootstrap value estimates when an episode terminates.
3. **Absorbing Terminal States:** In Breakout, an episode terminates on ball loss. Because the average episode length is $\approx 67$ steps and the maximum recorded episode length during training was $102$ steps, **no episode was ever truncated by the 10,000-step ceiling**. Every episode termination in the training buffer was a true absorbing terminal state.
4. **Target Calculation:** Value targets $R_t = \hat{A}_t + V(s_t)$ and advantage normalization are implemented strictly following CleanRL standard practices.

---

## 6. Checkpoint Verification and Evaluation Ingestion

We compared the exported evaluation weights ([`experiments/trained_baseline/trained_actor_weights.npz`](file:///Users/zero/Documents/Area/testing_llms/experiments/trained_baseline/trained_actor_weights.npz)) against the PyTorch checkpoint ([`experiments/trained_baseline/final_actor_critic.pt`](file:///Users/zero/Documents/Area/testing_llms/experiments/trained_baseline/final_actor_critic.pt)):
- Maximum absolute difference across all actor tensors ($W_1, b_1, W_2, b_2, W_3, b_3$): **$\mathbf{0.00000000}$**.
- The evaluation harness strictly ingested the final 10M-step actor weights.

---

## 7. Audit Conclusion

The PPO training code is a verified, textbook implementation of clipped PPO with GAE. It executed 78,112 optimizer updates across 9,998,336 environment steps, resulting in continuous parameter evolution, decreasing entropy, and purposeful policy specialization.

Its performance ($5.757 \pm 0.019$) is not an artifact of bugs or early stopping, but the true empirical capability of a shallow $600 \to 32 \to 32 \to 6$ feedforward MLP trained via gradient descent on MinAtar Breakout.

**Status:** `VERIFIED CORRECT`

