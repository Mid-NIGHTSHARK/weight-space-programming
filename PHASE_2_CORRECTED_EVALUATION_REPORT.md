# Phase 2 Corrected Evaluation Report

**Project:** MinAtar Breakout, Experiment 5 reproduction
**Baseline commit:** `8c15d156798bffb1befc7e85c06870c4981f9a72`
**Corrected evaluator commit:** `b8be6dc0134b80d6fefb69005ff364f799d1d793`
**Corrected analysis commit:** `3fa7ba5c786454f6bd71f375e7bf952edd9870a6`
**Evaluation completed:** 2026-11-06 15:32 UTC
**Analysis completed:** 2026-11-06 15:35 UTC

## Scope and evidence labels

This report describes Phase 2 changes and results. “Verified” means reproduced from repository files, source, recorded traces, or executed checks. “Interpretation” is a conclusion drawn from those facts. “Hypothesis” is a proposed causal account that the available evidence does not establish.

The repository did not contain a file named `PHASE_1_INDEPENDENT_FORENSIC_AUDIT.md`. Before implementation, I read the corresponding repository audit at `research/forensic_audit/FORENSIC_AUDIT_REPORT.md`, together with the related Phase 2B/2C audit material and the Phase 1 integration report. Those reports were treated as leads to verify, not as authority. Older reports and historical artifacts remain preserved as dated records.

## What Phase 1 identified and what changed

The audit raised concerns about using the same episode seed for policy sampling and environment randomness, matched comparisons analyzed as independent samples, hard-coded visualizer metrics, inaccurate causal explanations for several policies, Sonnet response-size metadata, and stale provenance descriptions.

Phase 2 adds a new evaluation path and dataset. It does not edit or regenerate any primary zero-shot weights, original prompts, raw responses, original manifests, historical CSVs, historical summaries, or prior forensic audit reports. Historical evaluation code remains callable as the legacy same-seed implementation. `experiments/corrected_evaluation/preserved_artifacts_sha256.json` records 180 pre-Phase-2 files; tests re-hash these files to guard against changes.

The correction to Sonnet's response length is additive: the file is 3,355 bytes with SHA-256 `7b6a0461c2f775cc08604c65672f145aa3efc1adbb5ce51ac59244e036cb85ee`. Its original manifest says 3,816 bytes. The original manifest is preserved; the Phase 2 artifact registry records the observed size and hash. The pre-existing forensic hash registry already recorded 3,355 bytes. Luna's committed raw response remains the exact baseline version: 3,168 bytes, SHA-256 `e037cc069e092820da6e0f08e54a7fad82af4eabe396695b559311bac068601c`.

## Random-number architecture

For each recorded base episode seed, the corrected runner uses:

```text
root = numpy.random.SeedSequence(base_episode_seed)
environment_child, policy_child = root.spawn(2)
environment_seed = environment_child.generate_state(1, dtype=uint32)[0]
policy_seed      = policy_child.generate_state(1, dtype=uint32)[0]
```

The environment receives `environment_seed`; the categorical policy sampler receives a separate `numpy.random.RandomState(policy_seed)`. The environment and policy therefore have distinct PRNG states, derived with distinct SeedSequence child spawn keys. A policy draw cannot advance the environment state, and an environment transition cannot advance the policy state. These are reproducible pseudo-random streams, not claims of mathematical independence from deterministic seed derivation.

Each episode CSV records the base, environment, and policy seeds. Episodes also store a trajectory digest, selected action counts, and executed action counts after sticky-action handling. Replaying a record re-derives and checks the seed pair. The corrected path retains temperature-1 softmax categorical sampling, MinAtar sticky-action probability 0.1, reset last action 0, MinAtar Breakout 1.0.15, and the 10,000-step limit. Game mechanics are unchanged.

The historical evaluator remains separate and still seeds its environment and policy sampler with the same episode integer. The two protocols are distinguishable in run manifests and in the visualizer. The corrected protocol is not claimed to improve policy quality; it changes random-number coupling and therefore can change episode outcomes.

## Corrected evaluation protocol and execution

The corrected configuration is `experiments/corrected_evaluation/evaluation_config.json`. Base seed groups are 42, 100, and 200; the existing deterministic per-group base-seed vectors were retained and the same 3,000 base seeds and derived seed triplets were used for every condition. Each condition has 1,000 episodes per group, 3,000 total.

Five primary zero-shot conditions were evaluated: Gemini Argon 4 Medium, GPT-6 Original, Claude Sonnet 5.5, DeepSeek Original, and GPT-6 Luna Extra High. Astra local reference weights, trained PPO, random-action, and zero-weight controls are separate comparison conditions. DeepSeek Corrected is a post-hoc researcher intervention and is excluded from this dataset and its primary ranking.

Before the full run, the configuration-locked pilot ran 10 episodes per condition across the three groups for all nine conditions: 90 episodes total. It passed 27/27 recorded replay checks (one checked episode per condition and group). The pilot is software validation only and is not pooled with the full results. A first analysis attempt after the full run failed before writing analysis artifacts due to a condition-registry key mismatch; the failure is recorded in `experiments/corrected_evaluation/full_run/analysis/analysis_attempt_log.json`. The CSVs were not edited, resumed, or regenerated; analysis succeeded after correcting the reader.

The final full run has 27,000 episodes and is stored separately under `experiments/corrected_evaluation/full_run/`. All nine conditions have 3,000 episodes. Run manifest status is `complete`, `resumed` is false, and the run is a fresh start. The evaluation source commit is `b8be6dc0134b80d6fefb69005ff364f799d1d793`; analysis source commit is `3fa7ba5c786454f6bd71f375e7bf952edd9870a6`. Python was 3.12.15; MinAtar 1.0.15; NumPy 2.5.4; pandas 3.1.1; SciPy 1.18.1; PyTorch 2.15.0; pytest 9.2.0; Matplotlib 3.11.2. The run manifest contains the exact configuration, registry, seed vectors, environment and policy seeds, run state, runtime versions, and hashes of the episode CSVs.

Phase 2 work is on the dedicated local branch `phase-2-corrected-evaluation`, based on the requested baseline. The evaluation outputs were committed as `03517f3db42385237d0356d2387a3363d908ff83`; visualizer integrity changes were committed as `b76517f2a2db237c9d776a572708c09b2b033a8e`; the subsequent collision-rule wording refinement is `d6847b167b0f5c3b218b80757d2717731690bb95`; and the explicit sticky-action regression test is `d7593414398c237286939ff015d22348f4357dbd`. Earlier implementation commits are `c1c5f7e` (independent RNG evaluator), `b8be6dc` (matched analysis and configured runner), and `3fa7ba5` (analysis registry-key fix). No commit was pushed.

## Corrected results

The table below is the corrected dataset. “SE” is the standard error of the mean. The confidence interval is a two-sided Student-t interval for the episode mean. Action distributions are available in `condition_metrics.json`; selected and executed actions are recorded separately.

| Rank | Condition | Episodes | Mean | SD | SE | 95% CI | Median | Min–max | Mean length | Median length | Max length | Cutoffs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Astra local reference | 3,000 | 504.852 | 243.575 | 4.447 | [496.133, 513.572] | 642 | 0–824 | 7,199.46 | 9,821 | 10,000 | 1,484 (49.47%) |
| 2 | Gemini Argon 4 Medium | 3,000 | 60.322 | 65.458 | 1.195 | [57.979, 62.665] | 38 | 0–475 | 914.85 | 558 | 7,910 | 0 |
| 3 | Claude Sonnet 5.5 | 3,000 | 10.587 | 11.284 | 0.206 | [10.183, 10.991] | 6 | 0–91 | 124.43 | 70 | 1,096 | 0 |
| 4 | Trained PPO baseline | 3,000 | 5.797 | 0.954 | 0.017 | [5.763, 5.831] | 6 | 0–7 | 67.77 | 70 | 80 | 0 |
| 5 | GPT-6 Original | 3,000 | 2.516 | 1.980 | 0.036 | [2.445, 2.587] | 2 | 0–17 | 32.72 | 28 | 178 | 0 |
| 6 | GPT-6 Luna Extra High | 3,000 | 1.230 | 1.270 | 0.023 | [1.185, 1.276] | 1 | 0–9 | 18.69 | 16 | 104 | 0 |
| 7 | Zero-weight control | 3,000 | 0.518 | 0.755 | 0.014 | [0.491, 0.545] | 0 | 0–5 | 11.29 | 6 | 60 | 0 |
| 8 | DeepSeek Original | 3,000 | 0.493 | 0.500 | 0.009 | [0.475, 0.511] | 0 | 0–1 | 10.93 | 6 | 16 | 0 |
| 9 | Random-action control | 3,000 | 0.492 | 0.721 | 0.013 | [0.466, 0.518] | 0 | 0–7 | 11.01 | 6 | 80 | 0 |

The five primary zero-shot models retain the historical order: Argon, Sonnet, GPT-6 Original, Luna, DeepSeek. Across all comparison conditions, zero-weight control ranks just above DeepSeek and random control just below it. That small ordering is not evidence of a meaningful separation: DeepSeek minus random paired mean difference is 0.001 score, 95% paired bootstrap CI [-0.028, 0.030], permutation p=0.9633 and Holm-adjusted p=0.9633.

### Selected action distributions

Counts below are selected action requests over all corrected episodes, in action order `no-op / left / up / right / down / fire`. They are not executed-action counts. Sticky-action handling can replace a selected action; the CSV and analysis JSON preserve both views.

| Primary model | Selected-action counts (no-op / left / up / right / down / fire) |
|---|---|
| Gemini Argon 4 Medium | 282,620 / 1,230,328 / 0 / 1,231,592 / 0 / 0 |
| GPT-6 Original | 58,614 / 19,096 / 2 / 20,440 / 1 / 3 |
| Claude Sonnet 5.5 | 33,165 / 169,425 / 0 / 170,694 / 0 / 0 |
| DeepSeek Original | 0 / 0 / 10,976 / 0 / 10,900 / 10,904 |
| GPT-6 Luna Extra High | 14,328 / 20,141 / 3 / 21,580 / 6 / 14 |

For the historical Sonnet CSVs, the selected counts were 34,413 / 175,322 / 0 / 176,847 / 0 / 0. The earlier action table was wrong; these are the corrected counts read directly from the original three episode CSVs. Historical CSVs do not record post-sticky executed actions, so no historical executed-action counts are inferred.

## Matched statistical comparisons

The analysis keeps each condition's episode paired by seed group, episode index, and the identical base/environment/policy seed triplet. It reports the mean within-pair score difference and a percentile bootstrap confidence interval by resampling whole episode pairs (9,999 replicates). The paired test randomly flips the sign of each within-pair difference (19,999 sign assignments sampled) and tests the absolute mean difference. Holm's step-down procedure adjusts all 36 pairwise p-values. The paired Cohen *dz* is the mean difference divided by the sample standard deviation of paired differences. Small p-values are reported as `5e-05`, the minimum resolution of this Monte Carlo procedure, never as zero.

This approach preserves the matched design; it does not use independent-sample Welch or Mann–Whitney tests. The sign-flip test assumes within-pair exchangeability under its null. The bootstrap intervals describe uncertainty across the sampled seed pairs. These inferences are conditional on the chosen deterministic seed-generation procedure and do not turn this one environment/configuration into evidence of broad model capability.

As one example, Argon minus Sonnet is 49.735 points (95% paired bootstrap CI [47.391, 52.041], paired Cohen *dz*=0.764, raw p=5e-05, Holm-adjusted p=0.0018). DeepSeek versus random has no evidence of a difference under this procedure. Nonsignificance is not evidence that the policies are equivalent.

No confirmatory equivalence test is conducted. The earlier ±25-point Astra margin was chosen post hoc; that exploratory result remains historical and is not promoted to a confirmatory claim. Full pairwise results and method metadata are in `paired_comparisons.json`.

## Historical comparison and ranking changes

The old evaluation remains an independent historical dataset and was not overwritten. Its episode counts differed: most generated policies and PPO had 3,000 episodes; random and zero controls had 1,000; local Astra had 4,000 (seed groups 42, 100, 200, 300). The corrected comparison uses 3,000 episodes for every condition. The change table below uses matched old/new seed vectors; Astra therefore uses only its matched 3,000 historical episodes, not the fourth group.

| Condition | Historical matched mean | Corrected mean | Mean paired change (corrected − historical) | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| Gemini Argon 4 Medium | 57.866 | 60.322 | +2.456 | [-0.678, 5.555] |
| GPT-6 Original | 2.493 | 2.516 | +0.023 | [-0.075, 0.121] |
| Claude Sonnet 5.5 | 10.957 | 10.587 | -0.371 | [-0.960, 0.232] |
| DeepSeek Original | 0.484 | 0.493 | +0.008 | [-0.017, 0.034] |
| GPT-6 Luna Extra High | 1.253 | 1.230 | -0.023 | [-0.087, 0.040] |
| Astra local reference, matched 3,000 | 507.459 | 504.852 | -2.606 | [-14.961, 9.639] |
| Trained PPO baseline | 5.757 | 5.797 | +0.040 | [-0.011, 0.090] |
| Random control, matched 1,000 | 0.485 | 0.481 | -0.004 | [-0.072, 0.064] |
| Zero-weight control, matched 1,000 | 0.517 | 0.528 | +0.011 | [-0.057, 0.078] |

The primary zero-shot ranking did not change. The all-condition ordering at the bottom changed from zero, random, DeepSeek to zero, DeepSeek, random. DeepSeek and random remain statistically indistinguishable in the matched test above. The matched mean changes are descriptive of the two protocols together: they combine RNG decoupling with different deterministic draws. They do not isolate random-stream coupling as the sole cause of any score change.

### Astra 800 versus 814 discrepancy

The checked-in paper/reference result at `_reference_repos/lm_understand-logs/exp5_write_from_scratch/eval/minatar_eval.json` reports 1,000 episodes, maximum 800, mean 517.411. The local historical Astra CSVs contain 4,000 episodes and maximum 814 (the value occurs in `run_seed100`). These are different samples, not conflicting values for the same sample. Corrected local Astra is another distinct 3,000-episode sample, maximum 824. The visualizer labels the local historical and corrected reevaluations separately and does not present either as the paper's published sample.

## Revised interpretation of policy mechanisms

The trajectory file `experiments/corrected_evaluation/failure_trace_replays.json` independently replays one historical seed-42, group-42 episode for each requested model and verifies score and length against its original CSV. It records ball/trail/paddle positions, selected and executed actions, sticky overrides, attempted and resolved transitions, collision/loss state, and logits. These five examples explain specific transitions only; they are not a representative or causal sample of aggregate failure.

MinAtar 1.0.15 moves the paddle first. It computes the ball's next x, clamps that coordinate at a side wall, then handles a descending ball if its current x equals the updated paddle position **or** the clamped next x equals that position. Otherwise the ball is lost. The trace records the attempted coordinate before clamping as well as the resulting state (for example, x=10 becomes x=9). A single projected target is not the collision rule.

| Model | Replayed historical episode | What the transition demonstrates |
|---|---|---|
| GPT-6 Original | Score 0, 6 steps. At final state ball (5,8), trail (4,7), paddle 4; selected/executed no-op; attempted (6,9); loss. | Paddle stayed at 4 while current/next x were 5/6. This sample contains no wall bounce. A general wall-reflection explanation is not demonstrated. |
| Gemini Argon | Score 128, 1,548 steps. Ball (7,8), trail (6,7), paddle 5; selected/executed right moves paddle to 6; attempted (8,9); loss. | Neither current x=7 nor attempted next x=8 equals paddle 6, so the environment loses the ball. This confirms the collision rule and one miss, not the aggregate cause of Argon's score. |
| Claude Sonnet | Score 15, 170 steps. Ball (9,8), trail (8,7), paddle 6; selected/executed right moves paddle to 7; attempted x=10 is wall-clamped to 9; terminal loss. | At this terminal transition the ball and trail coincide at x=9. The last selected action was right, not no-op; no following action is taken after termination. This disproves the previous claim that coincident ball/trail after a wall bounce caused a no-op in this trace. No replacement aggregate cause is established. |
| DeepSeek Original | Score 0, 6 steps. Ball (5,8), trail (4,7), paddle 4; selected/executed down; attempted (6,9); loss. | The paddle does not move. The logits in the trace are roughly +1,724 for Up, Down, and Fire, and historical/corrected action distributions show those inert actions dominate. This supports saturation for this specific policy/environment, not for DeepSeek generally. |
| GPT-6 Luna Extra High | Score 1, 16 steps. Ball (4,8), trail (5,7), paddle 7; selected/executed left moves paddle to 6; attempted (3,9); loss. | Neither current x=4 nor attempted next x=3 equals paddle 6. This verifies one miss but does not establish a mechanism for aggregate performance. |

For Luna, tensor inspection gives the following exact nonlinear computation for the active coordinate units:

```text
u(c) = 0.08 * (c - 4.5)
h_ball   = tanh(u(ball_x))
h_trail  = tanh(u(trail_x))
h_paddle = tanh(u(paddle_x))
h        = tanh(12*h_ball - 6*h_trail - 6*h_paddle)
logits = [1.5, -4*h, -6, 4*h, -6, -6]
          # [no-op, left, up, right, down, fire]
```

The expression `2*ball_x - trail_x - paddle_x` arises only by linearizing the first-layer `tanh` terms in the hidden preactivation: `12*u(ball_x) - 6*u(trail_x) - 6*u(paddle_x) = 0.48*(2*ball_x - trail_x - paddle_x)`. The exact hidden activation applies `tanh` at both hidden layers. The equation does not itself establish how a full episode ends or explain aggregate score.

## Visualizer and provenance safeguards

`visualize_comparison.py` now loads the corrected, complete dataset by default. `--dataset original` selects the historical CSVs. It shows the dataset/protocol, dynamically calculates historical summary values from hash-verified CSVs, and reads corrected metrics from the completed run artifact. There is no fixed-score fallback. Missing metrics, mismatched metric hashes, changed configuration/condition registry hashes, and canonical weight mismatches raise explicit errors. A displayed custom file receives no canonical score unless its SHA-256 matches the evaluated weight artifact. DeepSeek Corrected is labeled as a researcher intervention, never enters the zero-shot ranking, and receives no primary benchmark score.

The visualizer's benchmark summary distinguishes Astra local runs from the published 1,000-episode record. Panel action fields continue to distinguish selected from executed actions. Neural-network inspector text now describes Luna's exact nonlinear policy and limits the failure conclusions to observed evidence.

The original `research/forensic_audit/artifact_hashes.json` and original model manifests are retained as historical snapshots. Phase 2 records a separate current registry at `experiments/corrected_evaluation/phase2_artifact_registry.json`, including hashes/sizes for the new outputs and protected artifacts; this avoids rewriting the provenance record that existed before the correction. Provider/model labels and reasoning-effort settings remain locally recorded labels unless independent provider-side evidence exists. This phase has no provider-side attestation; it does not upgrade those labels to verified provider facts.

## Limitations and unresolved points

- SeedSequence creates reproducible, separated PRNG streams. It removes shared-state consumption; it does not prove formal statistical independence or demonstrate the magnitude of prior coupling effects.
- The corrected versus historical score changes cannot isolate the RNG architecture from the changed random draws.
- A matched deterministic seed design supports paired comparisons, but inferential generalization is conditional on treating these generated seed pairs as exchangeable samples. Sign-flip inference also assumes within-pair exchangeability under its null.
- The five historical trace replays are one episode per model. They correct or limit earlier explanations but cannot establish aggregate causation.
- The MinAtar Breakout environment, chosen weights, and episode protocol are narrow evidence. Results do not establish general model capability.
- No new confirmatory equivalence margin was set. The previous Astra TOST is exploratory only.

## Test results

The full existing and new test suite was run with `MPLCONFIGDIR=/private/tmp/phase2-mpl .venv/bin/pytest -q`. Pytest collected 134 tests: **125 passed, 6 failed, and 3 errored**. All nine failures/errors are in GUI checks that construct a Tk window, including screenshot capture. This host's Python 3.12.15/Tk 9.0 cannot initialize the Tk library: `tk scaling` evaluates to `NaN` in `icons.tcl`; the test process also logs Cocoa service connection errors. A second available Python 3.11.16 runtime uses Tk 9.0 and fails at the same startup step; the Homebrew Python 3.12 installation has no `_tkinter` module. The GUI checks were not skipped or weakened.

The data, metrics, pairing, RNG, preservation, replay, network, and visualizer data-integrity tests passed in that run. The focused visualizer/data/replay tests also passed independently: 17 passed. Therefore the tested data-loading and score-to-weight hash-binding paths pass, but launching the interactive Tk visualizer and its screenshot capture remains unverified on this host.

## Recommended actions before publication

1. Publish the corrected and historical datasets as separate, clearly labeled artifacts; cite the run and analysis commits and preserve both hashes.
2. Base inferential claims on the paired analysis artifacts, report effect sizes and intervals, and keep the sign-flip assumptions explicit. Do not claim equivalence from nonsignificance.
3. Describe score differences as results under two RNG protocols, not as a causal estimate of RNG coupling.
4. Keep trajectory-based explanations scoped to demonstrated transitions. Treat any aggregate mechanism as a hypothesis until tested over representative trajectory sets.
5. Retain local/provider provenance distinctions and correct the Sonnet byte-size discrepancy through the additive Phase 2 registry rather than rewriting its original manifest.
6. Describe the primary ranking as this fixed-weight MinAtar result; do not generalize it to broad capability.
