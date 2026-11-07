# Gemini Argon High — Iterative Development Report

## A. Summary

Three separate Argon High versions were reconstructed and evaluated using the existing Phase 2 corrected MinAtar Breakout evaluator. The original first-prompt assignments are the inline tensor block the user identified as the first Argon High output. The two supplied Python files are preserved byte-for-byte as follow-up 1 and follow-up 2.

All three versions passed structural and cross-framework forward-pass checks. The supplied follow-up-1 detector and representative-state assertions passed. The supplied follow-up-2 verification script did **not** complete: its probability assertion is impossible in float64 because `1.0 - 1e-40` rounds to `1.0`, while a probability cannot be greater than `1.0`. That source was not modified. A separate exhaustive comparison of its network against its own mathematical oracle found zero mismatches across 7,280 enumerated states, and actual MinAtar runs were performed independently.

The full benchmark completed with 3,000 episodes per version (9,000 total). Both follow-ups scored modestly higher than the supplied original assignments. Follow-up 2 did not show a measurable improvement over follow-up 1. All versions eventually lost the ball in every evaluated episode; none reached the 10,000-step cutoff.

## B. Version comparison

| Version | Provenance and changes | Nonzero parameters | Sparsity |
|---|---|---:|---:|
| Original first-prompt assignments | User-supplied inline assignment block. Four comparison units and one exact detector for a single left-start state. | 637 / 20,486 | 96.891% |
| Follow-up 1 | Researcher-directed follow-up. Adds exact detectors for three early left-start states and changes the second-layer left-command bias. | 647 / 20,486 | 96.842% |
| Follow-up 2 | Researcher-directed follow-up. Adds another left-start state, hold-at-column-5 states, and right-start states; strengthens the second-layer left-command inhibition. | 690 / 20,486 | 96.632% |

Follow-up 1 differs from the original at 15 tensor entries (10 in W1, 2 in b1, 2 in W2, 1 in b2); the nonzero count rises by 10. Follow-up 2 differs from follow-up 1 at 44 entries (24 in W1, 8 in b1, 11 in W2, 1 in b2); the nonzero count rises by 43. W3 and b3 are unchanged across all versions, so the six-action output routing is the same.

All tensors use the project architecture: input 600, hidden layers 32 and 32 with tanh, and 6 linear action logits; total 20,486 parameters. The inline original excerpt did not include array initialization or a complete provider response. Its reconstruction uses zero initialization, as used by the project and both supplied follow-up scripts. This is recorded as an assumption, not provider-authenticated provenance.

## C. Implementation verification

Each version reconstructed to shapes W1=(600,32), b1=(32,), W2=(32,32), b2=(32,), W3=(32,6), and b3=(6,). All values were finite; all assignment code executed without an indexing exception. Parameter counts were independently recounted from the resulting arrays.

| Version | W1 | b1 | W2 | b2 | W3 | b3 | Total | Weight SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original | 617 | 5 | 5 | 2 | 4 | 4 | 637 | `53c65f6ba95090ef88999ebe0dcc23339390c8579e62ec50eaaffc008c1d90f9` |
| Follow-up 1 | 623 | 7 | 7 | 2 | 4 | 4 | 647 | `9dbce900145f681fac6e081fa9cad927205a0ecbe7dfd181285eb0bc5209527f` |
| Follow-up 2 | 647 | 15 | 18 | 2 | 4 | 4 | 690 | `a88ee43463374fc79ea38c1ecfeb55da736930c5b3d4f5e201eca066c8041205` |

The NumPy project network and PyTorch reference returned identical logits (maximum absolute difference 0.0) on 16 binary 600-element probes for each version. This is a cross-implementation check, not a proof over every possible observation.

Re-running the same deterministic serializer on each reconstructed tensor set produced byte-identical `.npz` files and the exact registered SHA-256 values shown above.

The follow-up-1 source script passed its detector and representative-state assertions when run unchanged. The follow-up-2 script printed the expected 690 nonzero count and then failed at `assert probs[net_action] > 1.0 - 1e-40`. In float64, the right-hand side is exactly 1.0; the assertion cannot be true. The failure is in the supplied verification assertion, not evidence that the network has a probability above one. The script was kept unchanged. Independently, its network's argmax action matched its authored oracle on all 7,280 enumerated ball/trail/paddle states; the selected probability was exactly 1.0 for those states under that source's float64 softmax. This validates agreement with that oracle only, not equivalence of the oracle to all game behavior.

## D. Mathematical critique

The follow-up state encoders flatten the grid as `(row * 10 + column) * 6 + channel`, consistent with the project observation module. The channel order is paddle 0, empty 1, bricks 2, ball 3, trail 4, empty 5. The network weights use only paddle, ball, and trail channels; the supplied B/C mathematical encoders omit bricks, but this omission does not change those networks' logits because the corresponding W1 weights are zero.

The ball-position coefficient sequence `[1,2,4,6,8,10,12,14,16,17]` represents twice the current x-coordinate with endpoint adjustments. Subtracting the trail x-coordinate gives the next clamped horizontal column for the reachable MinAtar states, including the frame where a wall impact clamps the ball and flips its direction. This was checked against the actual Breakout implementation, where wall handling clamps x and changes `ball_dir`, and was observed in the replay traces. It should not be generalized to arbitrary impossible ball/trail combinations as a complete state model.

At row 8, the policies switch to comparing the current ball x with paddle position. This aligns with MinAtar's paddle rule: after the paddle action, a downward ball is caught if the updated paddle position equals the current ball x or the computed next x. Sticky action can still change the selected paddle movement, and the full evaluator preserved the 0.1 sticky-action mechanism rather than assuming that the mathematical argmax always executes.

The original's single special unit only activates for `(ball_y, ball_x, trail_y, trail_x, paddle_x)=(5,2,4,1,4)`. It does not generalize to other paddle positions or the whole opening path. Follow-up 1 adds exact-match detectors for the first three left-start states while the paddle is at column 4. Follow-up 2 adds more exact opening states, including a right-start path. These are narrow detectors, not a learned general trajectory representation. Actual categorical sampling means the network can choose other actions despite a preferred argmax; trajectory observations confirm that this occurs.

The second follow-up's more elaborate detector scheme is mathematically consistent with its own finite-state oracle on the enumerated domain, but the source verifier did not reach that loop due to the impossible probability assertion. Its intended extension is therefore partly supported by independent exhaustive logic and actual episodes, not by a passing supplied verifier.

## E. Actual environment testing

The evaluation used the actual project `MinAtarBreakoutEnv`, MinAtar 1.0.15, and its six-action mapping. The existing corrected evaluator derived environment and policy seeds with `numpy.random.SeedSequence(base_seed).spawn(2)`, passed the environment seed to MinAtar's environment `RandomState`, and used a distinct `RandomState` for `PolicyAgent` categorical softmax sampling. Sticky-action draws remained on the MinAtar environment stream; policy action sampling remained on the policy stream. The episode reset sets the sticky-action previous action to no-op. Existing corrected RNG tests passed.

The 10-episode-per-version pilot completed 30 episodes and replay-verified one episode in each seed group for each version (9 replay checks). The full evaluation also produced complete trajectory-digest-matched replays for left and right initial starts for all three versions. In seed group 42:

- Left start: base seed 1935803229, environment seed 2625102126, policy seed 1695836764; each version scored 70 over 866 steps, saw 85 wall direction changes and 71 row-8 bounces, and had 16 sticky overrides.
- Right start: base seed 1608637543, environment seed 126372768, policy seed 1417295617; each version scored 27 over 266 steps, saw 24 wall direction changes and 21 row-8 bounces. Sticky overrides were 14, 14, and 13 for original, follow-up 1, and follow-up 2 respectively.

Both traces include positive brick rewards and end in a terminal ball loss. Their complete per-step states, logits, probabilities, selected/executed actions, rewards, brick counts, sticky overrides, and trajectory hashes are in `evaluation/representative_trajectories.json`. Those digests match the corresponding rows in the full-run episode CSVs. These are representative paths, not exhaustive coverage of every collision or random sequence.

Runtime versions observed during final audit in the same project virtual environment: Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.15.0, pandas 3.1.1, pytest 9.2.0, and MinAtar 1.0.15. The original run manifest did not capture versions at run start; the observed versions are recorded as post-run audit metadata, not retroactively described as a launch-time capture.

## F. Corrected benchmark results

The copied evaluation configuration has SHA-256 `50292561321d3de7faf18cb6ba28ec02c294442be73ff80ae4744da474cc030b`. It uses groups 42, 100, and 200 with 1,000 episodes each, sticky probability 0.1, categorical stable-softmax sampling at temperature 1, and a 10,000-step cap. The full run has 3,000 rows per version and 9,000 total. All conditions used the same base-seed vector and the same environment/policy child-seed triplets. Seed matching and counts were rechecked directly from the raw CSVs.

| Version | Episodes | Mean ± SE | 95% Student-t CI for mean | Median | Min–max | Mean / median / max length | Cutoffs | Ball lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 3,000 | 59.9983 ± 1.1953 | [57.6546, 62.3421] | 38 | 0–626 | 902.29 / 557 / 9,024 | 0 (0%) | 3,000 |
| Follow-up 1 | 3,000 | 62.3677 ± 1.2029 | [60.0091, 64.7262] | 40 | 0–626 | 938.35 / 592 / 9,024 | 0 (0%) | 3,000 |
| Follow-up 2 | 3,000 | 62.1880 ± 1.1822 | [59.8699, 64.5061] | 40 | 0–626 | 937.97 / 596 / 9,024 | 0 (0%) | 3,000 |

Independent calculations from each raw CSV reproduced the means, medians, sample standard deviations, standard errors, lengths, minima, maxima, cutoff counts, and CSV hashes stored in the analysis outputs. Exact score counts and quantiles are available in `analysis/raw_recomputed_metrics.json`.

Seed-group mean scores (1,000 episodes per group):

| Version | Group 42 | Group 100 | Group 200 |
|---|---:|---:|---:|
| Original | 61.063 | 55.672 | 63.260 |
| Follow-up 1 | 63.944 | 57.867 | 65.292 |
| Follow-up 2 | 63.334 | 57.888 | 65.342 |

Selected action distributions were close to 10.2–10.7% no-op, 44.6–44.9% left, and 44.7–44.9% right. Actions up, down, and fire had zero counts. Sticky handling changed only small counts between selected and executed actions. Full counts and rates are in `analysis/condition_metrics.json`.

## G. Statistical comparisons

Paired comparisons used `(seed_group, episode_index)` as the pair key and checked exact equality of the base, environment, and policy seed identifiers. The existing Phase 2 functions were used unchanged: 9,999 bootstrap resamples of matched episode pairs for percentile 95% confidence intervals; 19,999 two-sided sign-flip permutations of paired mean differences; paired Cohen's dz with the sample SD of differences; and Holm correction across the three pairwise comparisons. We independently recomputed all three means, bootstrap intervals, sign-flip p-values, effect sizes, and Holm-adjusted p-values directly from the raw CSVs; the values matched the generated analysis exactly.

Differences below are **first named version minus second named version**:

| Comparison | Mean paired difference | Paired bootstrap 95% CI | Two-sided p | Holm p | Cohen's dz |
|---|---:|---:|---:|---:|---:|
| Original − Follow-up 1 | −2.3693 | [−3.1467, −1.6466] | 0.00005 | 0.00015 | −0.1122 |
| Original − Follow-up 2 | −2.1897 | [−3.4360, −1.0003] | 0.00045 | 0.00090 | −0.0637 |
| Follow-up 1 − Follow-up 2 | 0.1797 | [−0.8134, 1.1994] | 0.72455 | 0.72455 | 0.0065 |

The first two differences are statistically detectable under the project's procedure, but their paired standardized effects are small. Follow-up 2 did not significantly differ from follow-up 1; this is not evidence that the versions are equivalent. Confidence intervals around separate condition means were not used as a substitute for paired tests.

## H. Research interpretation

On this 3,000-episode MinAtar Breakout evaluation, the two researcher-directed follow-ups scored about 2.2–2.4 points above the supplied first-prompt version on average. The first follow-up is the highest-scoring version at 62.368, while the second follow-up is 62.188. The evidence supports a modest improvement for both follow-ups over this specific original assignment set, not a large or general improvement. There is no evidence that follow-up 2 improves on follow-up 1.

All versions are stochastic policies derived from fixed hand-authored tensor assignments; they are not trained networks. The follow-ups are not independent zero-shot generations. Results do not establish general LLM ability across environments, architectures, prompts, or repeated independent generations, and should not be merged into the project's primary five-model zero-shot ranking.

## I. Limitations and provenance

1. The inline first-prompt assignments are attributed to Gemini Argon High by the user, but the exact full response, provider receipt, model build, and generation settings for these tensors were not supplied. The local source is an explicit reconstruction from that assignment excerpt with project-standard zero initialization.
2. The repository contains a distinct archived Gemini run: `gemini/response_raw.txt` SHA-256 `6e593d588c827afd4f1ab2d81ae1a5c65585210e0345920fb887a76b24e248ac`; `gemini/generated_weights.npz` SHA-256 `455722b09750deb693093c823c13dd37b99eb470c957ce472262b1419014abb5`, 817 nonzero parameters. Its tensors differ from the supplied inline original at 813 tensor entries. That archive is also referenced by the Phase 2 primary condition labeled “Gemini Argon 4 Medium,” while `gemini/metadata.json` records `gemini-argon` and reasoning effort `high`. This repository naming/provenance conflict remains unresolved. The archived tensors were not substituted for the user-identified original.
3. Gemini/provider identity and reasoning-effort labels are recorded metadata/user attribution, not independently authenticated provider evidence.
4. Seed matching gives a shared schedule and independent random streams, but different actions lead to different states and episode lengths. Pairing is by common exogenous seed identifiers, not by identical resulting trajectories.
5. The second follow-up's supplied mathematical verifier still fails at its impossible probability assertion. The network was not silently changed to make it pass.
6. This is one environment, one architecture, and one sequence of related outputs. Episode count reduces uncertainty for these weight sets but does not replace independent model-generation replications.

The existing project test suite relevant to architecture, observations, MinAtar, evaluation, RNG separation, analysis, weight parsing, preservation, and the archived Gemini baseline passed **77/77** tests. This does not verify GUI behavior or broader website functionality. No website pages were implemented in this task.

## J. Reproducibility and preservation

The run was made on branch `phase-2-corrected-evaluation` at HEAD `e148ea353c3cf0307c69de43f5e49201d2c4478f`. The existing protected-artifact manifest covers 180 files; all 180 hashes and sizes matched both before implementation/evaluation and in the final check. The three historical Gemini files retained their original hashes. No tracked baseline file was modified, no branch or commit was created, and nothing was pushed.

Exact source hashes:

| Stage | Source path | SHA-256 |
|---|---|---|
| Original assignment reconstruction | `source/argon_high_original_inline_v1.py` | `153ee5b9867e947e397b30ea688dd220562ebe014318a34d9f6e311b461ddf93` |
| Follow-up 1 (verbatim supplied script) | `source/argon_high_followup_1.py` | `1460dcf411d31912a7de5286ee6b6b643230b7be4ba83c7dff045055ca8be317` |
| Follow-up 2 (verbatim supplied script) | `source/argon_high_followup_2.py` | `bfb27cdde52250e720c83de02127a3c5e1782bd5d17b389ba127b553c6351639` |

The three raw episode CSV hashes are recorded in `evaluation/full_run/run_manifest.json` and `artifact_registry.json`. Reproduction commands, including safe new output paths that refuse overwrites, are in `README.md`.

| Episode CSV | SHA-256 |
|---|---|
| `evaluation/full_run/conditions/argon_high_original/episode_scores.csv` | `a6d14220c51b14e88e1288a3e396731aa3e4fa2e8fdd3f9895f115735cf0da1c` |
| `evaluation/full_run/conditions/argon_high_followup_1/episode_scores.csv` | `dc2a1fca571ccede39c5d4b96f5a93a07a4ab00b092d990ffd177a112774a527` |
| `evaluation/full_run/conditions/argon_high_followup_2/episode_scores.csv` | `6f2dff042bc9d47fd5d4d2c9caa11a920f6a2bc89d55061c4a3c88cbc08b6cd0` |
