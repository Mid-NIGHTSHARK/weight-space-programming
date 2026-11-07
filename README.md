# Weight-Space Programming

**Research on whether large language models can write useful neural-network weights directly, without conventional training.**

[Overview](https://mid-nightshark.github.io/weight-space-programming/#overview) · [Live Arena](https://mid-nightshark.github.io/weight-space-programming/#arena) · [Models](https://mid-nightshark.github.io/weight-space-programming/#models) · [Results and downloads](https://mid-nightshark.github.io/weight-space-programming/#results) · [Methodology](https://mid-nightshark.github.io/weight-space-programming/#methodology) · [Browse the source repository](https://github.com/Mid-NIGHTSHARK/weight-space-programming)

**Project status:** research artifact with a corrected local evaluation and an independent review verdict of **verified with limitations**. Model/provider provenance is not independently authenticated, and interactive desktop-GUI tests remain unresolved on the recorded macOS/Tk setup. See [the corrected evaluation report](PHASE_2_CORRECTED_EVALUATION_REPORT.md) for the evidence and qualifications.

## Research question

Weight-space programming means specifying a neural network by writing its numerical parameters—the weights and biases that determine how inputs affect outputs—instead of training those parameters with a conventional optimization procedure.

This project asks: **Can a language model directly construct useful neural-network weights for a game-playing task without receiving training feedback?** It evaluates fixed weight sets in MinAtar Breakout, a small, fast, grid-based version of Breakout. The compact environment makes the observations, actions, and weight arrays inspectable while retaining moving objects, collisions, sticky actions, and stochastic action selection.

The repository studies a task related to Experiment 5 in the authors' [source article](https://rikiyat.github.io/blog/weight-space-programming/). This repository's corrected scores are local reevaluations of preserved weight files; they are not the same sample as the score reported by that article.

## How the experiment works

1. A model receives a prompt describing the game, network, and weight format.
2. It returns numerical weight assignments.
3. The assignments are parsed into a fixed network, without training on game outcomes.
4. The network selects actions while MinAtar Breakout runs thousands of episodes.
5. The evaluator records scores, episode lengths, actions, seeds, and replay information.

The project includes model responses, parsed weight archives, raw per-episode data, analysis outputs, and a static browser demonstration. The browser Arena is an educational JavaScript implementation; it does not reproduce the corrected evaluator's NumPy random streams and is not benchmark-equivalent.

## Network and environment

Each policy uses the same multilayer perceptron:

| Component | Specification |
|---|---|
| Input | 600 values: a 10 × 10 board with six channels at each cell |
| Hidden layers | 32 neurons, then 32 neurons; `tanh` activation after each |
| Output | 6 linear action logits |
| Parameters | 20,486 total: weights and biases across all three layers |
| Initialization for parsed assignment blocks | Zero-filled tensors, then the allowed assignments are applied |
| Action selection | Temperature-1 softmax followed by categorical sampling |

MinAtar's native Breakout observation has four channels. The project maps it to six channels, in order: paddle, empty danger, bricks, ball, ball trail, empty gauge. Values are flattened in row-major order with channels innermost: `(row * 10 + column) * 6 + channel`. The six actions are no-op, left, up, right, down, and fire. The corrected protocol uses MinAtar 1.0.15, sticky-action probability 0.1, reset last action 0, and a 10,000-step episode limit.

The implementation and protocol are documented in [the network code](network/architecture.py), [observation mapping](environment/observations.py), [action mapping](environment/actions.py), [corrected evaluator](evaluation/corrected_evaluate.py), and [evaluation configuration](experiments/corrected_evaluation/evaluation_config.json).

## Corrected benchmark results

The corrected dataset contains nine conditions and 3,000 episodes per condition (27,000 episodes total). Means and standard errors below are independently recomputed from the raw episode CSV files. Rows are grouped by scientific role; only the five rows marked **Primary zero-shot** belong in the zero-shot model ranking. The [detailed results page](https://mid-nightshark.github.io/weight-space-programming/#results) includes distributions, paired comparisons, and downloads.

| Condition | Role | Mean score ± SE | Median | Episodes |
|---|---|---:|---:|---:|
| Gemini Argon 4 Medium | Primary zero-shot | 60.322 ± 1.195 | 38 | 3,000 |
| Claude Sonnet 5.5 | Primary zero-shot | 10.587 ± 0.206 | 6 | 3,000 |
| GPT-6 Original | Primary zero-shot | 2.516 ± 0.036 | 2 | 3,000 |
| GPT-6 Luna Extra High | Primary zero-shot | 1.230 ± 0.023 | 1 | 3,000 |
| DeepSeek Original | Primary zero-shot | 0.493 ± 0.009 | 0 | 3,000 |
| Astra local reference | Handcrafted/reference controller | 504.852 ± 4.447 | 642 | 3,000 |
| Trained PPO baseline | Trained baseline | 5.797 ± 0.017 | 6 | 3,000 |
| Zero-weight control | Control | 0.518 ± 0.014 | 0 | 3,000 |
| Random-action control | Control | 0.492 ± 0.013 | 0 | 3,000 |

These are results for the listed fixed weight files and this one environment/protocol. They do not estimate performance across repeated independent generations of each model. The Astra row is a reference policy, PPO is trained, and the two controls are not model generations.

### Separate post-hoc and iterative conditions

**DeepSeek Corrected** is a researcher-created post-hoc intervention. It is not an original zero-shot output and is excluded from the nine-condition corrected dataset and primary ranking. Its response, weights, and analysis remain separately archived; see the [DeepSeek correction report](PHASE_3D_DEEPSEEK_CORRECTION_REPORT.md).

The three **Gemini Argon High iterative-development** versions are also separate from the primary zero-shot experiment. The first is a reconstruction from user-supplied inline assignments; the two follow-ups are researcher-directed scripts. They are not three independent model generations, and they are not interchangeable with Gemini Argon 4 Medium.

| Version | Mean score ± SE | Median | Episodes |
|---|---:|---:|---:|
| Original first-prompt assignments | 59.998 ± 1.195 | 38 | 3,000 |
| Follow-up 1 | 62.368 ± 1.203 | 40 | 3,000 |
| Follow-up 2 | 62.188 ± 1.182 | 40 | 3,000 |

Both follow-ups scored modestly above the supplied original, with small paired standardized effects. Follow-up 2 did not significantly differ from Follow-up 1; this does not establish equivalence. All 9,000 episodes ended in ball loss, with no time-limit cutoffs. The supplied Follow-up 2 verifier retains an impossible float64 probability assertion; an independent comparison against its finite-state oracle found no action mismatches over the tested 7,280 states. Provider identity and settings remain unauthenticated, and archived Gemini metadata conflicts with the Phase 2 Argon 4 Medium label. The [full iterative-development report](experiments/argon_high_iterative_development/report.md) documents these limits and links the raw records.

## Evaluation and analysis

The corrected evaluator creates a deterministic seed schedule using seed groups 42, 100, and 200. For each base episode seed, it derives separate environment and policy seeds with `numpy.random.SeedSequence(base_seed).spawn(2)`. MinAtar receives its own environment seed; categorical action sampling uses a separate `numpy.random.RandomState`. The same base and derived seed identifiers are used across conditions to support paired comparisons. The streams are separate PRNG instances; the seed derivation is reproducible, not a claim of mathematical independence.

For corrected comparisons, episodes are paired by seed group and episode index, with the base, environment, and policy seeds checked for exact agreement. The analysis uses bootstrap resampling of whole episode pairs, two-sided sign-flip permutation tests, paired Cohen's *dz*, and Holm correction across 36 pairwise comparisons. A nonsignificant comparison is not evidence of equivalence. See the [full methods and comparison tables](experiments/corrected_evaluation/full_run/analysis/analysis_report.md) and [Phase 2 corrected evaluation report](PHASE_2_CORRECTED_EVALUATION_REPORT.md).

The primary zero-shot ranking is Gemini Argon 4 Medium, Claude Sonnet 5.5, GPT-6 Original, GPT-6 Luna Extra High, and DeepSeek Original by mean score. It describes these five recorded artifacts on this benchmark only. Provider identity and some generation settings are locally recorded labels, not independently authenticated provider facts.

## Reproducibility and project map

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/). The package metadata declares the dependencies. From the repository root:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Run the full test suite with:

```bash
pytest -q
```

The full suite includes GUI tests. In the 2026-11-07 local run, 141 tests were collected: 132 passed, 6 failed, and 3 errored. All nine failures/errors were in screenshot or Tk GUI checks; Tk 9.0 startup failed while evaluating `tk scaling` as NaN, before the GUI assertions ran, and Cocoa service errors were also logged. A successful run of the non-GUI tests does not verify desktop GUI behavior. See the [Phase 2 report](PHASE_2_CORRECTED_EVALUATION_REPORT.md) for the earlier recorded test environment and outcomes.

To check the published static data bundle without writing files, and to run the browser/Python conformance tests used by Pages CI:

```bash
python scripts/export_research_site_data.py --check
pytest -q tests/test_site_export.py tests/test_browser_conformance.py
node --test tests/site_conformance.test.mjs
```

The main implementation and evidence are organized by purpose:

| Find | Location |
|---|---|
| MinAtar wrapper, observations, actions | [`environment/`](environment/) |
| Network, inference, weight parser and loader | [`network/`](network/) |
| Legacy and corrected evaluators, RNG, analysis | [`evaluation/`](evaluation/) |
| Primary corrected config, raw CSVs, metrics and registry | [`experiments/corrected_evaluation/`](experiments/corrected_evaluation/) |
| Iterative-development sources, weights, raw CSVs and reports | [`experiments/argon_high_iterative_development/`](experiments/argon_high_iterative_development/) |
| Preserved primary prompts, responses, weights and runs | [`experiments/strict_replication/`](experiments/strict_replication/) |
| Research background and forensic records | [`research/`](research/) |
| Static site source and exported datasets | [`docs/`](docs/) |
| Python and browser tests | [`tests/`](tests/) |

See the [developer and reproducibility guide](docs/DEVELOPMENT.md) for the execution path, safe reconstruction steps, run commands, artifact hashes, and website data flow. It distinguishes commands verified by inspection or CI from full evaluations that were not rerun for this documentation update.

## Integrity and limitations

- `experiments/corrected_evaluation/preserved_artifacts_sha256.json` records 180 pre-correction artifacts; the current files match those recorded hashes. The registry proves file identity against that snapshot, not scientific validity.
- `experiments/corrected_evaluation/phase2_artifact_registry.json` records the corrected run, code, configuration, metrics, and protected-file registry. `experiments/argon_high_iterative_development/artifact_registry.json` separately records 35 iterative-experiment files.
- The site exporter verifies its source data and generated bundle; `python scripts/export_research_site_data.py --check` checks the checked-in publication files without regenerating them.
- The model/provider labels and some reasoning-effort metadata are not provider-attested. The archived Gemini metadata and the Phase 2 “Gemini Argon 4 Medium” condition label conflict; the conflict is unresolved.
- The Tk GUI test failures remain unresolved on the audited host. The website Arena is a browser demonstration and does not share the benchmark's NumPy environment/policy random streams.
- Evidence comes from one compact environment, one network architecture, and a small number of fixed model outputs. Episode count reduces uncertainty for each fixed policy; it does not replace independent generations or establish general model capability.

Older reports and protocol files are retained as dated records. Where their early episode counts or evaluator details differ, use the explicitly labeled Phase 2 corrected dataset and report linked above for the corrected benchmark; do not treat historical and corrected outputs as one run.

## License

This project is distributed under the [MIT License](LICENSE).
