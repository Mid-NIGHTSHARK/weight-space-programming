# Gemini Argon High — Iterative Development Evaluation

This additional comparison is separate from the project's primary zero-shot model ranking.
All follow-ups are researcher-directed development stages. Model/provider identity is recorded, not independently authenticated.

## Corrected benchmark results

| Version | Episodes | Mean ± SE | 95% t interval | Median | Min–max | Mean / median / max length | Cutoffs | Ball lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Argon High — Original first-prompt assignments | 3000 | 59.9983 ± 1.1953 | [57.6546, 62.3421] | 38.0 | 0–626 | 902.29 / 557 / 9024 | 0 (0.000%) | 3000 |
| Argon High — First follow-up | 3000 | 62.3677 ± 1.2029 | [60.0091, 64.7262] | 40.0 | 0–626 | 938.35 / 592 / 9024 | 0 (0.000%) | 3000 |
| Argon High — Second follow-up | 3000 | 62.1880 ± 1.1822 | [59.8699, 64.5061] | 40.0 | 0–626 | 937.97 / 596 / 9024 | 0 (0.000%) | 3000 |

Seed-group mean scores (1,000 episodes per group):

| Version | Group 42 | Group 100 | Group 200 |
|---|---:|---:|---:|
| Argon High — Original first-prompt assignments | 61.063 | 55.672 | 63.260 |
| Argon High — First follow-up | 63.944 | 57.867 | 65.292 |
| Argon High — Second follow-up | 63.334 | 57.888 | 65.342 |

Selected action distribution (percent of all selected actions; only no-op/left/right had nonzero counts):

| Version | Selected no-op / left / right | Executed no-op / left / right |
|---|---:|---:|
| Argon High — Original first-prompt assignments | 10.24% / 44.86% / 44.90% | 10.24% / 44.86% / 44.90% |
| Argon High — First follow-up | 10.45% / 44.76% / 44.80% | 10.45% / 44.75% / 44.80% |
| Argon High — Second follow-up | 10.65% / 44.65% / 44.70% | 10.66% / 44.65% / 44.70% |

## Paired comparisons

Differences are condition A minus condition B. Confidence intervals use paired episode bootstrap; p-values use a two-sided sign-flip test; Holm correction covers all three pairwise comparisons.

| A − B | Mean difference | Paired bootstrap 95% CI | Raw p | Holm p | Cohen's dz |
|---|---:|---:|---:|---:|---:|
| Argon High — Original first-prompt assignments − Argon High — First follow-up | -2.3693 | [-3.1467, -1.6466] | 5e-05 | 0.00015 | -0.1122 |
| Argon High — Original first-prompt assignments − Argon High — Second follow-up | -2.1897 | [-3.4360, -1.0003] | 0.00045 | 0.0009 | -0.0637 |
| Argon High — First follow-up − Argon High — Second follow-up | 0.1797 | [-0.8134, 1.1994] | 0.72455 | 0.72455 | 0.0065 |

## Analysis limits

Episodes are paired by seed group and episode index, with exact equality of base, environment, and policy seed identifiers checked. A shared seed schedule does not make resulting trajectories identical when model actions differ. The confidence intervals and tests do not establish equivalence. The supplied second-follow-up mathematical script contains a floating-point assertion that cannot pass because `1.0 - 1e-40` rounds to `1.0`; the source was preserved unchanged, and this analysis uses the repository's existing statistical routines.
