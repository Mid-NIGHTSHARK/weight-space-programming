# Corrected MinAtar Breakout Evaluation Analysis

Dataset: `phase2_corrected_full`; matched episodes per condition: 3,000.

## Condition metrics

| Rank | Condition | Type | N | Mean | Median | SD | SE | 95% CI (mean) | Min | Max | Mean length | Median length | Max length | Cutoffs | Selected-action counts |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Astra reference (local reevaluation) | reference | 3000 | 504.8523 | 642.0 | 243.5745 | 4.4470 | [496.1328, 513.5719] | 0.0 | 824.0 | 7199.46 | 9821.0 | 10000 | 1484 (49.47%) | no_op=10938851, left=5328333, up=0, right=5331186, down=0, fire=0 |
| 2 | Gemini Argon 4 Medium | primary_zero_shot | 3000 | 60.3220 | 38.0 | 65.4576 | 1.1951 | [57.9787, 62.6653] | 0.0 | 475.0 | 914.85 | 558.0 | 7910 | 0 (0.00%) | no_op=282620, left=1230328, up=0, right=1231592, down=0, fire=0 |
| 3 | Claude Sonnet 5.5 | primary_zero_shot | 3000 | 10.5867 | 6.0 | 11.2835 | 0.2060 | [10.1827, 10.9906] | 0.0 | 91.0 | 124.43 | 70.0 | 1096 | 0 (0.00%) | no_op=33165, left=169425, up=0, right=170694, down=0, fire=0 |
| 4 | Trained PPO baseline | trained_baseline | 3000 | 5.7970 | 6.0 | 0.9543 | 0.0174 | [5.7628, 5.8312] | 0.0 | 7.0 | 67.77 | 70.0 | 80 | 0 (0.00%) | no_op=35307, left=22127, up=35119, right=23601, down=41591, fire=45565 |
| 5 | GPT-6 Original | primary_zero_shot | 3000 | 2.5160 | 2.0 | 1.9800 | 0.0361 | [2.4451, 2.5869] | 0.0 | 17.0 | 32.72 | 28.0 | 178 | 0 (0.00%) | no_op=58614, left=19096, up=2, right=20440, down=1, fire=3 |
| 6 | GPT-6 Luna Extra High | primary_zero_shot | 3000 | 1.2300 | 1.0 | 1.2700 | 0.0232 | [1.1845, 1.2755] | 0.0 | 9.0 | 18.69 | 16.0 | 104 | 0 (0.00%) | no_op=14328, left=20141, up=3, right=21580, down=6, fire=14 |
| 7 | Zero-weight control | zero_weight_control | 3000 | 0.5180 | 0.0 | 0.7553 | 0.0138 | [0.4910, 0.5450] | 0.0 | 5.0 | 11.29 | 6.0 | 60 | 0 (0.00%) | no_op=5602, left=5617, up=5706, right=5572, down=5845, fire=5524 |
| 8 | DeepSeek Original | primary_zero_shot | 3000 | 0.4927 | 0.0 | 0.5000 | 0.0091 | [0.4748, 0.5106] | 0.0 | 1.0 | 10.93 | 6.0 | 16 | 0 (0.00%) | no_op=0, left=0, up=10976, right=0, down=10900, fire=10904 |
| 9 | Random-action control | random_control | 3000 | 0.4917 | 0.0 | 0.7207 | 0.0132 | [0.4659, 0.5175] | 0.0 | 7.0 | 11.01 | 6.0 | 80 | 0 (0.00%) | no_op=5497, left=5641, up=5421, right=5524, down=5538, fire=5419 |

Selected-action counts are requests sent to MinAtar. Executed-action counts are separately recorded after sticky-action handling in `condition_metrics.json`.

## Matched comparisons

Differences are condition A minus condition B. Confidence intervals use paired bootstrap resampling of the matched episode pairs. P-values use two-sided sign-flip permutation tests under within-pair exchangeability and are Holm-adjusted across all condition pairs.

| A | B | Mean paired difference | 95% bootstrap CI | Cohen dz | p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| Gemini Argon 4 Medium | GPT-6 Original | 57.8060 | [55.4493, 60.1777] | 0.8830 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | Claude Sonnet 5.5 | 49.7353 | [47.3906, 52.0410] | 0.7639 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | DeepSeek Original | 59.8293 | [57.4840, 62.2077] | 0.9143 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | GPT-6 Luna Extra High | 59.0920 | [56.7800, 61.4934] | 0.9030 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | Astra reference (local reevaluation) | -444.5303 | [-453.4316, -435.4798] | -1.7812 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | Trained PPO baseline | 54.5250 | [52.1666, 56.8424] | 0.8331 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | Random-action control | 59.8303 | [57.5333, 62.2100] | 0.9139 | 5e-05 | 0.0018 |
| Gemini Argon 4 Medium | Zero-weight control | 59.8040 | [57.4419, 62.2053] | 0.9135 | 5e-05 | 0.0018 |
| GPT-6 Original | Claude Sonnet 5.5 | -8.0707 | [-8.4814, -7.6693] | -0.7067 | 5e-05 | 0.0018 |
| GPT-6 Original | DeepSeek Original | 2.0233 | [1.9500, 2.0970] | 0.9867 | 5e-05 | 0.0018 |
| GPT-6 Original | GPT-6 Luna Extra High | 1.2860 | [1.2037, 1.3690] | 0.5536 | 5e-05 | 0.0018 |
| GPT-6 Original | Astra reference (local reevaluation) | -502.3363 | [-510.9825, -493.4447] | -2.0615 | 5e-05 | 0.0018 |
| GPT-6 Original | Trained PPO baseline | -3.2810 | [-3.3597, -3.2020] | -1.4930 | 5e-05 | 0.0018 |
| GPT-6 Original | Random-action control | 2.0243 | [1.9493, 2.0980] | 0.9706 | 5e-05 | 0.0018 |
| GPT-6 Original | Zero-weight control | 1.9980 | [1.9227, 2.0740] | 0.9387 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | DeepSeek Original | 10.0940 | [9.6890, 10.4887] | 0.8942 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | GPT-6 Luna Extra High | 9.3567 | [8.9563, 9.7657] | 0.8265 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | Astra reference (local reevaluation) | -494.2657 | [-503.0587, -485.4160] | -2.0250 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | Trained PPO baseline | 4.7897 | [4.3867, 5.1977] | 0.4256 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | Random-action control | 10.0950 | [9.6976, 10.5103] | 0.8938 | 5e-05 | 0.0018 |
| Claude Sonnet 5.5 | Zero-weight control | 10.0687 | [9.6607, 10.4780] | 0.8904 | 5e-05 | 0.0018 |
| DeepSeek Original | GPT-6 Luna Extra High | -0.7373 | [-0.7857, -0.6883] | -0.5458 | 5e-05 | 0.0018 |
| DeepSeek Original | Astra reference (local reevaluation) | -504.3597 | [-513.1492, -495.4794] | -2.0706 | 5e-05 | 0.0018 |
| DeepSeek Original | Trained PPO baseline | -5.3043 | [-5.3400, -5.2683] | -5.1763 | 5e-05 | 0.0018 |
| DeepSeek Original | Random-action control | 0.0010 | [-0.0280, 0.0303] | 0.0012 | 0.9633 | 0.9633 |
| DeepSeek Original | Zero-weight control | -0.0253 | [-0.0557, 0.0050] | -0.0298 | 0.10455 | 0.31365 |
| GPT-6 Luna Extra High | Astra reference (local reevaluation) | -503.6223 | [-512.3913, -494.8692] | -2.0679 | 5e-05 | 0.0018 |
| GPT-6 Luna Extra High | Trained PPO baseline | -4.5670 | [-4.6233, -4.5090] | -2.8643 | 5e-05 | 0.0018 |
| GPT-6 Luna Extra High | Random-action control | 0.7383 | [0.6873, 0.7897] | 0.5107 | 5e-05 | 0.0018 |
| GPT-6 Luna Extra High | Zero-weight control | 0.7120 | [0.6587, 0.7647] | 0.4809 | 5e-05 | 0.0018 |
| Astra reference (local reevaluation) | Trained PPO baseline | 499.0553 | [490.2623, 507.6951] | 2.0490 | 5e-05 | 0.0018 |
| Astra reference (local reevaluation) | Random-action control | 504.3607 | [495.6462, 512.9626] | 2.0706 | 5e-05 | 0.0018 |
| Astra reference (local reevaluation) | Zero-weight control | 504.3343 | [495.7882, 513.0034] | 2.0706 | 5e-05 | 0.0018 |
| Trained PPO baseline | Random-action control | 5.3053 | [5.2627, 5.3473] | 4.5164 | 5e-05 | 0.0018 |
| Trained PPO baseline | Zero-weight control | 5.2790 | [5.2347, 5.3217] | 4.3845 | 5e-05 | 0.0018 |
| Random-action control | Zero-weight control | -0.0263 | [-0.0623, 0.0107] | -0.0256 | 0.1659 | 0.3318 |

## Corrected versus original historical results

These comparisons use the same base-seed vectors where historical episode CSVs exist. Changes are descriptive paired changes; they combine stream decoupling with different random draws and cannot isolate coupling as the sole cause. Astra's 3,000 matched episodes are distinct from its 4,000-episode local historical evaluation and the paper's separate 1,000-episode published result.

| Condition | Original N | Original mean | Corrected matched N | Corrected mean | Mean paired change | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| gemini_argon_4_medium | 3000 | 57.8660 | 3000 | 60.3220 | 2.4560 | [-0.6777, 5.5554] |
| gpt6_original | 3000 | 2.4930 | 3000 | 2.5160 | 0.0230 | [-0.0750, 0.1213] |
| claude_sonnet_55 | 3000 | 10.9573 | 3000 | 10.5867 | -0.3707 | [-0.9600, 0.2320] |
| deepseek_original | 3000 | 0.4843 | 3000 | 0.4927 | 0.0083 | [-0.0173, 0.0337] |
| gpt6_luna_extra_high | 3000 | 1.2530 | 3000 | 1.2300 | -0.0230 | [-0.0873, 0.0397] |
| astra_local_reference | 3000 | 507.4587 | 3000 | 504.8523 | -2.6063 | [-14.9610, 9.6385] |
| ppo_trained_baseline | 3000 | 5.7573 | 3000 | 5.7970 | 0.0397 | [-0.0110, 0.0903] |
| random_action_control | 1000 | 0.4850 | 1000 | 0.4810 | -0.0040 | [-0.0720, 0.0640] |
| zero_weight_control | 1000 | 0.5170 | 1000 | 0.5280 | 0.0110 | [-0.0570, 0.0780] |

No confirmatory equivalence test was conducted. The earlier Astra ±25-point margin is retained only as post-hoc exploratory work and is not used here.

