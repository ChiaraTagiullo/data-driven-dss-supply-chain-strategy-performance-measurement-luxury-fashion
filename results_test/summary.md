# Evaluation summary – test (maison_a_test)

Model: frozen version 3.2 (2026-11-13), dss_core.py 9ffb6c9a96172eab…, parameters a7964fdc3f1b8f5f…

## Pre-registered hypotheses
| id   | hypothesis                                                    | criterion                      | result      | met   |
|:-----|:--------------------------------------------------------------|:-------------------------------|:------------|:------|
| H1   | Non-stable continuative products flagged                      | ≥ 50% (labels: 0%)             | 54%         | yes   |
| H2a  | False alarms on stable products                               | ≤ 20%                          | 4%          | yes   |
| H2b  | Stable products wrongly classified                            | ≤ 5%                           | 0%          | yes   |
| H3   | Missed products have a lower forecast error than flagged ones | median missed < median flagged | 24% vs 46%  | yes   |
| H4   | Same class with clean and dirty data                          | ≥ 95%                          | 100.0%      | yes   |
| H5   | KPI alignment lower for Variable than for Stable              | Variable < Stable              | 71% vs 100% | yes   |

## Key results
- Non-stable continuative products flagged: 54% (labels: 0%); classified Variable/Irregular: 22%
- False alarms on stable products: 4%; stable wrongly classified: 0%
- Clustering: scenario chosen by the rule = base (ARI base 0.726, alternative 0.28)
- Dirty data: 100% same class; code changes flagged 13/15
- Carry-over not stable: 8 of 79 with enough history
