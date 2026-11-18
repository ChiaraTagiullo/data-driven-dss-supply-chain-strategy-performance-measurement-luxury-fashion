# Evaluation summary – dev (maison_a)

Model: frozen version 3.1 (2026-11-13), dss_core.py 47c403d5c8773aee…, parameters a7964fdc3f1b8f5f…

## Pre-registered hypotheses
| id   | hypothesis                                                    | criterion                      | result      | met   |
|:-----|:--------------------------------------------------------------|:-------------------------------|:------------|:------|
| H1   | Non-stable continuative products flagged                      | ≥ 50% (labels: 0%)             | 71%         | yes   |
| H2a  | False alarms on stable products                               | ≤ 20%                          | 12%         | yes   |
| H2b  | Stable products wrongly classified                            | ≤ 5%                           | 0%          | yes   |
| H3   | Missed products have a lower forecast error than flagged ones | median missed < median flagged | 28% vs 51%  | yes   |
| H4   | Same class with clean and dirty data                          | ≥ 95%                          | 100%        | yes   |
| H5   | KPI alignment lower for Variable than for Stable              | Variable < Stable              | 76% vs 100% | yes   |

## Key results
- Non-stable continuative products flagged: 71% (labels: 0%); classified Variable/Irregular: 27%
- False alarms on stable products: 12%; stable wrongly classified: 0%
- Clustering: scenario chosen by the rule = base (ARI base 0.652, alternative 0.219)
- Dirty data: 100% same class; code changes flagged 9/15
- Carry-over not stable: 14 of 78 with enough history
