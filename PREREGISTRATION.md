# Pre-registration of the final evaluation – Maison A DSS

**Date:** 13 November 2026 · **Author:** Chiara Tagiullo (master's thesis, Sapienza University of Rome)

This document fixes, **before the test data are opened**, what will be tested, how, and what will count as success. After the test, results are compared with these criteria and reported as they are, whether they meet them or not. Nothing in this document, in the model or in its parameters will be changed after the test.

## 1. What is evaluated

| Item | Value |
|---|---|
| Model | DSS version **3.1** (`dss_core.py`), default parameters |
| Model fingerprint (SHA-256) | `47c403d5c8773aeeaf606492bfc7e0dc117b2f3a281b6ba2d234ce52c7145702` |
| Parameters fingerprint (SHA-256) | `a7964fdc3f1b8f5f…` (full value in `VERSION.json`) |
| Evaluation code (SHA-256) | `dss_evaluation.py` `256721ef241c52a2…` · `stress_scenarios.json` `7077b27172a02f6d…` · `maison_a_generator.py` `5cdddfd4e0f0c2c1…` |

The evaluation notebook refuses to run on the test data if the model or the parameters differ from these fingerprints.

**Version history.** Versions 1, 2 and 3 were developed and compared on the development dataset only (seed 2026). Version 3.0 was frozen on 10 November 2026. During the dry run of the stress tests (13 November 2026, on datasets generated for development with seeds 9001–9005) a bug was found in the possible-code-change check: an old product code could be matched to two new codes, which stopped the run. Matching was made one-to-one (version 3.1). The fix does not change any output on the development data (clean and dirty data checked: identical results).

## 2. Data

- **Development dataset** (`maison_a`, seed 2026): used to design, improve and dry-run the DSS.
- **Test dataset** (`maison_a_test`, seed 2027): generated on 9 November 2026 with the same parameters, never opened during design.
- **Stress datasets** (seeds 3001–3005): generated at test time; each changes **one** assumption of the generator (section 5).

## 3. Procedure

1. Check the fingerprints of the model and the parameters.
2. Run the evaluation notebook **once** on the test dataset and on the five stress datasets.
3. Compare results with the criteria in sections 4 and 5 and report all of them, including those not met.

## 4. Hypotheses on the test dataset

Population for H1–H3: continuative products (Permanent, Carry-over) with enough history (DSS class other than "too little history"). A product is *flagged* when the DSS classifies it Variable or Irregular, marks its class "to confirm", or raises the declining-sales check. *Non-stable* = generated from the oscillating or declining profile.

| # | Hypothesis | Success criterion |
|---|---|---|
| H1 | The DSS recognises non-stable continuative products better than product line labels | non-stable products flagged **≥ 50%** (labels: 0%) |
| H2a | False alarms are limited | stable products flagged **≤ 20%** |
| H2b | Stable products are not misclassified | stable products classified Variable or Irregular **≤ 5%** |
| H3 | The products the DSS misses are the least problematic | median forecast error (planning on past average) of missed non-stable products **lower** than that of flagged ones |
| H4 | The DSS is robust to imperfect data | same DSS class with clean and dirty data for **≥ 95%** of products |
| H5 | The KPI gap lies where the thesis expects it | mean KPI alignment of the Variable class **lower** than that of the Stable class |

**Thresholds scenario.** No criterion: the rule fixed in design decision D4 is applied (the scenario with the highest agreement with k-means, ARI, is the default), whatever scenario it selects.

The criteria are deliberately less strict than the dry-run results on development data (71%, 12%, 0%, 99.6%), to allow for normal variation between datasets.

## 5. Stress tests

Each stress dataset changes one assumption; the DSS is not modified. There are no pass/fail criteria: the purpose is to describe the range of validity of the DSS. Expected behaviour:

| Scenario | Seed | Change | Expected behaviour |
|---|---|---|---|
| S1 – more unstable carry-over | 3001 | carry-over profiles 30% stable / 50% oscillating / 20% declining | similar detection rates; more products flagged in absolute terms |
| S2 – noisier sales | 3002 | season-to-season noise ×1.5 | higher detection, more false alarms |
| S3 – calmer sales | 3003 | season-to-season noise ×0.5 | lower detection (products barely oscillate), few false alarms |
| S4 – shorter history | 3004 | 6 seasons instead of 10 | more products with too little history, more classes to confirm |
| S5 – dirtier data | 3005 | error rates ×2 | lower robustness, but same class for more than 90% of products |

## 6. Reporting

All tables and figures produced by the notebook are reported in Chapter 5 (demonstration, verification, comparison with labels, clustering, sensitivity, dirty data, requirements, stress tests, hypotheses). Hypotheses not met are reported and discussed, not removed. For each check the thesis states what it does **not** show.

---

## Addendum 1 – Deviation from the protocol (13 November 2026, 12:28–12:45)

*Sections 1–6 above are left unchanged; this addendum records the only deviation.*

**What happened.** The first test run (12:28) stopped at its first step – running the DSS on the clean test data – with a software error, **before any result was computed or displayed**. The run log contains only the fingerprint check and the error message; no table, figure or indicator of the test data was produced.

**Cause.** In the test dataset, at least one active product has no sales recorded (possible in the generator when a product's random weekly sales are all zero; it never occurred in the development data). For such a product the number of seasons on the market was missing, and the step that writes the explanation failed.

**Fix (version 3.2).** Active products without sales get 0 seasons, zero revenue and the class "too little history", with the explanation "no sales recorded yet". The fix was made without looking at any result of the test data. Checks on the development dataset and on the five development stress datasets (clean and dirty data): all outputs are numerically identical to version 3.1.

**New fingerprint.** `dss_core.py` version 3.2: `9ffb6c9a96172eab1c65a697d111613f9e214af395cb100a6c62e2efb0c54d06`. Parameters, evaluation code, hypotheses, criteria, stress scenarios and seeds are unchanged.

**Procedure.** The test is run again, once, on version 3.2. Results are reported as stated in section 6.
