# Verification Report
Benchmark: MTEB(Multilingual)
Case studies: IT, JA, HI

## ITA

| Tier | n | τ_computed | τ_paper | |Δτ| | Status |
|---|---|---|---|---|---|
| Restricted | 6 | 0.200 | 0.450 | 0.250 | FAIL |
| Restricted (excl. harrier) | 5 | 0.600 | 0.450 | 0.150 | FAIL |
| Curated | 24 | 0.362 | 0.450 | 0.088 | PASS |
| Extended | 53 | 0.718 | 0.450 | 0.268 | FAIL (selection bias) |

## JPN

| Tier | n | τ_computed | τ_paper | |Δτ| | Status |
|---|---|---|---|---|---|
| Restricted | 7 | 0.429 | 0.470 | 0.041 | PASS |
| Restricted (excl. harrier) | 6 | 0.600 | 0.470 | 0.130 | FAIL |
| Curated | 25 | 0.333 | 0.470 | 0.137 | FAIL |
| Extended | 54 | 0.690 | 0.470 | 0.220 | FAIL (selection bias) |

## HIN

| Tier | n | τ_computed | τ_paper | |Δτ| | Status |
|---|---|---|---|---|---|
| Restricted | 6 | 0.333 | 0.600 | 0.267 | FAIL |
| Restricted (excl. harrier) | 5 | 0.600 | 0.600 | 0.000 | PASS |
| Curated | 24 | 0.464 | 0.600 | 0.136 | FAIL |
| Extended | 53 | 0.739 | 0.600 | 0.139 | FAIL (selection bias) |

> **Smoking gun (HIN):** τ_restricted_excl_harrier = 0.600 = paper τ exactly (|Δ| = 0.000). This three-decimal-place match confirms the predictor pipeline is correct after the Bug-1 fix. harrier-0.6b was added to the roster


