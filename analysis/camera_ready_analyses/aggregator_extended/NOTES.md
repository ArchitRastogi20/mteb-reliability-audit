# Extended-tier aggregator comparison: global baseline vs. language-balanced LOO

Script: `aggregator_extended.py` (run with `mteb_audit_data/.venv`; deterministic, SEED = 20260510, 10,000 bootstrap resamples).
Output table: `extended_aggregator_summary.csv` (per-language rows + `MEDIAN` rows, variant x roster).

## Question

The curated-tier experiment (`anon-repo/extended-experiments/exp7_alt_aggregators.py`, 25 models,
17 languages) showed that a language-balanced leave-one-out aggregator (`lang_macro_18lang_loo`:
per-language mean first, then macro-average over the other languages) cuts the median pairwise
inversion rate from 0.33 to 0.145 (median tau 0.71). Does the same intervention still help on the
**extended** tier — the larger, judgment-free roster (~28–57 models per language) where the
baseline global-vs-local disagreement is milder to begin with (median inversion ~15.4%)?

## Design

1. **Model sets.** The balanced aggregator needs multi-language coverage per model, but extended
   rosters are per-language. From `anon-repo/analysis/extended_roster.csv` (890 rows, all
   `included=True`, 57 distinct models):
   - **intersection** — models present in ALL 17 language rosters: **28 models**.
     (Spanish is the binding constraint: its roster has only 28 models, and all 28 appear everywhere else.)
   - **relaxed** — models present in >= 14 of 17 languages: **53 models**; the balanced aggregate
     averages over whichever of the other 16 languages each model actually covers.
   Since the intersection came out at 28 (>= the 15-model fallback threshold), the relaxed variant
   was not *required*; it is reported as a robustness check — and turns out to be the more
   informative roster (see calibration below).
2. **Evaluation.** Per language L, models are ranked by (a) the global Mean(Task) aggregate and
   (b) the LOO balanced aggregate (mean of per-language scores over languages != L within the
   set); each is compared to the local ranking (`lang_avg` for L) via Kendall tau (with 10k-resample
   bootstrap CI) and the pairwise inversion rate with ties counted 0.5 (an empirical check found
   **zero** exactly tied pairs in all 68 evaluations, so this coincides with exp7's drop-ties
   convention). Medians are taken across the 17 languages.
3. **Calibration anchors.** The pre-existing full-extended-roster baseline
   (`mteb_audit_data/summary_extended.csv`, global vs. local, no model-set restriction) has median
   inversion **0.1538 (15.4%)** — reproduced read-only, matching the expected anchor.

## Results

| roster | variant | median tau | median inversion |
|---|---|---|---|
| full roster (calibration, pre-existing) | global baseline | — | **0.154** |
| intersection (n=28) | global baseline | 0.847 | 0.077 |
| intersection (n=28) | balanced LOO    | 0.884 | **0.058** |
| relaxed (n=53)      | global baseline | 0.684 | 0.158 |
| relaxed (n=53)      | balanced LOO    | 0.856 | **0.072** |

## Headline

**Yes — the balanced aggregator still cuts inversions on the judgment-free tier.** On the relaxed
53-model roster — whose baseline (15.8%) essentially reproduces the full-roster calibration anchor
(15.4%), making it the like-for-like comparison — the language-balanced LOO aggregator cuts median
inversion from **15.8% to 7.2%** (a ~55% relative reduction; median tau 0.68 -> 0.86), winning in
16 of 17 languages (Spanish is an exact tie, see caveats). On the strict 28-model intersection the
cut is 7.7% -> 5.8% (tau 0.85 -> 0.88; balanced better in 14/17, baseline better for ita and deu,
spa tied) — smaller headroom because the intersection subset is already much easier to rank than
the full roster.

## Caveats

- **The intersection subset is not calibration-comparable to the full roster.** Its baseline
  inversion (7.7%) is half the full-roster figure (15.4%): the 28 models evaluated in all 17
  languages skew toward widely benchmarked, well-separated systems, so there are fewer
  near-ties to invert. The relaxed roster is the anchor-faithful comparison.
- **Spanish bounds everything.** Its extended roster has only 28 models — exactly the
  intersection set — so the spa row is identical across rosters and across intersection/relaxed,
  and it drags the intersection size down to 28.
- **Noisy local ground truth.** Extended-tier per-language scores rest on only 2–4 tasks per
  language (`n_tasks` in the roster: 2 for 10 languages, 3–4 for the rest), so per-language
  rankings are themselves noisy; per-language tau CIs in the output CSV should be read alongside
  the point estimates.
- **Relaxed-roster LOO averages over covered languages only** (14–16 of the other 16 per model),
  so balanced scores are not computed over an identical language basket for every model.
- **Data provenance deviations from the naive brief** (verified before running):
  `extended-experiments/data/lang_avg_per_model.csv` contains only `tier="curated"` rows (420 rows,
  no extended tier), so per-model per-language extended scores are read from `extended_roster.csv`'s
  own `lang_avg` column; `extended-experiments/data/mteb_agg_per_model.csv` covers only 33 models
  under short aliases, so the global aggregate comes from
  `anon-repo/mteb-ranking-audit/global/mteb_agg_extended.csv` (byte-identical to
  `mteb_audit_data/global/mteb_agg_extended.csv`), which shares the exact `org__model` keys with the
  roster (57/57 overlap) and holds each model's Mean(Task) aggregate over the 18 retrieval tasks.
