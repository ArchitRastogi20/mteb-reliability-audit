"""
Extended-tier aggregator comparison: baseline global aggregate vs. a
language-balanced leave-one-out (LOO) aggregate.

This extends the curated-tier comparison in
extended-experiments/exp7_alt_aggregators.py (25 models, 17
languages), which found that `lang_macro_18lang_loo` -- per-language mean
first, then macro-average across the OTHER 16 languages, leave-one-out --
cuts the median pairwise inversion rate relative to the global aggregate.
Here the same comparison is replicated on the extended tier, whose per-
language rosters are much larger (~28-57 models per language) but vary in
membership from language to language.

Design decisions (see NOTES.md for full discussion)
-----------------------------------------------------
1. Model-set construction. The language-balanced aggregator needs a model to
   have scores in multiple languages, but the extended roster is per-language
   (membership varies). Two model sets are built from extended_roster.csv:
     - "intersection": models present in the roster for ALL 17 languages.
     - "relaxed":       models present for >= 14 of the 17 languages; the
                         balanced aggregate is computed over whichever of the
                         other 16 languages that model actually covers.
   The intersection set came out to 28 models (>= 15), so the relaxed
   fallback was not strictly triggered -- it is still computed and reported
   as a bonus robustness check (28 vs. 53 models; see NOTES.md).
2. Per language L, the extended-intersection (or relaxed) models are ranked
   by (a) a global aggregate and (b) the LOO language-balanced aggregate
   (mean of per-language scores over languages != L, restricted to models
   in the given roster). Each ranking is compared to the local ranking
   (lang_avg for L) via Kendall tau (with bootstrap CI) and the pairwise
   inversion rate (ties = 0.5); medians are taken across the 17 languages.
3. Calibration: the full-extended-roster baseline (global vs. local, no
   model-set restriction) is read directly from mteb_audit_data's
   summary_extended.csv for reference; this script separately computes its
   own intersection/relaxed-subset baseline for a like-for-like comparison
   with the LOO aggregator (which can only be evaluated on those subsets).

Data provenance / deviations from the naive brief (full detail in NOTES.md)
-----------------------------------------------------------------------------
- extended-experiments/data/lang_avg_per_model.csv turned out to
  contain ONLY tier="curated" rows (420 rows, no "extended" tier present).
  Per-model, per-language extended-tier scores are read instead directly
  from extended_roster.csv's own `lang_avg` column -- which already IS the
  per-model-per-language score for the extended tier (that is what the file
  was built for).
- extended-experiments/data/mteb_agg_per_model.csv covers only 26
  unique models under short aliases (33 rows: 25 "curated" + 1 "extended" +
  7 "restricted" duplicates of curated models), not the 57-model extended
  roster (which uses "org__model"
  identifiers), and no alias-mapping table covering all 57 models exists.
  The global aggregate used here instead comes from
  mteb-ranking-audit/global/mteb_agg_extended.csv (byte-identical
  to mteb_audit_data/global/mteb_agg_extended.csv), which is keyed by the
  exact same "org__model" identifiers as extended_roster.csv (57/57 overlap,
  verified) and holds each model's Mean(Task) aggregate over the 18 MMTEB
  retrieval tasks -- exactly the "global aggregate" quantity this experiment
  needs, with no name-mapping guesswork required.

Reused methodology
------------------
`kendall_with_bootstrap` and `build_loo_macro_scores` below are adapted
line-for-line from exp7_alt_aggregators.py / extended-experiments/utils.py
(same SEED, same bootstrap procedure, same LOO-with-NaN-skipping logic),
kept as close to verbatim as the extended tier's data shape allows (no
English-task term, since the extended roster carries no per-task English
scores). `pairwise_inversion_rate` is reimplemented with an explicit
ties = 0.5 convention (exp7's version instead drops any tied pair from
both numerator and denominator); an empirical check in `main()` counts
tied pairs across every (variant, roster, language) combination evaluated
here and confirms the count is zero, so the two conventions coincide
exactly on this data.

Inputs
------
  analysis/extended_roster.csv
  mteb-ranking-audit/global/mteb_agg_extended.csv
  mteb_audit_data/summary_extended.csv   (read-only, calibration reference)

Outputs (this directory)
-------------------------
  extended_aggregator_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ── constants ──────────────────────────────────────────────────────────────
SEED = 20260510          # same value as extended-experiments/utils.py
N_BOOTSTRAP = 10_000      # same value as exp7_alt_aggregators.py

# Repo root derived from this file's location:
# <root>/.../analysis/aggregator_extended/aggregator_extended.py
REPO_ROOT = Path(__file__).resolve().parents[3]
ROSTER_CSV = REPO_ROOT / "analysis" / "extended_roster.csv"
GLOBAL_AGG_EXTENDED_CSV = (
    REPO_ROOT / "mteb-ranking-audit" / "global" / "mteb_agg_extended.csv"
)
BASELINE_SUMMARY_EXTENDED_CSV = REPO_ROOT / "mteb-ranking-audit" / "results" / "summary_extended.csv"
OUT_DIR = Path(__file__).resolve().parent

# Same 17 audit languages as extended-experiments/utils.py::AUDIT_LANGUAGES_ISO.
AUDIT_LANGUAGES_ISO = [
    "ita", "jpn", "hin", "fra", "ind", "tel", "kor", "swa", "fas",
    "tha", "rus", "zho", "deu", "ben", "ara", "spa", "vie",
]

RELAXED_MIN_LANGUAGES = 14   # design decision 1
INTERSECTION_FALLBACK_THRESHOLD = 15   # design decision 1: trigger relaxed if below this


# ── methodology reused from exp7_alt_aggregators.py / utils.py ───────────────

def kendall_with_bootstrap(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
    """Verbatim from exp7_alt_aggregators.py."""
    tau, _ = kendalltau(x, y)
    rng = np.random.default_rng(SEED)
    boot = []
    n_models = len(x)
    for _ in range(n):
        idx = rng.integers(0, n_models, n_models)
        t, _ = kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            boot.append(t)
    boot = np.asarray(boot)
    return tau, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_loo_macro_scores(
    lang_pivot: pd.DataFrame,
    model_list: list[str],
) -> dict[str, pd.Series]:
    """
    Leave-one-out macro-average aggregator. Adapted from
    exp7_alt_aggregators.build_loo_macro_scores: same mean-over-available-
    other-languages logic (NaNs skipped, i.e. "covered languages only" for
    the relaxed roster), minus the curated tier's English-task term (the
    extended roster has no per-task English scores attached).

    Returns dict: language_iso -> pd.Series(model -> LOO aggregate score),
    where the score for model M at target language X is the mean of M's
    lang_avg over all OTHER audit languages M actually has data for.
    """
    all_langs = list(lang_pivot.columns)
    result: dict[str, pd.Series] = {}
    for target_lang in all_langs:
        other_langs = [l for l in all_langs if l != target_lang]
        scores: dict[str, float] = {}
        for m in model_list:
            if m not in lang_pivot.index:
                scores[m] = float("nan")
                continue
            row = lang_pivot.loc[m]
            vals = [float(row[l]) for l in other_langs if not np.isnan(row[l])]
            scores[m] = float(np.mean(vals)) if vals else float("nan")
        result[target_lang] = pd.Series(scores)
    return result


def pairwise_inversion_rate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pairwise inversion rate with ties counted as 0.5 (design decision 2).
    All C(n,2) pairs count toward the denominator (unlike
    exp7_alt_aggregators.pairwise_inversion_rate, which drops any pair tied
    in x OR y from both numerator and denominator -- see NOTES.md for the
    empirical check that this dataset has zero exact ties, so the two
    conventions give identical numbers here).

    Concordant pair (agrees in x and y): +0
    Tied in x XOR y:                     +0.5
    Tied in both x and y:                +0  (no evidence of inversion)
    Discordant pair:                     +1
    """
    n = len(x)
    total = n * (n - 1) // 2
    if total == 0:
        return float("nan")
    inversions = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x_tie = x[i] == x[j]
            y_tie = y[i] == y[j]
            if x_tie and y_tie:
                continue
            if x_tie or y_tie:
                inversions += 0.5
                continue
            if (x[i] < x[j]) != (y[i] < y[j]):
                inversions += 1.0
    return inversions / total


def count_tied_pairs(x: np.ndarray, y: np.ndarray) -> int:
    """Diagnostic: number of pairs tied in x or y (used only to verify the
    ties=0.5 vs. skip-ties conventions coincide on this data)."""
    n = len(x)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or y[i] == y[j]:
                c += 1
    return c


# ── data loading ───────────────────────────────────────────────────────────

def load_roster_pivot() -> pd.DataFrame:
    """model x language_iso -> lang_avg (NaN where the model isn't in that
    language's extended roster)."""
    df = pd.read_csv(ROSTER_CSV)
    assert set(df["included"].unique()) == {True}, "expected all rows included=True"
    assert df.duplicated(subset=["language_iso", "model"]).sum() == 0, \
        "unexpected duplicate (language, model) rows in extended_roster.csv"
    assert set(df["language_iso"].unique()) == set(AUDIT_LANGUAGES_ISO), \
        "extended_roster.csv languages do not match the expected 17 audit languages"
    pivot = df.pivot(index="model", columns="language_iso", values="lang_avg")
    return pivot[AUDIT_LANGUAGES_ISO]


def load_global_agg_extended() -> pd.Series:
    df = pd.read_csv(GLOBAL_AGG_EXTENDED_CSV)
    s = df.set_index("model")["mteb_agg"]
    assert not s.index.duplicated().any(), "duplicate model rows in mteb_agg_extended.csv"
    return s


def read_full_roster_baseline_median() -> float:
    """Read-only: median inversion_rate across languages in the pre-existing
    full-extended-roster (global-vs-local, no model-set restriction)
    summary. Used only for the calibration figure quoted in NOTES.md."""
    df = pd.read_csv(BASELINE_SUMMARY_EXTENDED_CSV)
    return float(df["inversion_rate"].median())


# ── model-set construction (design decision 1) ────────────────────────────

def build_model_sets(pivot: pd.DataFrame) -> dict[str, list[str]]:
    coverage = pivot.notna().sum(axis=1)
    n_langs = len(AUDIT_LANGUAGES_ISO)
    intersection = sorted(coverage[coverage == n_langs].index.tolist())
    relaxed = sorted(coverage[coverage >= RELAXED_MIN_LANGUAGES].index.tolist())
    return {"intersection": intersection, "relaxed": relaxed}


# ── evaluation ──────────────────────────────────────────────────────────────

def evaluate_variant(
    variant: str,
    roster_name: str,
    models: list[str],
    pivot: pd.DataFrame,
    global_agg: pd.Series,
    loo_scores: dict[str, pd.Series] | None,
    tie_counter: list[int],
) -> list[dict]:
    rows = []
    for lang in AUDIT_LANGUAGES_ISO:
        local = pivot.loc[pivot.index.isin(models), lang].dropna()
        models_here = local.index.tolist()
        if len(models_here) < 5:
            continue
        y = local.to_numpy()

        if variant == "baseline_global":
            x = global_agg.reindex(models_here).to_numpy()
        elif variant == "lang_macro_loo":
            x = loo_scores[lang].reindex(models_here).to_numpy()
        else:
            raise ValueError(variant)

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_m, y_m = x[mask], y[mask]
        if len(x_m) < 5:
            continue

        tau, ci_lo, ci_hi = kendall_with_bootstrap(x_m, y_m)
        inv = pairwise_inversion_rate(x_m, y_m)
        tie_counter[0] += count_tied_pairs(x_m, y_m)
        rows.append({
            "variant":        variant,
            "roster":         roster_name,
            "language_iso":   lang,
            "n_models":       int(len(x_m)),
            "kendall_tau":    tau,
            "tau_ci_lo":      ci_lo,
            "tau_ci_hi":      ci_hi,
            "inversion_rate": inv,
        })
    return rows


def add_median_rows(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    out = list(rows)
    for (variant, roster), grp in df.groupby(["variant", "roster"]):
        out.append({
            "variant":        variant,
            "roster":         roster,
            "language_iso":   "MEDIAN",
            "n_models":       int(grp["n_models"].median()),
            "kendall_tau":    float(grp["kendall_tau"].median()),
            "tau_ci_lo":      float("nan"),
            "tau_ci_hi":      float("nan"),
            "inversion_rate": float(grp["inversion_rate"].median()),
        })
    return out


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    for p in (ROSTER_CSV, GLOBAL_AGG_EXTENDED_CSV, BASELINE_SUMMARY_EXTENDED_CSV):
        assert p.is_file(), f"input file not found: {p}"
    pivot = load_roster_pivot()
    global_agg = load_global_agg_extended()
    model_sets = build_model_sets(pivot)
    intersection_models = model_sets["intersection"]
    relaxed_models = model_sets["relaxed"]

    print(f"Extended-intersection model set (17/17 languages): {len(intersection_models)} models")
    print(f"Relaxed model set (>= {RELAXED_MIN_LANGUAGES}/17 languages): {len(relaxed_models)} models")
    if len(intersection_models) < INTERSECTION_FALLBACK_THRESHOLD:
        print(f"Intersection size < {INTERSECTION_FALLBACK_THRESHOLD}: relaxed variant is REQUIRED by design decision 1.")
    else:
        print(f"Intersection size >= {INTERSECTION_FALLBACK_THRESHOLD}: relaxed fallback not required; "
              f"computed anyway as a bonus robustness check.")

    missing_from_agg = [m for m in relaxed_models if m not in global_agg.index]
    assert not missing_from_agg, f"models missing a global aggregate: {missing_from_agg}"

    tie_counter = [0]
    all_rows: list[dict] = []
    for roster_name, models in (("intersection", intersection_models), ("relaxed", relaxed_models)):
        sub_pivot = pivot.loc[pivot.index.isin(models)]
        loo_scores = build_loo_macro_scores(sub_pivot, models)
        all_rows += evaluate_variant(
            "baseline_global", roster_name, models, pivot, global_agg, None, tie_counter
        )
        all_rows += evaluate_variant(
            "lang_macro_loo", roster_name, models, pivot, global_agg, loo_scores, tie_counter
        )

    full_rows = add_median_rows(all_rows)
    out_df = pd.DataFrame(full_rows)
    out_path = OUT_DIR / "extended_aggregator_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out_df)} rows)")

    print(f"\nTied pairs encountered across all {len(all_rows)} (variant, roster, language) "
          f"evaluations: {tie_counter[0]}  "
          f"({'ties=0.5 and skip-ties conventions are numerically identical on this data' if tie_counter[0] == 0 else 'CHECK: nonzero -- conventions may diverge'})")

    med = out_df[out_df["language_iso"] == "MEDIAN"].set_index(["variant", "roster"])

    full_roster_baseline_median = read_full_roster_baseline_median()
    print(f"\nFull-extended-roster baseline (global vs. local, no model-set restriction, "
          f"from mteb_audit_data/summary_extended.csv): median inversion = "
          f"{full_roster_baseline_median:.4f} ({full_roster_baseline_median * 100:.1f}%)")

    print("\nIntersection-subset (this experiment, n=%d models):" % len(intersection_models))
    print(f"  baseline_global   median tau = {med.loc[('baseline_global', 'intersection'), 'kendall_tau']:.4f}   "
          f"median inversion = {med.loc[('baseline_global', 'intersection'), 'inversion_rate']:.4f}")
    print(f"  lang_macro_loo    median tau = {med.loc[('lang_macro_loo', 'intersection'), 'kendall_tau']:.4f}   "
          f"median inversion = {med.loc[('lang_macro_loo', 'intersection'), 'inversion_rate']:.4f}")

    print("\nRelaxed-subset (bonus robustness check, n=%d models):" % len(relaxed_models))
    print(f"  baseline_global   median tau = {med.loc[('baseline_global', 'relaxed'), 'kendall_tau']:.4f}   "
          f"median inversion = {med.loc[('baseline_global', 'relaxed'), 'inversion_rate']:.4f}")
    print(f"  lang_macro_loo    median tau = {med.loc[('lang_macro_loo', 'relaxed'), 'kendall_tau']:.4f}   "
          f"median inversion = {med.loc[('lang_macro_loo', 'relaxed'), 'inversion_rate']:.4f}")


if __name__ == "__main__":
    main()
