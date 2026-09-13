"""
Extended-tier resampling: adjudicate between sample-size vs. selection-bias
explanations for the curated→extended Kendall τ jump (0.34 → 0.69 median).

Method: draw 100 random size-25 subsets of each language's extended-tier models
and recompute Kendall τ. If resampled τ ≈ 0.69, extended's high τ is driven by
model composition (multilingual-first self-selection into the benchmark), not by
having more models. If resampled τ ≈ 0.34, the jump is a sample-size artifact.

Produces:
  analysis/extended_resample_tau.csv -- per-language resampling stats (17 rows)
  analysis/resample_summary.json     -- cross-language summary + interpretation
  paper/resample_paragraph.tex       -- LaTeX paragraph with actual numbers

Data sources (pre-computed by 01_download.py and 03_analyze.py):
  global/mteb_agg_extended.csv
  <lang>/lang_avg_extended.csv
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent   # mteb-ranking-audit/
ANALYSIS = BASE / "analysis"
PAPER    = BASE / "paper"
ANALYSIS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

# ── Parameters ───────────────────────────────────────────────────────────────
SEED        = 20260510
N_SUBSETS   = 100
SUBSET_SIZE = 25

# Headline τ values from the full-tier analyses (used for interpretation).
CURATED_TAU_MEDIAN  = 0.34
EXTENDED_TAU_MEDIAN = 0.69

# ── Language list ─────────────────────────────────────────────────────────────
LANG_META = {
    "ita": ("Italian",    "case_studies"),
    "jpn": ("Japanese",   "case_studies"),
    "hin": ("Hindi",      "case_studies"),
    "ara": ("Arabic",     "audit_languages"),
    "zho": ("Chinese",    "audit_languages"),
    "deu": ("German",     "audit_languages"),
    "spa": ("Spanish",    "audit_languages"),
    "rus": ("Russian",    "audit_languages"),
    "fra": ("French",     "audit_languages"),
    "kor": ("Korean",     "audit_languages"),
    "ben": ("Bengali",    "audit_languages"),
    "vie": ("Vietnamese", "audit_languages"),
    "ind": ("Indonesian", "audit_languages"),
    "fas": ("Persian",    "audit_languages"),
    "swa": ("Swahili",    "audit_languages"),
    "tha": ("Thai",       "audit_languages"),
    "tel": ("Telugu",     "audit_languages"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_extended(iso3: str, subdir: str) -> pd.DataFrame | None:
    """Return merged (local, global) dataframe for the extended tier."""
    lang_path   = BASE / subdir / iso3 / "lang_avg_extended.csv"
    global_path = BASE / "global" / "mteb_agg_extended.csv"
    if not lang_path.exists() or not global_path.exists():
        return None
    lang_df   = pd.read_csv(lang_path)
    global_df = pd.read_csv(global_path)
    merged = lang_df.merge(global_df[["model", "mteb_agg"]], on="model", how="inner")
    return merged if len(merged) >= 2 else None


# ── Kendall τ and inversion rate ──────────────────────────────────────────────

def tau_and_inv(global_scores: np.ndarray, local_scores: np.ndarray) -> tuple[float, float]:
    """Compute Kendall τ and pairwise inversion rate on a subset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_val, _ = kendalltau(global_scores, local_scores)

    n       = len(global_scores)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return float("nan"), float("nan")

    g, l = global_scores, local_scores
    n_inv = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (g[i] > g[j]) != (l[i] > l[j])
    )
    return float(tau_val), n_inv / n_pairs


def resample_language(merged: pd.DataFrame, rng: np.random.Generator) -> dict:
    n = len(merged)
    global_scores = merged["mteb_agg"].values
    local_scores  = merged["lang_avg"].values

    taus, invs = [], []
    for _ in range(N_SUBSETS):
        idx = rng.choice(n, size=SUBSET_SIZE, replace=False)
        t, inv = tau_and_inv(global_scores[idx], local_scores[idx])
        if not np.isnan(t):
            taus.append(t)
            invs.append(inv)

    if not taus:
        return {"tau_median": float("nan"), "tau_iqr_lo": float("nan"),
                "tau_iqr_hi": float("nan"), "tau_p025": float("nan"),
                "tau_p975": float("nan"), "inv_median": float("nan"),
                "n_subsets_ok": 0}

    arr = np.array(taus)
    return {
        "tau_median":   float(np.median(arr)),
        "tau_iqr_lo":   float(np.percentile(arr, 25)),
        "tau_iqr_hi":   float(np.percentile(arr, 75)),
        "tau_p025":     float(np.percentile(arr, 2.5)),
        "tau_p975":     float(np.percentile(arr, 97.5)),
        "inv_median":   float(np.median(invs)),
        "n_subsets_ok": len(taus),
    }


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret(median_tau: float) -> str:
    """
    Classify the resampled median tau relative to curated and extended baselines.

    H1 (sample_size_artifact): extended τ is high just because n=57 > 25.
      Evidence: resampled τ at n=25 falls close to the curated level (0.34).

    H2 (selection_bias_in_extended): extended τ is high because all extended
      models are multilingual-first (self-selected into MMTEB). Even 25 random
      extended models preserve the high τ.
      Evidence: resampled τ at n=25 stays close to the extended level (0.69).

    Rules:
      selection_bias_in_extended -- resampled >= extended - 0.10 (>= 0.59)
      sample_size_artifact       -- resampled <= curated + 0.10 (<= 0.44)
      ambiguous                  -- otherwise
    """
    if median_tau >= EXTENDED_TAU_MEDIAN - 0.10:
        return "selection_bias_in_extended"
    if median_tau <= CURATED_TAU_MEDIAN + 0.10:
        return "sample_size_artifact"
    return "ambiguous"


# ── Output writers ────────────────────────────────────────────────────────────

def write_paragraph(summary: dict) -> None:
    med    = summary["cross_language_median_tau_resampled"]
    n_ql   = summary["n_languages_resamplable"]
    interp = summary["interpretation"]

    if np.isnan(med):
        paragraph = (
            "Resampling could not be completed (no language had $n_{\\text{ext}} "
            f"\\geq {SUBSET_SIZE}$).\n"
        )
    elif interp == "selection_bias_in_extended":
        paragraph = (
            f"\\paragraph{{Roster size vs.\\ selection bias.}}\n"
            f"A competing explanation for the curated-to-extended jump is mechanical:\n"
            f"larger samples resolve $\\tau$ better when true ranking signal is shared.\n"
            f"We rule this out by drawing {N_SUBSETS} random size-{SUBSET_SIZE} subsets of each\n"
            f"language's extended-tier models (where $n_{{\\text{{ext}}}} \\geq {SUBSET_SIZE}$;\n"
            f"{n_ql}/17 languages qualify) and recomputing per-language $\\tau$.\n"
            f"The median across-subset $\\tau$ is {med:.2f}\n"
            f"(vs.\\ {CURATED_TAU_MEDIAN:.2f} in the curated tier and "
            f"{EXTENDED_TAU_MEDIAN:.2f} in the full extended tier).\n"
            f"Random size-matched subsamples of the extended tier preserve its high $\\tau$,\n"
            f"confirming that the extended tier's inflated correlation is driven by\n"
            f"\\emph{{model composition}} (multilingual-first self-selection into the benchmark),\n"
            f"not by sample size. The curated tier's lower $\\tau$ is therefore a real signal:\n"
            f"it arises from the English-heavy additions that expose the predictor's failures.\n"
        )
    elif interp == "sample_size_artifact":
        paragraph = (
            f"\\paragraph{{Roster size vs.\\ selection bias.}}\n"
            f"Random size-{SUBSET_SIZE} subsamples of the extended tier yield median "
            f"$\\tau = {med:.2f}$,\n"
            f"close to the curated-tier level ({CURATED_TAU_MEDIAN:.2f}).\n"
            f"This suggests the extended tier's high $\\tau$ ({EXTENDED_TAU_MEDIAN:.2f}) "
            f"is a sample-size artifact; both\nrosters share a similar underlying correlation "
            f"at the same sample size.\n"
            f"Note: this result does not invalidate the hidden-failure finding,\n"
            f"which holds within each tier independently.\n"
        )
    else:
        paragraph = (
            f"\\paragraph{{Roster size vs.\\ selection bias.}}\n"
            f"Random size-{SUBSET_SIZE} subsamples of the extended tier yield median "
            f"$\\tau = {med:.2f}$\n"
            f"({n_ql}/17 languages resamplable), falling between the curated "
            f"({CURATED_TAU_MEDIAN:.2f}) and full-extended ({EXTENDED_TAU_MEDIAN:.2f}) baselines.\n"
            f"The result is ambiguous between sample-size and composition-bias accounts.\n"
        )

    out = PAPER / "resample_paragraph.tex"
    out.write_text(paragraph, encoding="utf-8")
    print(f"Wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng  = np.random.default_rng(SEED)
    rows = []

    for iso3, (lang_name, subdir) in LANG_META.items():
        merged = load_extended(iso3, subdir)
        if merged is None:
            print(f"  {iso3}: no extended data")
            rows.append({"language_iso": iso3, "n_extended": 0, "status": "no_data",
                         "tau_median": float("nan"), "tau_iqr_lo": float("nan"),
                         "tau_iqr_hi": float("nan"), "tau_p025": float("nan"),
                         "tau_p975": float("nan"), "inv_median": float("nan"),
                         "n_subsets_ok": 0})
            continue

        n_ext = len(merged)
        if n_ext < SUBSET_SIZE:
            print(f"  {iso3}: n={n_ext} < {SUBSET_SIZE} — not resamplable")
            rows.append({"language_iso": iso3, "n_extended": n_ext, "status": "not_resamplable",
                         "tau_median": float("nan"), "tau_iqr_lo": float("nan"),
                         "tau_iqr_hi": float("nan"), "tau_p025": float("nan"),
                         "tau_p975": float("nan"), "inv_median": float("nan"),
                         "n_subsets_ok": 0})
            continue

        stats = resample_language(merged, rng)
        print(
            f"  {iso3}: n={n_ext}, tau_median={stats['tau_median']:.3f} "
            f"[{stats['tau_iqr_lo']:.3f}, {stats['tau_iqr_hi']:.3f}]"
        )
        rows.append({"language_iso": iso3, "n_extended": n_ext, "status": "ok", **stats})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(ANALYSIS / "extended_resample_tau.csv", index=False)
    print(f"\nWrote analysis/extended_resample_tau.csv ({len(result_df)} rows)")

    valid_taus = result_df[result_df["status"] == "ok"]["tau_median"].dropna()
    cross_med  = float(valid_taus.median()) if len(valid_taus) > 0 else float("nan")
    n_ok       = int((result_df["status"] == "ok").sum())
    interp     = interpret(cross_med) if not np.isnan(cross_med) else "no_data"

    summary = {
        "cross_language_median_tau_resampled": cross_med,
        "curated_tier_median_tau":             CURATED_TAU_MEDIAN,
        "extended_tier_median_tau":            EXTENDED_TAU_MEDIAN,
        "n_languages_resamplable":             n_ok,
        "n_subsets_per_language":              N_SUBSETS,
        "subset_size":                         SUBSET_SIZE,
        "seed":                                SEED,
        "interpretation":                      interp,
    }
    (ANALYSIS / "resample_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote analysis/resample_summary.json")
    print(f"\nCross-language median tau (resampled): {cross_med:.3f}")
    print(f"Interpretation: {interp}")

    if interp == "sample_size_artifact":
        print(
            "\nNOTE: resampled tau ~= curated tau. Extended's high tau may partly reflect\n"
            "a sample-size artifact. Consider adding a caveat to the selection-bias framing.\n"
            "Hidden-failure findings hold independently and are not affected by this result.",
            file=sys.stderr,
        )

    write_paragraph(summary)


if __name__ == "__main__":
    main()
