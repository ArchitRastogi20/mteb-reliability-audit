"""
Experiment 1: Multilingual-only aggregator test.

PRIORITY: HIGHEST. Run this first. Closes the causal loop on the cluster
mechanism: the existing manuscript shows (a) cluster has shallower
English-multilingual gap, and (b) 11 of 18 predictor tasks are
English-derived. If filtering the aggregator to only the 7 multilingual
tasks materially reduces the inversion rate, that is direct causal
evidence for the English-task overweighting mechanism.

Inputs (from data/):
  - per_task_ndcg.csv
  - task_taxonomy.csv
  - lang_avg_per_model.csv  (curated tier rows, 17 audit languages)
  - mteb_agg_per_model.csv  (curated tier)

Procedure:
  1. For each curated-tier model, compute three aggregator scores:
        a. Mean(18) — recomputed from per_task_ndcg.csv. Should match the
           public-leaderboard mteb_agg within rounding (sanity check).
        b. Mean(7 multilingual) — average over the 7 multilingual tasks only.
        c. Mean(11 English-derived) — average over the English-derived
           tasks only (used as a contrast).
  2. For each of the 17 audit languages, rank the curated-tier models by
     each aggregator and by the language-specific lang_avg_ndcg. Compute:
        - Kendall tau (with 95 percent CI via 10000 model-resample bootstrap)
        - Pairwise inversion rate
        - Top-1 regret (best lang_avg_ndcg minus the lang_avg_ndcg of the
          model picked top-1 by the aggregator)
  3. Aggregate across languages: median tau, median inversion rate, mean
     top-1 regret. Compare the three aggregators side by side.

Outputs (to outputs/):
  - exp1_aggregator_comparison.csv
        rows: language x aggregator
        cols: kendall_tau, tau_ci_lo, tau_ci_hi, inversion_rate,
              top1_regret, n_models
  - exp1_summary_by_aggregator.csv
        rows: aggregator
        cols: median_tau, median_inversion_rate, mean_top1_regret,
              n_languages_FDR_significant
  - exp1_inversion_rate_by_aggregator.pdf  (and .png)
        per-language bar plot, three bars per language (one per aggregator),
        with the curated-tier headline 33 percent line for reference
  - SUMMARY.md entry with the headline finding

Acceptance:
  - The Mean(18) recomputed score reproduces mteb_agg within median |diff|
    < 0.005 NDCG; if larger, raise an error and stop (loader mismatch).
  - The output CSVs are non-empty and have one row per (language, aggregator)
    or per aggregator as specified.
  - The figure is publication-ready (vector PDF, sans-serif, no chart-junk).

Interpretation guide for the user (do not include in the figure):
  - If Mean(7 multilingual) inversion rate drops to <=20 percent (median),
    that is a strong causal result.
  - If it drops only slightly (e.g. 33 to 28 percent), the mechanism is
    real but partial; soften §4.4 framing accordingly.
  - If it does not drop, the cluster's English-multilingual gap is a marker
    rather than the mechanism; the structural-coverage argument stands but
    the English-overweighting framing needs revision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import utils

SEED = utils.SEED
N_BOOTSTRAP = 10000


def compute_aggregator_scores(
    per_task: pd.DataFrame, task_list: list[str]
) -> pd.Series:
    """Return mean NDCG@10 over the given task list, per model."""
    sub = per_task[per_task["task_name"].isin(task_list)]
    return sub.groupby("model_id")["ndcg_at_10"].mean()


def kendall_with_bootstrap(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
    """Return (tau, ci_lo, ci_hi) with model-level resample bootstrap."""
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


def pairwise_inversion_rate(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of pairs (i, j) where x and y disagree on the ordering."""
    n = len(x)
    inversions = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or y[i] == y[j]:
                continue
            total += 1
            if (x[i] < x[j]) != (y[i] < y[j]):
                inversions += 1
    return inversions / total if total else float("nan")


def top1_regret(scores: np.ndarray, lang_avg: np.ndarray) -> float:
    """Best lang_avg minus lang_avg of the model picked top-1 by `scores`."""
    picked = int(np.argmax(scores))
    return float(np.max(lang_avg) - lang_avg[picked])


def main() -> None:
    per_task = utils.load_per_task_ndcg()
    taxonomy = utils.load_task_taxonomy()
    lang_avg = utils.load_lang_avg_per_model()
    mteb_agg = utils.load_mteb_agg_per_model()

    # Restrict to curated tier.
    curated_models = mteb_agg[mteb_agg["tier"] == "curated"]["model_id"].tolist()
    per_task_curated = per_task[per_task["model_id"].isin(curated_models)]

    multilingual = taxonomy.loc[
        taxonomy["classification"] == "multilingual", "task_name"
    ].tolist()
    english_derived = taxonomy.loc[
        taxonomy["classification"] == "english_derived", "task_name"
    ].tolist()
    all_18 = multilingual + english_derived

    aggregators = {
        "mean_18_full": all_18,
        "mean_7_multilingual": multilingual,
        "mean_11_english": english_derived,
    }

    agg_scores = {
        name: compute_aggregator_scores(per_task_curated, tasks)
        for name, tasks in aggregators.items()
    }

    # Sanity: Mean(18) recomputed should match the public mteb_agg.
    public = mteb_agg.set_index("model_id")["mteb_agg"]
    diff = (agg_scores["mean_18_full"] - public).abs().median()
    if diff > 0.005:
        raise RuntimeError(
            f"Mean(18) recomputation differs from public mteb_agg: median |diff|={diff:.4f}. "
            "Check that per_task_ndcg.csv and mteb_agg_per_model.csv are aligned."
        )

    # Per-language analysis.
    rows = []
    for lang in utils.AUDIT_LANGUAGES_ISO:
        lang_rows = lang_avg[
            (lang_avg["language_iso"] == lang)
            & (lang_avg["tier"] == "curated")
            & (lang_avg["model_id"].isin(curated_models))
        ]
        if lang_rows.empty:
            continue
        models_here = lang_rows["model_id"].tolist()
        lang_vec = lang_rows.set_index("model_id").loc[models_here, "lang_avg_ndcg"].to_numpy()

        for agg_name, scores_series in agg_scores.items():
            agg_vec = scores_series.reindex(models_here).to_numpy()
            mask = ~np.isnan(agg_vec) & ~np.isnan(lang_vec)
            x, y = agg_vec[mask], lang_vec[mask]
            if len(x) < 5:
                continue
            tau, ci_lo, ci_hi = kendall_with_bootstrap(x, y)
            rows.append(
                {
                    "language_iso": lang,
                    "aggregator": agg_name,
                    "n_models": int(len(x)),
                    "kendall_tau": tau,
                    "tau_ci_lo": ci_lo,
                    "tau_ci_hi": ci_hi,
                    "inversion_rate": pairwise_inversion_rate(x, y),
                    "top1_regret": top1_regret(x, y),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(utils.OUTPUTS_DIR / "exp1_aggregator_comparison.csv", index=False)

    # Cross-language summary per aggregator.
    summary = (
        out.groupby("aggregator")
        .agg(
            median_tau=("kendall_tau", "median"),
            median_inversion_rate=("inversion_rate", "median"),
            mean_top1_regret=("top1_regret", "mean"),
            n_languages=("language_iso", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(utils.OUTPUTS_DIR / "exp1_summary_by_aggregator.csv", index=False)

    # Plot: per-language inversion rate, three bars per language.
    _plot_inversion_comparison(out)

    # Append to running summary.
    headline = summary.set_index("aggregator")
    utils.append_summary(
        "Experiment 1: Multilingual-only aggregator",
        [
            f"- Mean(18) recomputed median inversion rate: "
            f"{headline.loc['mean_18_full', 'median_inversion_rate']:.3f}",
            f"- Mean(7 multilingual) median inversion rate: "
            f"{headline.loc['mean_7_multilingual', 'median_inversion_rate']:.3f}",
            f"- Mean(11 English-derived) median inversion rate: "
            f"{headline.loc['mean_11_english', 'median_inversion_rate']:.3f}",
            f"- Mean(7 multilingual) mean top-1 regret: "
            f"{headline.loc['mean_7_multilingual', 'mean_top1_regret']:.4f}",
        ],
    )


def _plot_inversion_comparison(out: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})

    pivot = out.pivot(
        index="language_iso", columns="aggregator", values="inversion_rate"
    )
    pivot = pivot.sort_values("mean_18_full")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(pivot))
    width = 0.27

    ax.bar(x - width, pivot["mean_18_full"], width, label="Mean(18) full")
    ax.bar(x, pivot["mean_7_multilingual"], width, label="Mean(7) multilingual")
    ax.bar(x + width, pivot["mean_11_english"], width, label="Mean(11) English-derived")

    ax.axhline(0.33, linestyle="--", color="grey", linewidth=0.8,
               label="curated-tier headline (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Pairwise inversion rate")
    ax.set_title("Inversion rate by aggregator (curated tier)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(utils.OUTPUTS_DIR / "exp1_inversion_rate_by_aggregator.pdf")
    plt.savefig(utils.OUTPUTS_DIR / "exp1_inversion_rate_by_aggregator.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
