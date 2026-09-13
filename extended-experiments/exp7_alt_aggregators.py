"""
Experiment 7: Alternative aggregators — language-macro-average and 1/n_t reweighting.

Direct extension of Experiment 1. Tests three new aggregators:

  (A) lang_macro_18lang: Equal weight per language.
      Per-language score for 17 audit languages from lang_avg_per_model.csv;
      English score from mean over 11 english_derived tasks in per_task_ndcg.csv.
      Macro-average across 18 languages.

  (B) inv_nt_weighted: 1/n_t task reweighting.
      Each task weighted by 1/(number of languages it covers), normalised to
      sum to 1. Down-weights BelebeleRetrieval (122 languages) and
      MIRACLRetrievalHardNegatives (18 languages); up-weights monolingual tasks.

  (C) lang_macro_multilingual: Language-macro over multilingual tasks only.
      Macro-average of lang_avg_ndcg across 17 audit languages (from
      lang_avg_per_model.csv, which already averages the 7 multilingual tasks
      per language). Equal weight per language; English tasks excluded entirely.

  (D) lang_macro_18lang_loo: Leave-one-out variant of (A).
      When predicting performance for language X, the aggregator score for
      model M is the macro-average over all languages EXCEPT X, plus the
      English score. Eliminates the partial circularity of (A).

  (E) lang_macro_multilingual_loo: Leave-one-out variant of (C).
      When predicting for language X, aggregator = macro-average of
      lang_avg_ndcg over all audit languages EXCEPT X. No English score.

Aggregators (A)–(C) are self-inclusive (target language contributes 1/17 or
1/18 to its own predictor). Aggregators (D)–(E) are strictly held-out.
Evaluation mirrors Experiment 1: per-language Kendall tau and pairwise inversion
rate vs. ground-truth lang_avg_ndcg for 17 audit languages.

Inputs (from data/):
  - per_task_ndcg.csv
  - task_taxonomy.csv
  - lang_avg_per_model.csv
  - mteb_agg_per_model.csv

Outputs (to outputs/):
  - exp7_aggregator_comparison.csv
        rows: language_iso x aggregator
        cols: aggregator, language_iso, n_models, kendall_tau, tau_ci_lo,
              tau_ci_hi, inversion_rate, top1_regret
  - exp7_summary_by_aggregator.csv
        rows: aggregator
        cols: aggregator, median_tau, median_inversion_rate, mean_top1_regret,
              n_languages
  - exp7_inversion_rate_by_aggregator.pdf (and .png)
  - SUMMARY.md entry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import utils

SEED = utils.SEED
N_BOOTSTRAP = 10_000

# Number of distinct languages each MMTEB task covers.
# Used to compute 1/n_t weights (aggregator B).
TASK_N_LANGUAGES: dict[str, int] = {
    "AILAStatutes":                    1,
    "ArguAna":                         1,
    "BelebeleRetrieval":             122,   # 122 language scripts
    "CovidRetrieval":                  1,   # Chinese only
    "HagridRetrieval":                 1,
    "LEMBPasskeyRetrieval":            1,
    "LegalBenchCorporateLobbying":     1,
    "MIRACLRetrievalHardNegatives":   18,   # 18 MIRACL languages
    "MLQARetrieval":                   7,   # 7 query languages
    "SCIDOCS":                         1,
    "SpartQA":                         1,
    "StackOverflowQA":                 1,
    "StatcanDialogueDatasetRetrieval": 2,   # EN + FR
    "TRECCOVID":                       1,
    "TempReasonL1":                    1,
    "TwitterHjerneRetrieval":          1,   # Danish only
    "WikipediaRetrievalMultilingual": 16,   # 16 Wikipedia languages
    "WinoGrande":                      1,
}


# ── helpers (mirrors exp1) ────────────────────────────────────────────────────

def kendall_with_bootstrap(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
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
    n = len(x)
    inversions = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or y[i] == y[j]:
                continue
            total += 1
            if (x[i] < x[j]) != (y[i] < y[j]):
                inversions += 1
    return inversions / total if total else float("nan")


def top1_regret(scores: np.ndarray, lang_avg: np.ndarray) -> float:
    picked = int(np.argmax(scores))
    return float(np.max(lang_avg) - lang_avg[picked])


# ── aggregator builders ───────────────────────────────────────────────────────

def build_lang_macro_18lang(
    per_task: pd.DataFrame,
    lang_avg: pd.DataFrame,
    english_tasks: list[str],
    curated_models: list[str],
) -> pd.Series:
    """Mean over (17 audit-language scores + English score). Equal weight per language."""
    # Per-language scores for 17 audit languages.
    lang_means = (
        lang_avg[lang_avg["tier"] == "curated"]
        .groupby("model_id")["lang_avg_ndcg"]
        .agg(list)
    )
    # English score = mean over english_derived tasks.
    eng_means = (
        per_task[per_task["task_name"].isin(english_tasks)]
        .groupby("model_id")["ndcg_at_10"]
        .mean()
    )
    scores = {}
    for m in curated_models:
        lang_list = lang_means.get(m, [])
        eng = eng_means.get(m, float("nan"))
        all_vals = [v for v in lang_list if not np.isnan(v)]
        if not np.isnan(eng):
            all_vals.append(eng)
        scores[m] = float(np.mean(all_vals)) if all_vals else float("nan")
    return pd.Series(scores)


def build_inv_nt_weighted(
    per_task: pd.DataFrame,
    curated_models: list[str],
) -> pd.Series:
    """Weighted mean of task scores; weight_t = 1/n_t, normalised."""
    tasks = list(TASK_N_LANGUAGES.keys())
    weights = {t: 1.0 / TASK_N_LANGUAGES[t] for t in tasks}
    w_sum = sum(weights.values())
    norm_weights = {t: w / w_sum for t, w in weights.items()}

    sub = per_task[per_task["task_name"].isin(tasks) & per_task["model_id"].isin(curated_models)]
    scores = {}
    for m, grp in sub.groupby("model_id"):
        task_map = grp.set_index("task_name")["ndcg_at_10"].to_dict()
        available_w = sum(norm_weights[t] for t in task_map)
        if available_w == 0:
            scores[m] = float("nan")
            continue
        s = sum(task_map[t] * norm_weights[t] for t in task_map) / available_w
        scores[m] = float(s)
    return pd.Series(scores)


def build_lang_macro_multilingual(
    lang_avg: pd.DataFrame,
    curated_models: list[str],
) -> pd.Series:
    """Macro-average of lang_avg_ndcg over 17 audit languages. Multilingual tasks only."""
    sub = lang_avg[(lang_avg["tier"] == "curated") & (lang_avg["model_id"].isin(curated_models))]
    return sub.groupby("model_id")["lang_avg_ndcg"].mean()


def build_loo_macro_scores(
    lang_pivot: pd.DataFrame,
    eng_means: pd.Series | None,
    curated_models: list[str],
    include_english: bool,
) -> dict[str, pd.Series]:
    """
    Leave-one-out macro-average aggregator.

    For each target language X, returns a pd.Series of per-model aggregator scores
    computed as the macro-average over ALL audit languages EXCEPT X. If
    include_english=True, the English task mean (pre-computed in eng_means) is
    also included in the average.

    Args:
        lang_pivot: model_id × language_iso DataFrame of lang_avg_ndcg values.
        eng_means:  Per-model mean over english_derived tasks (None if not needed).
        curated_models: Ordered list of curated-tier model IDs.
        include_english: Whether to append the English score to the average.

    Returns:
        dict mapping language_iso -> pd.Series(model_id -> loo_score).
    """
    all_langs = list(lang_pivot.columns)
    result: dict[str, pd.Series] = {}

    for target_lang in all_langs:
        other_langs = [l for l in all_langs if l != target_lang]
        scores: dict[str, float] = {}
        for m in curated_models:
            if m not in lang_pivot.index:
                scores[m] = float("nan")
                continue
            row = lang_pivot.loc[m]
            vals = [float(row[l]) for l in other_langs if not np.isnan(row[l])]
            if include_english and eng_means is not None:
                eng = float(eng_means.get(m, float("nan")))
                if not np.isnan(eng):
                    vals.append(eng)
            scores[m] = float(np.mean(vals)) if vals else float("nan")
        result[target_lang] = pd.Series(scores)

    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    per_task = utils.load_per_task_ndcg()
    taxonomy = utils.load_task_taxonomy()
    lang_avg = utils.load_lang_avg_per_model()
    mteb_agg = utils.load_mteb_agg_per_model()

    curated_models = mteb_agg[mteb_agg["tier"] == "curated"]["model_id"].tolist()
    per_task_curated = per_task[per_task["model_id"].isin(curated_models)]

    english_tasks = taxonomy.loc[
        taxonomy["classification"] == "english_derived", "task_name"
    ].tolist()

    agg_scores = {
        "lang_macro_18lang": build_lang_macro_18lang(
            per_task_curated, lang_avg, english_tasks, curated_models
        ),
        "inv_nt_weighted": build_inv_nt_weighted(per_task_curated, curated_models),
        "lang_macro_multilingual": build_lang_macro_multilingual(lang_avg, curated_models),
    }

    # Precompute the model × language pivot and English means for LOO builders.
    _lang_pivot = (
        lang_avg[(lang_avg["tier"] == "curated") & (lang_avg["model_id"].isin(curated_models))]
        .pivot(index="model_id", columns="language_iso", values="lang_avg_ndcg")
    )
    _eng_means = (
        per_task_curated[per_task_curated["task_name"].isin(english_tasks)]
        .groupby("model_id")["ndcg_at_10"]
        .mean()
    )

    # LOO aggregators: aggregator score for (model M, language X) excludes X.
    loo_agg_scores: dict[str, dict[str, pd.Series]] = {
        "lang_macro_18lang_loo": build_loo_macro_scores(
            _lang_pivot, _eng_means, curated_models, include_english=True
        ),
        "lang_macro_multilingual_loo": build_loo_macro_scores(
            _lang_pivot, None, curated_models, include_english=False
        ),
    }

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

        # Self-inclusive aggregators (one score per model, language-independent).
        for agg_name, scores_series in agg_scores.items():
            agg_vec = scores_series.reindex(models_here).to_numpy()
            mask = ~np.isnan(agg_vec) & ~np.isnan(lang_vec)
            x, y = agg_vec[mask], lang_vec[mask]
            if len(x) < 5:
                continue
            tau, ci_lo, ci_hi = kendall_with_bootstrap(x, y)
            rows.append({
                "aggregator":     agg_name,
                "language_iso":   lang,
                "n_models":       int(len(x)),
                "kendall_tau":    tau,
                "tau_ci_lo":      ci_lo,
                "tau_ci_hi":      ci_hi,
                "inversion_rate": pairwise_inversion_rate(x, y),
                "top1_regret":    top1_regret(x, y),
            })

        # LOO aggregators (score depends on target language).
        for agg_name, lang_scores_dict in loo_agg_scores.items():
            if lang not in lang_scores_dict:
                continue
            agg_vec = lang_scores_dict[lang].reindex(models_here).to_numpy()
            mask = ~np.isnan(agg_vec) & ~np.isnan(lang_vec)
            x, y = agg_vec[mask], lang_vec[mask]
            if len(x) < 5:
                continue
            tau, ci_lo, ci_hi = kendall_with_bootstrap(x, y)
            rows.append({
                "aggregator":     agg_name,
                "language_iso":   lang,
                "n_models":       int(len(x)),
                "kendall_tau":    tau,
                "tau_ci_lo":      ci_lo,
                "tau_ci_hi":      ci_hi,
                "inversion_rate": pairwise_inversion_rate(x, y),
                "top1_regret":    top1_regret(x, y),
            })

    out = pd.DataFrame(rows)
    out.to_csv(utils.OUTPUTS_DIR / "exp7_aggregator_comparison.csv", index=False)

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
    summary.to_csv(utils.OUTPUTS_DIR / "exp7_summary_by_aggregator.csv", index=False)

    _plot_inversion(out)
    _plot_loo_comparison(out)

    h = summary.set_index("aggregator")
    utils.append_summary(
        "Experiment 7: Alternative aggregators (with LOO)",
        [
            "Self-inclusive aggregators (target language contributes 1/17 or 1/18 to its own predictor):",
            f"- lang_macro_18lang        inversion rate: "
            f"{h.loc['lang_macro_18lang','median_inversion_rate']:.3f}  "
            f"(tau = {h.loc['lang_macro_18lang','median_tau']:.3f})",
            f"- lang_macro_multilingual  inversion rate: "
            f"{h.loc['lang_macro_multilingual','median_inversion_rate']:.3f}  "
            f"(tau = {h.loc['lang_macro_multilingual','median_tau']:.3f})",
            f"- inv_nt_weighted          inversion rate: "
            f"{h.loc['inv_nt_weighted','median_inversion_rate']:.3f}  "
            f"(tau = {h.loc['inv_nt_weighted','median_tau']:.3f})",
            "Leave-one-out aggregators (target language strictly excluded from predictor):",
            f"- lang_macro_18lang_loo    inversion rate: "
            f"{h.loc['lang_macro_18lang_loo','median_inversion_rate']:.3f}  "
            f"(tau = {h.loc['lang_macro_18lang_loo','median_tau']:.3f})",
            f"- lang_macro_multilingual_loo inversion rate: "
            f"{h.loc['lang_macro_multilingual_loo','median_inversion_rate']:.3f}  "
            f"(tau = {h.loc['lang_macro_multilingual_loo','median_tau']:.3f})",
        ],
    )


def _plot_inversion(out: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})

    agg_order = ["lang_macro_18lang", "inv_nt_weighted", "lang_macro_multilingual"]
    agg_labels = {
        "lang_macro_18lang":       "Lang-macro (18 lang)",
        "inv_nt_weighted":         "1/n_t reweighted",
        "lang_macro_multilingual": "Lang-macro (multi only)",
    }

    pivot = out.pivot(index="language_iso", columns="aggregator", values="inversion_rate")
    pivot = pivot[agg_order]
    pivot = pivot.sort_values("lang_macro_multilingual")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(pivot))
    width = 0.27
    colours = ["tab:blue", "tab:orange", "tab:green"]
    for k, (col, colour) in enumerate(zip(agg_order, colours)):
        offset = (k - 1) * width
        ax.bar(x + offset, pivot[col], width, label=agg_labels[col], color=colour, alpha=0.85)

    ax.axhline(0.33, linestyle="--", color="grey", linewidth=0.8,
               label="headline curated-tier (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Pairwise inversion rate")
    ax.set_title("Inversion rate: alternative aggregators (curated tier, 17 languages)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(utils.OUTPUTS_DIR / "exp7_inversion_rate_by_aggregator.pdf")
    plt.savefig(utils.OUTPUTS_DIR / "exp7_inversion_rate_by_aggregator.png", dpi=200)
    plt.close()


def _plot_loo_comparison(out: pd.DataFrame) -> None:
    """Side-by-side comparison of self-inclusive vs LOO for the two macro aggregators."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})

    pairs = [
        ("lang_macro_18lang",       "lang_macro_18lang_loo",       "Lang-macro 18lang"),
        ("lang_macro_multilingual", "lang_macro_multilingual_loo", "Lang-macro multi-only"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (self_name, loo_name, title) in zip(axes, pairs):
        sub = out[out["aggregator"].isin([self_name, loo_name])]
        if sub.empty:
            continue
        pivot = sub.pivot(index="language_iso", columns="aggregator", values="inversion_rate")
        pivot = pivot.sort_values(loo_name if loo_name in pivot.columns else self_name)
        x = np.arange(len(pivot))
        width = 0.38
        if self_name in pivot.columns:
            ax.bar(x - width / 2, pivot[self_name], width,
                   label="Self-inclusive", color="tab:blue", alpha=0.85)
        if loo_name in pivot.columns:
            ax.bar(x + width / 2, pivot[loo_name], width,
                   label="Leave-one-out", color="tab:orange", alpha=0.85)
        ax.axhline(0.33, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Pairwise inversion rate")
    fig.suptitle("Self-inclusive vs. LOO language-macro aggregators", fontsize=10)
    plt.tight_layout()
    plt.savefig(utils.OUTPUTS_DIR / "exp7_loo_comparison.pdf")
    plt.savefig(utils.OUTPUTS_DIR / "exp7_loo_comparison.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
