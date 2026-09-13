"""
Experiment 6: Per-language top-1 regret breakdown.

The manuscript reports a curated-tier mean top-1 regret of 3.3 NDCG points
across 17 languages, with max on Japanese (5.5). A per-language breakdown
shows where the regret concentrates and lets a reader see whether it
correlates with structural features (e.g. number of MTEB tasks for the
language).

Inputs (from data/):
  - lang_avg_per_model.csv  (curated tier, 17 audit languages)
  - mteb_agg_per_model.csv  (curated tier, with global_rank)

Procedure:
  1. For each language, identify the curated-tier model with the best (lowest)
     global_rank by mteb_agg. This is the "leaderboard pick".
  2. Identify the model with the best lang_avg_ndcg for that language. This
     is the "language oracle".
  3. top_1_regret[language] = oracle_score - leaderboard_pick_score, in
     NDCG@10 points.
  4. Repeat for top-3 (mean of best 3 by mteb_agg vs best 3 by lang_avg)
     and top-5.
  5. Output a table and bar chart sorted by top-1 regret descending.
  6. Compute Spearman correlation between top-1 regret and the number of
     MTEB retrieval tasks for the language (n_t in {2, 3, 4}). The
     mechanism account predicts higher regret for languages with fewer
     tasks.

Inputs additionally needed:
  - For step 6: a small constant lookup of n_t per language. Hardcode it
     here (matches the manuscript Table 1).

Outputs (to outputs/):
  - exp6_top_k_regret.csv
        rows: language
        cols: language_iso, n_tasks, top1_regret, top3_regret,
              top5_regret, leaderboard_pick, oracle_pick
  - exp6_top1_regret_per_language.pdf  (and .png)
  - SUMMARY.md entry with the mean/median top-1 regret and the regret-vs-n_t
    Spearman correlation.

Acceptance:
  - Mean top-1 regret across 17 languages is approximately 0.033 (matches
    the manuscript's 3.3 NDCG points). If off by more than 0.005, raise.
  - Top-3 and top-5 regret are near zero on average, matching the manuscript
    finding that the binding cost is concentrated at top-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import utils

# Number of MMTEB(Multilingual) retrieval tasks per audit language (from Table 1).
N_TASKS_PER_LANG = {
    "fra": 2, "ind": 2, "tel": 2, "kor": 2, "swa": 2, "fas": 3, "tha": 2,
    "jpn": 2, "rus": 2, "ita": 2, "zho": 3, "deu": 4, "ben": 3, "ara": 3,
    "hin": 4, "spa": 3, "vie": 2,
}


def top_k_regret(scores_a: pd.Series, scores_b: pd.Series, k: int) -> float:
    """Mean of top-k by `scores_a` evaluated under `scores_b`, vs top-k by `scores_b`."""
    aligned = pd.concat([scores_a.rename("a"), scores_b.rename("b")], axis=1).dropna()
    by_a = aligned.sort_values("a", ascending=False).head(k)
    by_b = aligned.sort_values("b", ascending=False).head(k)
    return float(by_b["b"].mean() - by_a["b"].mean())


def main() -> None:
    lang_avg = utils.load_lang_avg_per_model()
    mteb_agg = utils.load_mteb_agg_per_model()

    curated_models = mteb_agg[mteb_agg["tier"] == "curated"]["model_id"].tolist()
    lang_avg_curated = lang_avg[
        (lang_avg["tier"] == "curated") & (lang_avg["model_id"].isin(curated_models))
    ]
    mteb_curated = mteb_agg[mteb_agg["tier"] == "curated"].set_index("model_id")["mteb_agg"]

    rows = []
    for lang in utils.AUDIT_LANGUAGES_ISO:
        sub = lang_avg_curated[lang_avg_curated["language_iso"] == lang]
        if sub.empty:
            continue
        sub = sub.set_index("model_id")["lang_avg_ndcg"]
        # Align mteb_curated to the same model index.
        agg = mteb_curated.reindex(sub.index)
        leaderboard_pick = agg.idxmax()
        oracle_pick = sub.idxmax()
        top1 = float(sub[oracle_pick] - sub[leaderboard_pick])

        rows.append(
            {
                "language_iso": lang,
                "n_tasks": N_TASKS_PER_LANG[lang],
                "top1_regret": top1,
                "top3_regret": top_k_regret(agg, sub, 3),
                "top5_regret": top_k_regret(agg, sub, 5),
                "leaderboard_pick": leaderboard_pick,
                "oracle_pick": oracle_pick,
            }
        )

    out = pd.DataFrame(rows).sort_values("top1_regret", ascending=False)
    out.to_csv(utils.OUTPUTS_DIR / "exp6_top_k_regret.csv", index=False)

    mean_top1 = out["top1_regret"].mean()
    if abs(mean_top1 - 0.033) > 0.005:
        raise RuntimeError(
            f"Mean top-1 regret {mean_top1:.4f} differs from manuscript value 0.033 "
            "by more than 0.005. Check inputs."
        )

    rho, p_rho = spearmanr(out["n_tasks"], out["top1_regret"])

    _plot_per_language(out)

    utils.append_summary(
        "Experiment 6: Per-language top-1 regret",
        [
            f"- Mean top-1 regret across 17 languages: {mean_top1:.4f} NDCG@10",
            f"- Median top-1 regret: {out['top1_regret'].median():.4f}",
            f"- Mean top-3 regret: {out['top3_regret'].mean():.4f}",
            f"- Mean top-5 regret: {out['top5_regret'].mean():.4f}",
            f"- Spearman(top1_regret, n_tasks) = {rho:.3f}, p = {p_rho:.3f}",
            "- Figure: outputs/exp6_top1_regret_per_language.pdf",
        ],
    )


def _plot_per_language(out: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colours = {2: "tab:blue", 3: "tab:orange", 4: "tab:red"}
    bar_colours = [colours[n] for n in out["n_tasks"]]
    ax.bar(out["language_iso"], out["top1_regret"], color=bar_colours)
    ax.set_ylabel("Top-1 regret (NDCG@10)")
    ax.axhline(out["top1_regret"].mean(), linestyle="--", color="black",
               linewidth=0.8, label=f"mean = {out['top1_regret'].mean():.3f}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    labels = [f"n_tasks = {k}" for k in colours]
    ax.legend(handles + [ax.get_lines()[0]], labels + ["mean"],
              loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Curated-tier top-1 regret by language", fontsize=10)
    plt.tight_layout()
    plt.savefig(utils.OUTPUTS_DIR / "exp6_top1_regret_per_language.pdf")
    plt.savefig(utils.OUTPUTS_DIR / "exp6_top1_regret_per_language.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
