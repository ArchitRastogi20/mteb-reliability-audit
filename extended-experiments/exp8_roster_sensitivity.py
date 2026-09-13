"""
Experiment 8: Curated-roster sensitivity.

Tests whether the per-language ranking disagreement (Kendall tau and pairwise
inversion rate) is stable when the 25-model curated roster is filtered to three
alternate sub-rosters. Uses the mean(18) aggregator throughout so results are
directly comparable to the headline curated-tier figures.

Roster variants:
  (A) no_instruction_tuned  (~20 models)
      Removes models that use task-specific instruction prefixes during embedding
      (following the InstructOR/GRIT paradigm). See INSTRUCTION_TUNED set below.

  (B) ge10_lang_coverage  (all 25 models pass; kept for completeness)
      Retains only models with valid scores in >= 10 of the 17 audit languages
      in lang_avg_per_model.csv. ml-e5-small (12 languages) passes; all 25 qualify.

  (C) family_balanced  (~18 models)
      At most one model per backbone family. Largest-by-parameter-count kept
      when a family has multiple representatives in the curated tier.

For each alternate roster, computes per-language Kendall tau and pairwise inversion
rate against lang_avg_ndcg ground truth, then aggregates across 17 languages.
Requires at least 5 models per language after filtering.

Inputs (from data/):
  - per_task_ndcg.csv
  - task_taxonomy.csv
  - lang_avg_per_model.csv
  - mteb_agg_per_model.csv

Outputs (to outputs/):
  - exp8_roster_sensitivity.csv
        rows: (roster_variant, language_iso)
        cols: roster_variant, language_iso, n_models, kendall_tau, inversion_rate
  - exp8_roster_summary.csv
        rows: roster_variant
        cols: roster_variant, n_models, median_tau, median_inversion_rate, n_languages
  - SUMMARY.md entry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import utils

SEED = utils.SEED
N_BOOTSTRAP = 10_000

# Models that use task-specific instruction prefixes during embedding inference.
INSTRUCTION_TUNED: set[str] = {
    "GritLM-7B",        # GRIT: explicitly instruction-tuned generative-representational model
    "KaLM-Gemma3-12B",  # Gemma3-based; tencent model card lists instruction prefix use
    "nemotron-8b",       # llama-embed-nemotron: instruction-following LLM backbone
    "inf-retriever-v1",  # Infly: instruction-following retrieval model
    "BOOM-4B",           # ICT-TIME/Querit: LLM-based with instruction prefix
}

# One representative per backbone family (largest by parameter count).
FAMILY_BALANCED: set[str] = {
    "KaLM-Gemma3-12B",   # Gemma3 family
    "Qwen3-8B",          # Qwen3 family (largest)
    "harrier-0.6b",      # unique
    "pplx-embed-4b",     # pplx family (largest)
    "nemotron-8b",       # unique
    "inf-retriever-v1",  # unique
    "F2LLM-v2-14B",      # F2LLM family (largest)
    "Seed1.6-embed",     # unique
    "bge-m3",            # unique
    "ml-e5-small",       # unique (multilingual-e5 family separate from bge)
    "jina-v5-small",     # jina-v5 family (largest in curated)
    "voyage-3.5",        # unique
    "zembed-1",          # unique
    "embeddinggemma-300m",# unique
    "BOOM-4B",           # unique
    "BidirLM-2.5B",      # BidirLM family (largest)
    "GritLM-7B",         # unique
    "granite-311m",      # granite family (only curated member)
}


# ── helpers ───────────────────────────────────────────────────────────────────

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
    return tau, float(np.percentile(np.asarray(boot), 2.5)), float(np.percentile(np.asarray(boot), 97.5))


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


def evaluate_roster(
    roster: list[str],
    agg_scores: pd.Series,
    lang_avg: pd.DataFrame,
) -> list[dict]:
    rows = []
    for lang in utils.AUDIT_LANGUAGES_ISO:
        lang_rows = lang_avg[
            (lang_avg["language_iso"] == lang)
            & (lang_avg["tier"] == "curated")
            & (lang_avg["model_id"].isin(roster))
        ]
        if lang_rows.empty:
            continue
        models_here = lang_rows["model_id"].tolist()
        lang_vec = lang_rows.set_index("model_id").loc[models_here, "lang_avg_ndcg"].to_numpy()
        agg_vec = agg_scores.reindex(models_here).to_numpy()
        mask = ~np.isnan(agg_vec) & ~np.isnan(lang_vec)
        x, y = agg_vec[mask], lang_vec[mask]
        if len(x) < 5:
            continue
        tau, _, _ = kendall_with_bootstrap(x, y)
        rows.append({
            "language_iso":   lang,
            "n_models":       int(len(x)),
            "kendall_tau":    tau,
            "inversion_rate": pairwise_inversion_rate(x, y),
        })
    return rows


def main() -> None:
    per_task = utils.load_per_task_ndcg()
    taxonomy = utils.load_task_taxonomy()
    lang_avg = utils.load_lang_avg_per_model()
    mteb_agg = utils.load_mteb_agg_per_model()

    curated_models = mteb_agg[mteb_agg["tier"] == "curated"]["model_id"].tolist()
    per_task_curated = per_task[per_task["model_id"].isin(curated_models)]
    all_tasks = taxonomy["task_name"].tolist()

    # Mean(18) aggregator scores for the full curated tier.
    agg_scores = (
        per_task_curated[per_task_curated["task_name"].isin(all_tasks)]
        .groupby("model_id")["ndcg_at_10"]
        .mean()
    )

    # Language-coverage count per model.
    lang_counts = (
        lang_avg[lang_avg["tier"] == "curated"]
        .groupby("model_id")["language_iso"]
        .nunique()
    )

    rosters: dict[str, list[str]] = {
        "headline_curated": curated_models,
        "no_instruction_tuned": [m for m in curated_models if m not in INSTRUCTION_TUNED],
        "ge10_lang_coverage": [
            m for m in curated_models if lang_counts.get(m, 0) >= 10
        ],
        "family_balanced": [m for m in curated_models if m in FAMILY_BALANCED],
    }

    for name, roster in rosters.items():
        print(f"  {name}: n={len(roster)} models")

    all_rows = []
    for variant, roster in rosters.items():
        lang_rows = evaluate_roster(roster, agg_scores, lang_avg)
        for r in lang_rows:
            r["roster_variant"] = variant
        all_rows.extend(lang_rows)

    out = pd.DataFrame(all_rows)[
        ["roster_variant", "language_iso", "n_models", "kendall_tau", "inversion_rate"]
    ]
    out.to_csv(utils.OUTPUTS_DIR / "exp8_roster_sensitivity.csv", index=False)

    summary = (
        out.groupby("roster_variant")
        .agg(
            n_models_median=("n_models", "median"),
            median_tau=("kendall_tau", "median"),
            median_inversion_rate=("inversion_rate", "median"),
            n_languages=("language_iso", "nunique"),
        )
        .reset_index()
    )
    # Add roster n_models (total, not per-language median).
    roster_sizes = pd.Series({k: len(v) for k, v in rosters.items()}, name="roster_n_models")
    summary = summary.merge(
        roster_sizes.reset_index().rename(columns={"index": "roster_variant", 0: "roster_n_models"}),
        on="roster_variant", how="left",
    )
    summary.to_csv(utils.OUTPUTS_DIR / "exp8_roster_summary.csv", index=False)

    h = summary.set_index("roster_variant")
    lines = ["- Roster variant | n | Median tau | Median inversion rate"]
    for variant in ["headline_curated", "no_instruction_tuned", "ge10_lang_coverage", "family_balanced"]:
        if variant in h.index:
            r = h.loc[variant]
            n = int(rosters[variant].__len__())
            lines.append(
                f"- {variant:<25} | {n:2d} | "
                f"{r['median_tau']:.3f}    | {r['median_inversion_rate']:.3f}"
            )

    utils.append_summary("Experiment 8: Curated-roster sensitivity", lines)


if __name__ == "__main__":
    main()
