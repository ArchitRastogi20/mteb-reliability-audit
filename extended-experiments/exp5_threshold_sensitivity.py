"""
Experiment 5: Hidden-failure threshold sensitivity at 17-language scale.

The existing manuscript appendix sweeps the hidden-failure threshold
(theta_lang, theta_MTEB) only on the IT/JA/HI restricted-roster analysis.
The headline 17-language cluster identification uses a fixed threshold-free
rank-based criterion: top-50 percent global rank AND bottom-25 percent
local rank, applied across the curated tier (n=25) per language.

This experiment extends the threshold sensitivity to the 17-language
curated-tier analysis. The natural question is whether the 5-model
cluster is stable under nearby cutoffs.

Inputs (from data/):
  - lang_avg_per_model.csv  (curated tier, 17 audit languages)
  - mteb_agg_per_model.csv  (curated tier, with global_rank)

Procedure:
  1. For each (theta_global, theta_local) on a small grid, identify the
     hidden-failure cells:
        global_rank <= theta_global * n_curated  AND
        local_rank  >= (1 - theta_local) * n_lang
     where:
        theta_global in {0.40, 0.50, 0.60}
        theta_local  in {0.20, 0.25, 0.30}
  2. For each combination, list models with hidden-failure incidence
     in >=3 languages. Compare against the headline 5-model cluster.

Outputs (to outputs/):
  - exp5_threshold_sensitivity.csv
        rows: (theta_global, theta_local) combinations
        cols: theta_global, theta_local, cluster_size_3plus,
              cluster_models (semicolon-separated), n_hidden_failure_cells
  - SUMMARY.md entry naming any combinations under which the cluster
    composition changes.

Acceptance:
  - At the headline (0.50, 0.25), the cluster size is 5 and the model list
    matches the manuscript: granite-311m, harrier-0.6b,
    Seed1.6-embedding-1215, inf-retriever-v1, granite-97m-r2.
  - Under more relaxed cutoffs, additional models may appear; under stricter
    cutoffs, the cluster should be a subset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import utils


def per_language_local_rank(lang_avg: pd.DataFrame) -> pd.DataFrame:
    """Compute local rank (1=best) per (language, tier)."""
    out = []
    for (lang, tier), group in lang_avg.groupby(["language_iso", "tier"]):
        ranked = group.sort_values("lang_avg_ndcg", ascending=False).copy()
        ranked["local_rank"] = np.arange(1, len(ranked) + 1)
        out.append(ranked)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    lang_avg = utils.load_lang_avg_per_model()
    mteb_agg = utils.load_mteb_agg_per_model()

    lang_avg_curated = lang_avg[lang_avg["tier"] == "curated"].copy()
    lang_avg_curated = per_language_local_rank(lang_avg_curated)
    mteb_curated = mteb_agg[mteb_agg["tier"] == "curated"][
        ["model_id", "global_rank"]
    ].copy()

    n_curated = mteb_curated["model_id"].nunique()
    merged = lang_avg_curated.merge(mteb_curated, on="model_id", how="inner")

    grid = [(g, l) for g in [0.40, 0.50, 0.60] for l in [0.20, 0.25, 0.30]]
    rows = []
    for theta_g, theta_l in grid:
        # Hidden failure: global rank in top theta_g, local rank in bottom theta_l.
        merged["is_top_global"] = (
            merged["global_rank"] <= np.ceil(theta_g * n_curated)
        )

        per_lang_n = merged.groupby("language_iso")["model_id"].nunique()

        def _local_threshold(row):
            n_lang = per_lang_n[row["language_iso"]]
            return row["local_rank"] >= np.ceil((1 - theta_l) * n_lang)

        merged["is_bottom_local"] = merged.apply(_local_threshold, axis=1)
        merged["is_hidden_failure"] = (
            merged["is_top_global"] & merged["is_bottom_local"]
        )

        per_model = (
            merged[merged["is_hidden_failure"]]
            .groupby("model_id")["language_iso"].nunique()
            .reset_index(name="n_languages")
        )
        cluster_3plus = per_model[per_model["n_languages"] >= 3]["model_id"].tolist()

        rows.append(
            {
                "theta_global": theta_g,
                "theta_local": theta_l,
                "cluster_size_3plus": len(cluster_3plus),
                "cluster_models": ";".join(sorted(cluster_3plus)),
                "n_hidden_failure_cells": int(merged["is_hidden_failure"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(utils.OUTPUTS_DIR / "exp5_threshold_sensitivity.csv", index=False)

    headline_row = out[(out["theta_global"] == 0.50) & (out["theta_local"] == 0.25)]
    headline_models = headline_row["cluster_models"].iloc[0].split(";")
    headline_size = int(headline_row["cluster_size_3plus"].iloc[0])

    expected = set(utils.CLUSTER_MODELS)
    found = set(headline_models)
    if found != expected:
        diff_msg = (
            f"missing: {expected - found}, extra: {found - expected}"
            if found != expected else ""
        )
        warning = f"Headline cluster composition differs from manuscript: {diff_msg}"
    else:
        warning = "Headline cluster composition reproduces the manuscript exactly."

    utils.append_summary(
        "Experiment 5: Threshold sensitivity (17-language)",
        [
            f"- Headline (0.50, 0.25) cluster size: {headline_size}",
            f"- Reproduction check: {warning}",
            "- Full grid in outputs/exp5_threshold_sensitivity.csv",
        ],
    )


if __name__ == "__main__":
    main()
