"""
Experiment 3: Stricter English taxonomy (robustness check).

The existing manuscript classifies 11 of 18 predictor tasks as
English-derived. Three of those eleven are arguable:
  - AILAStatutes is Indian legal text in English
  - LEMBPasskeyRetrieval is synthetic English
  - SpartQA, TempReasonL1, WinoGrande are reasoning tasks (English text but
    stylistically distinct from retrieval)

This experiment re-runs the cluster-profile Mann-Whitney test using a strict
six-task English list (the canonical retrieval-on-English-corpora subset):
  ArguAna, TRECCOVID, SCIDOCS, StackOverflowQA, HagridRetrieval,
  LegalBenchCorporateLobbying.

If the cluster's significantly shallower English-multilingual gap survives,
the result is robust to taxonomy. If it does not, the manuscript framing
needs to be softened before submission.

Inputs (from data/):
  - per_task_ndcg.csv
  - task_taxonomy.csv  (uses is_canonical_english column)
  - cluster_membership.csv

Procedure:
  1. Compute eng_mean using only the 6 canonical English tasks.
  2. Compute multi_mean as before (7 multilingual tasks).
  3. Compute delta = eng_mean - multi_mean.
  4. Run Mann-Whitney one-sided U test, cluster vs non-cluster.
  5. Compare against the 11-task taxonomy result side by side.

Outputs (to outputs/):
  - exp3_strict_taxonomy_results.csv
        rows: model
        cols: model_id, eng_mean_strict_6, multi_mean,
              delta_strict, delta_full, is_cluster_member
  - exp3_taxonomy_comparison.csv
        rows: taxonomy
        cols: cluster_median_delta, noncluster_median_delta, U, p_value
  - SUMMARY.md entry stating both p-values and whether the result holds.

Acceptance:
  - Both taxonomies produce a valid Mann-Whitney result.
  - If strict-6 p-value > 0.05, this is reported clearly so the user can
    decide whether to soften manuscript framing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import utils


def main() -> None:
    per_task = utils.load_per_task_ndcg()
    taxonomy = utils.load_task_taxonomy()
    cluster = utils.load_cluster_membership()

    multi_tasks = taxonomy.loc[
        taxonomy["classification"] == "multilingual", "task_name"
    ].tolist()
    eng_full = taxonomy.loc[
        taxonomy["classification"] == "english_derived", "task_name"
    ].tolist()
    eng_strict_6 = taxonomy.loc[taxonomy["is_canonical_english"], "task_name"].tolist()
    if len(eng_strict_6) != 6:
        raise RuntimeError(
            f"Expected exactly 6 canonical English tasks, found {len(eng_strict_6)}."
        )

    multi_mean = (
        per_task[per_task["task_name"].isin(multi_tasks)]
        .groupby("model_id")["ndcg_at_10"].mean()
    )
    eng_full_mean = (
        per_task[per_task["task_name"].isin(eng_full)]
        .groupby("model_id")["ndcg_at_10"].mean()
    )
    eng_strict_mean = (
        per_task[per_task["task_name"].isin(eng_strict_6)]
        .groupby("model_id")["ndcg_at_10"].mean()
    )

    df = (
        pd.DataFrame(
            {
                "multi_mean": multi_mean,
                "eng_full_mean": eng_full_mean,
                "eng_strict_mean": eng_strict_mean,
            }
        )
        .reset_index()
        .merge(cluster, on="model_id", how="left")
    )
    df["is_cluster_member"] = df["is_cluster_member"].fillna(False)
    df["delta_full"] = df["eng_full_mean"] - df["multi_mean"]
    df["delta_strict"] = df["eng_strict_mean"] - df["multi_mean"]
    df.to_csv(utils.OUTPUTS_DIR / "exp3_strict_taxonomy_results.csv", index=False)

    rows = []
    for label, col in [("english_derived_11", "delta_full"),
                       ("canonical_english_6", "delta_strict")]:
        cluster_d = df.loc[df["is_cluster_member"], col].to_numpy()
        other_d = df.loc[~df["is_cluster_member"], col].to_numpy()
        u, p = mannwhitneyu(cluster_d, other_d, alternative="greater")
        rows.append(
            {
                "taxonomy": label,
                "n_english_tasks": 11 if "11" in label else 6,
                "cluster_median_delta": float(np.median(cluster_d)),
                "noncluster_median_delta": float(np.median(other_d)),
                "U": float(u),
                "p_value": float(p),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(utils.OUTPUTS_DIR / "exp3_taxonomy_comparison.csv", index=False)

    p_full = float(comparison.loc[
        comparison["taxonomy"] == "english_derived_11", "p_value"
    ].iloc[0])
    p_strict = float(comparison.loc[
        comparison["taxonomy"] == "canonical_english_6", "p_value"
    ].iloc[0])

    verdict = (
        "result robust" if p_strict < 0.05
        else "result does NOT survive stricter taxonomy; soften framing"
    )

    utils.append_summary(
        "Experiment 3: Stricter English taxonomy",
        [
            f"- 11-task taxonomy: p = {p_full:.4f}",
            f"- 6-task strict taxonomy: p = {p_strict:.4f}",
            f"- Verdict: {verdict}",
        ],
    )


if __name__ == "__main__":
    main()
