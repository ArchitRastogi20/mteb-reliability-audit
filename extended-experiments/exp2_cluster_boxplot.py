"""
Experiment 2: Cluster vs non-cluster boxplot.

Visualization for the manuscript subsection on the hidden-failure cluster.
The existing manuscript reports the result as a Mann-Whitney p-value
(p=0.005) over n=5 vs n=21. A figure makes the gap visually legible and
lets reviewers see at a glance that no single point is driving the result.

Inputs (from data/):
  - per_task_ndcg.csv
  - task_taxonomy.csv
  - cluster_membership.csv

Procedure:
  1. For each curated-tier model (plus granite-97m-r2), compute:
        eng_mean   = mean NDCG@10 over the 11 English-derived tasks
        multi_mean = mean NDCG@10 over the 7 multilingual tasks
        delta      = eng_mean - multi_mean
  2. Split models by is_cluster_member (5 cluster, 21 non-cluster).
  3. Plot delta as a strip + box overlay, two groups side by side.
     - Each point labelled with its model_id.
     - Cluster median and non-cluster median annotated.
     - Mann-Whitney U p-value annotated.
  4. Save as PDF (vector) and PNG.

Outputs (to outputs/):
  - exp2_cluster_delta_values.csv
        rows: model
        cols: model_id, eng_mean, multi_mean, delta, is_cluster_member
  - exp2_cluster_boxplot.pdf  (and .png)
  - SUMMARY.md entry confirming the median values and p-value.

Visual style:
  - Sans-serif, embeddable fonts (pdf.fonttype = 42).
  - Cluster strip in a distinguishing colour (e.g. tab:red), non-cluster
    in tab:grey or tab:blue. Avoid red-green for accessibility.
  - Point labels offset to avoid overlap; if labels collide, use
    adjustText or skip non-cluster labels and show cluster labels only.
  - No 3D, no chart-junk. One-column figure (~3.4 inches wide if used in
    a two-column layout, otherwise 5 inches).

Acceptance:
  - Computed cluster median delta is approximately -0.026 and non-cluster
    median is approximately -0.120 (matches existing Table 3 in the
    manuscript). If off by more than 0.01, raise an error.
  - Mann-Whitney one-sided U-statistic and p-value reproduce the existing
    p=0.005 result. If off by more than 0.01 in p, raise an error.
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

    eng_tasks = taxonomy.loc[
        taxonomy["classification"] == "english_derived", "task_name"
    ].tolist()
    multi_tasks = taxonomy.loc[
        taxonomy["classification"] == "multilingual", "task_name"
    ].tolist()

    eng_mean = (
        per_task[per_task["task_name"].isin(eng_tasks)]
        .groupby("model_id")["ndcg_at_10"].mean()
    )
    multi_mean = (
        per_task[per_task["task_name"].isin(multi_tasks)]
        .groupby("model_id")["ndcg_at_10"].mean()
    )

    df = (
        pd.DataFrame({"eng_mean": eng_mean, "multi_mean": multi_mean})
        .reset_index()
        .merge(cluster, on="model_id", how="left")
    )
    df["delta"] = df["eng_mean"] - df["multi_mean"]
    df["is_cluster_member"] = df["is_cluster_member"].fillna(False)
    df.to_csv(utils.OUTPUTS_DIR / "exp2_cluster_delta_values.csv", index=False)

    cluster_delta = df.loc[df["is_cluster_member"], "delta"].to_numpy()
    other_delta = df.loc[~df["is_cluster_member"], "delta"].to_numpy()

    cluster_median = float(np.median(cluster_delta))
    other_median = float(np.median(other_delta))
    if abs(cluster_median - (-0.026)) > 0.01:
        raise RuntimeError(
            f"Cluster median delta {cluster_median:.4f} differs from manuscript value -0.026 "
            "by more than 0.01. Check inputs."
        )
    if abs(other_median - (-0.120)) > 0.01:
        raise RuntimeError(
            f"Non-cluster median delta {other_median:.4f} differs from manuscript value -0.120 "
            "by more than 0.01. Check inputs."
        )

    u_stat, p_value = mannwhitneyu(
        cluster_delta, other_delta, alternative="greater"
    )
    if abs(p_value - 0.005) > 0.01:
        raise RuntimeError(
            f"Mann-Whitney p={p_value:.4f} differs from manuscript value 0.005 by more "
            "than 0.01. Check inputs."
        )

    _plot_boxplot(df, cluster_median, other_median, p_value)

    utils.append_summary(
        "Experiment 2: Cluster vs non-cluster boxplot",
        [
            f"- Cluster median delta: {cluster_median:+.3f} (n={len(cluster_delta)})",
            f"- Non-cluster median delta: {other_median:+.3f} (n={len(other_delta)})",
            f"- Mann-Whitney U={u_stat:.0f}, one-sided p={p_value:.4f}",
            "- Figure: outputs/exp2_cluster_boxplot.pdf",
        ],
    )


def _plot_boxplot(
    df: pd.DataFrame, cluster_median: float, other_median: float, p_value: float
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    rng = np.random.default_rng(utils.SEED)
    groups = [
        ("Hidden-failure\ncluster (n=5)", df[df["is_cluster_member"]], "tab:red"),
        ("Other\n(n=21)", df[~df["is_cluster_member"]], "tab:gray"),
    ]
    positions = [1, 2]

    for pos, (label, sub, colour) in zip(positions, groups):
        deltas = sub["delta"].to_numpy()
        ax.boxplot(
            deltas,
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": "black"},
            medianprops={"color": "black", "linewidth": 1.4},
            showfliers=False,
        )
        jitter = rng.uniform(-0.10, 0.10, size=len(deltas))
        ax.scatter(
            np.full_like(deltas, pos) + jitter,
            deltas,
            color=colour,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            s=40,
            zorder=3,
        )
        # Label cluster points; skip labels for non-cluster to avoid clutter.
        if "cluster" in label.lower() and "Other" not in label:
            for j, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    row["model_id"],
                    xy=(pos + jitter[j], row["delta"]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=7,
                    va="center",
                )

    ax.axhline(0, linestyle=":", color="black", linewidth=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r"$\Delta = \mathrm{eng\_mean} - \mathrm{multi\_mean}$ (NDCG@10)")
    ax.set_title(f"Mann-Whitney one-sided p = {p_value:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(utils.OUTPUTS_DIR / "exp2_cluster_boxplot.pdf")
    plt.savefig(utils.OUTPUTS_DIR / "exp2_cluster_boxplot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
