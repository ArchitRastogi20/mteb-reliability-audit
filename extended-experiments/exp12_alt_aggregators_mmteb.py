"""Five aggregators over an 18-task MMTEB(Multilingual) Retrieval subset.

For each model, compute aggregated NDCG@10 with each aggregator, derive a
1-based rank (1 = best), then compare to per-language local-NDCG ranks for
IT / JA / HI via Kendall tau and pairwise inversion rate.

Operates over the curated 25-model roster (this is the population for which
per_task_ndcg.csv has full coverage). The original brief envisioned the
14-model standard set, but per_task_ndcg.csv only covers 7 of those 14
models; widening to the 25-model curated roster maximises sample size and
matches the data the canonical curated CSV actually exposes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from _common import (
    PER_TASK_NDCG, LOCAL_RANKS_CSV, OUTPUTS,
    kendall_tau, inversion_rate, dense_rank,
)

LANG_ISO = {"italian": "ita", "japanese": "jpn", "hindi": "hin"}


def arithmetic_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def min_score(x: np.ndarray) -> float:
    return float(np.min(x))


def median_score(x: np.ndarray) -> float:
    return float(np.median(x))


def harmonic_mean(x: np.ndarray, eps: float = 1e-6) -> float:
    safe = np.maximum(x, eps)
    return float(len(safe) / np.sum(1.0 / safe))


def trimmed_mean(x: np.ndarray, frac: float = 0.10) -> float:
    n = len(x); k = int(np.floor(n * frac))
    if k == 0:
        return float(np.mean(x))
    s = np.sort(x)
    return float(np.mean(s[k:n-k]))


AGGREGATORS = [
    ("Mean(Task)",       arithmetic_mean),
    ("Min-score",        min_score),
    ("Median(Task)",     median_score),
    ("Harmonic mean",    harmonic_mean),
    ("Trimmed mean 10%", lambda x: trimmed_mean(x, 0.10)),
]


def main():
    pt = pd.read_csv(PER_TASK_NDCG)
    pt = pt.dropna(subset=["ndcg_at_10"])

    # per_task_ndcg.csv uses short display names (e.g. "bge-m3"); map them
    # to canonical model_id via curated_local_ranks.csv.
    local = pd.read_csv(LOCAL_RANKS_CSV)
    name_map = dict(zip(local["display_name"], local["model_id"]))
    pt["model_id_canon"] = pt["model_id"].map(name_map)
    unmatched = sorted(set(pt[pt["model_id_canon"].isna()]["model_id"]))
    pt = pt.dropna(subset=["model_id_canon"])
    pt["model_id"] = pt["model_id_canon"]
    pt = pt.drop(columns=["model_id_canon"])

    if unmatched:
        print(f"NOTE: {len(unmatched)} display names did not map to a canonical "
              f"model_id; dropped: {unmatched}")

    roster = sorted(pt["model_id"].unique())
    print(f"roster (display-mapped): {len(roster)} models")

    # Build wide matrix model_id x task_name; require full coverage across tasks.
    coverage = pt.groupby("task_name")["model_id"].nunique()
    print(f"task count in source: {len(coverage)}")

    mat = (pt.pivot(index="model_id", columns="task_name", values="ndcg_at_10")
             .dropna())
    print(f"models with full coverage across all tasks: {len(mat)} "
          f"x {mat.shape[1]} tasks")

    if mat.shape[1] != 18:
        print(f"NOTE: expected 18 MMTEB(Multilingual) Retrieval tasks; "
              f"final matrix has {mat.shape[1]}")

    # For each aggregator: aggregate -> rank (1=best)
    agg_scores = {}
    agg_ranks = {}
    for name, fn in AGGREGATORS:
        s = mat.apply(fn, axis=1)
        agg_scores[name] = s
        agg_ranks[name]  = pd.Series(dense_rank(s.values), index=s.index)

    rows = []
    for agg_name, _ in AGGREGATORS:
        row = {"Aggregator": agg_name}
        for _lang_label, iso in LANG_ISO.items():
            sub = (local[local["language_iso"] == iso]
                   .set_index("model_id")["local_ndcg"])
            common = sorted(set(sub.index) & set(agg_scores[agg_name].index))
            if len(common) < 4:
                row[f"{iso}_n"] = len(common)
                row[f"{iso}_tau"] = float("nan")
                row[f"{iso}_inv_pct"] = float("nan")
                continue
            agg_r   = agg_ranks[agg_name].loc[common].values
            local_r = dense_rank(sub.loc[common].values)
            row[f"{iso}_n"]       = len(common)
            row[f"{iso}_tau"]     = round(kendall_tau(agg_r, local_r), 3)
            row[f"{iso}_inv_pct"] = round(100 * inversion_rate(agg_r, local_r), 1)
        rows.append(row)

    out = pd.DataFrame(rows, columns=["Aggregator",
                                      "ita_n", "ita_tau", "ita_inv_pct",
                                      "jpn_n", "jpn_tau", "jpn_inv_pct",
                                      "hin_n", "hin_tau", "hin_inv_pct"])
    path = OUTPUTS / "exp12_alt_aggregators.csv"
    out.to_csv(path, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\nwrote {path}")
    print(f"tasks used (n={mat.shape[1]}):")
    for t in sorted(mat.columns.tolist()):
        print(f"  {t}")


if __name__ == "__main__":
    main()
