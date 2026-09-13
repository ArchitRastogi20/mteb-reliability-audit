"""Per-language scatter of global curated-tier MTEB-aggregate rank against
local per-language NDCG rank, for every audit language with >= 2 retrieval
tasks. Annotates hidden failures (top-50% global, bottom-25% local).
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (
    ROSTER_CSV, LOCAL_RANKS_CSV, SUMMARY_CURATED, OUTPUTS, OUTPUTS,
    kendall_tau,
)

PARAM_BIN_EDGES_M = [0, 300, 1000, 8000, math.inf]  # millions
PARAM_BIN_LABELS  = ["<300M", "300M-1B", "1B-8B", ">8B"]
PARAM_BIN_COLORS  = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def _params_to_millions(p) -> float:
    p = str(p).strip().lower()
    if p in ("unk", "nan", ""):
        return float("nan")
    if p.endswith("b"):
        return float(p[:-1]) * 1000
    if p.endswith("m"):
        return float(p[:-1])
    try:
        return float(p)
    except ValueError:
        return float("nan")


def _bin_params(m: float) -> int:
    if math.isnan(m):
        return -1
    for i, hi in enumerate(PARAM_BIN_EDGES_M[1:]):
        if m < hi:
            return i
    return len(PARAM_BIN_LABELS) - 1


def is_hidden_failure(global_rank: int, n_global: int,
                      local_rank: int, n_local: int) -> bool:
    """top-50% global AND bottom-25% local within the per-language pool."""
    if global_rank > n_global / 2:
        return False
    if local_rank <= 0.75 * n_local:
        return False
    return True


def main() -> None:
    roster = pd.read_csv(ROSTER_CSV)
    local  = pd.read_csv(LOCAL_RANKS_CSV)
    summary = pd.read_csv(SUMMARY_CURATED).set_index("iso3")

    roster["params_m"] = roster["params"].map(_params_to_millions)
    roster["param_bin"] = roster["params_m"].map(_bin_params)
    n_global = len(roster)  # 25

    # Languages with >= 2 retrieval tasks per summary_curated.csv
    langs_with_data = local["language_iso"].unique().tolist()
    langs = [iso for iso in langs_with_data
             if iso in summary.index and summary.loc[iso, "n_tasks"] >= 2]
    # Stable order: iso3 alphabetical
    langs = sorted(langs)
    assert len(langs) == 17, f"expected 17 audit languages, got {len(langs)}: {langs}"

    rows, cols = 5, 4
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 8.5), constrained_layout=True)
    above_below: list[dict] = []

    for idx, iso in enumerate(langs):
        ax = axes[idx // cols, idx % cols]
        sub = (local[local["language_iso"] == iso]
               .merge(roster[["model_id", "global_rank_curated", "param_bin"]],
                      on="model_id", how="inner"))
        n_local = len(sub)
        x = sub["global_rank_curated"].astype(float).to_numpy()
        y = sub["local_rank"].astype(float).to_numpy()
        tau = kendall_tau(x, y)

        for b in range(len(PARAM_BIN_LABELS)):
            mask = sub["param_bin"].to_numpy() == b
            if mask.any():
                ax.scatter(x[mask], y[mask], c=PARAM_BIN_COLORS[b], s=22,
                           edgecolors="black", linewidths=0.3,
                           label=PARAM_BIN_LABELS[b] if idx == 0 else None)

        for _, r in sub.iterrows():
            if is_hidden_failure(int(r.global_rank_curated), n_global,
                                 int(r.local_rank), n_local):
                ax.scatter(r.global_rank_curated, r.local_rank,
                           marker="x", c="red", s=55, linewidths=1.4)

        lo, hi = 0.5, max(n_global, n_local) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", c="gray", lw=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.invert_yaxis(); ax.invert_xaxis()  # rank 1 in top-left

        lang_name = summary.loc[iso, "lang_name"]
        ax.set_title(f"{iso} {lang_name} (n={n_local}, tau={tau:+.2f})",
                     fontsize=7.5)
        ax.tick_params(labelsize=6)

        above = int(np.sum(y > x))
        below = int(np.sum(y < x))
        on    = int(np.sum(y == x))
        above_below.append({
            "iso3": iso, "lang_name": lang_name, "n_local": n_local,
            "above_diagonal": above, "below_diagonal": below,
            "on_diagonal": on,
            "kendall_tau": tau,
            "kendall_tau_summary_curated": float(summary.loc[iso, "kendall_tau"]),
        })

    # Hide unused subplots
    for k in range(len(langs), rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle("Curated tier: global MTEB-aggregate rank vs. per-language local rank",
                 fontsize=9)
    fig.supxlabel("global rank (curated, n=25)", fontsize=8)
    fig.supylabel("per-language local NDCG rank", fontsize=8)
    fig.legend(loc="lower right", fontsize=6, ncol=4,
               bbox_to_anchor=(1.0, -0.02))

    out_pdf = OUTPUTS / "exp9_rank_scatter_all_langs.pdf"
    out_png = OUTPUTS / "exp9_rank_scatter_all_langs.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)

    counts_df = pd.DataFrame(above_below)
    counts_df.to_csv(OUTPUTS / "exp9_above_below_counts.csv", index=False)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"wrote {OUTPUTS / 'exp9_above_below_counts.csv'}")
    print()
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
