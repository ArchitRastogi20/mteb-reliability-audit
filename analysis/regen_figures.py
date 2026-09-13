"""
Regenerate the headline figures with better fonts, sizes, and layout.
Fixes: fig1_rank_inversions (bigger), fig3_pareto (readable),
fig_rag_bytype (readable labels), fig_belebele_controlled (compact).
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]  # anon_repo root
OUT = ROOT / "analysis_output"
FIGS = OUT / "figures"

LANG_COLORS = {"italian": "#2ca02c", "japanese": "#1f77b4", "hindi": "#ff7f0e"}
TIER_COLORS = {"small": "#1f77b4", "medium": "#ff7f0e",
               "large": "#d62728", "xlarge": "#d62728", "qwen_sub1b": "#ff7f0e"}
TIER_COLORS["api"] = "#9467bd"  # purple — OpenAI/API embeddings


def big_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] {name}.pdf/png saved")


# -----------------------------------------------------------------------
# 1. fig1_gap_vs_params — slightly bigger
# -----------------------------------------------------------------------
def fig1_gap_vs_params():
    big_style()
    df = pd.read_csv(OUT / "unified_results.csv")
    df = df[(df.source != "missing") & (~df.gap.isna())]
    df = df[df.params_numeric > 0]  # skip API models (no public param count)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[(df.language == lang) & (~df.english_only)]
        if len(sub) < 3: continue
        xs = np.log10(sub.params_numeric.values)
        ys = sub.gap.values
        ax.scatter(xs, ys, color=LANG_COLORS[lang], s=80, alpha=0.85,
                   edgecolor="white", lw=0.8, zorder=3, label=f"{lang.capitalize()}")
        # regression line
        slope, intercept = np.polyfit(xs, ys, 1)
        x_range = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_range, slope * x_range + intercept,
                color=LANG_COLORS[lang], lw=1.5, ls="--", alpha=0.6)
    eng = df[df.english_only]
    if len(eng):
        ax.scatter(np.log10(eng.params_numeric), eng.gap, color="gray",
                   marker="x", s=70, lw=2, label="English-only", zorder=2)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Gap (lang-specific NDCG@10 - MTEB-agg)")
    ax.set_xticks([7.5, 8, 8.5, 9, 9.5, 10])
    ax.set_xticklabels(["33M", "100M", "316M", "1B", "3B", "10B"])
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig1_gap_vs_params")


# -----------------------------------------------------------------------
# 2. fig_rank_inversions — clearer scatter, bigger
# -----------------------------------------------------------------------
def fig_rank_inversions():
    big_style()
    df = pd.read_csv(OUT / "unified_results.csv")
    df = df[(~df.english_only) & (df.source != "missing") & (~df.lang_specific_avg.isna())]
    EXCL_IT = {"nvidia/llama-embed-nemotron-8b"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = df[df.language == lang].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCL_IT)]
        sub["mteb_rank"] = sub.mteb_agg_score.rank(ascending=False, method="min").astype(int)
        sub["lang_rank"] = sub.lang_specific_avg.rank(ascending=False, method="min").astype(int)
        n = len(sub)

        for _, r in sub.iterrows():
            color = TIER_COLORS.get(r.tier, "#888888")
            ax.scatter(r.mteb_rank, r.lang_rank, color=color, s=110, alpha=0.85,
                       edgecolor="white", lw=0.8, zorder=3)
            if abs(r.mteb_rank - r.lang_rank) >= 4:
                # offset above-or-below to avoid collision with diagonal
                dy = -8 if r.mteb_rank < r.lang_rank else 8
                ax.annotate(r.short_name, (r.mteb_rank, r.lang_rank),
                            fontsize=9, ha="left", va="bottom",
                            xytext=(4, dy), textcoords="offset points")

        ax.plot([0, n + 1], [0, n + 1], "k--", lw=1, alpha=0.5, zorder=1)
        # hidden failures
        hf = []
        if lang == "japanese":
            hf = [("snowflake", sub)]
        elif lang == "hindi":
            hf = [("snowflake", sub)]
        for name, _ in hf:
            r = sub[sub.short_name == name]
            if len(r):
                rr = r.iloc[0]
                ax.scatter(rr.mteb_rank, rr.lang_rank, marker="X",
                           color="red", s=200, edgecolor="white", lw=1.5,
                           zorder=5, label="hidden failure" if ax is axes[1] else None)
        tau, p = kendalltau(sub.mteb_rank.values, sub.lang_rank.values)
        ax.set_title(f"{lang.capitalize()}  ($\\tau$={tau:.2f}, p={p:.3f})")
        ax.set_xlabel("MTEB-aggregate rank")
        ax.set_ylabel("Language-specific rank")
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.invert_yaxis()
        ax.invert_xaxis()
        # ranks: 1 (best) is at top-right

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TIER_COLORS["small"],
               markersize=11, label="Small (<300M)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TIER_COLORS["medium"],
               markersize=11, label="Medium (300--600M)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TIER_COLORS["large"],
               markersize=11, label="Large (4--8B)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red",
               markersize=14, label="Hidden failure"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("MTEB Rank vs. Language-Specific Rank (above diagonal = MTEB undervalues)",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, "fig_rank_inversions")


# -----------------------------------------------------------------------
# 3. fig3_pareto — readable, with model labels for Pareto-frontier members
# -----------------------------------------------------------------------
def fig3_pareto():
    big_style()
    df = pd.read_csv(OUT / "unified_results.csv")
    df = df[(~df.english_only) & (df.source != "missing") & (~df.lang_specific_avg.isna())]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = df[df.language == lang].copy()
        sub = sub[sub.params_numeric > 0]
        sub = sub.sort_values("params_numeric")
        xs = np.log10(sub.params_numeric.values)
        lang_y = sub.lang_specific_avg.values
        mteb_y = sub.mteb_agg_score.values
        names = sub.short_name.values

        ax.scatter(xs, mteb_y, color="gray", marker="D", s=110, alpha=0.55,
                   label="MTEB-agg")
        ax.scatter(xs, lang_y, color=LANG_COLORS[lang], s=170, alpha=0.92,
                   edgecolor="white", lw=1.0, label="Lang-specific", zorder=3)

        order = np.argsort(xs)
        xs_o, ys_o = xs[order], lang_y[order]
        pareto = np.zeros(len(xs_o), dtype=bool)
        best = -np.inf
        for i, y in enumerate(ys_o):
            if y > best:
                pareto[i] = True
                best = y
        ax.plot(xs_o[pareto], ys_o[pareto], color=LANG_COLORS[lang],
                lw=2.5, alpha=0.7, zorder=2)

        # Label all models, not just a hand-picked few — but rotate to avoid overlap
        for i, name in enumerate(names):
            short = name.replace("multilingual-e5-", "ml-e5-").replace("snowflake-l-v2", "snowflake")
            ax.annotate(short, (xs[i], lang_y[i]), fontsize=10,
                        xytext=(6, 6), textcoords="offset points",
                        alpha=0.85)

        ax.set_xticks([8, 8.5, 9, 9.5, 10])
        ax.set_xticklabels(["100M", "316M", "1B", "3B", "10B"], fontsize=12)
        ax.set_xlabel("Parameters", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        if ax is axes[0]:
            ax.set_ylabel("NDCG@10", fontsize=13)
        ax.set_title(lang.capitalize(), fontsize=15)
        if ax is axes[0]:
            ax.legend(loc="lower right", framealpha=0.95, fontsize=12)
    fig.suptitle("Cost--Performance Pareto Frontiers (lang-specific in colour, MTEB-agg in grey)",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, "fig3_pareto")


# -----------------------------------------------------------------------
# 4. fig_rag_bytype — readable labels, 6 datasets including Italian
# -----------------------------------------------------------------------
def fig_rag_bytype():
    big_style()
    df = pd.read_csv(OUT / "unified_rag_bytype_results_v2.csv")
    df = df[df.short_name != "e5-small-v2"]  # English-only baseline; remove for clarity
    df = df[df.params_numeric > 0]  # skip API models (params=-1 would mis-sort to leftmost)

    DATASETS = ["it_finance", "it_law", "ja_finance", "ja_law", "hi_finance", "hi_law"]
    LABELS = {"it_finance": "IT/Finance", "it_law": "IT/Law",
              "ja_finance": "JA/Finance", "ja_law": "JA/Law",
              "hi_finance": "HI/Finance", "hi_law": "HI/Law"}
    BM25 = {"it_finance": 2.06, "it_law": 63.67,
            "ja_finance": 50.33, "ja_law": 30.96,
            "hi_finance": 92.31, "hi_law": 78.98}
    QT_COLORS = {"factual": "#1f77b4", "multi_hop": "#ff7f0e",
                 "summarization": "#2ca02c"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False)
    for ax, ds in zip(axes.flat, DATASETS):
        sub = df[df.dataset == ds].copy()
        order = sub.sort_values("params_numeric").short_name.unique()
        n_models = len(order)
        bw = 0.27
        for qi, qt in enumerate(["factual", "multi_hop", "summarization"]):
            qs = sub[sub.query_type == qt].set_index("short_name")
            ys = [qs.loc[m, "ndcg_at_10"] if m in qs.index else 0 for m in order]
            xs = np.arange(n_models) + (qi - 1) * bw
            label = qt.replace("_", "-") if ax is axes[0, 0] else None
            ax.bar(xs, ys, width=bw, color=QT_COLORS[qt], alpha=0.9, label=label)
        ax.axhline(BM25[ds], color="black", lw=2.0, ls="--", alpha=0.8,
                   label=f"BM25={BM25[ds]:.1f}")
        ax.set_xticks(np.arange(n_models))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=11)
        ax.set_title(LABELS[ds], fontsize=15)
        ax.set_ylabel("NDCG@10", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    fig.suptitle("Multilingual RAG: NDCG@10 by Query Type (six configurations; dashed = BM25)",
                 y=1.0, fontsize=16)
    fig.tight_layout()
    save(fig, "fig_rag_bytype")


# -----------------------------------------------------------------------
# 5. fig_belebele_controlled — clean, larger
# -----------------------------------------------------------------------
def fig_belebele_controlled():
    big_style()
    df = pd.read_csv(OUT / "unified_results.csv")
    df = df[(~df.english_only) & (df.source != "missing") & (~df.lang_specific_avg.isna())]
    df["belebele_gap"] = df.task_BelebeleRetrieval - df.mteb_agg_score

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = df[df.language == lang].copy()
        sub = sub[sub.params_numeric > 0]
        sub = sub.sort_values("params_numeric")
        ys = np.arange(len(sub))
        gaps = sub.belebele_gap.values
        colors = [TIER_COLORS.get(t, "#888888") for t in sub.tier]
        ax.barh(ys, gaps, color=colors, alpha=0.88, edgecolor="white", lw=0.9)
        ax.axvline(0, color="black", lw=1.0, ls=":")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub.short_name.values, fontsize=13)
        ax.set_xlabel("Belebele gap (task NDCG $-$ MTEB-agg)", fontsize=14)
        ax.set_title(lang.capitalize(), fontsize=16)
        ax.tick_params(axis="x", labelsize=13)
        ax.invert_yaxis()
    fig.suptitle("BelebeleRetrieval-Only Gap (controlled task, three languages)",
                 y=1.0, fontsize=16)
    fig.tight_layout()
    save(fig, "fig_belebele_controlled")


# -----------------------------------------------------------------------
# 6. NEW: fig_anchor_sensitivity — convert anchor sensitivity table to bar chart
# -----------------------------------------------------------------------
def fig_anchor_sensitivity():
    big_style()
    pairs = [
        ("ml-e5-small\nvs Qwen3-8B", 2.56, 1.61, 1.23),
        ("granite-107m\nvs Qwen3-8B", 2.66, 2.18, 1.16),
        ("granite-107m\nvs nemotron-8b", -0.45, 1.76, 1.00),
        ("jina-v5-nano\nvs Qwen3-4B", 15.88, 1.97, 1.93),
        ("ml-e5-small\nvs ml-e5-l-inst", 4.96, 5.71, 2.19),
    ]
    labels = [p[0] for p in pairs]
    it = [p[1] for p in pairs]
    ja = [p[2] for p in pairs]
    hi = [p[3] for p in pairs]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - w, it, width=w, label="Italian", color=LANG_COLORS["italian"], alpha=0.92,
           edgecolor="white", lw=0.7)
    ax.bar(x, ja, width=w, label="Japanese", color=LANG_COLORS["japanese"], alpha=0.92,
           edgecolor="white", lw=0.7)
    ax.bar(x + w, hi, width=w, label="Hindi", color=LANG_COLORS["hindi"], alpha=0.92,
           edgecolor="white", lw=0.7)
    ax.axhline(1, color="black", lw=1.5, ls="--", alpha=0.75, label="No-distortion null")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Inflation ratio $\\rho$", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Anchor-Pair Inflation Sensitivity Across (small, large) Choices",
                 fontsize=15)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(-1.5, 17.5)
    fig.tight_layout()
    save(fig, "fig_anchor_sensitivity")


def main():
    fig1_gap_vs_params()
    fig_rank_inversions()
    fig3_pareto()
    fig_rag_bytype()
    fig_belebele_controlled()
    fig_anchor_sensitivity()


if __name__ == "__main__":
    main()
