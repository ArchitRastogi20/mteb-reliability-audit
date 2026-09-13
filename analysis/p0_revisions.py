"""
revision P0.1, P0.2, P0.4 implementation.

P0.1 — MTEB-agg provenance: document exact aggregator + task list (text edit, no compute).
P0.2 — Replicate rank-inversion analysis using MMTEB language-filtered view as predictor.
P0.4 — Add additional languages via MMTEB language-filtered view (Swahili, Tamil, Bengali, Marathi).

Inputs:
- analysis_output/unified_results.csv (our 17-model × 3-language with MTEB-agg + lang-specific-avg)
- all_mteb/all_performance_per_language.csv (MMTEB leaderboard per-language averages, 314 models × 20 languages)

Outputs:
- analysis_output/stats/rev_P02_filtered_view.txt
- analysis_output/stats/rev_P04_extra_langs.txt
- analysis_output/figures/fig_filtered_view.png
- analysis_output/figures/fig_extra_langs.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]  # repo root
OUT = ROOT / "analysis_output"
STATS = OUT / "stats"
FIGS = OUT / "figures"

# Our 17 models, mapped to their leaderboard names in all_performance_per_language.csv
MODEL_MAP = {
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
    "intfloat/e5-small-v2": "e5-small-v2",
    "ibm-granite/granite-embedding-107m-multilingual": "granite-embedding-107m-multilingual",
    "jinaai/jina-embeddings-v5-text-nano": "jina-embeddings-v5-text-nano",
    "intfloat/multilingual-e5-base": "multilingual-e5-base",
    "intfloat/e5-large-v2": "e5-large-v2",
    "intfloat/multilingual-e5-large": "multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct": "multilingual-e5-large-instruct",
    "BAAI/bge-m3": "bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0",
    "microsoft/harrier-oss-v1-0.6b": "harrier-oss-v1-0.6b",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
    "Salesforce/SFR-Embedding-Mistral": "SFR-Embedding-Mistral",
    "intfloat/e5-mistral-7b-instruct": "e5-mistral-7b-instruct",
    "nvidia/llama-embed-nemotron-8b": "llama-embed-nemotron-8b",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
    "openai/text-embedding-3-large": "text-embedding-3-large",
    "openai/text-embedding-3-small": "text-embedding-3-small",
}
ENGLISH_ONLY = {"intfloat/e5-small-v2", "intfloat/e5-large-v2"}

LANG_CODES = {
    "italian": "ita-Latn",
    "japanese": "jpn-Jpan",
    "hindi": "hin-Deva",
    "swahili": "swa-Latn",
    "tamil": "tam-Taml",
    "bengali": "ben-Beng",
    "marathi": "mar-Deva",
    "telugu": "tel-Telu",
    "korean": "kor-Hang",
    "vietnamese": "vie-Latn",
    "arabic": "ara-Arab",
}


def load_filtered_view():
    """Per-language MMTEB averages for our 17 models."""
    df = pd.read_csv(ROOT / "analysis" / "mteb_csvs" / "all_mteb" / "all_performance_per_language.csv")
    # The model column doesn't have the leading index sometimes; check
    # Header row 1 is the model name column
    col_model = df.columns[1]  # "Model"
    keep_models = list(MODEL_MAP.values())
    sub = df[df[col_model].isin(keep_models)].copy()
    return sub, col_model


def load_unified():
    return pd.read_csv(OUT / "unified_results.csv")


def kendall_with_ci(x, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    point, p = kendalltau(x, y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        t, _ = kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            boot.append(t)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, p, lo, hi


def task_P02_filtered_view():
    """Replicate rank-inversion analysis using MMTEB language-filtered view as predictor."""
    df = load_unified()
    fv, col_model = load_filtered_view()
    rev_map = {v: k for k, v in MODEL_MAP.items()}

    lines = ["# P0.2 — Rank-inversion analysis: MTEB-overall vs MMTEB-language-filtered-view\n",
             "Question: how many of the 20-28% inversions reported in Table 3 SURVIVE",
             "if the practitioner uses the MMTEB language-filtered view instead of the headline aggregate?",
             ""]

    plot_data = []
    for lang in ["italian", "japanese", "hindi"]:
        code = LANG_CODES[lang]
        # Build the merged frame
        sub = df[(df.language == lang) & (~df.english_only) &
                 (df.source != "missing") & (~df.lang_specific_avg.isna())].copy()
        if lang == "italian":
            sub = sub[sub.model_id != "nvidia/llama-embed-nemotron-8b"]
        # join filtered view
        sub["mteb_lang_filtered"] = sub.model_id.map(
            lambda m: float(fv[fv[col_model] == MODEL_MAP[m]][code].iloc[0])
            if len(fv[fv[col_model] == MODEL_MAP[m]]) and code in fv.columns and
               not pd.isna(fv[fv[col_model] == MODEL_MAP[m]][code].iloc[0])
            else np.nan
        )
        sub2 = sub.dropna(subset=["mteb_lang_filtered"]).copy()
        if len(sub2) < 5:
            lines.append(f"## {lang}: insufficient filtered-view coverage ({len(sub2)} models)")
            continue
        n = len(sub2)
        # Three rank vectors
        r_overall = sub2.mteb_agg_score.rank(ascending=False, method="min").astype(int).values
        r_filtered = sub2.mteb_lang_filtered.rank(ascending=False, method="min").astype(int).values
        r_lang = sub2.lang_specific_avg.rank(ascending=False, method="min").astype(int).values

        # Inversions
        def inversions(a, b):
            inv, total = 0, 0
            for i in range(n):
                for j in range(i + 1, n):
                    if a[i] != a[j] and b[i] != b[j]:
                        total += 1
                        if (a[i] < a[j]) != (b[i] < b[j]):
                            inv += 1
            return inv, total

        inv_o, tot = inversions(r_overall, r_lang)
        inv_f, _ = inversions(r_filtered, r_lang)

        # Kendall taus
        tau_o, p_o, lo_o, hi_o = kendall_with_ci(r_overall, r_lang)
        tau_f, p_f, lo_f, hi_f = kendall_with_ci(r_filtered, r_lang)

        lines.append(f"## {lang.capitalize()} (n={n})")
        lines.append(f"  MTEB-overall  vs lang-specific:    "
                     f"tau={tau_o:.3f} [{lo_o:.3f},{hi_o:.3f}]  "
                     f"inversions={inv_o}/{tot} ({100*inv_o/tot:.1f}%)  p={p_o:.4f}")
        lines.append(f"  MMTEB-filtered vs lang-specific:   "
                     f"tau={tau_f:.3f} [{lo_f:.3f},{hi_f:.3f}]  "
                     f"inversions={inv_f}/{tot} ({100*inv_f/tot:.1f}%)  p={p_f:.4f}")
        residual = 100 * inv_f / tot
        original = 100 * inv_o / tot
        delta = original - residual
        lines.append(f"  Residual inversion AFTER filtering: {residual:.1f}% "
                     f"(was {original:.1f}%; filtering eliminates {delta:.1f} pp)")
        lines.append("")
        plot_data.append((lang, original, residual, n))

    # Summary
    lines.append("## SUMMARY for the paper")
    if plot_data:
        avg_orig = np.mean([d[1] for d in plot_data])
        avg_res = np.mean([d[2] for d in plot_data])
        lines.append(f"Average inversion rate (overall MTEB):  {avg_orig:.1f}%")
        lines.append(f"Average inversion rate (lang-filtered): {avg_res:.1f}%")
        lines.append(f"Filtering eliminates ~{avg_orig - avg_res:.1f} percentage points; "
                     f"residual inversions remain.")
        lines.append("")
        lines.append("INTERPRETATION:")
        if avg_res < 5:
            lines.append("  * The MMTEB language-filtered view fixes >95% of the practitioner-")
            lines.append("    visible inversions; the paper's contribution is best framed as")
            lines.append("    'the cost of consulting the headline instead of the filtered view'.")
        elif avg_res < 15:
            lines.append("  * The filtered view substantially reduces inversions but residual ")
            lines.append("    misalignment remains, motivating the BelebeleRetrieval-only probe.")
        else:
            lines.append("  * Even after language-filtering, substantial inversions remain (>15%),")

    (STATS / "rev_P02_filtered_view.txt").write_text("\n".join(lines))
    print("[P0.2] wrote rev_P02_filtered_view.txt")

    # Bar-chart figure
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    langs = [d[0].capitalize() for d in plot_data]
    orig = [d[1] for d in plot_data]
    res = [d[2] for d in plot_data]
    x = np.arange(len(langs))
    ax.bar(x - 0.2, orig, 0.4, color="#888888", label="MTEB-overall (headline)")
    ax.bar(x + 0.2, res, 0.4, color="#1f77b4", label="MMTEB language-filtered view")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylabel("Rank-inversion rate (%)")
    ax.set_title("Effect of using MMTEB's language-filtered view")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS / f"fig_filtered_view.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[P0.2] wrote fig_filtered_view.pdf/png")


def task_P04_extra_languages():
    """Run rank-inversion analysis on additional languages via MMTEB filtered view."""
    df = load_unified()
    fv, col_model = load_filtered_view()

    EXTRA = ["swahili", "tamil", "bengali", "marathi", "telugu", "korean", "vietnamese", "arabic"]
    lines = ["# P0.4 — Rank-inversion pattern on additional languages",
             "Using MMTEB language-filtered scores from all_mteb/all_performance_per_language.csv",
             "for our 17 models. Predictor = MTEB-overall-aggregate; outcome = MMTEB-lang-filtered.",
             ""]
    summary_rows = []

    for lang in EXTRA:
        code = LANG_CODES[lang]
        if code not in fv.columns:
            lines.append(f"## {lang} ({code}): not in leaderboard CSV")
            continue
        # Build joint frame: 17 models, MTEB-overall (from unified_results italian rows -- mteb_agg
        # is identical across rows for the same model), and lang-filtered from fv.
        # Get one row per model from unified.
        u = df[df.language == "italian"].drop_duplicates("model_id").copy()
        u = u[~u.english_only]
        merged = []
        for _, r in u.iterrows():
            lb_name = MODEL_MAP.get(r.model_id)
            if lb_name is None:
                continue
            row = fv[fv[col_model] == lb_name]
            if len(row) == 0 or pd.isna(row[code].iloc[0]):
                continue
            merged.append({
                "model_id": r.model_id,
                "short_name": r.short_name,
                "params": r.params_numeric,
                "mteb_agg": r.mteb_agg_score,
                "lang_filtered": float(row[code].iloc[0]),
            })
        sub = pd.DataFrame(merged)
        if len(sub) < 5:
            lines.append(f"## {lang} ({code}): insufficient ({len(sub)} models)")
            continue
        n = len(sub)
        ro = sub.mteb_agg.rank(ascending=False, method="min").astype(int).values
        rl = sub.lang_filtered.rank(ascending=False, method="min").astype(int).values

        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if (ro[i] < ro[j]) != (rl[i] < rl[j])
                  and ro[i] != ro[j] and rl[i] != rl[j])
        total = n * (n - 1) // 2
        tau, p, tau_lo, tau_hi = kendall_with_ci(ro, rl)
        lines.append(f"## {lang.capitalize()} ({code}, n={n})")
        lines.append(f"  Kendall tau = {tau:.3f}  [95% CI: {tau_lo:.3f}, {tau_hi:.3f}]  p={p:.4f}")
        lines.append(f"  Inversions: {inv}/{total} ({100*inv/total:.1f}%)")
        # Notable inversions
        sub["mteb_rank"] = ro
        sub["lang_rank"] = rl
        notable = sub[(sub.mteb_rank - sub.lang_rank).abs() >= 4].sort_values("mteb_rank")
        if len(notable):
            lines.append("  Notable inversions:")
            for _, r in notable.iterrows():
                lines.append(f"    {r.short_name}: MTEB rank {int(r.mteb_rank)} -> "
                             f"{lang} rank {int(r.lang_rank)} (delta={int(r.mteb_rank-r.lang_rank)})")
        lines.append("")
        summary_rows.append((lang, n, tau, 100*inv/total, p))

    # Summary table
    if summary_rows:
        lines.append("## SUMMARY across additional languages")
        lines.append(f"{'Language':12s} {'n':>4s} {'tau':>8s} {'inv%':>8s} {'p':>10s}")
        for lang, n, tau, ipct, p in summary_rows:
            lines.append(f"{lang:12s} {n:>4d} {tau:>8.3f} {ipct:>7.1f}% {p:>10.4f}")
        # Including IT/JA/HI from main table for context
        lines.append("---")
        lines.append("Plus paper's main 3 (already reported in Table 3):")
        lines.append(f"{'italian':12s}   14    0.451    27.5%      0.040")
        lines.append(f"{'japanese':12s}   15    0.467    26.7%      0.028")
        lines.append(f"{'hindi':12s}   15    0.600    20.0%      0.004")

    (STATS / "rev_P04_extra_langs.txt").write_text("\n".join(lines))
    print("[P0.4] wrote rev_P04_extra_langs.txt")

    # Figure: bar chart of inversion rates across all languages
    if summary_rows:
        all_rows = ([("italian", 14, 0.451, 27.5),
                     ("japanese", 15, 0.467, 26.7),
                     ("hindi", 15, 0.600, 20.0)] +
                    [(l, n, t, p) for l, n, t, p, _ in summary_rows])
        plt.rcParams.update({"font.family": "serif", "font.size": 12})
        fig, ax = plt.subplots(figsize=(10, 5))
        langs_ord = [r[0].capitalize() for r in all_rows]
        invs = [r[3] for r in all_rows]
        ns = [r[1] for r in all_rows]
        colors = ["#2ca02c"] * 3 + ["#ff7f0e"] * len(summary_rows)
        bars = ax.bar(langs_ord, invs, color=colors, alpha=0.85, edgecolor="white")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"n={n}", ha="center", fontsize=9)
        ax.axhline(20, color="black", lw=0.8, ls=":", alpha=0.5)
        ax.set_ylabel("Rank-inversion rate (%)")
        ax.set_title("MTEB-aggregate vs MMTEB-language-filtered: inversion rate across languages")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2ca02c", label="In paper (own evaluation)"),
                            Patch(color="#ff7f0e", label="New (leaderboard-only)")],
                  loc="upper right")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(FIGS / f"fig_extra_langs.{ext}", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print("[P0.4] wrote fig_extra_langs.pdf/png")


def main():
    task_P02_filtered_view()
    task_P04_extra_languages()


if __name__ == "__main__":
    main()
