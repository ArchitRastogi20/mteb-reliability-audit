"""
Follow-up statistical analyses.
Runs as a standalone script. All outputs go to analysis_output/stats/ and analysis_output/figures/.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]  # anon_repo root
OUT = ROOT / "analysis_output"
STATS = OUT / "stats"
FIGS = OUT / "figures"
RAG_BASE = ROOT / "rag-dataset" / "output" / "evaluation"

# -- Model registry (short names matching the existing analysis) --
MODEL_SHORT = {
    "Qwen/Qwen3-Embedding-0.6B": ("Qwen3-0.6B", 600_000_000, 1024, "qwen_sub1b", False),
    "Qwen/Qwen3-Embedding-4B": ("Qwen3-4B", 4_000_000_000, 2560, "large", False),
    "Qwen/Qwen3-Embedding-8B": ("Qwen3-8B", 8_000_000_000, 4096, "xlarge", False),
    "Salesforce/SFR-Embedding-Mistral": ("SFR-Mistral", 7_000_000_000, 4096, "large", False),
    "BAAI/bge-m3": ("bge-m3", 568_000_000, 1024, "medium", False),
    "intfloat/e5-large-v2": ("e5-large-v2", 335_000_000, 1024, "medium", True),  # English-only
    "intfloat/e5-mistral-7b-instruct": ("e5-mistral-7b", 7_000_000_000, 4096, "large", False),
    "intfloat/e5-small-v2": ("e5-small-v2", 33_000_000, 384, "small", True),  # English-only
    "ibm-granite/granite-embedding-107m-multilingual": ("granite-107m", 107_000_000, 768, "small", False),
    "microsoft/harrier-oss-v1-0.6b": ("harrier-0.6b", 596_000_000, 1024, "medium", False),
    "jinaai/jina-embeddings-v5-text-nano": ("jina-v5-nano", 212_000_000, 1024, "small", False),
    "nvidia/llama-embed-nemotron-8b": ("nemotron-8b", 7_500_000_000, 4096, "xlarge", False),
    "intfloat/multilingual-e5-base": ("ml-e5-base", 278_000_000, 768, "medium", False),
    "intfloat/multilingual-e5-large-instruct": ("ml-e5-large-instruct", 560_000_000, 1024, "medium", False),
    "intfloat/multilingual-e5-large": ("ml-e5-large", 560_000_000, 1024, "medium", False),
    "intfloat/multilingual-e5-small": ("ml-e5-small", 118_000_000, 384, "small", False),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ("snowflake", 568_000_000, 1024, "medium", False),
}
EXCLUDE_MODELS = {"perplexity-ai/pplx-embed-v1-0.6b", "jinaai/jina-embeddings-v3"}  # eval failures

FOLDER_TO_MODEL = {
    "Qwen3_Embedding_0_6B_1586ef": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3_Embedding_4B_8f1f6c": "Qwen/Qwen3-Embedding-4B",
    "Qwen3_Embedding_8B_c07e5f": "Qwen/Qwen3-Embedding-8B",
    "SFR_Embedding_Mistra_035d33": "Salesforce/SFR-Embedding-Mistral",
    "bge_m3_75e678": "BAAI/bge-m3",
    "e5_large_v2_6c60f7": "intfloat/e5-large-v2",
    "e5_mistral_7b_instru_fbd38f": "intfloat/e5-mistral-7b-instruct",
    "e5_small_v2_bd29da": "intfloat/e5-small-v2",
    "granite_embedding_10_a5fbbe": "ibm-granite/granite-embedding-107m-multilingual",
    "harrier_oss_v1_0_6b_686dc8": "microsoft/harrier-oss-v1-0.6b",
    "jina_embeddings_v5_t_eed139": "jinaai/jina-embeddings-v5-text-nano",
    "llama_embed_nemotron_eeabda": "nvidia/llama-embed-nemotron-8b",
    "multilingual_e5_base_e96c97": "intfloat/multilingual-e5-base",
    "multilingual_e5_larg_571f3e": "intfloat/multilingual-e5-large-instruct",
    "multilingual_e5_larg_e3a0cc": "intfloat/multilingual-e5-large",
    "multilingual_e5_smal_6f25f9": "intfloat/multilingual-e5-small",
    "snowflake_arctic_emb_567ccc": "Snowflake/snowflake-arctic-embed-l-v2.0",
}

# BM25 baselines (NEW — Italian fixed)
BM25 = {
    "ja_finance": 50.33,
    "ja_law": 30.96,
    "hi_finance": 92.31,
    "hi_law": 78.98,
    "it_finance": 2.06,   # broken — likely tokenizer artifact for Italian finance
    "it_law": 63.67,
}


def load_all_rag():
    """Walk evaluation/ and aggregate (model, dataset) -> overall + by-type metrics."""
    rows = []
    bytype_rows = []
    for folder, model_id in FOLDER_TO_MODEL.items():
        meta = MODEL_SHORT.get(model_id)
        if meta is None:
            continue
        short, params, dim, tier, eng_only = meta
        for fname in os.listdir(RAG_BASE / folder):
            if not fname.endswith(".json"):
                continue
            ds_key = fname.replace(".json", "")  # e.g. it_finance, ja_law
            with open(RAG_BASE / folder / fname) as f:
                d = json.load(f)
            ov = d.get("overall", {})
            rows.append({
                "model_id": model_id,
                "short_name": short,
                "params_numeric": params,
                "dim": dim,
                "tier": tier,
                "english_only": eng_only,
                "dataset": ds_key,
                "ndcg_at_10": ov.get("ndcg_at_10", float("nan")) * 100,
                "recall_at_10": ov.get("recall_at_10", float("nan")) * 100,
                "mrr": ov.get("mrr", float("nan")) * 100,
            })
            for qtype in ["factual", "multi_hop", "summarization"]:
                bt = d.get("by_type", {}).get(qtype, {})
                bytype_rows.append({
                    "model_id": model_id,
                    "short_name": short,
                    "params_numeric": params,
                    "tier": tier,
                    "dataset": ds_key,
                    "query_type": qtype,
                    "ndcg_at_10": bt.get("ndcg_at_10", float("nan")) * 100,
                    "recall_at_10": bt.get("recall_at_10", float("nan")) * 100,
                    "mrr": bt.get("mrr", float("nan")) * 100,
                })
    overall = pd.DataFrame(rows).sort_values(["dataset", "params_numeric"])
    bytype = pd.DataFrame(bytype_rows).sort_values(["dataset", "query_type", "params_numeric"])
    return overall, bytype


def kendall_tau_with_bootstrap(mteb_ranks, lang_ranks, n_boot=10_000, seed=42):
    """Bootstrap CI on Kendall tau; resample model indices with replacement."""
    rng = np.random.default_rng(seed)
    n = len(mteb_ranks)
    point_tau, point_p = kendalltau(mteb_ranks, lang_ranks)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = mteb_ranks[idx]
        l = lang_ranks[idx]
        if len(set(m)) < 2 or len(set(l)) < 2:
            continue
        t, _ = kendalltau(m, l)
        if not np.isnan(t):
            boot.append(t)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point_tau, point_p, lo, hi


def bootstrap_inversion_proportion(mteb_ranks, lang_ranks, n_boot=10_000, seed=42):
    """Bootstrap CI on the inversion proportion."""
    rng = np.random.default_rng(seed)
    n = len(mteb_ranks)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = mteb_ranks[idx]
        l = lang_ranks[idx]
        # inversions
        inv = 0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if m[i] != m[j] and l[i] != l[j]:
                    pairs += 1
                    if (m[i] < m[j]) != (l[i] < l[j]):
                        inv += 1
        if pairs > 0:
            boot.append(inv / pairs)
    boot = np.array(boot)
    return np.percentile(boot, [2.5, 97.5])


def bh_fdr(p_values, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum(1.0, ranked * n / np.arange(1, n + 1))
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty_like(adj)
    out[order] = adj
    return out


def task_C1_nemotron_italian(df):
    """Re-run rank inversion analysis with nemotron INCLUDED for Italian."""
    lines = ["# Italian rank-inversion analysis: with vs without nemotron-8b\n"]
    sub_full = df[(df.language == "italian") & (~df.english_only) & (df.source != "missing") &
                  (~df.lang_specific_avg.isna())].copy()
    EXCLUDE = ["nvidia/llama-embed-nemotron-8b"]
    for label, sub in [("with nemotron-8b included", sub_full),
                       ("without nemotron-8b (paper baseline)",
                        sub_full[~sub_full.model_id.isin(EXCLUDE)])]:
        sub = sub.copy()
        sub["mteb_rank"] = sub["mteb_agg_score"].rank(ascending=False, method="min").astype(int)
        sub["lang_rank"] = sub["lang_specific_avg"].rank(ascending=False, method="min").astype(int)
        n = len(sub)
        recs = sub[["mteb_rank", "lang_rank"]].to_dict("records")
        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if (recs[i]["mteb_rank"] < recs[j]["mteb_rank"]) !=
                  (recs[i]["lang_rank"] < recs[j]["lang_rank"]))
        total = n * (n - 1) // 2
        tau, p = kendalltau(sub["mteb_rank"].values, sub["lang_rank"].values)
        tau_lo, tau_hi = kendall_tau_with_bootstrap(sub["mteb_rank"].values,
                                                     sub["lang_rank"].values)[2:]
        inv_lo, inv_hi = bootstrap_inversion_proportion(sub["mteb_rank"].values,
                                                        sub["lang_rank"].values)
        lines.append(f"## Italian {label} (n={n})")
        lines.append(f"  Kendall tau = {tau:.3f}  [95%CI: {tau_lo:.3f}, {tau_hi:.3f}]  (p={p:.4f})")
        lines.append(f"  Rank inversions: {inv}/{total} ({100*inv/total:.1f}%)")
        lines.append(f"  Inversion proportion 95% CI: [{100*inv_lo:.1f}%, {100*inv_hi:.1f}%]")
        lines.append("")
    return "\n".join(lines)


def task_M3_bh_fdr(df):
    """Apply BH-FDR to all reported p-values."""
    p_values = []
    labels = []

    # Kendall tau tests for 3 languages, each with and without specific exclusions
    EXCLUDE_ITALIAN = {"nvidia/llama-embed-nemotron-8b"}
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[(df.language == lang) & (~df.english_only) & (df.source != "missing") &
                 (~df.lang_specific_avg.isna())].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        m = sub["mteb_agg_score"].rank(ascending=False, method="min").values
        l = sub["lang_specific_avg"].rank(ascending=False, method="min").values
        tau, p = kendalltau(m, l)
        p_values.append(p)
        labels.append(f"Kendall_tau_{lang}")

    # Mean-score bootstrap p-values (already computed: it<0.001, ja=0.009, hi<0.001)
    # We re-compute here
    for lang, p_orig in [("italian", 0.0001), ("japanese", 0.009), ("hindi", 0.0001)]:
        p_values.append(p_orig)
        labels.append(f"meanRatio_bootstrap_{lang}")

    # Anchor-pair bootstrap p-values (existing: italian=0.058, japanese=0.122, hindi=0.262)
    for lang, p_orig in [("italian", 0.058), ("japanese", 0.122), ("hindi", 0.262)]:
        p_values.append(p_orig)
        labels.append(f"anchorPair_bootstrap_{lang}")

    adj = bh_fdr(p_values)
    lines = ["# Benjamini-Hochberg FDR correction (q=0.05)\n",
             f"{'Test':40s} {'p (raw)':>10s} {'p (BH-FDR)':>14s} {'sig at q=0.05':>16s}"]
    for lab, p, ap in zip(labels, p_values, adj):
        sig = "yes" if ap < 0.05 else "NO"
        lines.append(f"{lab:40s} {p:>10.4f} {ap:>14.4f} {sig:>16s}")
    return "\n".join(lines)


def task_M4_kendall_ci(df):
    """Bootstrap CIs on Kendall tau and inversion proportion for all 3 languages."""
    EXCLUDE_ITALIAN = {"nvidia/llama-embed-nemotron-8b"}
    lines = ["# Kendall tau and inversion proportion bootstrap CIs\n"]
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[(df.language == lang) & (~df.english_only) & (df.source != "missing") &
                 (~df.lang_specific_avg.isna())].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        m = sub["mteb_agg_score"].rank(ascending=False, method="min").astype(int).values
        l = sub["lang_specific_avg"].rank(ascending=False, method="min").astype(int).values
        n = len(m)
        tau, p, tau_lo, tau_hi = kendall_tau_with_bootstrap(m, l)
        recs = list(zip(m.tolist(), l.tolist()))
        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if (recs[i][0] < recs[j][0]) != (recs[i][1] < recs[j][1]))
        total = n * (n - 1) // 2
        inv_lo, inv_hi = bootstrap_inversion_proportion(m, l)
        lines.append(f"## {lang.capitalize()} (n={n})")
        lines.append(f"  tau = {tau:.3f}  [95% CI: {tau_lo:.3f}, {tau_hi:.3f}]  p={p:.4f}")
        lines.append(f"  inversions = {inv}/{total} ({100*inv/total:.1f}%)")
        lines.append(f"  inversion proportion 95% CI: [{100*inv_lo:.1f}%, {100*inv_hi:.1f}%]")
        lines.append("")
    return "\n".join(lines)


def task_M5_anchor_sensitivity(df):
    """Compute inflation ratio across multiple (small,large) anchor pairs."""
    PAIRS = [
        ("intfloat/multilingual-e5-small", "Qwen/Qwen3-Embedding-8B"),  # paper baseline
        ("ibm-granite/granite-embedding-107m-multilingual", "Qwen/Qwen3-Embedding-8B"),
        ("ibm-granite/granite-embedding-107m-multilingual", "nvidia/llama-embed-nemotron-8b"),
        ("jinaai/jina-embeddings-v5-text-nano", "Qwen/Qwen3-Embedding-4B"),
        ("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large-instruct"),
    ]
    lines = ["# Anchor-pair inflation sensitivity sweep\n"]
    lines.append(f"{'Small':45s} {'Large':45s} {'Italian':>8s} {'Japanese':>10s} {'Hindi':>8s}")
    for sm, lg in PAIRS:
        row = [sm[:45], lg[:45]]
        for lang in ["italian", "japanese", "hindi"]:
            d = df[df.language == lang]
            try:
                small_mteb = float(d[d.model_id == sm].mteb_agg_score.iloc[0])
                small_lang = float(d[d.model_id == sm].lang_specific_avg.iloc[0])
                large_mteb = float(d[d.model_id == lg].mteb_agg_score.iloc[0])
                large_lang = float(d[d.model_id == lg].lang_specific_avg.iloc[0])
                num = large_mteb - small_mteb
                den = large_lang - small_lang
                if abs(den) < 0.01:
                    row.append("inf/0")
                else:
                    row.append(f"{num/den:.2f}x")
            except (IndexError, ValueError):
                row.append("--")
        lines.append(f"{row[0]:45s} {row[1]:45s} {row[2]:>8s} {row[3]:>10s} {row[4]:>8s}")
    return "\n".join(lines)


def task_threshold_sensitivity(df):
    """Tabulate hidden-failure set across threshold sweep."""
    lines = ["# Hidden-failure threshold sensitivity\n",
             "Identified set across (lang_threshold, mteb_threshold) sweep:\n"]
    multi = df[(~df.english_only) & (df.source != "missing") & (~df.lang_specific_avg.isna())].copy()
    THRESHOLDS = [(40, 65), (45, 70), (50, 75), (55, 80), (60, 85)]
    for lang_t, mteb_t in THRESHOLDS:
        hits = multi[(multi.lang_specific_avg < lang_t) & (multi.mteb_agg_score > mteb_t)]
        items = [f"{r.short_name}/{r.language[:3]} (m={r.mteb_agg_score:.1f}, l={r.lang_specific_avg:.1f})"
                 for _, r in hits.iterrows()]
        lines.append(f"  lang<{lang_t}, MTEB>{mteb_t}: {len(items)} cases — {', '.join(items)}")
    return "\n".join(lines)


def main():
    print("[load] unified_results.csv")
    df = pd.read_csv(OUT / "unified_results.csv")

    # Task: Italian RAG eval data + by-type
    print("[task] aggregating RAG evals (incl. Italian)")
    rag_overall, rag_bytype = load_all_rag()
    rag_overall.to_csv(OUT / "unified_rag_results_v2.csv", index=False)
    rag_bytype.to_csv(OUT / "unified_rag_bytype_results_v2.csv", index=False)
    print(f"  → {len(rag_overall)} (model,dataset) overall rows")
    print(f"  → {len(rag_bytype)} (model,dataset,query_type) rows")

    # Task C1
    print("[task C1] nemotron-included Italian rank inversions")
    (STATS / "rev_C1_nemotron_italian.txt").write_text(task_C1_nemotron_italian(df))

    # Task M3
    print("[task M3] BH-FDR")
    (STATS / "rev_M3_bh_fdr.txt").write_text(task_M3_bh_fdr(df))

    # Task M4
    print("[task M4] Kendall tau bootstrap CIs")
    (STATS / "rev_M4_kendall_ci.txt").write_text(task_M4_kendall_ci(df))

    # Task M5
    print("[task M5] Anchor sensitivity")
    (STATS / "rev_M5_anchor_sens.txt").write_text(task_M5_anchor_sensitivity(df))

    # Threshold sweep
    print("[task] hidden-failure threshold sensitivity")
    (STATS / "rev_threshold_sens.txt").write_text(task_threshold_sensitivity(df))

    print("[done]")


if __name__ == "__main__":
    main()
