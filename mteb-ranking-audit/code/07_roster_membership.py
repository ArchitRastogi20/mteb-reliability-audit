"""
Roster membership and local-rank documentation for the three-tier ranking audit.

Produces:
  analysis/roster_membership.csv   -- 25 curated-tier models with global ranks
  analysis/curated_local_ranks.csv -- per-(model, language) local ranks
  paper/appendix_roster_block.tex  -- LaTeX appendix block listing tier composition

Data sources (pre-computed by 01_download.py and 03_analyze.py):
  global/mteb_agg.csv              -- restricted-tier global predictor scores
  global/mteb_agg_curated.csv      -- curated-tier global predictor scores
  <lang>/lang_avg_curated.csv      -- per-language curated-tier averages
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent   # mteb-ranking-audit/
ANALYSIS = BASE / "analysis"
PAPER    = BASE / "paper"
ANALYSIS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

# ── Language list (Tamil excluded: n_tasks < 2 in curated/extended tiers) ───
LANG_META = {
    "ita": ("Italian",    "case_studies"),
    "jpn": ("Japanese",   "case_studies"),
    "hin": ("Hindi",      "case_studies"),
    "ara": ("Arabic",     "audit_languages"),
    "zho": ("Chinese",    "audit_languages"),
    "deu": ("German",     "audit_languages"),
    "spa": ("Spanish",    "audit_languages"),
    "rus": ("Russian",    "audit_languages"),
    "fra": ("French",     "audit_languages"),
    "kor": ("Korean",     "audit_languages"),
    "ben": ("Bengali",    "audit_languages"),
    "vie": ("Vietnamese", "audit_languages"),
    "ind": ("Indonesian", "audit_languages"),
    "fas": ("Persian",    "audit_languages"),
    "swa": ("Swahili",    "audit_languages"),
    "tha": ("Thai",       "audit_languages"),
    "tel": ("Telugu",     "audit_languages"),
}

# ── Roster definitions (must match 01_download.py exactly) ──────────────────
# Restricted tier: models with complete 18-task MMTEB coverage (2026-05 snapshot).
RESTRICTED = {
    "Qwen3-0.6B":   "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-4B":     "Qwen/Qwen3-Embedding-4B",
    "Qwen3-8B":     "Qwen/Qwen3-Embedding-8B",
    "nemotron-8b":  "nvidia/llama-embed-nemotron-8b",
    "harrier-0.6b": "microsoft/harrier-oss-v1-0.6b",
    "bge-m3":       "BAAI/bge-m3",
    "ml-e5-small":  "intfloat/multilingual-e5-small",
}

# Curated-only additions: restricted ∪ these = curated tier (n=25 total).
# This is a hand-selected, judgment-based, deployment-relevant roster, not a
# computed cutoff: no script in this repo computes MMTEB(Multilingual)
# language coverage or a top-quartile threshold for these additions. The
# roster is frozen and released in full as of the 2026-05 snapshot; the
# hardcoded list below IS the definition of curated-tier membership.
CURATED_ONLY = {
    "KaLM-Gemma3-12B":    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "pplx-embed-4b":      "perplexity-ai/pplx-embed-v1-4b",
    "inf-retriever-v1":   "infly/inf-retriever-v1",
    "F2LLM-v2-14B":       "codefuse-ai/F2LLM-v2-14B",
    "Seed1.6-embed":      "Bytedance/Seed1.6-embedding-1215",
    "F2LLM-v2-8B":        "codefuse-ai/F2LLM-v2-8B",
    "pplx-embed-0.6b":    "perplexity-ai/pplx-embed-v1-0.6b",
    "granite-311m":       "ibm-granite/granite-embedding-311m-multilingual-r2",
    "jina-v5-small":      "jinaai/jina-embeddings-v5-text-small",
    "F2LLM-v2-4B":        "codefuse-ai/F2LLM-v2-4B",
    "voyage-3.5":         "voyageai/voyage-3.5",
    "jina-v5-nano":       "jinaai/jina-embeddings-v5-text-nano",
    "zembed-1":           "zeroentropy/zembed-1",
    "embeddinggemma-300m":"google/embeddinggemma-300m",
    "BOOM-4B":            "ICT-TIME-and-Querit/BOOM_4B_v1",
    "BidirLM-2.5B":       "BidirLM/BidirLM-Omni-2.5B-Embedding",
    "GritLM-7B":          "GritLM/GritLM-7B",
    "BidirLM-1B":         "BidirLM/BidirLM-1B-Embedding",
}

MODEL_META = {
    "Qwen3-0.6B":         {"params": "0.6B",  "family": "Qwen3"},
    "Qwen3-4B":           {"params": "4B",    "family": "Qwen3"},
    "Qwen3-8B":           {"params": "8B",    "family": "Qwen3"},
    "nemotron-8b":        {"params": "8B",    "family": "Nemotron"},
    "harrier-0.6b":       {"params": "0.6B",  "family": "Harrier"},
    "bge-m3":             {"params": "568M",  "family": "BGE"},
    "ml-e5-small":        {"params": "117M",  "family": "E5"},
    "KaLM-Gemma3-12B":    {"params": "12B",   "family": "KaLM-Gemma"},
    "pplx-embed-4b":      {"params": "4B",    "family": "pplx-embed"},
    "inf-retriever-v1":   {"params": "7B",    "family": "Inf-Retriever"},
    "F2LLM-v2-14B":       {"params": "14B",   "family": "F2LLM"},
    "Seed1.6-embed":      {"params": "7B",    "family": "Seed"},
    "F2LLM-v2-8B":        {"params": "8B",    "family": "F2LLM"},
    "pplx-embed-0.6b":    {"params": "0.6B",  "family": "pplx-embed"},
    "granite-311m":       {"params": "311M",  "family": "Granite"},
    "jina-v5-small":      {"params": "570M",  "family": "Jina-v5"},
    "F2LLM-v2-4B":        {"params": "4B",    "family": "F2LLM"},
    "voyage-3.5":         {"params": "unk",   "family": "Voyage"},
    "jina-v5-nano":       {"params": "228M",  "family": "Jina-v5"},
    "zembed-1":           {"params": "unk",   "family": "ZEmbed"},
    "embeddinggemma-300m":{"params": "300M",  "family": "EmbeddingGemma"},
    "BOOM-4B":            {"params": "4B",    "family": "BOOM"},
    "BidirLM-2.5B":       {"params": "2.5B",  "family": "BidirLM"},
    "GritLM-7B":          {"params": "7B",    "family": "GritLM"},
    "BidirLM-1B":         {"params": "1B",    "family": "BidirLM"},
}

HF_LOOKUP = {**RESTRICTED, **CURATED_ONLY}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_global(suffix: str) -> pd.DataFrame:
    path = BASE / "global" / f"mteb_agg{suffix}.csv"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path).sort_values("mteb_agg", ascending=False).reset_index(drop=True)


def build_roster_membership(curated_df: pd.DataFrame, restricted_df: pd.DataFrame) -> pd.DataFrame:
    cur_ranks = {row["model"]: i + 1 for i, row in curated_df.iterrows()}
    res_ranks = {row["model"]: i + 1 for i, row in restricted_df.iterrows()}

    rows = []
    for display, hf_id in RESTRICTED.items():
        meta = MODEL_META.get(display, {})
        rows.append({
            "model_id":               hf_id,
            "display_name":           display,
            "tier":                   "restricted",
            "global_rank_curated":    cur_ranks.get(display),
            "global_rank_restricted": res_ranks.get(display),
            "params":                 meta.get("params", "unk"),
            "family":                 meta.get("family", "unk"),
        })
    for display, hf_id in CURATED_ONLY.items():
        meta = MODEL_META.get(display, {})
        rows.append({
            "model_id":               hf_id,
            "display_name":           display,
            "tier":                   "curated_only",
            "global_rank_curated":    cur_ranks.get(display),
            "global_rank_restricted": None,
            "params":                 meta.get("params", "unk"),
            "family":                 meta.get("family", "unk"),
        })
    return pd.DataFrame(rows)


def build_local_ranks() -> pd.DataFrame:
    rows = []
    for iso3, (lang_name, subdir) in LANG_META.items():
        path = BASE / subdir / iso3 / "lang_avg_curated.csv"
        if not path.exists():
            print(f"  skip {iso3}: lang_avg_curated.csv missing")
            continue
        df = pd.read_csv(path)
        df["local_rank"] = df["lang_avg"].rank(ascending=False, method="min").astype(int)
        for _, row in df.iterrows():
            rows.append({
                "model_id":     HF_LOOKUP.get(row["model"], row["model"]),
                "display_name": row["model"],
                "language_iso": iso3,
                "lang_name":    lang_name,
                "local_rank":   int(row["local_rank"]),
                "local_ndcg":   float(row["lang_avg"]),
            })
    return pd.DataFrame(rows)


def write_appendix_tex(membership: pd.DataFrame) -> None:
    curated_only_sorted = (
        membership[membership["tier"] == "curated_only"]
        .sort_values("display_name")
    )
    restricted_sorted = sorted(RESTRICTED.keys())

    entries = []
    for _, row in curated_only_sorted.iterrows():
        rank = row["global_rank_curated"]
        rank_str = f"rank {int(rank)}" if pd.notna(rank) else "rank~n/a"
        entries.append(f"\\texttt{{{row['display_name']}}} ({rank_str})")
    model_list = ",\n".join(entries)

    restr_list = ", ".join(f"\\texttt{{{m}}}" for m in restricted_sorted)
    n_additions = len(curated_only_sorted)

    tex = (
        f"\\paragraph{{Curated ($n{{=}}25$).}}"
        f" Restricted plus {n_additions} deployment-relevant\n"
        "additions. This is a hand-selected, judgment-based, deployment-relevant\n"
        "roster, not a computed cutoff; the roster is frozen and released in\n"
        "full below, and this list IS the definition of membership (2026-05\n"
        "snapshot):\n"
        f"{model_list}.\n\n"
        f"\\paragraph{{Restricted ($n{{=}}7$).}}"
        " Models with complete MTEB(Multilingual)\n"
        "retrieval results across all 18 predictor tasks (2026-05 snapshot):\n"
        f"{restr_list}.\n"
    )
    out = PAPER / "appendix_roster_block.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")


def cross_check_granite_ita(local_ranks: pd.DataFrame) -> None:
    row = local_ranks[
        (local_ranks["display_name"] == "granite-311m") &
        (local_ranks["language_iso"] == "ita")
    ]
    if row.empty:
        print("WARN: granite-311m / ita not found — cannot cross-check")
        return
    rank = int(row["local_rank"].iloc[0])
    if rank != 24:
        print(
            f"CRITICAL: granite-311m Italian local rank = {rank}, expected 24.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Cross-check PASSED: granite-311m Italian local rank = {rank}.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    curated_df    = load_global("_curated")
    restricted_df = load_global("")

    membership = build_roster_membership(curated_df, restricted_df)

    n_rows = len(membership)
    if n_rows != 25:
        print(f"ERROR: expected 25 curated-tier rows, got {n_rows}", file=sys.stderr)
        sys.exit(1)
    print(f"Curated tier: {n_rows} models. OK")

    membership.to_csv(ANALYSIS / "roster_membership.csv", index=False)
    print(f"Wrote analysis/roster_membership.csv ({n_rows} rows)")

    local_ranks = build_local_ranks()
    cross_check_granite_ita(local_ranks)

    n_local = len(local_ranks)
    expected = 17 * 25
    if n_local != expected:
        print(
            f"Note: curated_local_ranks has {n_local} rows "
            f"(expected up to {expected}; some models dropped due to missing task coverage)."
        )

    local_ranks.to_csv(ANALYSIS / "curated_local_ranks.csv", index=False)
    print(f"Wrote analysis/curated_local_ranks.csv ({n_local} rows)")

    write_appendix_tex(membership)


if __name__ == "__main__":
    main()
