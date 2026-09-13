"""Shared utilities for exp9..exp14 (extended analysis batch).

Pure NumPy/SciPy. No I/O beyond the path constants below.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, rankdata

REPO = Path(__file__).resolve().parents[1]  # repo root

# --- canonical input paths (repo layout) ----------------------------
ROSTER_CSV       = REPO / "mteb-ranking-audit" / "analysis" / "roster_membership.csv"
LOCAL_RANKS_CSV  = REPO / "mteb-ranking-audit" / "analysis" / "curated_local_ranks.csv"
SUMMARY_CURATED  = REPO / "mteb-ranking-audit" / "results" / "summary_curated.csv"
HIDDEN_FAILURES  = REPO / "mteb-ranking-audit" / "results" / "hidden_failures.csv"
PER_TASK_NDCG    = REPO / "extended-experiments" / "data" / "per_task_ndcg.csv"
UNIFIED_RESULTS  = REPO / "analysis" / "analysis_output" / "unified_results.csv"
RAG_OUTPUT       = REPO / "rag-dataset" / "output"
RAG_EVAL         = RAG_OUTPUT / "evaluation"
DOWNSTREAM_JSON  = REPO / "analysis" / "analysis_output" / "stats" / "downstream_rag.json"
MTEB_AGG_CSV     = REPO / "mteb-language-gap" / "config" / "mteb_agg_scores.csv"
DEPLOYMENT_CSV   = REPO / "rag-deployment-benchmark" / "results" / "deployment_results.csv"

# --- canonical output paths ----------------------------------------------
OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --- the 14-model standard set used by the headline correlation work ----
STANDARD_14 = frozenset({
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "Salesforce/SFR-Embedding-Mistral",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "ibm-granite/granite-embedding-107m-multilingual",
    "intfloat/e5-mistral-7b-instruct",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/multilingual-e5-small",
    "jinaai/jina-embeddings-v5-text-nano",
    "microsoft/harrier-oss-v1-0.6b",
})

# --- 6 RAG configs --------------------------------------------------------
RAG_CONFIGS = [
    ("it", "finance"), ("it", "law"),
    ("ja", "finance"), ("ja", "law"),
    ("hi", "finance"), ("hi", "law"),
]


def kendall_tau(x, y) -> float:
    """Kendall's tau-b. Returns 0.0 if input shorter than 2."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2:
        return 0.0
    tau, _ = kendalltau(x, y)
    return float(tau) if not np.isnan(tau) else 0.0


def inversion_rate(rank_a, rank_b) -> float:
    """Fraction of unordered pairs where the sign of rank_a[i]-rank_a[j]
    disagrees with the sign of rank_b[i]-rank_b[j]. Ties contribute 0.5."""
    a = np.asarray(rank_a, dtype=float); b = np.asarray(rank_b, dtype=float)
    n = len(a)
    if n < 2:
        return 0.0
    pairs = n * (n - 1) / 2
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    iu = np.triu_indices(n, k=1)
    sa = np.sign(da[iu]); sb = np.sign(db[iu])
    discordant = np.sum(sa * sb < 0)
    ties = np.sum((sa == 0) | (sb == 0))
    return float((discordant + 0.5 * ties) / pairs)


def dense_rank(values, ascending=False) -> np.ndarray:
    """1-based dense rank. ascending=False => rank 1 = largest."""
    arr = np.asarray(values, dtype=float)
    if ascending:
        return rankdata(arr, method="min").astype(int)
    return rankdata(-arr, method="min").astype(int)
