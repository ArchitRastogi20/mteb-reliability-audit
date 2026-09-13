# mteb-language-gap/scripts/q1_swa_summary.py
"""Compute Kendall τ + inversion rate for own-run Swahili-Belebele vs MTEB-agg.

Output: results/swa_lb/q1_summary.json
Compare to the §3.5 leaderboard-derived figures: τ=0.56, inv=22.0%.
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

from scripts.utils import sanitize_model_id

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "swa_lb"
OUT = RESULTS / "q1_summary.json"

# Headline MTEB-agg from the paper's Table 1 (predictor)
MTEB_AGG = {
    "ibm-granite/granite-embedding-107m-multilingual": 65.0,
    "intfloat/multilingual-e5-small": 77.0,
    "jinaai/jina-embeddings-v5-text-nano": 72.0,
    "intfloat/multilingual-e5-base": 79.7,
    "intfloat/multilingual-e5-large": 84.2,
    "intfloat/multilingual-e5-large-instruct": 83.8,
    "BAAI/bge-m3": 84.0,
    "Snowflake/snowflake-arctic-embed-l-v2.0": 82.3,
    "microsoft/harrier-oss-v1-0.6b": 78.0,
    "Qwen/Qwen3-Embedding-0.6B": 77.9,
    "Qwen/Qwen3-Embedding-4B": 86.2,
    "Salesforce/SFR-Embedding-Mistral": 80.3,
    "intfloat/e5-mistral-7b-instruct": 79.6,
    "nvidia/llama-embed-nemotron-8b": 87.0,
    "Qwen/Qwen3-Embedding-8B": 90.4,
}

def sanitize(model_id: str) -> str:
    return sanitize_model_id(model_id)

def get_score(d, key="ndcg_at_10"):
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() == key:
                return v
            r = get_score(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for x in d:
            r = get_score(x, key)
            if r is not None:
                return r
    return None

def kendall_tau(xs, ys):
    c = d = 0
    for i, j in combinations(range(len(xs)), 2):
        s = (xs[i] - xs[j]) * (ys[i] - ys[j])
        if s > 0: c += 1
        elif s < 0: d += 1
    total = c + d
    return ((c - d) / total if total else float("nan")), total, d

def main():
    pairs = []  # (model_id, mteb_agg, swa_ndcg)
    for model_id, agg in MTEB_AGG.items():
        # Find the model dir (sanitized, may have a trailing hash from cache)
        san_id = sanitize(model_id)
        candidate_dir = RESULTS / san_id
        if not candidate_dir.exists():
            print(f"[WARN] no result dir for {model_id} (looked for {san_id})")
            continue
        candidates = [candidate_dir]
        result_file = candidates[0] / "BelebeleRetrieval.json"
        if not result_file.exists():
            print(f"[WARN] {result_file} missing")
            continue
        ndcg = get_score(json.loads(result_file.read_text()))
        if ndcg is None:
            print(f"[WARN] no ndcg_at_10 in {result_file}")
            continue
        pairs.append((model_id, agg, ndcg * 100))

    pairs.sort(key=lambda p: -p[1])  # sort by MTEB-agg desc
    aggs = [p[1] for p in pairs]
    swa  = [p[2] for p in pairs]
    tau, total, inv = kendall_tau(aggs, swa)

    summary = {
        "task": "BelebeleRetrieval",
        "lang": "swa",
        "n_models": len(pairs),
        "kendall_tau_vs_mteb_agg": tau,
        "inversions": inv,
        "total_pairs": total,
        "inversion_rate_pct": (inv / total * 100) if total else None,
        "models": [{"model_id": m, "mteb_agg": a, "swa_belebele_ndcg10": s}
                   for (m, a, s) in pairs],
        "compare_to_paper_3.5": {
            "leaderboard_filtered_tau": 0.56,
            "leaderboard_filtered_inv_pct": 22.0,
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Q1 Swahili-Belebele own-run summary ===")
    print(f"  n_models   = {len(pairs)}")
    print(f"  Kendall τ  = {tau:+.3f}")
    print(f"  inversions = {inv}/{total} ({inv/total*100:.1f}%)" if total else "  inversions = N/A")
    print(f"  → paper's leaderboard-only τ=0.56 / inv=22.0%")
    print(f"\nWritten: {OUT}")

if __name__ == "__main__":
    main()
