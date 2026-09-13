#!/usr/bin/env python3
# scripts/eval_hybrid_rrf.py
"""Q7: Reciprocal Rank Fusion (RRF) hybrid sparse+dense retrieval evaluation.

Reads persisted run_dicts from output/evaluation/runs/, fuses BM25 + dense
rankings via RRF, computes NDCG@10/recall@10/mrr against qrels.

Usage:
    python scripts/eval_hybrid_rrf.py --all            # all (model, dataset) combos
    python scripts/eval_hybrid_rrf.py \
        --bm25-run output/evaluation/runs/BM25_hi_law.json \
        --dense-run output/evaluation/runs/BAAI_bge_m3_hi_law.json \
        --lang hi --domain law --model bge-m3 --k 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "output" / "evaluation" / "runs"
HYBRID_DIR = REPO_ROOT / "output" / "evaluation" / "hybrid"
OUTPUT_DIR = REPO_ROOT / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranx import Qrels, Run, evaluate


# ── RRF math ──────────────────────────────────────────────────────────────────

def rrf_fuse(runs: list[dict[str, dict[str, float]]], k: int = 60) -> dict[str, dict[str, float]]:
    """Reciprocal Rank Fusion.

    For each query and each run, doc d gets score 1/(k + rank_d). Sum across runs.

    Args:
        runs: list of run_dicts. Each run_dict maps qid -> {doc_id: score}.
              Within each query, docs are assumed ordered by score desc.
        k: RRF constant. Standard value is 60 (Cormack et al. 2009).

    Returns:
        A fused run_dict with the same qid keys, mapping doc_id -> RRF score.
    """
    fused: dict[str, dict[str, float]] = {}
    qids: set[str] = set()
    for r in runs:
        qids.update(r.keys())
    for qid in qids:
        scores: dict[str, float] = {}
        for run in runs:
            ranking = run.get(qid, {})
            sorted_docs = sorted(ranking.items(), key=lambda kv: -kv[1])
            for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if scores:
            fused[qid] = scores
    return fused


# ── Qrels loader ──────────────────────────────────────────────────────────────

def load_qrels(lang: str, domain: str) -> dict[str, dict[str, int]]:
    qrels_path = OUTPUT_DIR / lang / domain / "qrels" / "test.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, did, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


# ── Per-cell hybrid evaluation ────────────────────────────────────────────────

def evaluate_hybrid(bm25_path: Path, dense_path: Path, lang: str, domain: str,
                    model_label: str, k: int = 60) -> dict:
    bm25 = json.loads(bm25_path.read_text())
    dense = json.loads(dense_path.read_text())
    fused = rrf_fuse([bm25, dense], k=k)

    qrels = load_qrels(lang, domain)
    qrels_str = {q: dict(r) for q, r in qrels.items()}

    def metrics(run_dict):
        if not run_dict or not qrels_str:
            return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
        res = evaluate(Qrels(qrels_str), Run(run_dict),
                       ["ndcg@10", "recall@10", "mrr"], make_comparable=True)
        return {"ndcg_at_10": float(res["ndcg@10"]),
                "recall_at_10": float(res["recall@10"]),
                "mrr": float(res["mrr"])}

    return {
        "model": model_label,
        "lang": lang,
        "domain": domain,
        "k_rrf": k,
        "bm25_only": metrics(bm25),
        "dense_only": metrics(dense),
        "hybrid_rrf": metrics(fused),
        "n_queries_fused": len(fused),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_COMBOS = [
    ("hi", "law"),
    ("ja", "finance"),
]
DEFAULT_DENSE_MODELS = [
    ("jina-v5-nano", "jina_embeddings_v5_t_eed139"),
    ("bge-m3",       "bge_m3_75e678"),
    ("Qwen3-8B",     "Qwen3_Embedding_8B_c07e5f"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run all (model, dataset) combos.")
    parser.add_argument("--bm25-run", type=Path, help="Path to BM25 run JSON.")
    parser.add_argument("--dense-run", type=Path, help="Path to dense run JSON.")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--model", type=str, help="Label for output file.")
    parser.add_argument("--k", type=int, default=60, help="RRF constant (default 60).")
    args = parser.parse_args()

    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    if args.all:
        for lang, domain in DEFAULT_COMBOS:
            bm25_path = RUNS_DIR / f"BM25_{lang}_{domain}.json"
            if not bm25_path.exists():
                print(f"[WARN] missing {bm25_path}; run eval_bm25.py --save-runs first")
                continue
            for label, fname_prefix in DEFAULT_DENSE_MODELS:
                dense_path = RUNS_DIR / f"{fname_prefix}_{lang}_{domain}.json"
                if not dense_path.exists():
                    print(f"[WARN] missing {dense_path}; run eval_rag.py --model {label} --save-runs first")
                    continue
                r = evaluate_hybrid(bm25_path, dense_path, lang, domain, label, k=args.k)
                out_file = HYBRID_DIR / f"{label}_{lang}_{domain}.json"
                out_file.write_text(json.dumps(r, indent=2))
                results.append(r)
                print(f"\n=== {label} on {lang}-{domain} (k={args.k}) ===")
                for kind, m in [("bm25_only", r["bm25_only"]),
                                ("dense_only", r["dense_only"]),
                                ("hybrid_rrf", r["hybrid_rrf"])]:
                    print(f"  {kind:12s}: ndcg@10={m['ndcg_at_10']:.3f}")

        summary = {"k": args.k, "results": results}
        (HYBRID_DIR / "q7_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written: {HYBRID_DIR / 'q7_summary.json'}")
    else:
        if not (args.bm25_run and args.dense_run and args.lang and args.domain and args.model):
            parser.error("--all OR (--bm25-run + --dense-run + --lang + --domain + --model) required")
        r = evaluate_hybrid(args.bm25_run, args.dense_run, args.lang, args.domain, args.model, k=args.k)
        out_file = HYBRID_DIR / f"{args.model}_{args.lang}_{args.domain}.json"
        out_file.write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
