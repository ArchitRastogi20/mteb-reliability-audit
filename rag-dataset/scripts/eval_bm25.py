#!/usr/bin/env python3
"""
BM25 Baseline Evaluation for custom LAG dataset.

Usage:
    python scripts/eval_bm25.py                         # all 4 combos
    python scripts/eval_bm25.py --lang ja --domain finance
    python scripts/eval_bm25.py --resume
    python scripts/eval_bm25.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from joblib import Parallel, delayed
from rank_bm25 import BM25Okapi
from ranx import Qrels, Run, evaluate

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
EVAL_DIR = OUTPUT_DIR / "evaluation" / "bm25"
COMBOS = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law"), ("it", "finance"), ("it", "law")]
TOP_K = 10


def _cpu_workers() -> int:
    yaml_path = REPO_ROOT / "config" / "hardware.yaml"
    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text())
        return cfg.get("torch_num_threads", max(1, (os.cpu_count() or 4) - 2))
    return max(1, (os.cpu_count() or 4) - 2)


def _tokenize(text: str, lang: str) -> list[str]:
    if lang == "hi":
        return text.split()
    if lang == "ja":
        return list(text)
    return text.lower().split()


def load_combo_data(output_dir: Path, lang: str, domain: str) -> dict:
    base = output_dir / lang / domain

    corpus: dict[str, str] = {}
    with (base / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                corpus[row["_id"]] = row.get("text", "")

    queries: dict[str, str] = {}
    with (base / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels" / "test.tsv").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[did] = rel

    query_types: dict[str, str] = {}
    with (base / "full_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                query_types[row["_id"]] = row["query_type"]

    return {"corpus": corpus, "queries": queries, "qrels": qrels, "query_types": query_types}


def _compute_metrics(run_dict: dict, qrels_str: dict, save_run_path: Path | None = None) -> dict[str, float]:
    if not run_dict or not qrels_str:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    if save_run_path is not None:
        save_run_path.parent.mkdir(parents=True, exist_ok=True)
        save_run_path.write_text(json.dumps(run_dict, indent=2))
    result = evaluate(
        Qrels(qrels_str),
        Run(run_dict),
        ["ndcg@10", "recall@10", "mrr"],
        make_comparable=True,
    )
    return {
        "ndcg_at_10": float(result["ndcg@10"]),
        "recall_at_10": float(result["recall@10"]),
        "mrr": float(result["mrr"]),
    }


def run_bm25(lang: str, domain: str, n_workers: int, output_dir: Path = OUTPUT_DIR, save_runs: bool = False) -> dict:
    t0 = time.time()
    data = load_combo_data(output_dir, lang, domain)

    corpus = data["corpus"]
    queries = data["queries"]
    qrels = data["qrels"]
    query_types = data["query_types"]

    doc_ids = list(corpus.keys())
    print(f"[BM25] {lang}/{domain}: indexing {len(doc_ids)} docs…", flush=True)
    tokenized_corpus = [_tokenize(corpus[did], lang) for did in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    q_ids = list(queries.keys())
    tokenized_queries = [_tokenize(queries[qid], lang) for qid in q_ids]
    top_k = TOP_K

    all_scores = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(bm25.get_scores)(tq) for tq in tokenized_queries
    )

    run_dict: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(q_ids):
        scores = all_scores[qi]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        run_dict[qid] = {doc_ids[i]: float(scores[i]) for i in top_indices}

    qrels_str = {
        str(k): {str(dk): int(dv) for dk, dv in v.items()}
        for k, v in qrels.items()
    }

    save_run_path = None
    if save_runs:
        runs_dir = OUTPUT_DIR / "evaluation" / "runs"
        save_run_path = runs_dir / f"BM25_{lang}_{domain}.json"
    overall = _compute_metrics(run_dict, qrels_str, save_run_path)

    all_types = ["factual", "multi_hop", "summarization", "unanswerable"]
    by_type: dict[str, dict[str, float]] = {}
    for qtype in all_types:
        type_qids = [qid for qid in q_ids if query_types.get(qid) == qtype]
        if not type_qids:
            by_type[qtype] = {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
            continue
        type_run = {qid: run_dict[qid] for qid in type_qids}
        type_qrels = {qid: qrels_str[qid] for qid in type_qids if qid in qrels_str}
        by_type[qtype] = _compute_metrics(type_run, type_qrels)

    elapsed = time.time() - t0
    result = {
        "model_id": "BM25",
        "lang": lang,
        "domain": domain,
        "num_docs": len(corpus),
        "num_queries": len(q_ids),
        "overall": overall,
        "by_type": by_type,
        "evaluation_time_sec": round(elapsed, 2),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{lang}_{domain}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(
        f"[BM25] {lang}/{domain}: NDCG@10={overall['ndcg_at_10']:.4f}  "
        f"recall@10={overall['recall_at_10']:.4f}  mrr={overall['mrr']:.4f}  ({elapsed:.1f}s)"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 Baseline for RAG dataset")
    parser.add_argument("--lang", choices=["ja", "hi", "it"])
    parser.add_argument("--domain", choices=["finance", "law"])
    parser.add_argument("--resume", action="store_true", help="Skip combos with existing results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without evaluating")
    parser.add_argument("--save-runs", action="store_true",
                        help="Q7: save per-query top-10 run_dicts for hybrid RRF fusion.")
    args = parser.parse_args()

    combos = COMBOS
    if args.lang and args.domain:
        combos = [(args.lang, args.domain)]
    elif args.lang:
        combos = [(ll, d) for ll, d in COMBOS if ll == args.lang]
    elif args.domain:
        combos = [(ll, d) for ll, d in COMBOS if d == args.domain]

    if args.resume:
        combos = [(ll, d) for ll, d in combos if not (EVAL_DIR / f"{ll}_{d}.json").exists()]

    n_workers = _cpu_workers()
    print(f"Hardware: {os.cpu_count()} CPU cores  workers={n_workers}")

    if args.dry_run:
        print(f"\nWould evaluate {len(combos)} combos:")
        for lang, domain in combos:
            print(f"  {lang}/{domain}")
        return

    for lang, domain in combos:
        run_bm25(lang, domain, n_workers, save_runs=args.save_runs)


if __name__ == "__main__":
    main()
