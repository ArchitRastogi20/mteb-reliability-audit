#!/usr/bin/env python3
"""
BM25 Baseline for MTEB leaderboard datasets.

Usage:
    python scripts/eval_bm25.py --lang all
    python scripts/eval_bm25.py --lang jpn
    python scripts/eval_bm25.py --lang hin --resume
    python scripts/eval_bm25.py --lang ita --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from joblib import Parallel, delayed
from rank_bm25 import BM25Okapi
from ranx import Qrels, Run, evaluate

from scripts.lb_evaluator import LANG_CONFIGS
DATASET_CACHE_DIR = Path("/workspace/dataset_cache")
RESULTS_DIR = REPO_ROOT / "results" / "bm25"
TOP_K = 10


def _cpu_workers() -> int:
    yaml_path = REPO_ROOT / "config" / "hardware.yaml"
    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text())
        return cfg.get("torch_num_threads", max(1, (os.cpu_count() or 4) - 2))
    return max(1, (os.cpu_count() or 4) - 2)


def _tokenize(text: str, lang_code: str) -> list[str]:
    if lang_code == "hin":
        return text.split()
    if lang_code == "jpn":
        return list(text)
    return text.lower().split()


def _task_cache_path(task_name: str, hf_subset: str, split: str) -> Path:
    key = f"{task_name}__{hf_subset}__{split}".replace("/", "-")
    return DATASET_CACHE_DIR / f"{key}.json"


def run_bm25_task(
    task_name: str, lang_code: str, hf_subset: str, split: str, n_workers: int
) -> tuple[float, float]:
    cache_path = _task_cache_path(task_name, hf_subset, split)
    data = json.loads(cache_path.read_text())

    corpus: dict[str, str] = data["corpus"]
    queries: dict[str, str] = data["queries"]
    qrels: dict = data["qrels"]

    t0 = time.time()

    doc_ids = list(corpus.keys())
    print(f"[BM25] {lang_code}/{task_name}: indexing {len(doc_ids)} docs…", flush=True)
    tokenized_corpus = [_tokenize(corpus[did], lang_code) for did in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    q_ids = list(queries.keys())
    tokenized_queries = [_tokenize(queries[qid], lang_code) for qid in q_ids]

    all_scores = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(bm25.get_scores)(tq) for tq in tokenized_queries
    )

    run_dict: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(q_ids):
        scores = all_scores[qi]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        run_dict[str(qid)] = {str(doc_ids[i]): float(scores[i]) for i in top_indices}

    qrels_str = {
        str(k): {str(dk): int(dv) for dk, dv in v.items()}
        for k, v in qrels.items()
    }

    ndcg = float(evaluate(Qrels(qrels_str), Run(run_dict), "ndcg@10", make_comparable=True))
    elapsed = time.time() - t0

    print(f"[BM25] {lang_code}/{task_name}: NDCG@10={ndcg:.4f}  ({elapsed:.1f}s)")
    return ndcg, elapsed


def evaluate_lang(lang_code: str, n_workers: int, resume: bool) -> dict[str, float]:
    lang_cfg = LANG_CONFIGS[lang_code]
    results_lang_dir = RESULTS_DIR / f"{lang_code}_lb"
    results_lang_dir.mkdir(parents=True, exist_ok=True)

    task_results: dict[str, float] = {}
    for task_name in lang_cfg.tasks:
        out_path = results_lang_dir / f"{task_name}.json"

        if resume and out_path.exists():
            cached = json.loads(out_path.read_text())
            ndcg = cached["scores"]["test"][0]["ndcg_at_10"]
            task_results[task_name] = ndcg
            print(f"[BM25] {lang_code}/{task_name}: NDCG@10={ndcg:.4f}  (resumed)")
            continue

        task_cfg = lang_cfg.task_config[task_name]
        ndcg, elapsed = run_bm25_task(
            task_name, lang_code,
            task_cfg["hf_subset"], task_cfg["split"],
            n_workers,
        )
        task_results[task_name] = ndcg
        out_path.write_text(json.dumps({
            "dataset_revision": "N/A",
            "evaluation_time": round(elapsed, 2),
            "scores": {"test": [{"ndcg_at_10": round(ndcg, 6), "main_score": round(ndcg, 6)}]},
        }, indent=2))

    return task_results


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 Baseline for MTEB leaderboard")
    parser.add_argument("--lang", choices=["jpn", "hin", "ita", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with existing results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without evaluating")
    args = parser.parse_args()

    langs = ["jpn", "hin", "ita"] if args.lang == "all" else [args.lang]
    n_workers = _cpu_workers()
    print(f"Hardware: {os.cpu_count()} CPU cores  workers={n_workers}")

    if args.dry_run:
        print(f"\nWould evaluate langs: {langs}")
        for lang in langs:
            for task in LANG_CONFIGS[lang].tasks:
                print(f"  {lang}/{task}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for lang_code in langs:
        results = evaluate_lang(lang_code, n_workers, args.resume)
        avg = sum(results.values()) / len(results) if results else 0.0
        lang_cfg = LANG_CONFIGS[lang_code]
        summary[lang_code] = {
            **{t: round(v * 100, 2) for t, v in results.items()},
            lang_cfg.avg_label: round(avg * 100, 2),
        }
        print(f"[BM25] {lang_code}: {lang_cfg.avg_label}={avg * 100:.2f}")

    (RESULTS_DIR / "bm25_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {RESULTS_DIR / 'bm25_summary.json'}")


if __name__ == "__main__":
    main()
