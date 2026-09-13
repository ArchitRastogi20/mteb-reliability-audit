#!/usr/bin/env python3
"""
Pre-download and cache all MTEB datasets to /workspace/dataset_cache/.

Both eval_bm25.py and lb_evaluator.py read from this cache — running this
script first means evaluation never blocks on a network download.

Usage:
    python scripts/cache_datasets.py              # all languages
    python scripts/cache_datasets.py --lang jpn   # single language
    python scripts/cache_datasets.py --force       # re-download even if cached
    python scripts/cache_datasets.py --dry-run     # show what would be cached
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DATASET_CACHE_DIR = Path("/workspace/dataset_cache")
REPO_ROOT = Path(__file__).parent.parent

# Make `scripts.*` importable when run directly as `python scripts/cache_datasets.py`
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _task_cache_path(task_name: str, hf_subset: str, split: str) -> Path:
    key = f"{task_name}__{hf_subset}__{split}".replace("/", "-")
    return DATASET_CACHE_DIR / f"{key}.json"


def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def _load_belebele(hf_subset: str) -> tuple[dict, dict, dict]:
    from datasets import load_dataset
    lang_config = hf_subset.split("-")[0]
    revision = "979a211276faa22f671e69d096634193567cfd05"
    ds = load_dataset("mteb/belebele", lang_config, revision=revision, split="test")

    link_to_cid: dict[str, str] = {}
    corpus: dict[str, str] = {}
    for row in ds:
        link = row["link"]
        if link not in link_to_cid:
            cid = f"C{len(link_to_cid)}"
            link_to_cid[link] = cid
            corpus[cid] = row["flores_passage"]

    question_to_qid: dict[str, str] = {}
    queries: dict[str, str] = {}
    for row in ds:
        q = row["question"]
        if q not in question_to_qid:
            qid = f"Q{len(question_to_qid)}"
            question_to_qid[q] = qid
            queries[qid] = q

    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = question_to_qid[row["question"]]
        cid = link_to_cid[row["link"]]
        qrels.setdefault(qid, {})[cid] = 1

    return corpus, queries, qrels


def _load_mteb_task(task_name: str, hf_subset: str, split: str) -> tuple[dict, dict, dict]:
    import mteb
    t = mteb.get_task(task_name)
    t.load_data(eval_splits=[split], hf_subsets=[hf_subset])
    data = t.dataset[hf_subset][split]
    corpus = {row["id"]: _corpus_text(row) for row in data["corpus"]}
    queries = {row["id"]: row["text"] for row in data["queries"]}
    qrels = {
        str(k): {str(dk): int(dv) for dk, dv in v.items()}
        for k, v in data["relevant_docs"].items()
    }
    return corpus, queries, qrels


def cache_task(task_name: str, hf_subset: str, split: str, force: bool) -> bool:
    cache_path = _task_cache_path(task_name, hf_subset, split)
    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text())
        print(
            f"  [skip] {cache_path.name}  "
            f"(corpus={len(data['corpus'])}, queries={len(data['queries'])})"
        )
        return False

    t0 = time.time()
    print(f"  [download] {task_name}  subset={hf_subset}  split={split} …", flush=True)
    if task_name == "BelebeleRetrieval":
        corpus, queries, qrels = _load_belebele(hf_subset)
    else:
        corpus, queries, qrels = _load_mteb_task(task_name, hf_subset, split)

    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels}))
    elapsed = time.time() - t0
    print(
        f"  [cached]   {cache_path.name}  "
        f"corpus={len(corpus)}, queries={len(queries)}  ({elapsed:.1f}s)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache MTEB datasets to disk")
    parser.add_argument(
        "--lang", choices=["jpn", "hin", "ita", "all"], default="all",
        help="Which language(s) to cache (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cached, don't download")
    args = parser.parse_args()

    from scripts.lb_evaluator import LANG_CONFIGS

    langs = ["jpn", "hin", "ita"] if args.lang == "all" else [args.lang]

    tasks_to_cache: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for lang in langs:
        cfg = LANG_CONFIGS[lang]
        for task_name in cfg.tasks:
            task_cfg = cfg.task_config[task_name]
            key = (task_name, task_cfg["hf_subset"], task_cfg["split"])
            if key not in seen:
                seen.add(key)
                tasks_to_cache.append((lang, *key))

    print(f"Cache dir: {DATASET_CACHE_DIR}")
    print(f"Tasks to process: {len(tasks_to_cache)}\n")

    if args.dry_run:
        for lang, task_name, hf_subset, split in tasks_to_cache:
            path = _task_cache_path(task_name, hf_subset, split)
            status = "exists" if path.exists() else "missing"
            print(f"  [{status}] {path.name}  (lang={lang})")
        return

    downloaded = skipped = 0
    for _lang, task_name, hf_subset, split in tasks_to_cache:
        was_downloaded = cache_task(task_name, hf_subset, split, args.force)
        if was_downloaded:
            downloaded += 1
        else:
            skipped += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} already cached.")


if __name__ == "__main__":
    main()
