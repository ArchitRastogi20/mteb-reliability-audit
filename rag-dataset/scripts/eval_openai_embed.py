# rag-dataset/scripts/eval_openai_embed.py
"""Evaluate OpenAI embedding models on BelebeleRetrieval (IT/JA/HI) and the
unified Multilingual RAG benchmark (6 configs).

Usage:
  python scripts/eval_openai_embed.py --mode belebele --model text-embedding-3-large
  python scripts/eval_openai_embed.py --mode rag      --model text-embedding-3-large
  python scripts/eval_openai_embed.py --mode all      --model text-embedding-3-small
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from ranx import Qrels, Run, evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: eval_rag imports torch/sentence_transformers/chromadb at module level.
# To avoid that dependency chain, we inline the three pure helper functions we
# need (load_combo_data, compute_metrics, compute_metrics_by_type) directly here.
# They only require json/numpy/ranx/pathlib, all already imported above.
from scripts._openai_embed_lib import EmbedJob, embed_all_async, EMBED_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_openai_embed")

OUTPUT_DIR = REPO_ROOT / "output"
EVAL_DIR = OUTPUT_DIR / "evaluation"
ANALYSIS_OUT = REPO_ROOT.parent / "analysis_output"

LANG_CODE = {"ita_Latn": "italian", "jpn_Jpan": "japanese", "hin_Deva": "hindi"}
RAG_CONFIGS = [
    ("it", "finance"), ("it", "law"),
    ("ja", "finance"), ("ja", "law"),
    ("hi", "finance"), ("hi", "law"),
]


# ── Inlined helpers from eval_rag.py (avoids loading torch/chromadb) ─────────

def load_combo_data(output_dir: Path, lang: str, domain: str) -> dict:
    """Load corpus, queries, qrels, and query_types for one combo."""
    base = output_dir / lang / domain

    corpus: dict[str, str] = {}
    with (base / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus[row["_id"]] = row.get("text", "")

    queries: dict[str, str] = {}
    with (base / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels" / "test.tsv").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, did, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[did] = rel

    query_types: dict[str, str] = {}
    with (base / "full_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            query_types[row["_id"]] = row["query_type"]

    logger.debug(
        "Loaded %s/%s: corpus=%d, queries=%d, qrels=%d",
        lang, domain, len(corpus), len(queries), len(qrels),
    )
    return {"corpus": corpus, "queries": queries, "qrels": qrels, "query_types": query_types}


def compute_metrics(
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_embs: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute NDCG@10, Recall@10, MRR for a set of query/corpus embeddings."""
    if not qrels:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}

    scores = query_embs @ corpus_embs.T
    top_k = min(10, scores.shape[1])
    top_k_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]

    run_dict: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        top = top_k_idx[qi]
        top_sorted = top[np.argsort(scores[qi, top])[::-1]]
        run_dict[str(qid)] = {str(corpus_ids[di]): float(scores[qi, di]) for di in top_sorted}

    qrels_str = {
        str(qid): {str(did): int(s) for did, s in rels.items()}
        for qid, rels in qrels.items()
    }

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


def compute_metrics_by_type(
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_embs: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    query_types: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by query type."""
    all_types = ["factual", "multi_hop", "summarization", "unanswerable"]
    results: dict[str, dict[str, float]] = {}

    for qtype in all_types:
        type_query_ids = [qid for qid in query_ids if query_types.get(qid) == qtype]
        if not type_query_ids:
            results[qtype] = {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
            continue

        type_mask = [i for i, qid in enumerate(query_ids) if qid in set(type_query_ids)]
        type_query_embs = query_embs[type_mask]
        type_qrels = {qid: qrels[qid] for qid in type_query_ids if qid in qrels}

        results[qtype] = compute_metrics(
            corpus_embs, corpus_ids,
            type_query_embs, type_query_ids,
            type_qrels,
        )

    return results


# ── Belebele loader ───────────────────────────────────────────────────────────

def load_belebele(langs: list[str]) -> dict[str, dict]:
    """Load Belebele corpus + queries + qrels per language from HuggingFace MTEB datasets.

    NOTE: The spec assumed a pre-built retrieval split layout (corpus/queries/default) at
    'mteb/BelebeleRetrieval', but that dataset does not exist on the Hub.  The actual MTEB
    BelebeleRetrieval task (mteb>=1.0) builds retrieval on-the-fly from 'mteb/belebele'
    (config = lang code, split = 'test'), which stores the raw MRC rows with fields
    'link', 'flores_passage', and 'question'.  We replicate that logic here:
      - corpus  : unique flores_passage texts, keyed C0..CN  (deduped by link URL)
      - queries : unique question texts, keyed Q0..QM
      - qrels   : each question's link -> the C-id that holds that link's passage
    Function signature is unchanged from the spec.
    """
    out = {}
    for lang in langs:
        # 'mteb/belebele' has per-language configs; the only split is 'test'
        ds = load_dataset("mteb/belebele", lang, split="test")

        # Build corpus: deduplicate passages by link
        link_to_cid: dict[str, str] = {}
        corpus: dict[str, str] = {}
        for row in ds:
            link = row["link"]
            if link not in link_to_cid:
                cid = f"C{len(link_to_cid)}"
                link_to_cid[link] = cid
                corpus[cid] = row["flores_passage"]

        # Build queries: deduplicate questions
        question_to_qid: dict[str, str] = {}
        queries: dict[str, str] = {}
        for row in ds:
            q = row["question"]
            if q not in question_to_qid:
                qid = f"Q{len(question_to_qid)}"
                question_to_qid[q] = qid
                queries[qid] = q

        # Build qrels: question -> passage (via shared link)
        qrels: dict[str, dict[str, int]] = {}
        for row in ds:
            qid = question_to_qid[row["question"]]
            cid = link_to_cid[row["link"]]
            qrels.setdefault(qid, {})[cid] = 1

        out[lang] = {"corpus": corpus, "queries": queries, "qrels": qrels}
        logger.info("Belebele %s: %d corpus, %d queries, %d qrels",
                    lang, len(corpus), len(queries), len(qrels))
    return out


# ── Orchestrator helpers ──────────────────────────────────────────────────────

async def embed_corpus_and_queries(
    corpus: dict[str, str], queries: dict[str, str], model: str,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], int, float]:
    """Embed corpus + queries; return (corp_mat, corp_ids, q_mat, q_ids, tokens, cost)."""
    corp_ids = sorted(corpus.keys())
    corp_texts = [corpus[i] for i in corp_ids]
    q_ids = sorted(queries.keys())
    q_texts = [queries[i] for i in q_ids]

    corp_mat, t_c, c_c = await embed_all_async(
        EmbedJob(texts=corp_texts, model=model)
    )
    q_mat, t_q, c_q = await embed_all_async(
        EmbedJob(texts=q_texts, model=model)
    )
    return corp_mat, corp_ids, q_mat, q_ids, t_c + t_q, c_c + c_q


async def eval_belebele(model: str, sanitized_folder: str) -> dict:
    data = load_belebele(["ita_Latn", "jpn_Jpan", "hin_Deva"])
    out: dict[str, dict] = {}
    total_tokens = 0
    total_cost = 0.0
    for lang_code, d in data.items():
        logger.info("-> Belebele %s with %s", lang_code, model)
        corp_mat, corp_ids, q_mat, q_ids, tokens, cost = await embed_corpus_and_queries(
            d["corpus"], d["queries"], model
        )
        metrics = compute_metrics(corp_mat, corp_ids, q_mat, q_ids, d["qrels"])
        out[LANG_CODE[lang_code]] = {
            "lang_code": lang_code,
            "ndcg_at_10": metrics["ndcg_at_10"],
            "recall_at_10": metrics["recall_at_10"],
            "mrr": metrics["mrr"],
            "tokens_used": tokens,
            "cost_usd": cost,
        }
        total_tokens += tokens
        total_cost += cost
        logger.info("  NDCG@10=%.4f Recall@10=%.4f MRR=%.4f cost=$%.3f",
                    metrics["ndcg_at_10"], metrics["recall_at_10"], metrics["mrr"], cost)
    out_path = EVAL_DIR / "openai_belebele.json"
    payload = {model: out, "total_tokens": total_tokens, "total_cost_usd": total_cost}
    # Merge if file exists (other model already evaluated)
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        prev[model] = out
        prev["total_tokens"] = prev.get("total_tokens", 0) + total_tokens
        prev["total_cost_usd"] = prev.get("total_cost_usd", 0.0) + total_cost
        payload = prev
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Belebele done: $%.3f total, written to %s", total_cost, out_path)
    return out


async def eval_rag_one(
    lang: str, domain: str, model: str, sanitized_folder: str
) -> dict:
    data = load_combo_data(OUTPUT_DIR, lang, domain)
    if not data["queries"]:
        logger.warning("No queries for %s/%s", lang, domain)
        return {}
    t0 = time.time()
    corp_mat, corp_ids, q_mat, q_ids, tokens, cost = await embed_corpus_and_queries(
        data["corpus"], data["queries"], model
    )
    overall = compute_metrics(corp_mat, corp_ids, q_mat, q_ids, data["qrels"])
    by_type = compute_metrics_by_type(
        corp_mat, corp_ids, q_mat, q_ids, data["qrels"], data["query_types"]
    )
    elapsed = time.time() - t0
    payload = {
        "model_id": f"openai/{model}",
        "lang": lang,
        "domain": domain,
        "overall": overall,
        "by_type": by_type,
        "evaluation_time_sec": elapsed,
        "tokens_used": tokens,
        "cost_usd": cost,
        "embed_dim": EMBED_DIM[model],
    }
    folder = EVAL_DIR / sanitized_folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{lang}_{domain}.json").write_text(json.dumps(payload, indent=2))
    logger.info(
        "RAG %s/%s with %s: NDCG@10=%.4f cost=$%.3f",
        lang, domain, model, overall["ndcg_at_10"], cost,
    )
    return payload


async def eval_rag_all(model: str, sanitized_folder: str) -> list[dict]:
    results = []
    for lang, domain in RAG_CONFIGS:
        r = await eval_rag_one(lang, domain, model, sanitized_folder)
        results.append(r)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

# Hardcoded sanitized folder names — DO NOT change these without recomputing
# via the snippet in SPEC_openai_embedding_eval.md.
SANITIZED_FOLDER = {
    "text-embedding-3-large": "text_embedding_3_lar_af1f17",
    "text-embedding-3-small": "text_embedding_3_sma_ade2d3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["belebele", "rag", "all"], required=True,
        help="Which eval to run.",
    )
    parser.add_argument(
        "--model",
        choices=["text-embedding-3-large", "text-embedding-3-small"],
        required=True,
    )
    args = parser.parse_args()

    folder = SANITIZED_FOLDER[args.model]
    if args.mode in ("belebele", "all"):
        asyncio.run(eval_belebele(args.model, folder))
    if args.mode in ("rag", "all"):
        asyncio.run(eval_rag_all(args.model, folder))


if __name__ == "__main__":
    main()
