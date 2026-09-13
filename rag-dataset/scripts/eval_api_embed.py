"""Multi-provider API embedding eval for Belebele + RAG.

Supports: gemini (text-embedding-004 / gemini-embedding-001),
          cohere (embed-multilingual-v3.0),
          voyage (voyage-3-large or voyage-multilingual-2).

Reuses the orchestrator/loader plumbing from eval_openai_embed.py.

Usage:
  python scripts/eval_api_embed.py --provider gemini --mode all
  python scripts/eval_api_embed.py --provider cohere --mode rag
  python scripts/eval_api_embed.py --provider voyage --mode belebele
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse data loaders + metric helpers from the openai eval script
from scripts.eval_openai_embed import (
    LANG_CODE,
    RAG_CONFIGS,
    load_belebele,
    load_combo_data,
    compute_metrics,
    compute_metrics_by_type,
)

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_api_embed")

OUTPUT_DIR = REPO_ROOT / "output"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# Provider config
PROVIDERS = {
    "gemini": {
        "model_id": "gemini-embedding-2",  # newer, also free, less restrictive quota
        "dim": 3072,
        "price_per_1m": 0.0,    # free tier
        "batch_size": 100,
    },
    "cohere": {
        "model_id": "embed-multilingual-v3.0",
        "dim": 1024,
        "price_per_1m": 0.10,
        "batch_size": 96,        # Cohere caps at 96 inputs / call
    },
    "voyage": {
        "model_id": "voyage-3-large",
        "dim": 1024,
        "price_per_1m": 0.18,
        "batch_size": 128,
    },
}
DEFAULT_SEMAPHORE = 1           # all three rate-limit aggressively on free tier


def sanitize_model_id(model_id: str) -> str:
    name = model_id.split("/")[-1]
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name)[:20]
    h = hashlib.md5(model_id.encode()).hexdigest()[:6]
    return f"{safe}_{h}"


def truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ── Provider-specific async embed ───────────────────────────────────────────

async def embed_gemini(texts: list[str], model: str) -> tuple[np.ndarray, int]:
    """Embed via Gemini REST batchEmbedContents endpoint (SDK doesn't batch
    for the gemini-embedding-2 family — single-input only — so we use REST)."""
    import aiohttp

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment / .env")

    BATCH = 100
    all_vecs: list[list[float]] = []
    total_chars = 0
    sem = asyncio.Semaphore(DEFAULT_SEMAPHORE)

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:batchEmbedContents?key={api_key}")

    async def run_batch(session: "aiohttp.ClientSession", batch: list[str]):
        payload = {"requests": [
            {"model": f"models/{model}",
             "content": {"parts": [{"text": t}]},
             "taskType": "SEMANTIC_SIMILARITY"}
            for t in batch
        ]}
        async with sem:
            for attempt in range(8):
                try:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as r:
                        if r.status == 429:
                            t = await r.text()
                            raise RuntimeError(f"gemini 429: {t[:120]}")
                        if r.status >= 400:
                            t = await r.text()
                            raise RuntimeError(f"gemini {r.status}: {t[:200]}")
                        d = await r.json()
                        return [e["values"] for e in d["embeddings"]]
                except Exception as e:
                    if attempt == 7:
                        raise
                    delay = min(2 ** attempt, 60)
                    logger.warning("gemini retry %d after %s (sleeping %ds)",
                                   attempt + 1, str(e)[:80], delay)
                    await asyncio.sleep(delay)

    async with aiohttp.ClientSession() as session:
        batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
        for b in batches:
            total_chars += sum(len(t) for t in b)
        # Gemini free tier rate-limits hard on parallel requests; run
        # sequentially with a generous inter-batch sleep.
        for b in batches:
            vecs = await run_batch(session, b)
            all_vecs.extend(vecs)
            await asyncio.sleep(4.0)

    arr = np.asarray(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr, total_chars // 4


async def embed_cohere(texts: list[str], model: str, input_type: str = "search_document") -> tuple[np.ndarray, int]:
    """Embed via cohere SDK. Cohere differentiates query vs document."""
    import cohere

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set in environment / .env")
    client = cohere.Client(api_key)

    BATCH = 96  # Cohere hard cap
    all_vecs: list[list[float]] = []
    total_chars = 0

    def _embed_batch(batch: list[str]):
        resp = client.embed(
            texts=batch,
            model=model,
            input_type=input_type,
            embedding_types=["float"],
        )
        return resp.embeddings.float_

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(DEFAULT_SEMAPHORE)

    async def run_batch(batch):
        async with sem:
            for attempt in range(6):
                try:
                    return await loop.run_in_executor(None, _embed_batch, batch)
                except Exception as e:
                    if attempt == 5:
                        raise
                    delay = min(2 ** attempt, 30)
                    logger.warning("cohere retry %d after %s (sleeping %ds)", attempt + 1, e, delay)
                    await asyncio.sleep(delay)
            return []

    batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    for b in batches:
        total_chars += sum(len(t) for t in b)
    coros = [run_batch(b) for b in batches]
    batched = await asyncio.gather(*coros)
    for b in batched:
        all_vecs.extend(b)
    arr = np.asarray(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr, total_chars // 4


async def embed_voyage(texts: list[str], model: str, input_type: str = "document") -> tuple[np.ndarray, int]:
    """Embed via voyage REST API."""
    import aiohttp

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY not set — add it to .env then re-run")

    # Free tier: 3 RPM, 10K TPM. Use small batches and 22s pacing.
    BATCH = 8
    all_vecs: list[list[float]] = []
    total_chars = 0
    sem = asyncio.Semaphore(1)

    async def run_batch(session: "aiohttp.ClientSession", batch: list[str]):
        url = "https://api.voyageai.com/v1/embeddings"
        payload = {"input": batch, "model": model, "input_type": input_type}
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        async with sem:
            for attempt in range(8):
                try:
                    async with session.post(url, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status == 429:
                            t = await r.text()
                            raise RuntimeError(f"voyage 429: {t[:120]}")
                        if r.status >= 400:
                            t = await r.text()
                            raise RuntimeError(f"voyage {r.status}: {t[:200]}")
                        d = await r.json()
                        return [it["embedding"] for it in d["data"]]
                except Exception as e:
                    if attempt == 7:
                        raise
                    delay = min(25 + 5 * attempt, 90)
                    logger.warning("voyage retry %d after %s (sleeping %ds)", attempt + 1, str(e)[:80], delay)
                    await asyncio.sleep(delay)

    async with aiohttp.ClientSession() as session:
        batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
        for b in batches:
            total_chars += sum(len(t) for t in b)
        # Sequential with 22s sleep — 3 RPM free tier
        for i, b in enumerate(batches):
            vecs = await run_batch(session, b)
            all_vecs.extend(vecs)
            if i + 1 < len(batches):
                await asyncio.sleep(22.0)
    arr = np.asarray(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr, total_chars // 4


# ── Generic dispatch ────────────────────────────────────────────────────────

async def embed_texts(texts: list[str], provider: str, role: str = "document") -> tuple[np.ndarray, int]:
    """role ∈ {"document","query"} — only Cohere/Voyage care."""
    cfg = PROVIDERS[provider]
    model = cfg["model_id"]
    if provider == "gemini":
        return await embed_gemini(texts, model)
    if provider == "cohere":
        ct = "search_document" if role == "document" else "search_query"
        return await embed_cohere(texts, model, input_type=ct)
    if provider == "voyage":
        return await embed_voyage(texts, model, input_type=role)
    raise ValueError(provider)


async def embed_corpus_and_queries(
    corpus: dict[str, str], queries: dict[str, str], provider: str,
):
    cfg = PROVIDERS[provider]
    # truncate to a generous char limit (most providers accept ≥8k tokens)
    MAX_CHARS = 30_000
    corp_ids = sorted(corpus.keys())
    corp_texts = [truncate_chars(corpus[i], MAX_CHARS) for i in corp_ids]
    q_ids = sorted(queries.keys())
    q_texts = [truncate_chars(queries[i], MAX_CHARS) for i in q_ids]

    cm, ct = await embed_texts(corp_texts, provider, role="document")
    qm, qt = await embed_texts(q_texts, provider, role="query")
    return cm, corp_ids, qm, q_ids, ct + qt, (ct + qt) * cfg["price_per_1m"] / 1_000_000


# ── Orchestrators ───────────────────────────────────────────────────────────

async def eval_belebele(provider: str, sanitized_folder: str) -> dict:
    cfg = PROVIDERS[provider]
    data = load_belebele(["ita_Latn", "jpn_Jpan", "hin_Deva"])
    out: dict[str, dict] = {}
    total_tokens = 0
    total_cost = 0.0
    for lang_code, d in data.items():
        logger.info("→ Belebele %s with %s/%s", lang_code, provider, cfg["model_id"])
        cm, c_ids, qm, q_ids, tokens, cost = await embed_corpus_and_queries(
            d["corpus"], d["queries"], provider
        )
        m = compute_metrics(cm, c_ids, qm, q_ids, d["qrels"])
        out[LANG_CODE[lang_code]] = {
            "lang_code": lang_code,
            "ndcg_at_10": m["ndcg_at_10"],
            "recall_at_10": m["recall_at_10"],
            "mrr": m["mrr"],
            "tokens_used": tokens,
            "cost_usd": cost,
        }
        total_tokens += tokens
        total_cost += cost
        logger.info("  NDCG@10=%.4f Recall@10=%.4f MRR=%.4f cost=$%.3f",
                    m["ndcg_at_10"], m["recall_at_10"], m["mrr"], cost)

    out_path = EVAL_DIR / "openai_belebele.json"
    payload: dict[str, Any] = {}
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    payload[cfg["model_id"]] = out
    payload["total_tokens"] = payload.get("total_tokens", 0) + total_tokens
    payload["total_cost_usd"] = payload.get("total_cost_usd", 0.0) + total_cost
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Belebele done for %s: $%.3f total -> %s", provider, total_cost, out_path)
    return out


async def eval_rag_one(lang: str, domain: str, provider: str, sanitized_folder: str) -> dict:
    cfg = PROVIDERS[provider]
    data = load_combo_data(OUTPUT_DIR, lang, domain)
    if not data["queries"]:
        logger.warning("no queries for %s/%s", lang, domain)
        return {}
    t0 = time.time()
    cm, c_ids, qm, q_ids, tokens, cost = await embed_corpus_and_queries(
        data["corpus"], data["queries"], provider
    )
    overall = compute_metrics(cm, c_ids, qm, q_ids, data["qrels"])
    by_type = compute_metrics_by_type(cm, c_ids, qm, q_ids, data["qrels"], data["query_types"])
    elapsed = time.time() - t0
    payload = {
        "model_id": f"{provider}/{cfg['model_id']}",
        "lang": lang, "domain": domain,
        "overall": overall, "by_type": by_type,
        "evaluation_time_sec": elapsed, "tokens_used": tokens,
        "cost_usd": cost, "embed_dim": cfg["dim"],
    }
    folder = EVAL_DIR / sanitized_folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{lang}_{domain}.json").write_text(json.dumps(payload, indent=2))
    logger.info("RAG %s/%s with %s: NDCG@10=%.4f cost=$%.3f",
                lang, domain, provider, overall["ndcg_at_10"], cost)
    return payload


async def eval_rag_all(provider: str, sanitized_folder: str) -> list[dict]:
    out = []
    for lang, domain in RAG_CONFIGS:
        r = await eval_rag_one(lang, domain, provider, sanitized_folder)
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), required=True)
    p.add_argument("--mode", choices=["belebele", "rag", "all"], required=True)
    args = p.parse_args()

    cfg = PROVIDERS[args.provider]
    folder = sanitize_model_id(f"{args.provider}/{cfg['model_id']}")
    logger.info("[start] provider=%s model=%s folder=%s",
                args.provider, cfg["model_id"], folder)
    if args.mode in ("belebele", "all"):
        asyncio.run(eval_belebele(args.provider, folder))
    if args.mode in ("rag", "all"):
        asyncio.run(eval_rag_all(args.provider, folder))


if __name__ == "__main__":
    main()
