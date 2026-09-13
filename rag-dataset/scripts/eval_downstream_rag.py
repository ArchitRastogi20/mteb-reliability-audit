"""B2: Downstream RAG answer-quality eval.

For each (model, RAG config), sample N queries, embed corpus + queries with
the model, retrieve top-K docs, generate an answer with gpt-5.4-mini using
those K docs as context, then have gpt-5.4-mini LLM-judge the answer
against the ground-truth keypoints.

Goal: show whether retrieval rank inversions (Table 4 NDCG@10) translate to
answer-quality inversions. If a model with NDCG@10=31.2 produces lower
keypoint coverage than a model with NDCG@10=57.3 on the same queries, we
have evidence that the leaderboard-vs-deployment gap is consequential, not
just academic.

Scope (configurable):
- Models: 4-5 selected from different tiers + the within-family inversion
  (text-embedding-3-small vs -large on hi-fin)
- Configs: 2-3 chosen for variety (different languages, different RAG
  difficulty)
- Queries: 50 per config (random sample from factual + multi_hop pools;
  exclude unanswerable)

Output: analysis_output/stats/downstream_rag.json + .csv with per-(model,
config) keypoint-coverage rate, alongside the retrieval NDCG@10 the eval
already reported.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_openai_embed import load_combo_data
from scripts._openai_embed_lib import EmbedJob, embed_all_async
from scripts.eval_api_embed import (
    embed_texts as api_embed_texts,
    sanitize_model_id,
    PROVIDERS as API_PROVIDERS,
)

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_downstream_rag")

OUT_DIR = PROJECT_ROOT / "analysis_output" / "stats"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
OUTPUT_DIR = REPO_ROOT / "output"

# Default sample
N_QUERIES = 50
TOP_K = 3
JUDGE_MODEL = "gpt-5.4-mini"
ANSWER_MODEL = "gpt-5.4-mini"

# Default model+config matrix — chosen to test the within-API inversion
# and contrast with strong open-weight models.
DEFAULT_MATRIX = [
    # (provider, model_id_or_short, config_lang, config_domain)
    ("openai", "text-embedding-3-large", "hi", "finance"),
    ("openai", "text-embedding-3-small", "hi", "finance"),  # NDCG@10=31.2 (inverted)
    ("openai", "text-embedding-3-large", "it", "finance"),
    ("openai", "text-embedding-3-small", "it", "finance"),
    ("cohere",  "embed-multilingual-v3.0", "hi", "finance"),
    ("gemini",  "gemini-embedding-2", "hi", "finance"),
]


def load_full_queries(lang: str, domain: str) -> list[dict]:
    """Load full_queries.jsonl which has keypoints + reference answers."""
    p = OUTPUT_DIR / lang / domain / "full_queries.jsonl"
    qs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qs.append(json.loads(line))
    # Filter out unanswerable; restrict to factual + multi_hop for fair test
    qs = [q for q in qs if q.get("query_type") in ("factual", "multi_hop")]
    return qs


def load_corpus(lang: str, domain: str) -> dict[str, str]:
    p = OUTPUT_DIR / lang / domain / "corpus.jsonl"
    out: dict[str, str] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["_id"]] = r.get("text", "")
    return out


async def embed_for_provider(provider: str, model: str,
                             texts: list[str], role: str) -> np.ndarray:
    if provider == "openai":
        # Use the openai embed via existing library
        from scripts._openai_embed_lib import EmbedJob, embed_all_async
        arr, _, _ = await embed_all_async(EmbedJob(texts=texts, model=model))
        return arr
    # cohere / gemini / voyage via shared dispatch
    arr, _ = await api_embed_texts(texts, provider, role=role)
    return arr


async def generate_answer(client, query: str, contexts: list[str], lang: str) -> str:
    """Use gpt-5.4-mini to answer the query using the retrieved contexts."""
    sys_prompt = (
        f"You are a {lang} retrieval-augmented question-answering assistant. "
        f"Answer the user's question using ONLY the provided reference passages. "
        f"If the answer is not present in the passages, reply 'Information not "
        f"available in retrieved passages.' Answer in the same language as the question."
    )
    ctx = "\n\n".join(f"[Doc {i+1}]\n{c}" for i, c in enumerate(contexts))
    user_prompt = f"Reference passages:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    resp = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        max_completion_tokens=600,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


async def judge_answer(client, query: str, answer: str,
                       keypoints: list[str], lang: str) -> dict[str, Any]:
    """LLM-judge: count how many keypoints the answer covers (0..N)."""
    if not keypoints:
        return {"covered": 0, "total": 0, "rate": 0.0}
    sys_prompt = (
        "You are a strict factual grader. Given a question, an answer, and a "
        "list of ground-truth atomic facts (keypoints), decide which keypoints "
        "the answer correctly conveys. Output a JSON object with keys "
        "'covered' (list of keypoint indices, 0-indexed) and 'reasoning' "
        "(short string). A keypoint is covered only if its factual content is "
        "stated or directly implied by the answer."
    )
    kp_list = "\n".join(f"{i}. {k}" for i, k in enumerate(keypoints))
    user_prompt = (
        f"Question: {query}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        f"Keypoints:\n{kp_list}\n\n"
        f"Output JSON only."
    )
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        max_completion_tokens=400,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        d = json.loads(raw)
        covered = list(set(int(i) for i in d.get("covered", []) if isinstance(i, (int, str))))
        covered = [c for c in covered if 0 <= c < len(keypoints)]
        return {"covered": len(covered), "total": len(keypoints),
                "rate": len(covered) / len(keypoints)}
    except Exception:
        return {"covered": 0, "total": len(keypoints), "rate": 0.0}


async def eval_one(provider: str, model: str, lang: str, domain: str,
                   n_queries: int, top_k: int, seed: int = 42) -> dict[str, Any]:
    logger.info("[%s/%s] %s/%s/%s — %d queries", provider, model, lang, domain,
                provider, n_queries)
    corpus = load_corpus(lang, domain)
    queries = load_full_queries(lang, domain)
    rng = random.Random(seed)
    sampled = rng.sample(queries, min(n_queries, len(queries)))

    # Embed corpus + sampled queries
    corp_ids = sorted(corpus.keys())
    corp_texts = [corpus[i][:30_000] for i in corp_ids]
    q_ids = [q["_id"] for q in sampled]
    q_texts = [q["question"][:8_000] for q in sampled]

    corp_emb = await embed_for_provider(provider, model, corp_texts, role="document")
    q_emb = await embed_for_provider(provider, model, q_texts, role="query")

    # Top-K
    scores = q_emb @ corp_emb.T
    topk_idx = np.argsort(-scores, axis=1)[:, :top_k]
    retrieved = [[corp_ids[j] for j in row] for row in topk_idx]

    # Now generate answers + judge
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(8)

    async def process(i: int, q: dict, top_doc_ids: list[str]):
        async with sem:
            contexts = [corpus[d][:6_000] for d in top_doc_ids]
            try:
                ans = await generate_answer(client, q["question"], contexts, lang)
            except Exception as e:
                logger.warning("answer-gen failed q=%s: %s", q["_id"], str(e)[:80])
                ans = ""
            kp = q.get("ground_truth", {}).get("keypoints", [])
            try:
                judge = await judge_answer(client, q["question"], ans, kp, lang)
            except Exception as e:
                logger.warning("judge failed q=%s: %s", q["_id"], str(e)[:80])
                judge = {"covered": 0, "total": len(kp), "rate": 0.0}
            return {
                "qid": q["_id"], "qtype": q.get("query_type"),
                "top_docs": top_doc_ids, "answer": ans, **judge,
            }

    coros = [process(i, q, retrieved[i]) for i, q in enumerate(sampled)]
    per_query = await asyncio.gather(*coros)
    await client.close()

    rates = [p["rate"] for p in per_query]
    return {
        "provider": provider, "model": model,
        "lang": lang, "domain": domain,
        "n_queries": len(sampled),
        "top_k": top_k,
        "mean_keypoint_coverage": float(np.mean(rates)),
        "median_keypoint_coverage": float(np.median(rates)),
        "per_query": per_query,
    }


async def main_async(matrix, n_queries: int, top_k: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    detailed: dict[str, Any] = {}
    for (provider, model, lang, domain) in matrix:
        try:
            r = await eval_one(provider, model, lang, domain, n_queries, top_k)
            summary.append({
                "provider": provider, "model": model, "lang": lang, "domain": domain,
                "n": r["n_queries"], "mean_kp": r["mean_keypoint_coverage"],
                "median_kp": r["median_keypoint_coverage"],
            })
            detailed[f"{provider}/{model}/{lang}_{domain}"] = r
            logger.info("DONE %s/%s on %s/%s: mean kp coverage = %.3f",
                        provider, model, lang, domain, r["mean_keypoint_coverage"])
        except Exception as e:
            logger.exception("FAILED %s/%s/%s/%s: %s", provider, model, lang, domain, e)
            summary.append({"provider": provider, "model": model, "lang": lang,
                            "domain": domain, "n": 0, "mean_kp": None,
                            "median_kp": None, "error": str(e)[:200]})
    out_path.write_text(json.dumps({"summary": summary, "detailed": detailed}, indent=2))
    logger.info("wrote %s", out_path)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=N_QUERIES)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--matrix", default="default",
                   help="'default' or 'tiny' (3 entries for smoke)")
    p.add_argument("--out", default=str(OUT_DIR / "downstream_rag.json"))
    args = p.parse_args()

    if args.matrix == "tiny":
        matrix = DEFAULT_MATRIX[:3]
    else:
        matrix = DEFAULT_MATRIX
    print(f"Matrix: {len(matrix)} cells, {args.n_queries} queries each, top-k={args.top_k}")
    summary = asyncio.run(main_async(matrix, args.n_queries, args.top_k, Path(args.out)))
    print("\n=== Summary ===")
    for s in summary:
        kp = s.get("mean_kp")
        kp_s = f"{kp:.3f}" if kp is not None else "FAILED"
        print(f"  {s['provider']:8s} {s['model']:30s} {s['lang']}/{s['domain']:7s} "
              f"n={s['n']} mean_kp={kp_s}")


if __name__ == "__main__":
    main()
