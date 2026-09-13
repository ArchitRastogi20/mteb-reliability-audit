"""R5c — Open-weight pair replication of the 1.71× answer-quality propagation.

Runs the same retrieve→generate→judge pipeline as scripts/eval_downstream_rag.py
on an open-weight model pair (BAAI/bge-m3 vs Snowflake/snowflake-arctic-embed-l-v2.0)
on ja_finance, with two independent LLM judges (gpt-5.4-mini and judge-llm-small).

Goal: address the "single-vendor" concern that the within-OpenAI
1.71× propagation result is OpenAI-specific.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# Must be set before any HF library import so HF Hub resolves the right cache dir.
os.environ.setdefault("HF_HUB_CACHE", "/workspace/model_cache")

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_R5c")

# --- Constants ---
MODEL_PAIR = [
    ("hf", "BAAI/bge-m3", "ja", "finance"),
    ("hf", "Snowflake/snowflake-arctic-embed-l-v2.0", "ja", "finance"),
]
JUDGES = ["gpt-5.4-mini", "judge-llm-small"]
N_QUERIES = 50
TOP_K = 3
ANSWER_MODEL = "gpt-5.4-mini"
SEED = 42

OUT_DIR = REPO_ROOT / "analysis_output" / "stats"
OUT_R5C = OUT_DIR / "rev_R5c_open_weight_propagation.json"
OUT_DOWNSTREAM = OUT_DIR / "downstream_rag.json"
OUTPUT_DIR = REPO_ROOT / "output"

# --- Per-model HF prefixes (from config/models.yaml) ---
HF_MODEL_PREFIXES: dict[str, dict[str, str]] = {
    "BAAI/bge-m3": {"passage": "", "query": ""},
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "passage": "",
        "query": "Represent this sentence for searching relevant passages: ",
    },
}


def embed_hf_model(model_id: str, texts: list[str], role: str = "passage") -> np.ndarray:
    """Encode `texts` with the given HuggingFace SentenceTransformer model.

    `role` is "passage" or "query" — controls which prefix to apply.
    Returns a (N, D) float32 numpy array. Logs encode wall-time + peak VRAM
    (measured from model load through encode, i.e. peak_vram_load_and_encode).
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if model_id not in HF_MODEL_PREFIXES:
        raise ValueError(f"Unknown model {model_id}; add prefixes to HF_MODEL_PREFIXES")

    valid_roles = {"passage", "query"}
    if role not in valid_roles:
        raise ValueError(f"role must be one of {valid_roles!r}, got {role!r}")

    prefix = HF_MODEL_PREFIXES[model_id][role]
    prefixed = [prefix + t for t in texts]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = SentenceTransformer(model_id, device="cuda", trust_remote_code=True)
    model.eval()

    try:
        _model_dtype = next(model.parameters()).dtype
        _autocast_dtype = _model_dtype if _model_dtype in (torch.bfloat16, torch.float16) else torch.float16
        with torch.amp.autocast("cuda", dtype=_autocast_dtype):
            embs = model.encode(
                prefixed, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
        elapsed = time.time() - t0
        peak_mb = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
        logger.info("embed_hf_model[%s, role=%s, n=%d]: t=%.1fs, peak_vram_load_and_encode=%dMB",
                    model_id, role, len(texts), elapsed, peak_mb)
    finally:
        del model
        torch.cuda.empty_cache()
    return embs.astype(np.float32, copy=False)


def load_full_queries(lang: str, domain: str) -> list[dict]:
    """Load full_queries.jsonl which has keypoints + reference answers."""
    p = OUTPUT_DIR / lang / domain / "full_queries.jsonl"
    qs: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qs.append(json.loads(line))
    # Restrict to factual + multi_hop (matches existing eval_downstream_rag.py)
    qs = [q for q in qs if q.get("query_type") in ("factual", "multi_hop")]
    return qs


def load_corpus(lang: str, domain: str) -> dict[str, str]:
    p = OUTPUT_DIR / lang / domain / "corpus.jsonl"
    out: dict[str, str] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            out[doc["_id"]] = doc.get("text", "")
    return out


def sample_queries(qs: list[dict], n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(qs, min(n, len(qs)))


def retrieve_top_k(corpus_embs: np.ndarray, corpus_ids: list[str],
                   query_embs: np.ndarray, top_k: int) -> list[list[str]]:
    """Cosine-similarity top-K retrieval. Embeddings must be L2-normalized."""
    sims = query_embs @ corpus_embs.T  # (Q, C)
    top_k_idx = np.argsort(-sims, axis=1)[:, :top_k]
    return [[corpus_ids[i] for i in row] for row in top_k_idx]


async def generate_answer_openai(client, query: str, contexts: list[str], lang: str) -> str:
    """Generate an answer with gpt-5.4-mini given top-K retrieved contexts."""
    ctx_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    lang_name = {"ja": "Japanese", "hi": "Hindi", "it": "Italian"}.get(lang, lang)
    prompt = (
        f"You are answering a {lang_name} factual question using ONLY the contexts below.\n"
        "If the contexts do not contain the answer, say so honestly.\n\n"
        f"{ctx_block}\n\nQuestion: {query}\n\nAnswer in {lang_name}:"
    )
    resp = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=400,
    )
    return resp.choices[0].message.content.strip()


async def judge_gpt(client, query: str, answer: str, keypoints: list[str]) -> tuple[int, int]:
    """Return (covered, total) keypoint counts as judged by gpt-5.4-mini."""
    kp_block = "\n".join(f"  {i+1}. {k}" for i, k in enumerate(keypoints))
    prompt = (
        "You are a strict judge of retrieval-augmented answer quality.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"GOLD KEYPOINTS (each is a discrete fact a perfect answer should contain):\n{kp_block}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"Count exactly how many of the {len(keypoints)} gold keypoints are explicitly covered "
        f"by the candidate answer (paraphrase counts; partial mention does not). "
        f"Reply with ONLY a single integer 0-{len(keypoints)}, no other text."
    )
    resp = await client.chat.completions.create(
        model=JUDGES[0],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=10,
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", txt)
    covered = min(int(m.group()) if m else 0, len(keypoints))
    return covered, len(keypoints)


def judge_claude(anth_client, query: str, answer: str, keypoints: list[str]) -> tuple[int, int]:
    """Return (covered, total) keypoint counts as judged by judge-llm-small.

    Synchronous on purpose — judge-LLM SDK's async surface is overkill for ~100 calls.
    """
    kp_block = "\n".join(f"  {i+1}. {k}" for i, k in enumerate(keypoints))
    prompt = (
        "You are a strict judge of retrieval-augmented answer quality.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"GOLD KEYPOINTS (each is a discrete fact a perfect answer should contain):\n{kp_block}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"Count exactly how many of the {len(keypoints)} gold keypoints are explicitly covered "
        f"by the candidate answer (paraphrase counts; partial mention does not). "
        f"Reply with ONLY a single integer 0-{len(keypoints)}, no other text."
    )
    msg = anth_client.messages.create(
        model=JUDGES[1],
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = msg.content[0].text.strip()
    m = re.search(r"\d+", txt)
    covered = min(int(m.group()) if m else 0, len(keypoints))
    return covered, len(keypoints)


async def eval_one_model(
    provider: str, model_id: str, lang: str, domain: str,
    n_queries: int, top_k: int,
    openai_client, anth_client,
) -> dict:
    """Run R5c for a single (model, config). Returns a per-query result list + summary."""
    assert provider == "hf", "R5c only supports HuggingFace open-weight models"

    logger.info("=" * 60)
    logger.info("eval_one_model: %s on %s/%s", model_id, lang, domain)

    corpus = load_corpus(lang, domain)
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[i] for i in corpus_ids]

    all_qs = load_full_queries(lang, domain)
    qs = sample_queries(all_qs, n_queries)
    q_ids = [q["_id"] for q in qs]
    q_texts = [q["question"] for q in qs]

    # 1. Encode corpus + queries
    c_embs = embed_hf_model(model_id, corpus_texts, role="passage")
    q_embs = embed_hf_model(model_id, q_texts, role="query")

    # 2. Retrieve top-K
    top_doc_ids = retrieve_top_k(c_embs, corpus_ids, q_embs, top_k)

    # 3. Generate answers + judge
    per_query: list[dict] = []
    for i, q in enumerate(qs):
        contexts = [corpus[d] for d in top_doc_ids[i]]
        try:
            answer = await generate_answer_openai(openai_client, q["question"], contexts, lang)
        except Exception as e:
            logger.warning("answer-gen failed on %s: %s", q["_id"], e)
            answer = ""
        keypoints = q.get("ground_truth", {}).get("keypoints", []) or []
        if not keypoints:
            logger.warning("no keypoints for %s; skipping judge", q["_id"])
            continue
        try:
            gpt_cov, gpt_tot = await judge_gpt(openai_client, q["question"], answer, keypoints)
        except Exception as e:
            logger.warning("gpt judge failed on %s: %s", q["_id"], e)
            gpt_cov, gpt_tot = 0, len(keypoints)
        try:
            claude_cov, claude_tot = judge_claude(anth_client, q["question"], answer, keypoints)
        except Exception as e:
            logger.warning("LLM judge failed on %s: %s", q["_id"], e)
            claude_cov, claude_tot = 0, len(keypoints)

        per_query.append({
            "qid": q["_id"],
            "qtype": q["query_type"],
            "top_docs": top_doc_ids[i],
            "answer": answer,
            "gpt_covered": gpt_cov, "gpt_total": gpt_tot, "gpt_rate": gpt_cov / gpt_tot if gpt_tot else 0,
            "judge_covered": claude_cov, "judge_total": claude_tot,
            "judge_rate": claude_cov / claude_tot if claude_tot else 0,
        })
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d done", i + 1, len(qs))

    n_judged = len(per_query)
    gpt_mean = float(np.mean([p["gpt_rate"] for p in per_query])) if n_judged else 0.0
    claude_mean = float(np.mean([p["judge_rate"] for p in per_query])) if n_judged else 0.0
    return {
        "provider": provider, "model": model_id,
        "lang": lang, "domain": domain,
        "n_queries": n_judged, "top_k": top_k,
        "gpt_mean_keypoint_coverage": gpt_mean,
        "claude_mean_keypoint_coverage": claude_mean,
        "per_query": per_query,
    }


async def main_async() -> dict:
    """Run R5c on the bge-m3 vs snowflake pair, dual-judge."""
    from openai import AsyncOpenAI
    from judge_llm_sdk import JudgeClient

    openai_client = AsyncOpenAI(api_key=os.environ["CHATGPT_API_KEY"])
    anth_client = JudgeClient(api_key=os.environ["JUDGE_API_KEY"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed: dict[str, dict] = {}
    summary: list[dict] = []
    for provider, model_id, lang, domain in MODEL_PAIR:
        result = await eval_one_model(
            provider, model_id, lang, domain,
            n_queries=N_QUERIES, top_k=TOP_K,
            openai_client=openai_client, anth_client=anth_client,
        )
        key = f"{provider}/{model_id}/{lang}_{domain}"
        detailed[key] = result
        summary.append({k: v for k, v in result.items() if k != "per_query"})

    # bge-m3 / snowflake ratio under both judges
    bge = next(s for s in summary if "bge-m3" in s["model"])
    snw = next(s for s in summary if "snowflake" in s["model"])

    def ratio(num: float, den: float) -> float | None:
        return None if den == 0 else round(num / den, 3)

    headline = {
        "pair": "bge-m3 / snowflake-arctic-embed-l-v2.0",
        "config": "ja_finance",
        "n_queries_per_model": min(s["n_queries"] for s in summary),
        "gpt_judge_ratio": ratio(bge["gpt_mean_keypoint_coverage"], snw["gpt_mean_keypoint_coverage"]),
        "claude_judge_ratio": ratio(bge["claude_mean_keypoint_coverage"], snw["claude_mean_keypoint_coverage"]),
        "claim_holds_under_both_judges": (
            bge["gpt_mean_keypoint_coverage"] > snw["gpt_mean_keypoint_coverage"]
            and bge["claude_mean_keypoint_coverage"] > snw["claude_mean_keypoint_coverage"]
        ),
    }

    out = {
        "judge_models": JUDGES,
        "answer_model": ANSWER_MODEL,
        "seed": SEED,
        "headline": headline,
        "per_model_summary": summary,
        "detailed": detailed,
    }
    OUT_R5C.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", OUT_R5C)

    # Append to existing downstream_rag.json (preserve prior schema)
    if OUT_DOWNSTREAM.exists():
        existing = json.loads(OUT_DOWNSTREAM.read_text())
    else:
        existing = {"summary": [], "detailed": {}}
    for s in summary:
        existing["summary"].append({
            "provider": "open-weight",
            "model": s["model"],
            "lang": s["lang"],
            "domain": s["domain"],
            "n": s["n_queries"],
            "mean_kp": s["gpt_mean_keypoint_coverage"],
            "median_kp": float(np.median([p["gpt_rate"] for p in detailed[f"hf/{s['model']}/{s['lang']}_{s['domain']}"]["per_query"]] or [0.0])),
        })
    for k, v in detailed.items():
        existing["detailed"][k] = {
            "provider": "open-weight",
            "model": v["model"], "lang": v["lang"], "domain": v["domain"],
            "n_queries": v["n_queries"], "top_k": v["top_k"],
            "mean_keypoint_coverage": v["gpt_mean_keypoint_coverage"],
            "median_keypoint_coverage": float(np.median([p["gpt_rate"] for p in v["per_query"]] or [0.0])),
            "per_query": v["per_query"],
        }
    OUT_DOWNSTREAM.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.info("Appended to %s", OUT_DOWNSTREAM)

    print("\n=== R5c HEADLINE ===")
    print(json.dumps(headline, indent=2))
    return out


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
