# rag-dataset/scripts/_openai_embed_lib.py
"""Shared OpenAI embedding helpers: token-aware truncation, async batched embed, cost."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBED_DIM = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}
PRICE_PER_1M = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}
MAX_INPUT_TOKENS = 8192
MAX_BATCH_SIZE = 256          # OpenAI accepts up to 2048 inputs but 256 is well under rate limits
MAX_BATCH_TOKENS = 250_000    # OpenAI hard limit is 300K tokens per request; use 250K for headroom
DEFAULT_SEMAPHORE = 20

# tiktoken encodings are cached per process
_ENC_CACHE: dict[str, tiktoken.Encoding] = {}


def _enc(model: str) -> tiktoken.Encoding:
    if model not in _ENC_CACHE:
        try:
            _ENC_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENC_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENC_CACHE[model]


def truncate_to_tokens(text: str, max_tokens: int, model: str) -> str:
    """Truncate text so it fits within max_tokens for the given model."""
    enc = _enc(model)
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def estimate_cost_usd(num_tokens: int, model: str) -> float:
    return num_tokens * PRICE_PER_1M[model] / 1_000_000


def count_tokens(text: str, model: str) -> int:
    return len(_enc(model).encode(text))


@dataclass
class EmbedJob:
    texts: list[str]
    model: str
    semaphore: int = DEFAULT_SEMAPHORE
    budget_tokens: int = 40_000_000  # ~$5 hard cap on -large


async def embed_batch_async(
    client: AsyncOpenAI,
    batch: list[str],
    model: str,
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> list[list[float]]:
    """Embed one batch with exponential backoff on rate limit / transient errors."""
    delay = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.embeddings.create(input=batch, model=model)
                return [d.embedding for d in resp.data]
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning("embed_batch retry %d after error: %s", attempt + 1, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        # Unreachable; satisfies type checker
        return []


async def embed_all_async(job: EmbedJob) -> tuple[np.ndarray, int, float]:
    """Embed all texts in `job` and return (matrix, total_tokens, total_cost_usd)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    sem = asyncio.Semaphore(job.semaphore)
    async with AsyncOpenAI(api_key=api_key) as client:
        # Pre-truncate
        safe_texts = [truncate_to_tokens(t, MAX_INPUT_TOKENS, job.model) for t in job.texts]
        token_counts = [count_tokens(t, job.model) for t in safe_texts]
        total_tokens = sum(token_counts)
        if total_tokens > job.budget_tokens:
            raise RuntimeError(
                f"Token budget exceeded: needs {total_tokens}, budget {job.budget_tokens}"
            )

        # Build token-aware batches: each batch stays under MAX_BATCH_SIZE items
        # AND under MAX_BATCH_TOKENS tokens (OpenAI hard limit 300K; we use 250K).
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0
        for text, tok_count in zip(safe_texts, token_counts):
            if current_batch and (
                len(current_batch) >= MAX_BATCH_SIZE
                or current_tokens + tok_count > MAX_BATCH_TOKENS
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(text)
            current_tokens += tok_count
        if current_batch:
            batches.append(current_batch)
        t0 = time.time()
        coros = [embed_batch_async(client, b, job.model, sem) for b in batches]
        batched = await asyncio.gather(*coros)
        elapsed = time.time() - t0
        flat = [vec for b in batched for vec in b]
        matrix = np.asarray(flat, dtype=np.float32)
        # OpenAI returns L2-normalised vectors already, but renormalise defensively
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
    cost = estimate_cost_usd(total_tokens, job.model)
    logger.info(
        "embedded %d items, %d tokens, %.2fs, ~$%.3f", len(safe_texts), total_tokens, elapsed, cost
    )
    return matrix, total_tokens, cost
