from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from tqdm.asyncio import tqdm

from src.client.async_openai import NANO, AsyncOpenAIClient
from src.models.query_models import AnswerValidation, QARPair

# Thresholds per query type.
# Summarization uses a lower bar because GT references are only 2-3 excerpts
# from a full document — partial coverage is structurally expected even with
# perfect retrieval. This is a dataset characteristic, not a quality failure.
_SCORE_THRESHOLDS: dict[str, float] = {
    "factual": 0.5,
    "multi_hop": 0.5,
    "summarization": 0.3,
    "unanswerable": 0.0,  # not validated
}


def _validation_messages(
    question: str,
    expected_answer: str,
    generated_answer: str,
    keypoints: list[str],
) -> list[dict]:
    kp_text = "\n".join(f"- {kp}" for kp in keypoints) if keypoints else expected_answer
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating a RAG-generated answer against an expected answer. "
                "Check if the generated answer correctly addresses the question and "
                "covers the key factual points. Score 0.0–1.0 where 1.0 = all keypoints covered."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Expected keypoints:\n{kp_text}\n\n"
                f"Generated answer: {generated_answer}\n\n"
                "Does the generated answer cover the expected keypoints? "
                "Provide is_valid (true if score >= 0.5), a score 0.0–1.0, and a brief reason."
            ),
        },
    ]


async def _validate_one(
    pair: QARPair,
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
) -> QARPair:
    result: AnswerValidation = await client.complete(
        messages=_validation_messages(
            pair.question, pair.answer, pair.generated_answer, pair.keypoints
        ),
        model=NANO,
        response_format=AnswerValidation,
        semaphore=semaphore,
    )
    threshold = _SCORE_THRESHOLDS.get(pair.query_type, 0.5)
    return pair.model_copy(update={"answer_valid": result.is_valid and result.score >= threshold})


async def run_answer_validation(
    queries: list[QARPair],
    client: AsyncOpenAIClient,
    semaphore_limit: int = 50,
) -> list[QARPair]:
    """
    Validate generated_answer coverage against expected answer + keypoints.
    Skips unanswerable queries and queries without a generated answer.
    Uses asyncio for API calls + ThreadPoolExecutor via asyncio.to_thread for any CPU work.
    """
    semaphore = asyncio.Semaphore(semaphore_limit)

    to_validate = [
        p for p in queries
        if p.query_type != "unanswerable" and p.valid and p.generated_answer
    ]
    passthrough = [
        p for p in queries
        if p.query_type == "unanswerable" or not p.valid or not p.generated_answer
    ]

    validated: list[QARPair] = []
    if to_validate:
        tasks = [_validate_one(p, client, semaphore) for p in to_validate]
        validated = await tqdm.gather(*tasks, desc="Answer validation")

    # Merge preserving original order
    validated_map = {p.query_id: p for p in validated}
    return [
        validated_map.get(p.query_id, p)
        for p in queries
    ]
