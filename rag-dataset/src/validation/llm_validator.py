from __future__ import annotations

import asyncio
from typing import Literal

from tqdm.asyncio import tqdm

from src.client.async_openai import NANO, AsyncOpenAIClient
from src.models.corpus_models import Document
from src.models.query_models import QARPair, ValidationResult
from src.pipeline.beir_writer import _strip_markdown


def _answerable_messages(pair: QARPair, doc_content: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a quality reviewer for a RAG evaluation dataset. "
                "Given a question, its expected answer, reference passages, and the source document, "
                "decide if the question is genuinely answerable from the document and if the references support the answer. "
                "Respond with is_valid=true only if both conditions hold."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {pair.question}\n"
                f"Expected answer: {pair.answer}\n"
                f"References: {pair.references}\n\n"
                f"Source document:\n{doc_content}\n\n"
                "Is this question answerable from the document, and do the references directly support the answer?"
            ),
        },
    ]


def _unanswerable_messages(pair: QARPair, corpus_sample: list[str]) -> list[dict]:
    sample_text = "\n---\n".join(corpus_sample[:5])
    return [
        {
            "role": "system",
            "content": (
                "You are a quality reviewer for a RAG evaluation dataset. "
                "Given a question and a sample of corpus documents, decide if the question "
                "could be answered from any of the documents. "
                "Respond with is_valid=true if the question is genuinely unanswerable from this corpus."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {pair.question}\n\n"
                f"Corpus sample (first 5 documents):\n{sample_text}\n\n"
                "Is this question unanswerable from the corpus? (is_valid=true means it IS unanswerable — good)"
            ),
        },
    ]


async def _validate_one(
    pair: QARPair,
    doc_index: dict[str, Document],
    corpus_sample: list[str],
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
) -> QARPair:
    if pair.query_type == "unanswerable":
        messages = _unanswerable_messages(pair, corpus_sample)
    else:
        doc = doc_index.get(pair.doc_id or "")
        if doc is None:
            return pair.model_copy(update={"valid": False})
        messages = _answerable_messages(pair, _strip_markdown(doc.content))

    result: ValidationResult = await client.complete(
        messages=messages,
        model=NANO,
        response_format=ValidationResult,
        semaphore=semaphore,
    )
    return pair.model_copy(update={"valid": result.is_valid})


async def run_validation(
    queries: list[QARPair],
    documents: list[Document],
    client: AsyncOpenAIClient,
    semaphore_limit: int = 50,
) -> list[QARPair]:
    """Validate all queries and return updated list with valid flags set."""
    doc_index = {doc.doc_id: doc for doc in documents}
    corpus_sample = [doc.content[:500] for doc in documents[:5]]
    semaphore = asyncio.Semaphore(semaphore_limit)

    tasks = [
        _validate_one(pair, doc_index, corpus_sample, client, semaphore)
        for pair in queries
    ]
    return await tqdm.gather(*tasks, desc="Validation")
