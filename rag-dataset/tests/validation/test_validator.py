from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.models.corpus_models import Document
from src.models.query_models import QARPair, ValidationResult
from src.validation.llm_validator import run_validation


def _make_doc(idx: int) -> Document:
    return Document(
        doc_id=f"ja_finance_{idx:04d}",
        language="ja",
        domain="finance",
        content=f"Document {idx} content about finance.",
    )


def _make_pair(idx: int, qtype: str = "factual") -> QARPair:
    return QARPair(
        query_id=f"ja_finance_{qtype}_{idx:04d}",
        doc_id=f"ja_finance_{idx:04d}" if qtype != "unanswerable" else None,
        language="ja",
        domain="finance",
        query_type=qtype,  # type: ignore[arg-type]
        question="Q?",
        answer="A.",
        references=["ref"],
    )


@pytest.mark.asyncio
async def test_answerable_valid(tmp_path):
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "factual")]
    docs = [_make_doc(0)]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=ValidationResult(is_valid=True, reason="Answerable.")
    ):
        results = await run_validation(queries, docs, client)

    assert results[0].valid is True


@pytest.mark.asyncio
async def test_answerable_invalid(tmp_path):
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "factual")]
    docs = [_make_doc(0)]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=ValidationResult(is_valid=False, reason="Not answerable.")
    ):
        results = await run_validation(queries, docs, client)

    assert results[0].valid is False


@pytest.mark.asyncio
async def test_unanswerable_validated(tmp_path):
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "unanswerable")]
    docs = [_make_doc(0)]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=ValidationResult(is_valid=True, reason="Genuinely unanswerable.")
    ):
        results = await run_validation(queries, docs, client)

    assert results[0].valid is True


@pytest.mark.asyncio
async def test_missing_doc_marks_invalid(tmp_path):
    """If doc_id is not in documents list, pair is marked invalid without LLM call."""
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    pair = _make_pair(0, "factual")  # doc_id = ja_finance_0000
    docs: list[Document] = []  # empty — doc not found

    with patch.object(client, "complete", new_callable=AsyncMock) as mock:
        results = await run_validation([pair], docs, client)

    mock.assert_not_called()
    assert results[0].valid is False


@pytest.mark.asyncio
async def test_mixed_query_types(tmp_path):
    """All 4 query types processed, correct valid flags returned."""
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [
        _make_pair(0, "factual"),
        _make_pair(0, "multi_hop"),
        _make_pair(0, "summarization"),
        _make_pair(0, "unanswerable"),
    ]
    docs = [_make_doc(0)]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=ValidationResult(is_valid=True, reason="ok")
    ):
        results = await run_validation(queries, docs, client)

    assert len(results) == 4
    assert all(r.valid for r in results)
