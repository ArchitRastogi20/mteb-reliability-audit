from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.models.config_models import FinanceConfig, FinanceEvent
from src.models.corpus_models import Document
from src.models.query_models import GeneratedQuery, QARPair, UnanswerableQuery
from src.pipeline.stage3_queries import (
    _checkpoint_path,
    _entity_names,
    _load_checkpoint,
    run_stage3,
)


def _make_doc(idx: int, lang: str = "ja", domain: str = "finance") -> Document:
    return Document(
        doc_id=f"{lang}_{domain}_{idx:04d}",
        language=lang,   # type: ignore[arg-type]
        domain=domain,   # type: ignore[arg-type]
        content=f"Article content for doc {idx}",
    )


def _make_finance_config(name: str = "Corp") -> FinanceConfig:
    return FinanceConfig(
        company_name=name,
        industry="Banking",
        founded_year=2000,
        headquarters="Tokyo",
        events=[
            FinanceEvent(date="2023-Q1", description="M", financial_impact="+10%"),
            FinanceEvent(date="2023-Q2", description="I", financial_impact="100M"),
        ],
        revenue="50B",
        profit_margin="10%",
    )


def test_checkpoint_path(tmp_path: Path):
    assert _checkpoint_path("ja", "finance", tmp_path).name == "queries_ja_finance.jsonl"


def test_load_checkpoint_empty(tmp_path: Path):
    assert _load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_reads_pair(tmp_path: Path):
    pair = QARPair(
        query_id="ja_finance_factual_0000",
        doc_id="ja_finance_0000",
        language="ja",
        domain="finance",
        query_type="factual",
        question="Q?",
        answer="A.",
        references=["ref"],
    )
    ckpt = tmp_path / "queries_ja_finance.jsonl"
    ckpt.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    result = _load_checkpoint(ckpt)
    assert "ja_finance_factual_0000" in result
    assert result["ja_finance_factual_0000"].question == "Q?"


def test_entity_names_finance():
    configs = [_make_finance_config("Alpha"), _make_finance_config("Beta")]
    assert _entity_names(configs) == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_run_stage3_skips_completed(tmp_path: Path):
    """Already-checkpointed query_ids are not re-generated."""
    pair = QARPair(
        query_id="ja_finance_factual_0000",
        doc_id="ja_finance_0000",
        language="ja",
        domain="finance",
        query_type="factual",
        question="Existing Q?",
        answer="Existing A.",
        references=["ref"],
    )
    ckpt = tmp_path / "queries_ja_finance.jsonl"
    ckpt.write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    docs = [_make_doc(i) for i in range(250)]
    configs = [_make_finance_config(f"Corp{i}") for i in range(250)]
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)

    call_count = 0

    async def mock_complete(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        rf = kwargs.get("response_format")
        if rf is GeneratedQuery:
            return GeneratedQuery(question="Q?", answer="A.", references=["r"])
        return UnanswerableQuery(question="Unans Q?")

    with patch.object(client, "complete", side_effect=mock_complete):
        results = await run_stage3("ja", "finance", docs, configs, client, tmp_path)

    # 500 total - 1 already done = 499 new calls
    assert call_count == 499
    assert len(results) == 500
    # The pre-existing one is returned correctly
    existing = next(r for r in results if r.query_id == "ja_finance_factual_0000")
    assert existing.question == "Existing Q?"


@pytest.mark.asyncio
async def test_run_stage3_query_types(tmp_path: Path):
    """Correct query types and doc_ids are assigned."""
    docs = [_make_doc(i) for i in range(250)]
    configs = [_make_finance_config(f"Corp{i}") for i in range(250)]
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)

    async def mock_complete(*args, **kwargs):
        rf = kwargs.get("response_format")
        if rf is GeneratedQuery:
            return GeneratedQuery(question="Q?", answer="A.", references=["r"])
        return UnanswerableQuery(question="Unans Q?")

    with patch.object(client, "complete", side_effect=mock_complete):
        results = await run_stage3("ja", "finance", docs, configs, client, tmp_path)

    factual = [r for r in results if r.query_type == "factual"]
    multihop = [r for r in results if r.query_type == "multi_hop"]
    summ = [r for r in results if r.query_type == "summarization"]
    unans = [r for r in results if r.query_type == "unanswerable"]

    assert len(factual) == 175
    assert len(multihop) == 175
    assert len(summ) == 100
    assert len(unans) == 50
    assert len(results) == 500

    # Unanswerable queries have no doc_id
    assert all(r.doc_id is None for r in unans)
    assert all(r.answer == "" for r in unans)
    assert all(r.references == [] for r in unans)

    # Factual queries reference docs 0–174
    factual_doc_ids = {r.doc_id for r in factual}
    assert all(d.startswith("ja_finance_") for d in factual_doc_ids)
