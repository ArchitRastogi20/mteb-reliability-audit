from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.models.config_models import FinanceConfig, FinanceEvent
from src.models.corpus_models import Document, GeneratedArticle
from src.pipeline.stage2_article import (
    _checkpoint_path,
    _load_checkpoint,
    run_stage2,
)


def _finance_config(name: str = "Corp") -> FinanceConfig:
    return FinanceConfig(
        company_name=name,
        industry="Banking",
        founded_year=2000,
        headquarters="Tokyo",
        events=[
            FinanceEvent(date="2023-Q1", description="Merger", financial_impact="+30%"),
            FinanceEvent(date="2023-Q2", description="IPO", financial_impact="500M"),
        ],
        revenue="100B",
        profit_margin="15%",
    )


def test_checkpoint_path(tmp_path: Path):
    path = _checkpoint_path("ja", "finance", tmp_path)
    assert path.name == "articles_ja_finance.jsonl"


def test_load_checkpoint_empty(tmp_path: Path):
    assert _load_checkpoint(tmp_path / "missing.jsonl") == {}


def test_load_checkpoint_reads_document(tmp_path: Path):
    doc = Document(doc_id="ja_finance_0000", language="ja", domain="finance", content="Test content")
    ckpt = tmp_path / "articles_ja_finance.jsonl"
    data = {"_index": 0, **doc.model_dump()}
    ckpt.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")

    result = _load_checkpoint(ckpt)
    assert 0 in result
    assert result[0].doc_id == "ja_finance_0000"
    assert result[0].content == "Test content"


@pytest.mark.asyncio
async def test_run_stage2_skips_completed(tmp_path: Path):
    doc = Document(doc_id="ja_finance_0000", language="ja", domain="finance", content="Existing")
    ckpt = tmp_path / "articles_ja_finance.jsonl"
    data = {"_index": 0, **doc.model_dump()}
    ckpt.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")

    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    configs = [_finance_config()]

    with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
        results = await run_stage2("ja", "finance", configs, client, tmp_path)

    mock_complete.assert_not_called()
    assert results[0].content == "Existing"


@pytest.mark.asyncio
async def test_run_stage2_generates_article(tmp_path: Path):
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    configs = [_finance_config("NewCorp"), _finance_config("OldCorp")]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=GeneratedArticle(content="Generated article text here.")
    ):
        results = await run_stage2("ja", "finance", configs, client, tmp_path)

    assert len(results) == 2
    assert results[0].doc_id == "ja_finance_0000"
    assert results[1].doc_id == "ja_finance_0001"
    assert results[0].language == "ja"
    assert results[0].domain == "finance"
    assert "Generated" in results[0].content

    ckpt = tmp_path / "articles_ja_finance.jsonl"
    lines = [l for l in ckpt.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_run_stage2_doc_id_format(tmp_path: Path):
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=GeneratedArticle(content="Article")
    ):
        results = await run_stage2("hi", "law", [_finance_config()], client, tmp_path)

    assert results[0].doc_id == "hi_law_0000"
    assert results[0].language == "hi"
    assert results[0].domain == "law"
