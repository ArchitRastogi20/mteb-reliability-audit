from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.models.config_models import FinanceConfig, FinanceEvent
from src.pipeline.stage1_config import _load_checkpoint, _checkpoint_path, run_stage1


def _make_finance_config(name: str = "Corp") -> FinanceConfig:
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


def test_load_checkpoint_empty(tmp_path: Path):
    result = _load_checkpoint(tmp_path / "nonexistent.jsonl")
    assert result == {}


def test_load_checkpoint_reads_existing(tmp_path: Path):
    cfg = _make_finance_config("CorpA")
    ckpt = tmp_path / "configs_ja_finance.jsonl"
    data = {"_index": 0, **cfg.model_dump()}
    ckpt.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")

    result = _load_checkpoint(ckpt)
    assert 0 in result
    assert result[0]["company_name"] == "CorpA"


def test_checkpoint_path(tmp_path: Path):
    path = _checkpoint_path("ja", "finance", tmp_path)
    assert path.name == "configs_ja_finance.jsonl"
    assert path.parent == tmp_path


@pytest.mark.asyncio
async def test_run_stage1_skips_completed(tmp_path: Path):
    """Items already in checkpoint are not re-generated."""
    cfg = _make_finance_config("CorpA")
    ckpt = tmp_path / "configs_ja_finance.jsonl"
    data = {"_index": 0, **cfg.model_dump()}
    ckpt.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")

    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)

    with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = _make_finance_config("CorpB")
        results = await run_stage1("ja", "finance", count=1, client=client, checkpoint_dir=tmp_path)

    # complete() should not be called since index 0 is already checkpointed
    mock_complete.assert_not_called()
    assert len(results) == 1
    assert results[0].company_name == "CorpA"


@pytest.mark.asyncio
async def test_run_stage1_generates_missing(tmp_path: Path):
    """Missing items are generated and appended to checkpoint."""
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    generated = _make_finance_config("NewCorp")

    with patch.object(client, "complete", new_callable=AsyncMock, return_value=generated):
        results = await run_stage1("ja", "finance", count=2, client=client, checkpoint_dir=tmp_path)

    assert len(results) == 2
    # Checkpoint file should have 2 entries
    ckpt = tmp_path / "configs_ja_finance.jsonl"
    lines = [l for l in ckpt.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_run_stage1_law_domain(tmp_path: Path):
    """Law domain uses LawConfig model."""
    from src.models.config_models import LawConfig, LawParty

    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    generated = LawConfig(
        case_name="State v. Doe",
        court_name="High Court",
        case_date="2023-01-01",
        parties=[LawParty(name="Doe", role="defendant")],
        charges=["Fraud"],
        key_facts=["Fact 1", "Fact 2", "Fact 3"],
        verdict="Guilty",
        sentence="2 years",
    )

    with patch.object(client, "complete", new_callable=AsyncMock, return_value=generated):
        results = await run_stage1("hi", "law", count=1, client=client, checkpoint_dir=tmp_path)

    assert len(results) == 1
    assert results[0].case_name == "State v. Doe"
