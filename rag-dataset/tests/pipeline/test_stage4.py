from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.models.corpus_models import Document
from src.models.query_models import QARPair, RefinedReferences
from src.pipeline.stage4_refine import (
    _checkpoint_path,
    _load_checkpoint,
    run_stage4,
)
from src.pipeline.beir_writer import write_beir_dataset


def _make_doc(idx: int) -> Document:
    return Document(
        doc_id=f"ja_finance_{idx:04d}",
        language="ja",
        domain="finance",
        content=f"Document content {idx}",
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
        references=["original ref"],
    )


def test_checkpoint_path(tmp_path: Path):
    assert _checkpoint_path("ja", "finance", tmp_path).name == "queries_ja_finance_refined.jsonl"


def test_load_checkpoint_empty(tmp_path: Path):
    assert _load_checkpoint(tmp_path / "nope.jsonl") == {}


@pytest.mark.asyncio
async def test_unanswerable_passes_through(tmp_path: Path):
    """Unanswerable queries are returned without any LLM call."""
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "unanswerable")]
    docs = [_make_doc(0)]

    with patch.object(client, "complete", new_callable=AsyncMock) as mock:
        results = await run_stage4("ja", "finance", queries, docs, client, tmp_path)

    mock.assert_not_called()
    assert len(results) == 1
    assert results[0].query_type == "unanswerable"


@pytest.mark.asyncio
async def test_run_stage4_refines_references(tmp_path: Path):
    """References are updated from LLM response."""
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "factual")]
    docs = [_make_doc(0)]

    with patch.object(
        client, "complete", new_callable=AsyncMock,
        return_value=RefinedReferences(references=["refined ref"])
    ):
        results = await run_stage4("ja", "finance", queries, docs, client, tmp_path)

    assert results[0].references == ["refined ref"]


@pytest.mark.asyncio
async def test_run_stage4_skips_completed(tmp_path: Path):
    """Already-refined queries are not re-processed."""
    refined_pair = _make_pair(0, "factual")
    refined_pair = refined_pair.model_copy(update={"references": ["already refined"]})
    ckpt = tmp_path / "queries_ja_finance_refined.jsonl"
    ckpt.write_text(refined_pair.model_dump_json() + "\n", encoding="utf-8")

    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    queries = [_make_pair(0, "factual")]
    docs = [_make_doc(0)]

    with patch.object(client, "complete", new_callable=AsyncMock) as mock:
        results = await run_stage4("ja", "finance", queries, docs, client, tmp_path)

    mock.assert_not_called()
    assert results[0].references == ["already refined"]


# --- BEIR writer tests ---

def test_write_beir_corpus(tmp_path: Path):
    docs = [_make_doc(0), _make_doc(1)]
    queries: list[QARPair] = []
    write_beir_dataset("ja", "finance", docs, queries, tmp_path)

    corpus_path = tmp_path / "ja" / "finance" / "corpus.jsonl"
    assert corpus_path.exists()
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["_id"] == "ja_finance_0000"
    assert record["title"] == ""
    assert record["text"] == "Document content 0"


def test_write_beir_queries_only_valid(tmp_path: Path):
    docs = [_make_doc(0)]
    q_valid = _make_pair(0, "factual")
    q_invalid = _make_pair(0, "factual")
    q_invalid = q_invalid.model_copy(update={"query_id": "ja_finance_factual_0099", "valid": False})
    write_beir_dataset("ja", "finance", docs, [q_valid, q_invalid], tmp_path)

    queries_path = tmp_path / "ja" / "finance" / "queries.jsonl"
    lines = queries_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # only valid query written


def test_write_beir_qrels_excludes_unanswerable(tmp_path: Path):
    docs = [_make_doc(0)]
    q_factual = _make_pair(0, "factual")
    q_unans = _make_pair(0, "unanswerable")
    write_beir_dataset("ja", "finance", docs, [q_factual, q_unans], tmp_path)

    qrels_path = tmp_path / "ja" / "finance" / "qrels" / "test.tsv"
    lines = [l for l in qrels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1  # only factual gets a qrel entry
    assert "unanswerable" not in lines[0]
    parts = lines[0].split("\t")
    assert parts[0] == q_factual.query_id
    assert parts[2] == "ja_finance_0000"
    assert parts[3] == "1"
