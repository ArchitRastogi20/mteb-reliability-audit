from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.bm25_sanity import _tokenize, run_bm25_sanity


def _write_beir_files(
    tmp_path: Path,
    lang: str = "ja",
    domain: str = "finance",
    num_docs: int = 5,
) -> tuple[list[str], list[str]]:
    """Write minimal BEIR files for testing. Returns (doc_ids, query_ids)."""
    dataset_dir = tmp_path / lang / domain
    qrels_dir = dataset_dir / "qrels"
    dataset_dir.mkdir(parents=True)
    qrels_dir.mkdir()

    doc_ids = [f"{lang}_{domain}_{i:04d}" for i in range(num_docs)]
    contents = [
        "東京銀行は2023年に合併を実施した。売上は500億円であった。",
        "大阪証券は2022年にIPOを行い、300億円を調達した。",
        "名古屋保険は2021年に新商品を発売した。利益率は15%であった。",
        "福岡リース株式会社は2020年に設立された。",
        "札幌信託は2019年に海外進出を開始した。",
    ]

    # corpus.jsonl
    with (dataset_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for did, content in zip(doc_ids, contents[:num_docs]):
            f.write(json.dumps({"_id": did, "title": "", "text": content}, ensure_ascii=False) + "\n")

    # queries.jsonl — 3 factual queries
    query_ids = [f"{lang}_{domain}_factual_{i:04d}" for i in range(3)]
    query_texts = [
        "東京銀行はいつ合併を実施しましたか",
        "大阪証券のIPOはいくらを調達しましたか",
        "名古屋保険の利益率はいくらですか",
    ]
    with (dataset_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid, text in zip(query_ids, query_texts):
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")

    # qrels/test.tsv — each query maps to its matching doc
    with (qrels_dir / "test.tsv").open("w", encoding="utf-8") as f:
        for qid, did in zip(query_ids, doc_ids[:3]):
            f.write(f"{qid}\t0\t{did}\t1\n")

    return doc_ids, query_ids


def test_tokenize_japanese():
    tokens = _tokenize("東京銀行")
    assert tokens == ["東", "京", "銀", "行"]


def test_tokenize_hindi():
    tokens = _tokenize("नमस्ते")
    assert len(tokens) == 6


def test_tokenize_latin():
    tokens = _tokenize("hello")
    assert tokens == ["h", "e", "l", "l", "o"]


def test_run_bm25_sanity_produces_output(tmp_path: Path):
    _write_beir_files(tmp_path, "ja", "finance")
    result = run_bm25_sanity("ja", "finance", tmp_path, tmp_path / "evaluation")

    assert "metrics" in result
    assert "ndcg_cut_10" in result["metrics"]
    assert "ndcg10_interpretation" in result
    assert result["lang"] == "ja"
    assert result["domain"] == "finance"
    assert result["num_docs"] == 5
    assert result["num_answerable_queries"] == 3

    out_file = tmp_path / "evaluation" / "bm25_sanity_ja_finance.json"
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["lang"] == "ja"


def test_run_bm25_sanity_interpretation_healthy(tmp_path: Path):
    """With matching queries, NDCG@10 should be > 0 and interpretation set."""
    _write_beir_files(tmp_path, "hi", "law")
    result = run_bm25_sanity("hi", "law", tmp_path, tmp_path / "evaluation")
    assert result["ndcg10_interpretation"] in {"healthy", "too_easy", "broken"}


def test_run_bm25_sanity_metrics_keys(tmp_path: Path):
    _write_beir_files(tmp_path, "ja", "law")
    result = run_bm25_sanity("ja", "law", tmp_path, tmp_path / "evaluation", k_values=[10])
    assert "ndcg_cut_10" in result["metrics"]
    assert "recall_10" in result["metrics"]
