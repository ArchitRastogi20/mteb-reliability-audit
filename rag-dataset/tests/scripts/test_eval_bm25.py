from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml


def _make_combo_dir(tmp_path: Path, lang: str, domain: str) -> None:
    base = tmp_path / lang / domain
    (base / "qrels").mkdir(parents=True)

    if lang == "ja":
        corpus_texts = ["金融市場の動向", "株式取引の仕組み", "銀行システムの概要"]
        query_texts = ["金融市場", "株式"]
    else:
        corpus_texts = ["वित्त बाजार की जानकारी", "शेयर बाजार का विश्लेषण", "बैंक प्रणाली"]
        query_texts = ["वित्त", "शेयर"]

    corpus = [{"_id": f"d{i+1}", "text": t} for i, t in enumerate(corpus_texts)]
    (base / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in corpus)
    )

    queries = [{"_id": f"q{i+1}", "text": t} for i, t in enumerate(query_texts)]
    (base / "queries.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in queries)
    )

    # q1→d1 and q2→d2 are the relevant pairs
    (base / "qrels" / "test.tsv").write_text("q1\t0\td1\t1\nq2\t0\td2\t1\n")

    full_queries = [
        {"_id": "q1", "query_type": "factual"},
        {"_id": "q2", "query_type": "multi_hop"},
    ]
    (base / "full_queries.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in full_queries)
    )


def test_tokenize_japanese():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("日本語", "ja") == ["日", "本", "語"]


def test_tokenize_hindi():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("यह परीक्षण है", "hi") == ["यह", "परीक्षण", "है"]


def test_tokenize_default_lowercases():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("Hello World", "it") == ["hello", "world"]


def test_cpu_workers_fallback(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert m._cpu_workers() == 6


def test_cpu_workers_from_yaml(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "hardware.yaml").write_text(
        yaml.dump({"torch_num_threads": 20})
    )
    assert m._cpu_workers() == 20


def test_load_combo_data(tmp_path):
    from scripts.eval_bm25 import load_combo_data
    _make_combo_dir(tmp_path, "ja", "finance")
    data = load_combo_data(tmp_path, "ja", "finance")
    assert set(data["corpus"].keys()) == {"d1", "d2", "d3"}
    assert set(data["queries"].keys()) == {"q1", "q2"}
    assert data["qrels"]["q1"]["d1"] == 1
    assert data["query_types"]["q1"] == "factual"
    assert data["query_types"]["q2"] == "multi_hop"


def test_run_bm25_output_schema(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    _make_combo_dir(tmp_path, "ja", "finance")
    eval_dir = tmp_path / "evaluation" / "bm25"
    monkeypatch.setattr(m, "EVAL_DIR", eval_dir)

    result = m.run_bm25("ja", "finance", n_workers=1, output_dir=tmp_path)

    assert result["model_id"] == "BM25"
    assert result["lang"] == "ja"
    assert result["domain"] == "finance"
    assert result["num_docs"] == 3
    assert result["num_queries"] == 2
    assert 0.0 <= result["overall"]["ndcg_at_10"] <= 1.0
    assert 0.0 <= result["overall"]["recall_at_10"] <= 1.0
    assert 0.0 <= result["overall"]["mrr"] <= 1.0
    assert set(result["by_type"].keys()) == {
        "factual", "multi_hop", "summarization", "unanswerable"
    }


def test_run_bm25_writes_json_file(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    _make_combo_dir(tmp_path, "hi", "finance")
    eval_dir = tmp_path / "evaluation" / "bm25"
    monkeypatch.setattr(m, "EVAL_DIR", eval_dir)

    m.run_bm25("hi", "finance", n_workers=1, output_dir=tmp_path)

    out = eval_dir / "hi_finance.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["lang"] == "hi"
    assert "overall" in data
    assert "by_type" in data


def test_run_bm25_correct_ranking(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    _make_combo_dir(tmp_path, "ja", "finance")
    eval_dir = tmp_path / "evaluation" / "bm25"
    monkeypatch.setattr(m, "EVAL_DIR", eval_dir)

    result = m.run_bm25("ja", "finance", n_workers=1, output_dir=tmp_path)

    # d1 contains "金融市場の動向" which shares characters with q1="金融市場",
    # so BM25 should rank it first (MRR=1.0 means relevant doc is ranked first)
    assert result["overall"]["mrr"] >= 0.5  # relevant doc in top 2
