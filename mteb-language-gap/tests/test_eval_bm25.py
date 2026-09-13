from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def cache_dir(tmp_path):
    data = {
        "corpus": {"d1": "日本語テスト", "d2": "英語テスト", "d3": "中国語テスト"},
        "queries": {"q1": "日本語", "q2": "英語"},
        "qrels": {"q1": {"d1": 1}, "q2": {"d2": 1}},
    }
    (tmp_path / "BelebeleRetrieval__jpn_Jpan-jpn_Jpan__test.json").write_text(
        json.dumps(data)
    )
    return tmp_path


def test_tokenize_japanese():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("日本語", "jpn") == ["日", "本", "語"]


def test_tokenize_hindi():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("यह परीक्षण", "hin") == ["यह", "परीक्षण"]


def test_tokenize_italian():
    from scripts.eval_bm25 import _tokenize
    assert _tokenize("Test Italiano", "ita") == ["test", "italiano"]


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
        yaml.dump({"torch_num_threads": 42})
    )
    assert m._cpu_workers() == 42


def test_task_cache_path(tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    monkeypatch.setattr(m, "DATASET_CACHE_DIR", tmp_path)
    path = m._task_cache_path("BelebeleRetrieval", "jpn_Jpan-jpn_Jpan", "test")
    assert path == tmp_path / "BelebeleRetrieval__jpn_Jpan-jpn_Jpan__test.json"


def test_run_bm25_task_returns_ndcg_and_elapsed(cache_dir, monkeypatch):
    import scripts.eval_bm25 as m
    monkeypatch.setattr(m, "DATASET_CACHE_DIR", cache_dir)

    ndcg, elapsed = m.run_bm25_task(
        "BelebeleRetrieval", "jpn", "jpn_Jpan-jpn_Jpan", "test", n_workers=1
    )

    assert 0.0 <= ndcg <= 1.0
    assert elapsed >= 0.0


def test_run_bm25_task_correct_ranking(cache_dir, monkeypatch):
    import scripts.eval_bm25 as m
    monkeypatch.setattr(m, "DATASET_CACHE_DIR", cache_dir)

    ndcg, _ = m.run_bm25_task(
        "BelebeleRetrieval", "jpn", "jpn_Jpan-jpn_Jpan", "test", n_workers=1
    )

    # d1 = "日本語テスト" exactly matches q1 = "日本語" at character level → high NDCG
    assert ndcg > 0.5


def test_evaluate_lang_writes_task_files(cache_dir, tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    from scripts.lb_evaluator import LANG_CONFIGS

    monkeypatch.setattr(m, "DATASET_CACHE_DIR", cache_dir)
    results_dir = tmp_path / "bm25"
    monkeypatch.setattr(m, "RESULTS_DIR", results_dir)

    original_tasks = LANG_CONFIGS["jpn"].tasks
    LANG_CONFIGS["jpn"].tasks = ["BelebeleRetrieval"]
    try:
        results = m.evaluate_lang("jpn", n_workers=1, resume=False)
    finally:
        LANG_CONFIGS["jpn"].tasks = original_tasks

    assert "BelebeleRetrieval" in results
    out = results_dir / "jpn_lb" / "BelebeleRetrieval.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "scores" in data
    assert data["scores"]["test"][0]["ndcg_at_10"] >= 0.0
    assert data["scores"]["test"][0]["main_score"] == data["scores"]["test"][0]["ndcg_at_10"]


def test_evaluate_lang_resume_skips_existing(cache_dir, tmp_path, monkeypatch):
    import scripts.eval_bm25 as m
    from scripts.lb_evaluator import LANG_CONFIGS

    monkeypatch.setattr(m, "DATASET_CACHE_DIR", cache_dir)
    results_dir = tmp_path / "bm25"
    monkeypatch.setattr(m, "RESULTS_DIR", results_dir)

    (results_dir / "jpn_lb").mkdir(parents=True)
    (results_dir / "jpn_lb" / "BelebeleRetrieval.json").write_text(json.dumps({
        "dataset_revision": "N/A",
        "evaluation_time": 1.0,
        "scores": {"test": [{"ndcg_at_10": 0.999, "main_score": 0.999}]},
    }))

    original_tasks = LANG_CONFIGS["jpn"].tasks
    LANG_CONFIGS["jpn"].tasks = ["BelebeleRetrieval"]
    try:
        results = m.evaluate_lang("jpn", n_workers=1, resume=True)
    finally:
        LANG_CONFIGS["jpn"].tasks = original_tasks

    assert results["BelebeleRetrieval"] == pytest.approx(0.999)
