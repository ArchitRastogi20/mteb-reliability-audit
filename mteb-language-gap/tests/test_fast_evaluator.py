import numpy as np
import pytest


def test_load_task_data_structure():
    """_load_task_data returns corpus, queries, qrels dicts with correct types."""
    from scripts.fast_evaluator import _load_task_data
    td = _load_task_data("JaGovFaqsRetrieval")
    assert isinstance(td["corpus"], dict)
    assert isinstance(td["queries"], dict)
    assert isinstance(td["qrels"], dict)
    assert len(td["corpus"]) > 0
    assert len(td["queries"]) > 0
    sample_text = next(iter(td["corpus"].values()))
    assert isinstance(sample_text, str) and len(sample_text) > 0
    sample_qrel = next(iter(td["qrels"].values()))
    assert isinstance(sample_qrel, dict)


def test_compute_ndcg10_perfect_retrieval():
    """NDCG@10 = 1.0 when the relevant doc is ranked first."""
    from scripts.fast_evaluator import _compute_ndcg10
    corpus_ids = ["d1", "d2", "d3"]
    corpus_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    query_ids = ["q1"]
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels = {"q1": {"d1": 1}}
    _, ndcg = _compute_ndcg10(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test")
    assert abs(ndcg - 1.0) < 1e-4


def test_compute_ndcg10_worst_retrieval():
    """NDCG@10 < 0.5 when relevant doc is ranked last."""
    from scripts.fast_evaluator import _compute_ndcg10
    corpus_ids = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11"]
    n = len(corpus_ids)
    corpus_embs = np.eye(n, dtype=np.float32)
    query_embs = corpus_embs[[-1]]
    query_ids = ["q1"]
    qrels = {"q1": {"d1": 1}}
    _, ndcg = _compute_ndcg10(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test")
    assert ndcg < 0.5


def test_compute_ndcg10_returns_float_in_range():
    from scripts.fast_evaluator import _compute_ndcg10
    rng = np.random.default_rng(0)
    n_corpus, n_queries, dim = 50, 10, 16
    corpus_ids = [f"d{i}" for i in range(n_corpus)]
    query_ids = [f"q{i}" for i in range(n_queries)]
    corpus_embs = rng.random((n_corpus, dim), dtype=np.float32)
    query_embs = rng.random((n_queries, dim), dtype=np.float32)
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(n_queries)}
    _, ndcg = _compute_ndcg10(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test")
    assert isinstance(ndcg, float)
    assert 0.0 <= ndcg <= 1.0


def test_all_results_exist(tmp_path):
    from scripts.fast_evaluator import FastJapaneseEvaluator, TASKS
    evaluator = FastJapaneseEvaluator.__new__(FastJapaneseEvaluator)
    evaluator.results_dir = tmp_path
    import json
    for t in TASKS:
        (tmp_path / f"{t}.json").write_text(json.dumps({"scores": {"test": [{"main_score": 0.5}]}}))
    assert evaluator.all_results_exist()


def test_not_all_results_exist(tmp_path):
    from scripts.fast_evaluator import FastJapaneseEvaluator, TASKS
    evaluator = FastJapaneseEvaluator.__new__(FastJapaneseEvaluator)
    evaluator.results_dir = tmp_path
    import json
    for t in TASKS[:-1]:
        (tmp_path / f"{t}.json").write_text(json.dumps({"scores": {"test": [{"main_score": 0.5}]}}))
    assert not evaluator.all_results_exist()
