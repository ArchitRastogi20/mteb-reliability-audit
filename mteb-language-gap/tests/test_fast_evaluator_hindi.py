import numpy as np
import pytest


def test_load_task_data_hindi_structure():
    """_load_task_data_hindi returns corpus/queries/qrels dicts with correct types."""
    from scripts.fast_evaluator_hindi import _load_task_data_hindi
    td = _load_task_data_hindi("WikipediaRetrievalMultilingual")
    assert isinstance(td["corpus"], dict)
    assert isinstance(td["queries"], dict)
    assert isinstance(td["qrels"], dict)
    assert len(td["corpus"]) > 0
    assert len(td["queries"]) > 0
    sample_text = next(iter(td["corpus"].values()))
    assert isinstance(sample_text, str) and len(sample_text) > 0
    sample_qrel = next(iter(td["qrels"].values()))
    assert isinstance(sample_qrel, dict)


def test_task_config_has_all_tasks():
    from scripts.fast_evaluator_hindi import TASKS, TASK_CONFIG
    for t in TASKS:
        assert t in TASK_CONFIG, f"{t} missing from TASK_CONFIG"
        assert "hf_subset" in TASK_CONFIG[t]
        assert "split" in TASK_CONFIG[t]


def test_miracl_uses_dev_split():
    from scripts.fast_evaluator_hindi import TASK_CONFIG
    assert TASK_CONFIG["MIRACLRetrieval"]["split"] == "dev"


def test_compute_ndcg10_hindi_perfect():
    """NDCG@10 = 1.0 when the relevant doc is ranked first."""
    from scripts.fast_evaluator_hindi import _compute_ndcg10_hindi
    corpus_ids = ["d1", "d2", "d3"]
    corpus_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    query_ids = ["q1"]
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels = {"q1": {"d1": 1}}
    _, ndcg = _compute_ndcg10_hindi(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test")
    assert abs(ndcg - 1.0) < 1e-4


def test_all_results_exist_hindi(tmp_path):
    from scripts.fast_evaluator_hindi import FastHindiEvaluator, TASKS
    evaluator = FastHindiEvaluator.__new__(FastHindiEvaluator)
    evaluator.results_dir = tmp_path
    import json
    for t in TASKS:
        (tmp_path / f"{t}.json").write_text(
            json.dumps({"scores": {"test": [{"main_score": 0.5}]}})
        )
    assert evaluator.all_results_exist()


def test_not_all_results_exist_hindi(tmp_path):
    from scripts.fast_evaluator_hindi import FastHindiEvaluator, TASKS
    evaluator = FastHindiEvaluator.__new__(FastHindiEvaluator)
    evaluator.results_dir = tmp_path
    import json
    for t in TASKS[:-1]:
        (tmp_path / f"{t}.json").write_text(
            json.dumps({"scores": {"test": [{"main_score": 0.5}]}})
        )
    assert not evaluator.all_results_exist()
