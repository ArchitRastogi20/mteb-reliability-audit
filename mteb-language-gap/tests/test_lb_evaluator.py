from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path


def test_all_lang_codes_defined():
    from scripts.lb_evaluator import LANG_CONFIGS
    assert set(LANG_CONFIGS.keys()) == {"jpn", "hin", "ita"}


def test_language_configs_have_required_fields():
    from scripts.lb_evaluator import LANG_CONFIGS, LanguageConfig
    for code, cfg in LANG_CONFIGS.items():
        assert isinstance(cfg, LanguageConfig)
        assert len(cfg.tasks) >= 2
        for task in cfg.tasks:
            assert task in cfg.task_config, f"{code}: {task} missing from task_config"
            assert "hf_subset" in cfg.task_config[task]
            assert "split" in cfg.task_config[task]
            assert task in cfg.task_short, f"{code}: {task} missing from task_short"
        assert cfg.results_base.parts[-1] == f"{code}_lb"
        assert cfg.mteb_csv.exists(), f"{code}: MTEB CSV not found at {cfg.mteb_csv}"


def test_jpn_has_2_tasks():
    from scripts.lb_evaluator import LANG_CONFIGS
    assert len(LANG_CONFIGS["jpn"].tasks) == 2


def test_hin_has_4_tasks():
    from scripts.lb_evaluator import LANG_CONFIGS
    assert len(LANG_CONFIGS["hin"].tasks) == 4


def test_ita_has_2_tasks():
    from scripts.lb_evaluator import LANG_CONFIGS
    assert len(LANG_CONFIGS["ita"].tasks) == 2


def test_compute_ndcg10_perfect_score():
    from scripts.lb_evaluator import _compute_ndcg10
    corpus_ids = ["d1", "d2", "d3"]
    corpus_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    query_ids = ["q1"]
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels = {"q1": {"d1": 1}}
    _, ndcg = _compute_ndcg10(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test_task")
    assert abs(ndcg - 1.0) < 1e-4


def test_compute_ndcg10_zero_score():
    from scripts.lb_evaluator import _compute_ndcg10
    corpus_ids = ["d1", "d2", "d3"]
    corpus_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    query_ids = ["q1"]
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    # Relevant doc is d2, but query is most similar to d1
    qrels = {"q1": {"d2": 1}}
    _, ndcg = _compute_ndcg10(corpus_ids, corpus_embs, query_ids, query_embs, qrels, "test_task")
    assert ndcg < 1.0


def test_all_results_exist_true(tmp_path):
    from scripts.lb_evaluator import LeaderboardEvaluator, LANG_CONFIGS
    cfg = LANG_CONFIGS["jpn"]
    ev = LeaderboardEvaluator.__new__(LeaderboardEvaluator)
    ev.results_dir = tmp_path
    ev.lang_config = cfg
    for task in cfg.tasks:
        (tmp_path / f"{task}.json").write_text(
            json.dumps({"scores": {"test": [{"main_score": 0.75}]}})
        )
    assert ev.all_results_exist()


def test_all_results_exist_false(tmp_path):
    from scripts.lb_evaluator import LeaderboardEvaluator, LANG_CONFIGS
    cfg = LANG_CONFIGS["jpn"]
    ev = LeaderboardEvaluator.__new__(LeaderboardEvaluator)
    ev.results_dir = tmp_path
    ev.lang_config = cfg
    # Only write first task
    (tmp_path / f"{cfg.tasks[0]}.json").write_text(
        json.dumps({"scores": {"test": [{"main_score": 0.75}]}})
    )
    assert not ev.all_results_exist()


def test_model_loaded_with_model_kwargs_not_spread(tmp_path):
    """model_kwargs from YAML must be passed as model_kwargs=dict, not **dict."""
    from unittest.mock import patch, MagicMock, call
    import numpy as np

    captured = {}

    def fake_st(local_path, device, trust_remote_code, model_kwargs=None, **spread_kwargs):
        captured["model_kwargs"] = model_kwargs
        captured["spread_kwargs"] = spread_kwargs
        m = MagicMock()
        m.eval.return_value = None
        dim = 8
        def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True, **kw):
            return np.random.default_rng(0).random((len(texts), dim), dtype=np.float32)
        m.encode.side_effect = fake_encode
        return m

    from scripts.lb_evaluator import LeaderboardEvaluator, LANG_CONFIGS
    import chromadb

    lang_cfg = LANG_CONFIGS["jpn"]
    fake_chroma = str(tmp_path / "chroma")

    # Seed result files so all_results_exist() → False
    results_dir = lang_cfg.results_base / "bge_m3_aabbcc"
    results_dir.mkdir(parents=True, exist_ok=True)

    task_data = {
        "task_name": lang_cfg.tasks[0],
        "corpus": {"d0": "doc0", "d1": "doc1"},
        "queries": {"q0": "query0"},
        "qrels": {"q0": {"d0": 1}},
    }

    with patch("scripts.lb_evaluator.SentenceTransformer", side_effect=fake_st), \
         patch("scripts.lb_evaluator._load_task_data", return_value=task_data), \
         patch("scripts.lb_evaluator.sanitize_model_id", return_value="bge_m3_aabbcc"):
        ev = LeaderboardEvaluator(
            model_id="BAAI/bge-m3",
            local_model_path=str(tmp_path),
            model_config={
                "tier": "medium", "passage_prefix": "", "query_prefix": "",
                "model_kwargs": {"use_safetensors": True},
            },
            lang_config=lang_cfg,
            batch_sizes={"medium": 16},
            chroma_path=fake_chroma,
        )
        ev.run()

    assert captured.get("model_kwargs") == {"use_safetensors": True}, (
        "model_kwargs must be passed as named arg, not spread into **kwargs"
    )
    assert "use_safetensors" not in captured.get("spread_kwargs", {}), (
        "use_safetensors must NOT appear as a top-level SentenceTransformer kwarg"
    )


def test_collection_names_have_lang_prefix():
    from scripts.lb_evaluator import LANG_CONFIGS, _corpus_col, _query_col
    for lang_code, cfg in LANG_CONFIGS.items():
        for task in cfg.tasks:
            short = cfg.task_short[task]
            c_name = _corpus_col("test/model-id", task, cfg)
            q_name = _query_col("test/model-id", task, cfg)
            assert short in c_name
            assert short in q_name
