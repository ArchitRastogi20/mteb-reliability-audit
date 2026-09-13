from __future__ import annotations
import json
from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_corpus(tmp_path: Path, lang: str, domain: str, n: int = 5) -> Path:
    base = tmp_path / lang / domain
    base.mkdir(parents=True, exist_ok=True)
    with (base / "corpus.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"_id": f"{lang}_{domain}_{i:04d}", "title": "", "text": f"doc {i}"}) + "\n")
    return base


def _make_queries(base: Path, lang: str, domain: str, n: int = 4) -> None:
    with (base / "queries.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"_id": f"{lang}_{domain}_q{i:04d}", "text": f"query {i}"}) + "\n")


def _make_qrels(base: Path, lang: str, domain: str, n: int = 4) -> None:
    qrels_dir = base / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    with (qrels_dir / "test.tsv").open("w") as f:
        for i in range(n):
            f.write(f"{lang}_{domain}_q{i:04d}\t0\t{lang}_{domain}_{i:04d}\t1\n")


def _make_full_queries(base: Path, lang: str, domain: str, n: int = 4) -> None:
    types = ["factual", "factual", "multi_hop", "unanswerable"]
    with (base / "full_queries.jsonl").open("w") as f:
        for i in range(n):
            row = {
                "_id": f"{lang}_{domain}_q{i:04d}",
                "query_type": types[i % len(types)],
                "language": lang,
                "domain": domain,
                "question": f"query {i}",
                "ground_truth": {"doc_id": None, "answer": "", "references": [], "keypoints": []},
                "prediction": {"content": "", "answer_valid": True},
            }
            f.write(json.dumps(row) + "\n")


def _make_combo(tmp_path: Path, lang: str = "ja", domain: str = "finance", n: int = 5) -> Path:
    base = _make_corpus(tmp_path, lang, domain, n)
    _make_queries(base, lang, domain, min(n, 4))
    _make_qrels(base, lang, domain, min(n - 1, 3))
    _make_full_queries(base, lang, domain, min(n, 4))
    return base


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_combo_data_keys(tmp_path):
    from scripts.eval_rag import load_combo_data
    _make_combo(tmp_path)
    data = load_combo_data(tmp_path, "ja", "finance")
    assert set(data.keys()) == {"corpus", "queries", "qrels", "query_types"}


def test_load_combo_data_corpus(tmp_path):
    from scripts.eval_rag import load_combo_data
    _make_combo(tmp_path, n=5)
    data = load_combo_data(tmp_path, "ja", "finance")
    assert len(data["corpus"]) == 5
    assert all(v for v in data["corpus"].values())


def test_load_combo_data_query_types(tmp_path):
    from scripts.eval_rag import load_combo_data
    _make_combo(tmp_path, n=4)
    data = load_combo_data(tmp_path, "ja", "finance")
    assert all(qt in {"factual", "multi_hop", "summarization", "unanswerable"}
               for qt in data["query_types"].values())


def test_load_combo_data_qrels_tsv(tmp_path):
    from scripts.eval_rag import load_combo_data
    _make_combo(tmp_path, n=5)
    data = load_combo_data(tmp_path, "ja", "finance")
    for qid, rels in data["qrels"].items():
        for did, score in rels.items():
            assert isinstance(score, int)
            assert score == 1


def test_compute_metrics_returns_three_metrics(tmp_path):
    import numpy as np
    from scripts.eval_rag import compute_metrics
    n_corpus, n_query, dim = 10, 5, 4
    rng = np.random.default_rng(0)
    corpus_embs = rng.random((n_corpus, dim), dtype=np.float32)
    query_embs = rng.random((n_query, dim), dtype=np.float32)
    corpus_ids = [f"d{i}" for i in range(n_corpus)]
    query_ids = [f"q{i}" for i in range(n_query)]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(n_query)}
    metrics = compute_metrics(corpus_embs, corpus_ids, query_embs, query_ids, qrels)
    assert set(metrics.keys()) == {"ndcg_at_10", "recall_at_10", "mrr"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_compute_metrics_empty_qrels():
    import numpy as np
    from scripts.eval_rag import compute_metrics
    rng = np.random.default_rng(1)
    corpus_embs = rng.random((5, 4), dtype=np.float32)
    query_embs = rng.random((3, 4), dtype=np.float32)
    metrics = compute_metrics(
        corpus_embs, ["d0","d1","d2","d3","d4"],
        query_embs, ["q0","q1","q2"],
        {},  # empty qrels (unanswerable)
    )
    assert metrics == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}


def test_compute_metrics_perfect_retrieval():
    import numpy as np
    from scripts.eval_rag import compute_metrics
    dim = 4
    # Identical query and corpus vectors → cosine similarity = 1.0 → rank 1 always
    vecs = np.eye(3, dim, dtype=np.float32)
    corpus_ids = ["d0", "d1", "d2"]
    query_ids = ["q0", "q1", "q2"]
    qrels = {"q0": {"d0": 1}, "q1": {"d1": 1}, "q2": {"d2": 1}}
    metrics = compute_metrics(vecs, corpus_ids, vecs, query_ids, qrels)
    assert metrics["ndcg_at_10"] > 0.9
    assert metrics["mrr"] > 0.9


def test_compute_metrics_by_type_basic():
    import numpy as np
    from scripts.eval_rag import compute_metrics_by_type
    dim = 4
    # 3 queries: factual, multi_hop, unanswerable
    vecs = np.eye(3, dim, dtype=np.float32)
    corpus_ids = ["d0", "d1", "d2"]
    query_ids = ["q0", "q1", "q2"]
    qrels = {"q0": {"d0": 1}, "q1": {"d1": 1}}  # q2 (unanswerable) has no entry
    query_types = {"q0": "factual", "q1": "multi_hop", "q2": "unanswerable"}
    result = compute_metrics_by_type(vecs, corpus_ids, vecs, query_ids, qrels, query_types)
    assert set(result.keys()) == {"factual", "multi_hop", "summarization", "unanswerable"}
    # Perfect retrieval for factual and multi_hop
    assert result["factual"]["ndcg_at_10"] > 0.9
    assert result["multi_hop"]["ndcg_at_10"] > 0.9
    # Unanswerable always 0
    assert result["unanswerable"] == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}


def test_compute_metrics_by_type_absent_type_returns_zeros():
    import numpy as np
    from scripts.eval_rag import compute_metrics_by_type
    rng = np.random.default_rng(42)
    corpus_embs = rng.random((5, 4), dtype=np.float32)
    query_embs = rng.random((3, 4), dtype=np.float32)
    corpus_ids = [f"d{i}" for i in range(5)]
    query_ids = [f"q{i}" for i in range(3)]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(3)}
    query_types = {f"q{i}": "factual" for i in range(3)}  # no multi_hop, summarization, unanswerable
    result = compute_metrics_by_type(corpus_embs, corpus_ids, query_embs, query_ids, qrels, query_types)
    assert result["multi_hop"] == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    assert result["summarization"] == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    assert result["unanswerable"] == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}


def test_compute_metrics_by_type_unknown_query_type_excluded():
    import numpy as np
    from scripts.eval_rag import compute_metrics_by_type
    rng = np.random.default_rng(7)
    corpus_embs = rng.random((5, 4), dtype=np.float32)
    query_embs = rng.random((2, 4), dtype=np.float32)
    corpus_ids = [f"d{i}" for i in range(5)]
    query_ids = ["q0", "q1"]
    qrels = {"q0": {"d0": 1}, "q1": {"d1": 1}}
    query_types = {"q0": "factual"}  # q1 has no type entry → excluded from all buckets
    result = compute_metrics_by_type(corpus_embs, corpus_ids, query_embs, query_ids, qrels, query_types)
    # q1 is absent from all types, so only factual has results
    assert result["multi_hop"] == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    # factual has q0 only — should compute without crash
    assert set(result["factual"].keys()) == {"ndcg_at_10", "recall_at_10", "mrr"}
