import json
from pathlib import Path


def _make_result(model_id: str, lang: str, domain: str, ndcg: float) -> dict:
    return {
        "model_id": model_id,
        "lang": lang,
        "domain": domain,
        "overall": {"ndcg_at_10": ndcg, "recall_at_10": ndcg - 0.05, "mrr": ndcg - 0.02},
        "by_type": {
            "factual":       {"ndcg_at_10": ndcg + 0.1, "recall_at_10": ndcg + 0.1, "mrr": ndcg + 0.1},
            "multi_hop":     {"ndcg_at_10": ndcg - 0.1, "recall_at_10": ndcg - 0.1, "mrr": ndcg - 0.1},
            "summarization": {"ndcg_at_10": ndcg - 0.2, "recall_at_10": ndcg - 0.2, "mrr": ndcg - 0.2},
            "unanswerable":  {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
        },
        "evaluation_time_sec": 10.0,
        "peak_vram_mb": 1000,
    }


def _write_results(eval_dir: Path, san_id: str, results: list[dict]) -> None:
    for r in results:
        out = eval_dir / san_id / f"{r['lang']}_{r['domain']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r))


def test_build_overall_rows_single_model(tmp_path):
    from scripts.rag_report import build_overall_rows
    from scripts.rag_utils import sanitize_model_id
    eval_dir = tmp_path / "evaluation"
    san_id = sanitize_model_id("intfloat/multilingual-e5-small")
    combos = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]
    results = [_make_result("intfloat/multilingual-e5-small", l, d, 0.6) for l, d in combos]
    _write_results(eval_dir, san_id, results)

    model_map = {"intfloat/multilingual-e5-small": {"params": "118M", "dim": 384}}
    rows = build_overall_rows(eval_dir, model_map, metric="ndcg_at_10")
    assert len(rows) == 1
    assert rows[0]["Model"] == "multilingual-e5-small"
    assert rows[0]["Params"] == "118M"
    assert "ja/fin" in rows[0]
    assert "AvgRAG" in rows[0]


def test_build_overall_rows_sorted_by_avg(tmp_path):
    from scripts.rag_report import build_overall_rows
    from scripts.rag_utils import sanitize_model_id
    eval_dir = tmp_path / "evaluation"
    combos = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]
    for model_id, ndcg in [("org/model-a", 0.5), ("org/model-b", 0.8)]:
        san = sanitize_model_id(model_id)
        results = [_make_result(model_id, l, d, ndcg) for l, d in combos]
        _write_results(eval_dir, san, results)

    model_map = {
        "org/model-a": {"params": "100M", "dim": 384},
        "org/model-b": {"params": "500M", "dim": 768},
    }
    rows = build_overall_rows(eval_dir, model_map, metric="ndcg_at_10")
    assert rows[0]["Model"] == "model-b"  # higher score first


def test_write_markdown_table(tmp_path):
    from scripts.rag_report import write_markdown_table
    rows = [
        {"Model": "model-a", "Params": "118M", "AvgRAG": "65.2"},
        {"Model": "model-b", "Params": "568M", "AvgRAG": "72.1"},
    ]
    out = tmp_path / "table.md"
    write_markdown_table(rows, out)
    content = out.read_text()
    assert "| Model |" in content
    assert "model-a" in content
    assert "model-b" in content
    assert "---" in content


def test_update_tables_creates_files(tmp_path, monkeypatch):
    from scripts.rag_report import update_tables
    from scripts.rag_utils import sanitize_model_id
    eval_dir = tmp_path / "evaluation"
    tables_dir = tmp_path / "tables"
    combos = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]
    san = sanitize_model_id("intfloat/multilingual-e5-small")
    results = [_make_result("intfloat/multilingual-e5-small", l, d, 0.6) for l, d in combos]
    _write_results(eval_dir, san, results)

    model_yaml = tmp_path / "models.yaml"
    import yaml
    model_yaml.write_text(yaml.dump({"models": [
        {"id": "intfloat/multilingual-e5-small", "params": "118M", "dim": 384, "tier": "small"}
    ]}))
    update_tables(eval_dir=eval_dir, tables_dir=tables_dir, model_yaml=model_yaml)
    assert (tables_dir / "rag_overall.md").exists()
    assert (tables_dir / "rag_by_type.md").exists()
    assert (eval_dir / "summary.json").exists()


def test_build_bytype_rows_returns_correct_columns(tmp_path):
    from scripts.rag_report import build_bytype_rows
    from scripts.rag_utils import sanitize_model_id
    eval_dir = tmp_path / "evaluation"
    san_id = sanitize_model_id("intfloat/multilingual-e5-small")
    combos = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]
    results = [_make_result("intfloat/multilingual-e5-small", l, d, 0.6) for l, d in combos]
    _write_results(eval_dir, san_id, results)

    model_map = {"intfloat/multilingual-e5-small": {"params": "118M", "dim": 384}}
    rows = build_bytype_rows(eval_dir, model_map, query_type="factual", metric="ndcg_at_10")
    assert len(rows) == 1
    assert rows[0]["Model"] == "multilingual-e5-small"
    assert "ja/fin" in rows[0]
    assert "AvgRAG" in rows[0]
