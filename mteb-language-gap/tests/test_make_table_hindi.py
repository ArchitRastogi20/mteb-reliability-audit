import json
import pytest
from pathlib import Path

TASKS_HINDI = [
    "WikipediaRetrievalMultilingual", "MIRACLRetrieval", "IndicQARetrieval",
    "MintakaRetrieval", "WebFAQRetrieval", "MLQARetrieval",
]


def _write_synthetic_results(results_dir: Path, model_san_id: str, task_scores: dict) -> None:
    d = results_dir / model_san_id
    d.mkdir(parents=True)
    for task, score in task_scores.items():
        (d / f"{task}.json").write_text(json.dumps({
            "scores": {"test": [{"main_score": score}]}
        }))


def _write_config(config_dir: Path) -> None:
    import yaml
    cfg = {"models": [
        {"id": "org/model-small", "params": "118M", "dim": 384, "tier": "small",
         "batch_size": 4096, "passage_prefix": "", "query_prefix": ""},
        {"id": "org/model-large", "params": "7B", "dim": 4096, "tier": "large",
         "batch_size": 512, "passage_prefix": "", "query_prefix": ""},
    ]}
    (config_dir / "models_hindi.yaml").write_text(yaml.dump(cfg))
    with open(config_dir / "mteb_agg_scores.csv", "w") as f:
        f.write("model_id,mteb_agg,source\norg/model-small,70.0,test\norg/model-large,85.0,test\n")


def test_hindi_markdown_table_produced(tmp_path):
    from scripts.utils import sanitize_model_id
    from scripts.make_table_hindi import build_rows_hindi, write_markdown_hindi

    results_dir = tmp_path / "results"
    config_dir = tmp_path / "config"
    tables_dir = tmp_path / "tables"
    config_dir.mkdir(); tables_dir.mkdir()
    _write_config(config_dir)

    scores = {t: 0.80 for t in TASKS_HINDI}
    _write_synthetic_results(results_dir, sanitize_model_id("org/model-small"), scores)
    _write_synthetic_results(results_dir, sanitize_model_id("org/model-large"), {t: 0.88 for t in TASKS_HINDI})

    import yaml, csv as csvmod
    cfg = yaml.safe_load((config_dir / "models_hindi.yaml").read_text())
    model_configs = {m["id"]: m for m in cfg["models"]}
    agg_scores = {}
    for row in csvmod.DictReader(open(config_dir / "mteb_agg_scores.csv")):
        agg_scores[row["model_id"]] = float(row["mteb_agg"])

    rows = build_rows_hindi(results_dir, model_configs, agg_scores)
    assert len(rows) == 2

    out = tables_dir / "test_hindi.md"
    write_markdown_hindi(rows, out)
    content = out.read_text()
    assert "| " in content
    assert "model-small" in content or "model_small" in content


def test_hindi_gap_computed_correctly(tmp_path):
    from scripts.utils import sanitize_model_id
    from scripts.make_table_hindi import build_rows_hindi

    results_dir = tmp_path / "results"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_config(config_dir)

    scores = {t: 0.8 for t in TASKS_HINDI}  # AvgHIN = 80.0
    _write_synthetic_results(results_dir, sanitize_model_id("org/model-small"), scores)

    import yaml, csv as csvmod
    cfg = yaml.safe_load((config_dir / "models_hindi.yaml").read_text())
    model_configs = {m["id"]: m for m in cfg["models"]}
    agg_scores = {"org/model-small": 70.0}

    rows = build_rows_hindi(results_dir, model_configs, agg_scores)
    assert len(rows) == 1
    # gap = 80.0 - 70.0 = +10.0
    assert rows[0]["gap"] == "+10.0"
