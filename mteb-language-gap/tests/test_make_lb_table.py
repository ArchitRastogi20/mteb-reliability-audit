from __future__ import annotations
import csv
import json
from pathlib import Path
import pytest
import yaml


TASKS_JPN = ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives"]
TASKS_HIN = ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives", "MLQARetrieval", "WikipediaRetrievalMultilingual"]
TASKS_ITA = ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives", "WikipediaRetrievalMultilingual"]


def _write_results(results_dir: Path, san_id: str, tasks: list[str], score: float) -> None:
    d = results_dir / san_id
    d.mkdir(parents=True)
    for task in tasks:
        (d / f"{task}.json").write_text(json.dumps({
            "scores": {"test": [{"main_score": score}]}
        }))


def _write_mteb_csv(path: Path, tasks: list[str], model_names: list[str], score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Model"] + tasks)
        for i, name in enumerate(model_names):
            w.writerow([i, name] + [str(score)] * len(tasks))


def _write_agg_csv(path: Path, model_ids: list[str], agg: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "mteb_agg", "source"])
        for mid in model_ids:
            w.writerow([mid, str(agg), "test"])


def _write_model_yaml(path: Path, model_ids: list[str]) -> None:
    models = [
        {"id": mid, "params": "100M", "dim": 384, "tier": "small",
         "passage_prefix": "", "query_prefix": ""}
        for mid in model_ids
    ]
    path.write_text(yaml.dump({"models": models}))


def test_jpn_table_has_correct_columns(tmp_path):
    from scripts.lb_evaluator import LANG_CONFIGS
    from scripts.make_lb_table import build_rows, write_markdown
    from scripts.utils import sanitize_model_id

    cfg = LANG_CONFIGS["jpn"]
    model_id = "org/test-model"
    san_id = sanitize_model_id(model_id)

    _write_results(tmp_path / "results/jpn_lb", san_id, TASKS_JPN, 0.75)
    _write_mteb_csv(tmp_path / "mteb.csv", TASKS_JPN, ["test-model"], 80.0)
    _write_agg_csv(tmp_path / "config/agg.csv", [model_id], 70.0)
    _write_model_yaml(tmp_path / "config/models.yaml", [model_id])

    rows = build_rows(
        lang_code="jpn",
        results_dir=tmp_path / "results/jpn_lb",
        mteb_csv_path=tmp_path / "mteb.csv",
        agg_csv_path=tmp_path / "config/agg.csv",
        model_yaml_path=tmp_path / "config/models.yaml",
    )
    assert len(rows) == 1
    assert "AvgJPN" in rows[0]
    assert "Gap" in rows[0]


def test_gap_formula(tmp_path):
    from scripts.lb_evaluator import LANG_CONFIGS
    from scripts.make_lb_table import build_rows
    from scripts.utils import sanitize_model_id

    san_id = sanitize_model_id("org/test-model")
    model_id = "org/test-model"

    _write_results(tmp_path / "results/hin_lb", san_id, TASKS_HIN, 0.80)  # AvgHIN = 80.0
    _write_mteb_csv(tmp_path / "mteb.csv", TASKS_HIN[:2], ["test-model"], 85.0)
    _write_agg_csv(tmp_path / "config/agg.csv", [model_id], 70.0)  # MTEB_agg = 70.0
    _write_model_yaml(tmp_path / "config/models.yaml", [model_id])

    rows = build_rows(
        lang_code="hin",
        results_dir=tmp_path / "results/hin_lb",
        mteb_csv_path=tmp_path / "mteb.csv",
        agg_csv_path=tmp_path / "config/agg.csv",
        model_yaml_path=tmp_path / "config/models.yaml",
    )
    assert len(rows) == 1
    # Gap = AvgHIN(80.0) - MTEB_agg(70.0) = +10.0
    assert rows[0]["Gap"] == "+10.0"


def test_markdown_table_format(tmp_path):
    from scripts.make_lb_table import build_rows, write_markdown
    from scripts.utils import sanitize_model_id

    san_id = sanitize_model_id("org/test-model")
    model_id = "org/test-model"
    out_path = tmp_path / "table.md"

    _write_results(tmp_path / "results/jpn_lb", san_id, TASKS_JPN, 0.75)
    _write_mteb_csv(tmp_path / "mteb.csv", TASKS_JPN, ["test-model"], 80.0)
    _write_agg_csv(tmp_path / "config/agg.csv", [model_id], 70.0)
    _write_model_yaml(tmp_path / "config/models.yaml", [model_id])

    rows = build_rows(
        lang_code="jpn",
        results_dir=tmp_path / "results/jpn_lb",
        mteb_csv_path=tmp_path / "mteb.csv",
        agg_csv_path=tmp_path / "config/agg.csv",
        model_yaml_path=tmp_path / "config/models.yaml",
    )
    write_markdown(rows, out_path, lang_code="jpn")
    content = out_path.read_text()
    assert "| " in content
    assert "*MTEB scores" in content  # footnote present


def test_missing_mteb_csv_shows_na(tmp_path):
    from scripts.make_lb_table import build_rows
    from scripts.utils import sanitize_model_id

    san_id = sanitize_model_id("org/test-model")
    model_id = "org/test-model"

    _write_results(tmp_path / "results/ita_lb", san_id, TASKS_ITA, 0.75)
    # CSV has only 2 tasks (not WikipediaRetrievalMultilingual)
    _write_mteb_csv(tmp_path / "mteb.csv", TASKS_ITA[:2], ["test-model"], 85.0)
    _write_agg_csv(tmp_path / "config/agg.csv", [model_id], 70.0)
    _write_model_yaml(tmp_path / "config/models.yaml", [model_id])

    rows = build_rows(
        lang_code="ita",
        results_dir=tmp_path / "results/ita_lb",
        mteb_csv_path=tmp_path / "mteb.csv",
        agg_csv_path=tmp_path / "config/agg.csv",
        model_yaml_path=tmp_path / "config/models.yaml",
    )
    assert len(rows) == 1
    # WikipediaRetrievalMultilingual not in CSV → MTEB ref should be N/A
    wiki_label = "WikiIta_mteb"
    assert rows[0].get(wiki_label) in ("N/A", None, "")
