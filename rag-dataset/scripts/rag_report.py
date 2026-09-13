from __future__ import annotations
import json
import logging
from pathlib import Path

import yaml

from scripts.rag_utils import sanitize_model_id

REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
TABLES_DIR = REPO_ROOT / "tables"
MODEL_YAML = REPO_ROOT / "config" / "models.yaml"

COMBOS = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]
COMBO_LABELS = {
    ("ja", "finance"): "ja/fin",
    ("ja", "law"):     "ja/law",
    ("hi", "finance"): "hi/fin",
    ("hi", "law"):     "hi/law",
}
QUERY_TYPES = ["factual", "multi_hop", "summarization", "unanswerable"]
METRICS = ["ndcg_at_10", "recall_at_10", "mrr"]

logger = logging.getLogger("rag_eval")


def _load_result(result_dir: Path, lang: str, domain: str) -> dict | None:
    path = result_dir / f"{lang}_{domain}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_overall_rows(
    eval_dir: Path,
    model_map: dict[str, dict],
    metric: str = "ndcg_at_10",
) -> list[dict]:
    san_to_mid = {sanitize_model_id(mid): mid for mid in model_map}
    rows: list[dict] = []

    for result_dir in sorted(eval_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith("."):
            continue
        model_id = san_to_mid.get(result_dir.name)
        if model_id is None:
            continue
        mc = model_map[model_id]
        short_name = model_id.split("/")[-1]

        row: dict = {"Model": short_name, "Params": mc["params"]}
        scores: list[float] = []
        for lang, domain in COMBOS:
            label = COMBO_LABELS[(lang, domain)]
            result = _load_result(result_dir, lang, domain)
            if result is None:
                row[label] = "N/A"
            else:
                raw = result.get("overall", {}).get(metric)
                if raw is None:
                    row[label] = "N/A"
                else:
                    val = raw * 100
                    scores.append(val)
                    row[label] = f"{val:.1f}"

        if not scores:
            continue
        avg = sum(scores) / len(scores)
        row["AvgRAG"] = f"{avg:.1f}"
        rows.append(row)

    rows.sort(key=lambda r: float(r.get("AvgRAG", "0") or "0"), reverse=True)
    return rows


def build_bytype_rows(
    eval_dir: Path,
    model_map: dict[str, dict],
    query_type: str,
    metric: str = "ndcg_at_10",
) -> list[dict]:
    san_to_mid = {sanitize_model_id(mid): mid for mid in model_map}
    rows: list[dict] = []

    for result_dir in sorted(eval_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith("."):
            continue
        model_id = san_to_mid.get(result_dir.name)
        if model_id is None:
            continue
        mc = model_map[model_id]
        short_name = model_id.split("/")[-1]

        row: dict = {"Model": short_name, "Params": mc["params"]}
        scores: list[float] = []
        for lang, domain in COMBOS:
            label = COMBO_LABELS[(lang, domain)]
            result = _load_result(result_dir, lang, domain)
            if result is None:
                row[label] = "N/A"
            else:
                raw = result.get("by_type", {}).get(query_type, {}).get(metric)
                if raw is None:
                    row[label] = "N/A"
                else:
                    val = raw * 100
                    scores.append(val)
                    row[label] = f"{val:.1f}"

        if not scores:
            continue
        row["AvgRAG"] = f"{sum(scores)/len(scores):.1f}"
        rows.append(row)

    rows.sort(key=lambda r: float(r.get("AvgRAG", "0") or "0"), reverse=True)
    return rows


def write_markdown_table(rows: list[dict], path: Path, mode: str = "w") -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(str(row.get(h, 'N/A')) for h in headers)} |")
    with path.open(mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_summary(eval_dir: Path, model_map: dict[str, dict]) -> dict:
    san_to_mid = {sanitize_model_id(mid): mid for mid in model_map}
    summary: dict = {}
    for result_dir in sorted(eval_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith("."):
            continue
        model_id = san_to_mid.get(result_dir.name)
        if model_id is None:
            continue
        model_results: dict = {}
        for lang, domain in COMBOS:
            result = _load_result(result_dir, lang, domain)
            if result:
                model_results[f"{lang}_{domain}"] = result
        if model_results:
            summary[model_id] = model_results
    return summary


def update_tables(
    eval_dir: Path = EVAL_DIR,
    tables_dir: Path = TABLES_DIR,
    model_yaml: Path = MODEL_YAML,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_map = {
        m["id"]: m
        for m in yaml.safe_load(model_yaml.read_text())["models"]
    }

    if not eval_dir.is_dir():
        logger.warning("eval_dir does not exist: %s", eval_dir)
        return

    overall_path = tables_dir / "rag_overall.md"
    overall_path.write_text("")
    for metric in METRICS:
        rows = build_overall_rows(eval_dir, model_map, metric=metric)
        if rows:
            metric_label = metric.replace("_at_10", "@10").replace("_", " ").upper()
            with overall_path.open("a", encoding="utf-8") as fh:
                fh.write(f"## {metric_label}\n\n")
            write_markdown_table(rows, overall_path, mode="a")
            with overall_path.open("a", encoding="utf-8") as fh:
                fh.write("\n")

    bytype_path = tables_dir / "rag_by_type.md"
    bytype_path.write_text("")
    for qtype in QUERY_TYPES:
        qtype_has_rows = False
        for metric in METRICS:
            rows = build_bytype_rows(eval_dir, model_map, qtype, metric=metric)
            if rows:
                if not qtype_has_rows:
                    with bytype_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"# Query Type: {qtype}\n\n")
                    qtype_has_rows = True
                metric_label = metric.replace("_at_10", "@10").replace("_", " ").upper()
                with bytype_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"## {metric_label}\n\n")
                write_markdown_table(rows, bytype_path, mode="a")
                with bytype_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n")

    # Summary JSON
    summary = _build_summary(eval_dir, model_map)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Tables updated: %s", tables_dir)
