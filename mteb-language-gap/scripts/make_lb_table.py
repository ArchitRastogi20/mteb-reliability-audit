#!/usr/bin/env python3
"""
Reads results/{lang}_lb/ + MTEB leaderboard CSV + mteb_agg_scores_subset.csv
→ tables/{lang}_lb_comparison.md
"""
from __future__ import annotations
import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
import yaml

from scripts.lb_evaluator import LANG_CONFIGS, LanguageConfig
from scripts.utils import sanitize_model_id

REPO_ROOT = Path(__file__).parent.parent
CONFIG_FILE = REPO_ROOT / "config" / "models.yaml"
AGG_CSV = REPO_ROOT / "config" / "mteb_agg_scores_subset.csv"
TABLES_DIR = REPO_ROOT / "tables"

logger = logging.getLogger("mteb_lb")

TASK_DISPLAY: dict[str, dict[str, str]] = {
    "jpn": {
        "BelebeleRetrieval": "Belebele",
        "MIRACLRetrievalHardNegatives": "MIRACL_HN",
    },
    "hin": {
        "BelebeleRetrieval": "Belebele",
        "MIRACLRetrievalHardNegatives": "MIRACL_HN",
        "MLQARetrieval": "MLQA",
        "WikipediaRetrievalMultilingual": "WikiHi",
    },
    "ita": {
        "BelebeleRetrieval": "Belebele",
        "WikipediaRetrievalMultilingual": "WikiIta",
    },
    "swa": {
        "BelebeleRetrieval": "Belebele",
    },
}


def _clean_model_name(name: str) -> str:
    m = re.search(r"\[([^\]]+)\]", name)
    return m.group(1) if m else name


def _load_mteb_csv(path: Path) -> dict[str, dict[str, float]]:
    """Returns {short_model_name: {task_name: score}}."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        task_cols = [c for c in (reader.fieldnames or []) if c not in ("", "Model")]
        for row in reader:
            name = _clean_model_name(row["Model"])
            result[name] = {}
            for task in task_cols:
                val = row.get(task, "").strip()
                if val:
                    try:
                        result[name][task] = float(val)
                    except ValueError:
                        pass
    return result


def build_rows(
    lang_code: str,
    results_dir: Path,
    mteb_csv_path: Path,
    agg_csv_path: Path,
    model_yaml_path: Path,
) -> list[dict]:
    cfg = LANG_CONFIGS[lang_code]
    display = TASK_DISPLAY[lang_code]

    model_configs = {
        m["id"]: m
        for m in yaml.safe_load(model_yaml_path.read_text())["models"]
    }
    san_to_mid = {sanitize_model_id(mid): mid for mid in model_configs}

    agg_scores: dict[str, float] = {}
    with open(agg_csv_path) as f:
        for row in csv.DictReader(f):
            agg_scores[row["model_id"]] = float(row["mteb_agg"])

    mteb_task_scores = _load_mteb_csv(mteb_csv_path)

    rows = []
    if not results_dir.exists():
        return rows

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        san_id = model_dir.name
        model_id = san_to_mid.get(san_id)
        if model_id is None:
            continue
        mc = model_configs[model_id]
        short_name = model_id.split("/")[-1]

        task_ndcgs: list[float] = []
        row: dict[str, str] = {
            "Model": short_name,
            "Params": mc["params"],
            "Dim": str(mc["dim"]),
        }

        for task in cfg.tasks:
            label = display[task]
            task_file = model_dir / f"{task}.json"
            if not task_file.exists():
                row[f"{label}_ours"] = "N/A"
                row[f"{label}_mteb"] = "N/A"
                row[f"Δ{label}"] = "N/A"
                continue

            our_ndcg = json.loads(task_file.read_text())["scores"]["test"][0]["main_score"]
            our_score = our_ndcg * 100
            task_ndcgs.append(our_ndcg)
            row[f"{label}_ours"] = f"{our_score:.1f}"

            mteb_ref: float | None = None
            for model_short, task_dict in mteb_task_scores.items():
                if model_short.lower() in short_name.lower() or short_name.lower() in model_short.lower():
                    mteb_ref = task_dict.get(task)
                    break

            if mteb_ref is not None:
                row[f"{label}_mteb"] = f"{mteb_ref:.1f}"
                row[f"Δ{label}"] = f"{our_score - mteb_ref:+.1f}"
            else:
                row[f"{label}_mteb"] = "N/A"
                row[f"Δ{label}"] = "N/A"

        if not task_ndcgs:
            continue

        avg_lang = float(np.mean(task_ndcgs)) * 100
        agg = agg_scores.get(model_id, float("nan"))
        gap = avg_lang - agg if not np.isnan(agg) else float("nan")

        row["MTEB_agg"] = f"{agg:.1f}" if not np.isnan(agg) else "N/A"
        row[cfg.avg_label] = f"{avg_lang:.1f}"
        row["Gap"] = f"{gap:+.1f}" if not np.isnan(gap) else "N/A"

        rows.append(row)

    rows.sort(key=lambda r: float(r.get(cfg.avg_label, "0") or "0"), reverse=True)
    return rows


def write_markdown(rows: list[dict], path: Path, lang_code: str) -> None:
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
    lines.append("")
    lines.append(
        f"*MTEB scores = task-level averages across **all** language subsets on the MTEB leaderboard "
        f"(not {lang_code.upper()}-specific)."
    )
    path.write_text("\n".join(lines) + "\n")


def update_table(lang_code: str) -> None:
    cfg = LANG_CONFIGS[lang_code]
    out = TABLES_DIR / f"{lang_code}_lb_comparison.md"
    rows = build_rows(
        lang_code=lang_code,
        results_dir=cfg.results_base,
        mteb_csv_path=cfg.mteb_csv,
        agg_csv_path=AGG_CSV,
        model_yaml_path=CONFIG_FILE,
    )
    write_markdown(rows, out, lang_code)
    logger.info("Table updated: %s (%d models)", out, len(rows))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["jpn", "hin", "ita", "swa", "all"], default="all")
    args = parser.parse_args()

    langs = ["jpn", "hin", "ita", "swa"] if args.lang == "all" else [args.lang]
    for lang in langs:
        update_table(lang)
        out = TABLES_DIR / f"{lang}_lb_comparison.md"
        if out.exists():
            print(f"[{lang}] {out}")


if __name__ == "__main__":
    main()
