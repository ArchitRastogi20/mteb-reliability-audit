# analysis/integrate_api_results.py
"""Fold OpenAI + Cohere + Gemini embedding eval results into analysis CSVs.

Updates:
  - analysis_output/unified_rag_results_v2.csv          (overall RAG metrics)
  - analysis_output/unified_rag_bytype_results_v2.csv   (per-query-type RAG)
  - analysis_output/openai_belebele_per_lang.csv        (Belebele per-language)
  - all_mteb/all_performance_per_language.csv           (appended rows)

Run: python3 integrate_api_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / "rag-dataset" / "output" / "evaluation"
OUT = ROOT / "analysis_output"

API_MODELS = {
    "text-embedding-3-large": {
        "folder": "text_embedding_3_lar_af1f17",
        "dim": 3072, "tier": "api", "english_only": False,
        "short_name": "openai-3-large",
        "leaderboard_name": "text-embedding-3-large",
        "model_id_prefix": "openai/",
    },
    "text-embedding-3-small": {
        "folder": "text_embedding_3_sma_ade2d3",
        "dim": 1536, "tier": "api", "english_only": False,
        "short_name": "openai-3-small",
        "leaderboard_name": "text-embedding-3-small",
        "model_id_prefix": "openai/",
    },
    "embed-multilingual-v3.0": {
        "folder": "embed_multilingual_v_672ca2",
        "dim": 1024, "tier": "api", "english_only": False,
        "short_name": "cohere-v3",
        "leaderboard_name": "embed-multilingual-v3.0",
        "model_id_prefix": "cohere/",
    },
    "gemini-embedding-2": {
        "folder": "gemini_embedding_2_72c72b",
        "dim": 3072, "tier": "api", "english_only": False,
        "short_name": "gemini-2",
        "leaderboard_name": "gemini-embedding-2",
        "model_id_prefix": "gemini/",
    },
}


def integrate_rag_overall() -> None:
    df = pd.read_csv(OUT / "unified_rag_results_v2.csv")
    # Drop any pre-existing API rows so the script is idempotent
    df = df[~df.short_name.isin(m["short_name"] for m in API_MODELS.values())]

    new_rows = []
    for model_name, meta in API_MODELS.items():
        folder = EVAL / meta["folder"]
        if not folder.exists():
            continue
        for json_file in sorted(folder.glob("*.json")):
            d = json.loads(json_file.read_text())
            new_rows.append({
                "model_id": d.get("model_id", f"{meta['model_id_prefix']}{model_name}"),
                "short_name": meta["short_name"],
                "params_numeric": -1,
                "dim": d.get("embed_dim", meta["dim"]),
                "tier": "api",
                "english_only": False,
                "dataset": json_file.stem,
                "ndcg_at_10": d["overall"]["ndcg_at_10"] * 100,
                "recall_at_10": d["overall"]["recall_at_10"] * 100,
                "mrr": d["overall"]["mrr"] * 100,
            })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(OUT / "unified_rag_results_v2.csv", index=False)
    print(f"[rag-overall] appended {len(new_rows)} rows ({len(df)} total)")


def integrate_rag_bytype() -> None:
    df = pd.read_csv(OUT / "unified_rag_bytype_results_v2.csv")
    df = df[~df.short_name.isin(m["short_name"] for m in API_MODELS.values())]

    new_rows = []
    for model_name, meta in API_MODELS.items():
        folder = EVAL / meta["folder"]
        if not folder.exists():
            continue
        for json_file in sorted(folder.glob("*.json")):
            d = json.loads(json_file.read_text())
            for qtype, m in d.get("by_type", {}).items():
                new_rows.append({
                    "model_id": d.get("model_id", f"{meta['model_id_prefix']}{model_name}"),
                    "short_name": meta["short_name"],
                    "params_numeric": -1,
                    "tier": "api",
                    "dataset": json_file.stem,
                    "query_type": qtype,
                    "ndcg_at_10": m["ndcg_at_10"] * 100,
                    "recall_at_10": m["recall_at_10"] * 100,
                    "mrr": m["mrr"] * 100,
                })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(OUT / "unified_rag_bytype_results_v2.csv", index=False)
    print(f"[rag-bytype] appended {len(new_rows)} rows ({len(df)} total)")


def integrate_belebele() -> None:
    bel = json.loads((EVAL / "openai_belebele.json").read_text())
    rows = []
    for model_name, langs in bel.items():
        if model_name not in API_MODELS:
            continue
        for lang, m in langs.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "Model": API_MODELS[model_name]["leaderboard_name"],
                "language": lang,
                "BelebeleRetrieval": m["ndcg_at_10"] * 100,
                "Recall_at_10": m["recall_at_10"] * 100,
                "MRR": m["mrr"] * 100,
            })
    pd.DataFrame(rows).to_csv(OUT / "openai_belebele_per_lang.csv", index=False)
    print(f"[belebele] wrote {len(rows)} rows")


def append_to_all_mteb_per_language() -> None:
    csv_path = ROOT / "mteb_csvs" / "all_mteb" / "all_performance_per_language.csv"
    df = pd.read_csv(csv_path)
    df = df[~df[df.columns[1]].isin(m["leaderboard_name"] for m in API_MODELS.values())]

    bel = json.loads((EVAL / "openai_belebele.json").read_text())
    LANG_TO_COL = {"japanese": "jpn-Jpan", "hindi": "hin-Deva"}
    new_rows = []
    for model_name, langs in bel.items():
        if model_name not in API_MODELS:
            continue
        meta = API_MODELS[model_name]
        row = {col: pd.NA for col in df.columns}
        row[df.columns[1]] = meta["leaderboard_name"]
        for lang, m in langs.items():
            if not isinstance(m, dict):
                continue
            col = LANG_TO_COL.get(lang)
            if col and col in df.columns:
                row[col] = m["ndcg_at_10"] * 100
        new_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"[all_mteb] appended {len(new_rows)} rows")


def main() -> None:
    integrate_rag_overall()
    integrate_rag_bytype()
    integrate_belebele()
    append_to_all_mteb_per_language()
    print("\nNext: rerun regen_figures.py + p0_revisions.py + recompile <paper>/main.tex")


if __name__ == "__main__":
    main()
