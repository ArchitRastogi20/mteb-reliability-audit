# analysis/integrate_openai_results.py
"""Fold OpenAI embedding eval results into the existing analysis CSVs.

After running:
  - analysis_output/unified_rag_results_v2.csv          (overall RAG metrics)
  - analysis_output/unified_rag_bytype_results_v2.csv   (per-query-type RAG)
  - analysis_output/openai_belebele_per_lang.csv        (Belebele per-language; new file)
  - all_mteb/all_performance_per_language.csv           (appended rows for the 2 models)

Run: python3 integrate_openai_results.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / "rag-dataset" / "output" / "evaluation"
OUT = ROOT / "analysis_output"

# Actual sanitized folder names (computed via hashlib.md5 in eval_openai_embed.py):
#   text-embedding-3-large -> text_embedding_3_lar_af1f17
#   text-embedding-3-small -> text_embedding_3_sma_ade2d3
OPENAI_MODELS = {
    "text-embedding-3-large": {
        "folder": "text_embedding_3_lar_af1f17",  # actual hash, not spec's 8a7b2c
        "params": "—", "dim": 3072,
        "tier": "api", "english_only": False,
        "short_name": "openai-3-large",
        "leaderboard_name": "text-embedding-3-large",
    },
    "text-embedding-3-small": {
        "folder": "text_embedding_3_sma_ade2d3",  # actual hash, not spec's e4f1d9
        "params": "—", "dim": 1536,
        "tier": "api", "english_only": False,
        "short_name": "openai-3-small",
        "leaderboard_name": "text-embedding-3-small",
    },
}


def integrate_rag_overall() -> None:
    df = pd.read_csv(OUT / "unified_rag_results_v2.csv")
    new_rows = []
    for model, meta in OPENAI_MODELS.items():
        folder = EVAL / meta["folder"]
        for json_file in sorted(folder.glob("*.json")):
            d = json.loads(json_file.read_text())
            new_rows.append({
                "model_id": d["model_id"],
                "short_name": meta["short_name"],
                "params_numeric": -1,  # API model; no public param count
                "dim": d["embed_dim"],
                "tier": "api",
                "english_only": False,
                "dataset": json_file.stem,
                "ndcg_at_10": d["overall"]["ndcg_at_10"] * 100,
                "recall_at_10": d["overall"]["recall_at_10"] * 100,
                "mrr": d["overall"]["mrr"] * 100,
            })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(OUT / "unified_rag_results_v2.csv", index=False)
    print(f"[rag-overall] appended {len(new_rows)} rows")


def integrate_rag_bytype() -> None:
    df = pd.read_csv(OUT / "unified_rag_bytype_results_v2.csv")
    new_rows = []
    for model, meta in OPENAI_MODELS.items():
        folder = EVAL / meta["folder"]
        for json_file in sorted(folder.glob("*.json")):
            d = json.loads(json_file.read_text())
            for qtype, m in d["by_type"].items():
                new_rows.append({
                    "model_id": d["model_id"],
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
    print(f"[rag-bytype] appended {len(new_rows)} rows")


def integrate_belebele() -> None:
    bel = json.loads((EVAL / "openai_belebele.json").read_text())
    rows = []
    for model, langs in bel.items():
        if model not in OPENAI_MODELS:
            continue  # skip top-level totals (total_tokens, total_cost_usd keys)
        for lang, m in langs.items():
            rows.append({
                "Model": OPENAI_MODELS[model]["leaderboard_name"],
                "language": lang,
                "BelebeleRetrieval": m["ndcg_at_10"] * 100,
                "Recall_at_10": m["recall_at_10"] * 100,
                "MRR": m["mrr"] * 100,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "openai_belebele_per_lang.csv", index=False)
    print(f"[belebele] wrote {len(df)} rows to openai_belebele_per_lang.csv")


def append_to_all_mteb_per_language() -> None:
    """Append openai-{large,small} rows to all_mteb/all_performance_per_language.csv
    so p0_revisions.py picks them up automatically.

    Belebele JSON has language keys: "italian", "japanese", "hindi".
    Mapping to all_performance_per_language.csv column names:
      - "italian" -> "ita-Latn"  (NOT present in CSV; will be silently skipped)
      - "japanese" -> "jpn-Jpan" (present; will be written)
      - "hindi"    -> "hin-Deva" (present; will be written)

    For all other language columns we leave NaN — the downstream code already
    tolerates missing values (fillna in p0_revisions.py / regen_figures.py).
    """
    csv_path = ROOT / "mteb_csvs" / "all_mteb" / "all_performance_per_language.csv"
    df = pd.read_csv(csv_path)
    bel = json.loads((EVAL / "openai_belebele.json").read_text())

    # Column names in all_performance_per_language.csv use hyphen codes,
    # but the Belebele JSON uses human-readable language names.
    LANG_CODE_TO_COL = {
        "italian": "ita-Latn",  # NOT in CSV — column will be skipped silently
        "japanese": "jpn-Jpan",
        "hindi": "hin-Deva",
    }

    new_rows = []
    for model, langs in bel.items():
        if model not in OPENAI_MODELS:
            continue
        meta = OPENAI_MODELS[model]
        row = {col: pd.NA for col in df.columns}
        row[df.columns[1]] = meta["leaderboard_name"]   # 'Model' col (index 1)
        for lang, m in langs.items():
            col = LANG_CODE_TO_COL.get(lang)
            if col and col in df.columns:
                # Belebele-only score used as the language-filtered score for that language.
                # This is an approximation (the public CSV averages multiple tasks) but is
                # the best single-task proxy we have for these models on jpn/hin.
                row[col] = m["ndcg_at_10"] * 100
        new_rows.append(row)

    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"[all_mteb] appended {len(new_rows)} rows to all_performance_per_language.csv")


def main() -> None:
    integrate_rag_overall()
    integrate_rag_bytype()
    integrate_belebele()
    append_to_all_mteb_per_language()
    print("\nNext: re-run regen_figures.py and p0_revisions.py to refresh figures + stats.")


if __name__ == "__main__":
    main()
