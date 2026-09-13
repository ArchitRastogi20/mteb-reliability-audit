"""
Shared data-loading and utility functions for the analysis scripts.

All loaders return tidy pandas DataFrames with a normalised NDCG@10 scale
(floats in [0, 1]). If your CSVs use the 0-100 scale, the loaders detect
and convert.

Run `python utils.py` to validate that all expected files load correctly.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SEED = 20260510

# Task lists (mirrors data/README.md; do not edit in scripts, edit here).
ENGLISH_DERIVED_TASKS_FULL = [
    "ArguAna",
    "TRECCOVID",
    "SCIDOCS",
    "StackOverflowQA",
    "HagridRetrieval",
    "LegalBenchCorporateLobbying",
    "AILAStatutes",
    "LEMBPasskeyRetrieval",
    "SpartQA",
    "TempReasonL1",
    "WinoGrande",
]

ENGLISH_DERIVED_TASKS_STRICT_6 = [
    "ArguAna",
    "TRECCOVID",
    "SCIDOCS",
    "StackOverflowQA",
    "HagridRetrieval",
    "LegalBenchCorporateLobbying",
]

MULTILINGUAL_TASKS = [
    "BelebeleRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MLQARetrieval",
    "WikipediaRetrievalMultilingual",
    "StatcanDialogueDatasetRetrieval",
    "CovidRetrieval",
    "TwitterHjerneRetrieval",
]

CLUSTER_MODELS = [
    "granite-311m",
    "harrier-0.6b",
    "Seed1.6-embed",
    "inf-retriever-v1",
    "granite-97m-r2",
]

AUDIT_LANGUAGES_ISO = [
    "ita", "jpn", "hin", "fra", "ind", "tel", "kor", "swa", "fas",
    "tha", "rus", "zho", "deu", "ben", "ara", "spa", "vie",
]


def _normalise_ndcg(series: pd.Series) -> pd.Series:
    """If values look like 0-100, divide by 100. Otherwise pass through."""
    if series.max() > 1.5:
        return series / 100.0
    return series


def load_per_task_ndcg() -> pd.DataFrame:
    """Returns columns: model_id, task_name, ndcg_at_10."""
    df = pd.read_csv(DATA_DIR / "per_task_ndcg.csv")
    df["ndcg_at_10"] = _normalise_ndcg(df["ndcg_at_10"])
    return df


def load_task_taxonomy() -> pd.DataFrame:
    """Returns columns: task_name, classification, is_canonical_english."""
    df = pd.read_csv(DATA_DIR / "task_taxonomy.csv")
    if "is_canonical_english" not in df.columns:
        df["is_canonical_english"] = df["task_name"].isin(
            ENGLISH_DERIVED_TASKS_STRICT_6
        )
    return df


def load_lang_avg_per_model() -> pd.DataFrame:
    """Returns columns: model_id, language_iso, lang_avg_ndcg, tier."""
    df = pd.read_csv(DATA_DIR / "lang_avg_per_model.csv")
    df["lang_avg_ndcg"] = _normalise_ndcg(df["lang_avg_ndcg"])
    return df


def load_mteb_agg_per_model() -> pd.DataFrame:
    """Returns columns: model_id, mteb_agg, tier, global_rank."""
    df = pd.read_csv(DATA_DIR / "mteb_agg_per_model.csv")
    df["mteb_agg"] = _normalise_ndcg(df["mteb_agg"])
    return df


def load_cluster_membership() -> pd.DataFrame:
    """Returns columns: model_id, is_cluster_member, n_languages_affected."""
    return pd.read_csv(DATA_DIR / "cluster_membership.csv")


def append_summary(experiment_id: str, lines: list[str]) -> None:
    """Append a short result block to outputs/SUMMARY.md."""
    summary_path = OUTPUTS_DIR / "SUMMARY.md"
    block = [f"\n## {experiment_id}\n", *lines, ""]
    with summary_path.open("a") as f:
        f.write("\n".join(block) + "\n")


if __name__ == "__main__":
    print("Validating data loaders...")
    for name, fn in [
        ("per_task_ndcg", load_per_task_ndcg),
        ("task_taxonomy", load_task_taxonomy),
        ("lang_avg_per_model", load_lang_avg_per_model),
        ("mteb_agg_per_model", load_mteb_agg_per_model),
        ("cluster_membership", load_cluster_membership),
    ]:
        try:
            df = fn()
            print(f"  ok: {name}: {len(df)} rows, columns = {list(df.columns)}")
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
    print("Done.")
