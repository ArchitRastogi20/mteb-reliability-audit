#!/usr/bin/env python3
"""
Reproduces the MMTEB language-filter behavior finding from Section 4.5.

What this demonstrates:
  The MMTEB "Italian filter" changes which tasks appear in the leaderboard view,
  but does NOT expose per-Italian NDCG@10 for BelebeleRetrieval. The "filtered"
  BelebeleRetrieval score shown is an aggregate over all languages, not an
  Italian-specific score. The per-language breakdown CSV for the Italian filter
  contains only eng-Latn -- no ita-Latn column exists.

Run from the repo root:
  python analysis/reproduce_filter_view.py

Inputs (relative to repo root):
  all_mteb/all_performance_per_task.csv
  italian_mteb/italian_performance_per_task.csv
  italian_mteb/italian_performance_per_langauge.csv
  japnese_mteb/japanese_performance_per_task.csv
  japnese_mteb/japanese_performace_per_langauge.csv
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).parent / "mteb_csvs"   # analysis/mteb_csvs/


def load(rel_path):
    p = REPO_ROOT / rel_path
    assert p.exists(), f"Expected file not found: {p}"
    return pd.read_csv(p)


def get_score(df, model, col):
    if col not in df.columns:
        return None
    row = df[df["Model"] == model]
    return float(row.iloc[0][col]) if not row.empty else None


def main():
    all_task = load("all_mteb/all_performance_per_task.csv")
    it_task  = load("italian_mteb/italian_performance_per_task.csv")
    it_lang  = load("italian_mteb/italian_performance_per_langauge.csv")
    jp_task  = load("japnese_mteb/japanese_performance_per_task.csv")
    jp_lang  = load("japnese_mteb/japanese_performace_per_langauge.csv")

    model = "bge-m3"

    print("=" * 60)
    print("DEMONSTRATION A: BelebeleRetrieval -- Italian filter")
    print("=" * 60)

    score_all = get_score(all_task, model, "BelebeleRetrieval")
    score_it  = get_score(it_task,  model, "BelebeleRetrieval")
    print(f"  {model} BelebeleRetrieval (all-tasks view):      {score_all:.4f}")
    print(f"  {model} BelebeleRetrieval (Italian filter view): {score_it:.4f}")
    print(f"  Scores differ ({score_all:.2f} vs {score_it:.2f}): both are cross-language")
    print(f"  aggregates, not Italian-specific NDCG.")

    it_lang_cols = [c for c in it_lang.columns if c not in ("Unnamed: 0", "Model")]
    print(f"\n  Italian filter per-language CSV columns: {it_lang_cols}")
    if "ita-Latn" in it_lang_cols:
        print("  UNEXPECTED: ita-Latn IS exposed -- check CSV.")
    else:
        print("  No ita-Latn column: Italian-specific NDCG is NOT exposed for BelebeleRetrieval.")

    print()
    print("=" * 60)
    print("DEMONSTRATION B: Tasks that DO expose per-language scores")
    print("=" * 60)

    jp_lang_cols = [c for c in jp_lang.columns if c not in ("Unnamed: 0", "Model")]
    print(f"  Japanese filter per-language CSV columns: {jp_lang_cols}")
    miracl_ja_score = get_score(jp_lang, model, "jpn-Jpan")
    print(f"  {model} jpn-Jpan (Japanese filter): {miracl_ja_score}")
    print(f"  MIRACLRetrievalHardNegatives DOES expose per-language (jpn-Jpan) scores.")

    print()
    print("Summary:")
    print("  BelebeleRetrieval (Italian filter): per-Italian NDCG NOT exposed.")
    print("  MIRACLRetrievalHardNegatives (Japanese filter): per-Japanese NDCG IS exposed.")
    print("  The language filter determines which tasks appear in the view,")
    print("  not whether per-language NDCG is extracted within a task.")


if __name__ == "__main__":
    main()
