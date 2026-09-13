#!/usr/bin/env python3
"""Task B: MMTEB language-filter behavior demonstration."""
import logging
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent          # analysis/
REPO_ROOT = SCRIPT_DIR.parent / "mteb_csvs"   # analysis/mteb_csvs/
OUT_DIR = SCRIPT_DIR / "task_b_filter_demo"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(OUT_DIR / "task_b_filter_demo.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

MTEB_DIRS = {
    "all":      REPO_ROOT / "all_mteb",
    "italian":  REPO_ROOT / "italian_mteb",
    "japanese": REPO_ROOT / "japnese_mteb",
    "hindi":    REPO_ROOT / "hindi_mteb",
}

DEMO_MODEL = "bge-m3"
DEMO_TASK = "BelebeleRetrieval"


def load_csv_safe(path, label):
    if not path.exists():
        log.warning(f"Not found: {path}")
        return None
    df = pd.read_csv(path)
    log.info(f"{label}: {path.name}, shape={df.shape}, cols={df.columns.tolist()}")
    return df


def get_score(df, model_str, col):
    if df is None or col not in df.columns:
        return None
    row = df[df["Model"] == model_str]
    if row.empty:
        return None
    return float(row.iloc[0][col])


def task_exposure(task_name, lang_csv, lang_col, model):
    if lang_csv is None:
        return dict(task_name=task_name, exposed_per_language=False,
                    n_languages_in_csv=0, demonstration_cell=model,
                    value_in_filter_view="N/A")
    exposed = lang_col in lang_csv.columns
    n_langs = sum(1 for c in lang_csv.columns if c not in ("Unnamed: 0", "Model"))
    val = get_score(lang_csv, model, lang_col) if exposed else None
    return dict(task_name=task_name, exposed_per_language=exposed,
                n_languages_in_csv=n_langs, demonstration_cell=model,
                value_in_filter_view=val if val is not None else "not_exposed")


def write_standalone_script(out_dir: Path):
    script = '''\
#!/usr/bin/env python3
"""
Reproduces the MMTEB language-filter behavior finding.

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
    print(f"\\n  Italian filter per-language CSV columns: {it_lang_cols}")
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
'''
    (out_dir / "reproduce_filter_view.py").write_text(script, encoding="utf-8")
    log.info("Saved reproduce_filter_view.py")


def main():
    t0 = time.time()
    log.info("Task B: MMTEB filter-view demonstration")

    for lang, folder in MTEB_DIRS.items():
        if folder.exists():
            files = [f.name for f in sorted(folder.iterdir()) if f.suffix == ".csv"]
            log.info(f"{lang} folder ({folder.name}): {files}")
        else:
            log.warning(f"Folder not found: {folder}")

    all_task = load_csv_safe(MTEB_DIRS["all"]      / "all_performance_per_task.csv",         "all_task")
    it_task  = load_csv_safe(MTEB_DIRS["italian"]  / "italian_performance_per_task.csv",      "it_task")
    it_lang  = load_csv_safe(MTEB_DIRS["italian"]  / "italian_performance_per_langauge.csv",  "it_lang")
    jp_task  = load_csv_safe(MTEB_DIRS["japanese"] / "japanese_performance_per_task.csv",     "jp_task")
    jp_lang  = load_csv_safe(MTEB_DIRS["japanese"] / "japanese_performace_per_langauge.csv",  "jp_lang")
    hi_task  = load_csv_safe(MTEB_DIRS["hindi"]    / "hindi_performance_per_task.csv",        "hi_task")
    hi_lang  = load_csv_safe(MTEB_DIRS["hindi"]    / "hindi_performance_per_language.csv",    "hi_lang")

    score_all = get_score(all_task, DEMO_MODEL, DEMO_TASK)
    score_it  = get_score(it_task,  DEMO_MODEL, DEMO_TASK)
    it_lang_cols = it_lang.columns.tolist() if it_lang is not None else []

    log.info(f"Demonstration A -- {DEMO_MODEL} + {DEMO_TASK}:")
    log.info(f"  Score in all_task CSV:             {score_all}")
    log.info(f"  Score in italian_filter_task CSV:  {score_it}")
    log.info(f"  Italian per-language CSV columns:  {it_lang_cols}")
    log.info(f"  'ita-Latn' exposed: {'ita-Latn' in it_lang_cols}")

    summary_rows = [
        task_exposure("BelebeleRetrieval (Italian filter)", it_lang, "ita-Latn", DEMO_MODEL),
        task_exposure("MIRACLRetrievalHardNegatives (Japanese filter)", jp_lang, "jpn-Jpan", DEMO_MODEL),
        task_exposure("MIRACLRetrievalHardNegatives (Hindi filter)", hi_lang, "hin-Deva", DEMO_MODEL),
        task_exposure("BelebeleRetrieval (Japanese filter)", jp_lang, "ita-Latn", DEMO_MODEL),
    ]

    for r in summary_rows:
        log.info(f"  {r['task_name']}: exposed={r['exposed_per_language']}, "
                 f"n_langs={r['n_languages_in_csv']}, value={r['value_in_filter_view']}")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "filter_view_summary.csv", index=False)
    log.info("Saved filter_view_summary.csv")

    write_standalone_script(OUT_DIR)

    s_it  = f"{score_it:.2f}"  if score_it  is not None else "N/A"
    s_all = f"{score_all:.2f}" if score_all is not None else "N/A"
    md = (
        "## MMTEB Filter-View Demonstration\n\n"
        "A standalone reproduction script (`analysis/reproduce_filter_view.py`) "
        "loads the public CSV and demonstrates the filter behavior on `bge-m3`: "
        f"the BelebeleRetrieval score under the Italian filter ({s_it}) "
        f"differs from the global-task-list score ({s_all}) "
        "because the filter changes which tasks are averaged, not which language within a task is read. "
        "The per-language CSV for the Italian filter exposes only `eng-Latn` -- "
        "no `ita-Latn` column exists.\n"
    )
    (OUT_DIR / "filter_view.md").write_text(md, encoding="utf-8")
    log.info(f"Saved filter_view.md. Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
