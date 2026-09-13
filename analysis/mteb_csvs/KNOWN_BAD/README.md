# KNOWN_BAD — mislabelled "Italian" CSVs

Two files that were shipped under `analysis/mteb_csvs/italian_mteb/` are wrong.
They are quarantined here rather than deleted, so the release does not lose the
record that they existed and were checked.

## What is wrong with each file

- **`italian_performance_per_task.csv`** — byte-identical (md5
  `e6d8e841285b544c1a24e85e9ae25c45`) to
  `analysis/mteb_csvs/japnese_mteb/japanese_performance_per_task.csv`. Its
  columns are `BelebeleRetrieval` and `MIRACLRetrievalHardNegatives` —
  Japanese's task set for this audit, not Italian's (Italian's is Belebele +
  Wikipedia-based retrieval; MIRACL has no Italian split at all). This is
  Japanese data mislabelled as Italian.

- **`italian_summary.csv`** — byte-identical (md5
  `9e4853d296869883b01c3bdd2e04d4cb`) to
  `analysis/mteb_csvs/japnese_mteb/japanese_summary.csv`. Same problem: the
  Japanese summary, mislabelled as Italian.

**The correct Italian per-task and summary extracts are not present anywhere in
this release.** No replacement data has been fabricated or backfilled. Anyone
needing them must re-pull from the MMTEB leaderboard/results repo at the
snapshot documented in `mteb-ranking-audit/code/01_download.py` (results-repo
commit dated 2026-05-08, cloned 2026-05-09).

## What is NOT wrong: `italian_performance_per_langauge.csv`

That file was briefly quarantined here and has been **restored** to
`analysis/mteb_csvs/italian_mteb/`. It is correct data, and it is in fact
evidence for one of the paper's claims.

It has a single score column, `eng-Latn`, and no `ita-Latn` column. That looks
like a defect at first glance, but it is the expected and intended content:

| Filter view | Language columns exposed |
|---|---|
| Hindi | `ara-Arab, deu-Latn, eng-Latn, hin-Deva, spa-Latn, vie-Latn, zho-Hans` |
| Japanese | `eng-Latn, jpn-Jpan` |
| **Italian** | **`eng-Latn` only** |

The absence of `ita-Latn` is the finding: the MMTEB "Italian filter" view does
not expose a per-Italian retrieval score. This is precisely what
`analysis/reproduce_filter_view.py` and
`analysis/robustness_analyses/task_b_filter_demo.py` were written to
demonstrate.

Two further checks confirm the file is sound rather than spurious: its 180
models are a **complete subset** of the 215 models in the Japanese
per-language file (zero models appear in the Italian file that do not appear in
the Japanese one), and its structure is identical to its siblings in every
respect except the missing Italian column.

## Consumers

Four scripts referenced these filenames. Their paths assumed an older layout in
which the per-language folders sat at the repository root rather than under
`analysis/mteb_csvs/`; all four have since been repointed:

- `analysis/analysis.py`
- `analysis/reproduce_filter_view.py`
- `analysis/robustness_analyses/task_b_filter_demo.py`
- `mteb-language-gap/scripts/lb_evaluator.py`

`task_b_filter_demo.py` runs to completion and reproduces the filter-view
demonstration, degrading gracefully over the two missing files.
`reproduce_filter_view.py` asserts on the missing per-task file and stops —
deliberately, since silently substituting Japanese data would produce Japanese
numbers labelled Italian.
