# Fresh-snapshot robustness check (P9)

Fresh sparse clone of the public results repository: commit `044b132709ad4f080ecb46c7289336b88b1412a6` (2026-07-09), vs the frozen audit clone `9490b40` (2026-05-08). Only the 25 curated model dirs were checked out (`git clone --filter=blob:none --sparse` + `git sparse-checkout set`, clone at `C:\tmp\results_fresh_2026_07`). Roster and the 18-task global-aggregate definition held fixed at the audit's frozen values; only scores are fresh. Extraction replicates `01_download.py` (ndcg_at_10; consensus tasks >=50% of models-with-data; full-coverage lang-avg; lexicographically-last revision dir, recorded per model in `fresh_revisions.csv`).

## Result: full replication

| Metric | Frozen (2026-05) | Fresh (2026-07) |
|---|---|---|
| Median inversion (curated) | 0.330 | 0.330 |
| Median tau | 0.340 | 0.340 |
| Coverage rho(n_t, inversion) | -0.59 | -0.581 (p=0.014) |
| Hidden-failure cells | 16/10/6/3 (=35) | 16/10/6/3 (=35, same models) |
| Max per-language tau shift | — | ±0.022 (fas) |

All 25 curated models retain complete 18-task global coverage in the fresh snapshot. 6 of 17 languages are bit-identical (ara, ben, hin, rus, spa, vie); 11 shift by <=0.022 tau / <=0.011 inversion.

Files: `fresh_snapshot_check.py`, `fresh_snapshot_summary.csv`, `fresh_hidden_failures.csv`, `fresh_revisions.csv`. Add all to the released repo at resubmission.

## Re-running this script

`fresh_snapshot_check.py` reads its fresh clone from the `FRESH_RESULTS_DIR`
environment variable (it raises a clear error if unset) rather than a
hardcoded path, since the clone itself is not shipped in this release tree
(third-party, large, and machine-local). To re-run the check: take a fresh
sparse clone of the public results repository
(`https://github.com/embeddings-benchmark/results`) at a later snapshot date
than the frozen audit clone (`9490b40`, 2026-05-08) -- e.g. with
`git clone --filter=blob:none --sparse` + `git sparse-checkout set` for just
the 25 curated model dirs, as was done for the 044b132 (2026-07-09) clone
above -- then set `FRESH_RESULTS_DIR` to that clone's `results/` directory
before running the script.
