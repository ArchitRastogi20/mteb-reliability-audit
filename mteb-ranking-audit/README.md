# MTEB Multilingual Ranking Audit

The audit measures how well the global MTEB leaderboard score predicts model performance
on individual non-English languages. It uses Kendall τ correlation and inversion rate as
the primary metrics, and defines a "hidden failure" as a model that ranks in the global
top-50% but falls into the local bottom-25% for a specific language.

## Benchmark version

All results use **MTEB(Multilingual)** (the multilingual sub-benchmark of MTEB).
The global predictor is the mean NDCG over the 18 official MTEB(Multilingual) retrieval tasks.

## How to run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install mteb numpy pandas scipy statsmodels matplotlib

python code/01_download.py             # clones results repo, builds per-language CSVs
python code/02_verify.py               # cross-checks IT/JA/HI against reference τ values
python code/03_analyze.py              # restricted tier → results/summary.csv
python code/03_analyze.py --curated    # curated tier → results/summary_curated.csv
python code/03_analyze.py --extended   # extended tier → results/summary_extended.csv
python code/04_plot.py                 # restricted plots
python code/04_plot.py --curated       # curated plots
python code/04_plot.py --extended      # extended plots
python code/05_audit_artifacts.py      # sensitivity_table, tau_vs_roster_tier.png, hidden_failures.csv
python code/06_followup_analyses.py    # snowflake_audit.csv, inversion_vs_task_count.png, hidden_failure_cluster.csv
python code/07_roster_membership.py    # analysis/roster_membership.csv, analysis/curated_local_ranks.csv
python code/08_english_profile.py      # analysis/task_taxonomy.csv, analysis/eng_vs_multi_profile.csv
python code/09_extended_resample.py    # analysis/extended_resample_tau.csv, analysis/resample_summary.json
python code/10_cluster_table.py        # analysis/extended_roster.csv, paper/table_hf_cluster.tex
```

Scripts 08–10 require no new downloads. Script 08 (`08_english_profile.py`) reads
MTEB result JSONs from `mteb_results/results/` (populated by `01_download.py`) or
from a path set in the `MTEB_RESULTS_DIR` environment variable.

Released analysis artifacts referenced in the paper:
- `analysis/extended_roster.csv` — per-language inclusion flags for granite-97m-r2
- `analysis/task_taxonomy.csv`   — 18 MMTEB task classifications with rationale

Total expected runtime: under 30 minutes (dominated by `git clone` of the results repo).

