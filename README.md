# A Reliability Audit of Aggregated MTEB Retrieval Scores

Code, configuration, and released results for:

> **A Reliability Audit of Aggregated MTEB Retrieval Scores: Coverage Asymmetry and Language-Specific Rank Inversions Across 17 Languages**
> Archit Rastogi. *AACL-IJCNLP 2026* (main conference).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What this is

Practitioners pick multilingual embedding models off the MTEB leaderboard's
headline aggregated score. This repository contains the audit showing that the
same global score can correspond to very different per-language behaviour, and
that the disagreement is large enough — at deployment-realistic roster sizes —
to change which model you would choose.

Two quantities do the work throughout:

| Term | Definition |
|---|---|
| **MTEB-agg** | The headline leaderboard number: a model's mean over every retrieval task it was evaluated on. |
| **Lang-avg** | The same model's mean over only the retrieval tasks that exist for *one* language. |
| **Pairwise inversion rate** | The fraction of model pairs the two orderings disagree about. 0% = identical rankings; ~50% = unrelated rankings. |

### Headline findings

| Finding | Number | Reproduced by |
|---|---|---|
| Median pairwise inversion rate, deployment-relevant roster (n=25) | **33%** | `mteb-ranking-audit/results/summary_curated.csv` |
| Per-language Kendall τ reliable after Benjamini–Hochberg FDR | **11 / 17 languages** | `mteb-ranking-audit/results/sensitivity_table.csv` |
| Broad MMTEB-eligible tier agreement (composition effect) | **τ = 0.69** | `mteb-ranking-audit/analysis/resample_summary.json` |
| Inversion rate falls as a language has more retrieval tasks | **ρ = −0.59** (curated) | `mteb-ranking-audit/results/summary_curated.csv` |
| **Mitigation:** language-balanced aggregator (per-language mean, then macro-average) | **33% → 14.5%**, τ = 0.71 (leave-one-out) | `extended-experiments/outputs/exp7_summary_by_aggregator.csv` |
| Hidden-failure cluster: globally top-half, locally bottom-quartile | 5 models; `granite-311m` in **16/17** languages | `mteb-ranking-audit/results/hidden_failure_cluster.csv` |

---

## Start here: reproduce the headline result on a laptop

**The core audit needs no GPU, no API key, and no paid service.** It is pure
pandas/numpy/scipy over a public leaderboard snapshot, and runs in roughly
30 minutes.

It does need **network and disk**: step one shallow-clones the public MTEB
results repository, which is multi-gigabyte and dominates the runtime. Every
step after it is offline and takes seconds.

```bash
git clone https://github.com/ArchitRastogi20/mteb-reliability-audit.git
cd mteb-reliability-audit

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install mteb numpy pandas scipy statsmodels matplotlib

cd mteb-ranking-audit/code
python 01_download.py        # clones the public MMTEB results snapshot (~30 min)
python 02_verify.py          # cross-checks downloaded scores against the paper
python 03_analyze.py         # Kendall tau + bootstrap CI + inversion rate
python 03_analyze.py --curated
python 03_analyze.py --extended
python 05_audit_artifacts.py # -> ../sensitivity_table.csv  (Table 1; see Known issues 7)
python 06_followup_analyses.py
```

That gives you Table 1 — the 17-language × 3-tier audit — and the
hidden-failure cluster, from scratch.

Then reproduce the **mitigation** (also CPU-only, seconds):

```bash
cd ../../extended-experiments
pip install -r requirements.txt
python exp7_alt_aggregators.py   # -> outputs/exp7_summary_by_aggregator.csv
```

Everything else in this repository — the own-run GPU sweeps, the RAG benchmark,
the deployment benchmark — is supporting evidence that requires hardware,
money, or both. See [Cost and hardware](#cost-and-hardware).

---

## Repository layout

Six sub-projects, each corresponding to one track of the paper.

```
mteb-reliability-audit/
├── mteb-ranking-audit/        # ★ Core three-tier rank-inversion audit (CPU only)
├── extended-experiments/      # ★ Robustness battery + the headline mitigation (CPU only)
├── analysis/                  # Aggregation, statistics, figures, contamination check
│   └── camera_ready_analyses/ #   Late analyses added for the camera-ready
├── mteb-language-gap/         # Own-run MMTEB sweep for IT/JA/HI/SW   (GPU)
├── rag-dataset/               # Multilingual RAG benchmark + eval     (GPU + paid API)
├── rag-deployment-benchmark/  # Latency / throughput / VRAM sweep     (GPU)
├── .env.example               # Required environment variables (no secrets)
├── CITATION.cff
└── LICENSE                    # MIT
```

★ = runnable with no GPU and no API key. These two sub-projects produce every
number in the paper's abstract.

---

### `mteb-ranking-audit/` — the core audit

The three-tier audit itself. Downloads the public MMTEB(Multilingual)
leaderboard snapshot (18 retrieval tasks, 2026-05) and compares aggregated
rankings against per-language rankings across three nested model rosters:
restricted (n=7), curated (n=25), and extended (n=28–57).

| Script | Purpose | Cost |
|---|---|---|
| `01_download.py` | Fetch the MMTEB snapshot — MTEB SDK first, falls back to cloning the results repo | network, ~30 min |
| `02_verify.py` | Cross-check downloaded scores against the paper's Table 5 across all tiers | cheap |
| `03_analyze.py` | Kendall τ + bootstrap CI, inversion rate, top-k regret. `--curated` / `--extended` select tier | cheap |
| `04_plot.py` | Summary plots from `results/summary*.csv` | cheap |
| `05_audit_artifacts.py` | **Table 1** (`sensitivity_table.csv`), `tau_vs_roster_tier.png`, `hidden_failures.csv` | cheap |
| `06_followup_analyses.py` | `hidden_failure_cluster.csv`, `task_asymmetry_table.csv`, `snowflake_audit.csv` | cheap |
| `07_roster_membership.py` | Roster membership + per-language local ranks | cheap |
| `08_english_profile.py` | English-vs-multilingual task profile for the cluster | cheap |
| `09_extended_resample.py` | 100 random size-25 draws from the extended tier per language | cheap |
| `10_cluster_table.py` | Hidden-failure cluster table (global rank vs per-language rank) | cheap |

Run `01` → `10` in numeric order. Everything after `01` states "no new
downloads required".

> **On roster membership.** The curated roster is a **hand-selected,
> judgment-based, deployment-relevant list**, hardcoded as `MODELS_CURATED` /
> `CURATED_ONLY`. It is not computed by any script from a coverage or
> top-quartile rule. The released 25-model list *is* the definition — which is
> why it ships in full, in `analysis/roster_membership.csv`. The argument rests
> instead on three guards: the roster-sensitivity table
> (`extended-experiments/outputs/exp8_roster_summary.csv`), the judgment-free
> extended tier, and the fresh-snapshot replication.

---

### `extended-experiments/` — robustness battery and the mitigation

Fourteen independent experiments over pre-aggregated CSVs. `run_all.py` runs
them in priority order; each is independent, so a failure in one does not block
the others. Pure NumPy/SciPy — **except** `exp11`, which makes paid API calls.

| Script | Purpose |
|---|---|
| `exp1_multilingual_only_aggregator.py` | Does restricting the aggregator to the 7 multilingual tasks help? |
| `exp2_cluster_boxplot.py` | English-vs-multilingual gap, cluster vs non-cluster |
| `exp3_stricter_english_taxonomy.py` | Robustness to a stricter 6-task English list |
| `exp4_null_baseline.py` | Monte-Carlo null distribution of inversion rate (10,000 draws) |
| `exp5_threshold_sensitivity.py` | Hidden-failure (θ_global, θ_local) threshold grid |
| `exp6_per_language_regret.py` | Per-language top-1/3/5 selection regret |
| **`exp7_alt_aggregators.py`** | **The paper's headline mitigation: 14.5% / τ=0.71** |
| `exp8_roster_sensitivity.py` | Four roster variants (Table `tab:roster-sensitivity`) |
| `exp9_curated_rank_scatter.py` | Global vs local rank scatter, all 17 languages |
| `exp10_paired_bootstrap_contrasts.py` | Paired bootstrap (B=10,000) for the task-fixed probes |
| `exp11_closed_book_baseline.py` | Closed-book RAG baseline — **paid API**, hard $10 ceiling, SQLite cache prevents re-spend |
| `exp12_alt_aggregators_mmteb.py` | Five aggregators over the 18-task subset |
| `exp13_params_vs_querytype_corr.py` | log(params) vs per-query-type NDCG@10 — ⚠ see Known issues |
| `exp14_mteb_to_rag_propagation.py` | MTEB-agg rank vs RAG-coverage rank, all 6 configs — ⚠ see Known issues |

> **On the aggregator's discovery order.** The language-macro aggregator was
> found *post hoc*, not pre-registered. `exp1` tested only Mean(18) /
> Mean(7 multilingual) / Mean(11 English); `lang_macro` first appears two hours
> later in `exp7`, alongside a competing 1/n_t reweighting. The guards against
> over-fitting are leave-one-out evaluation, consistency across all 17
> languages, and replication on the judgment-free extended tier. The
> leave-one-out variant is headlined — rather than the self-inclusive one,
> which scores better at 13.7% — because the self-inclusive variant leaks the
> target language into its own predictor.

---

### `analysis/` — aggregation, statistics, figures

The final aggregation layer. Consumes outputs from every other sub-project and
produces the paper's figures plus most `analysis_output/stats/*` files.

```bash
cd analysis
python analysis.py                 # main aggregation pass
python analysis_revisions.py       # core revision analyses (P0, M-series)
python analysis_revisions_R1.py    # monolingual-controlled view
python analysis_revisions_S4.py    # query-type-ratio sensitivity
python p0_revisions.py             # MMTEB filtered-view replication + extra languages
python contamination_check.py      # 8-gram overlap vs Wikipedia and Belebele
python regen_figures.py
python integrate_api_results.py    # merge Cohere / Gemini / Voyage rows
python integrate_openai_results.py
```

`robustness_analyses/` holds six standalone appendix checks: Borda replication,
per-pair significance, own-run τ, Nemotron query-type triangulation, MIRACL-only
probe, and the MMTEB filter-view demonstration. All are cheap except
`task_c_per_query_bootstrap.py`, which requires a GPU.

#### `analysis/camera_ready_analyses/`

Five analyses added late, during camera-ready preparation, each cited by the
final paper. Their output CSVs ship alongside the code, so the numbers are
inspectable even where the script needs an input this repository does not carry.

| Analysis | Backs | Runnable from this repo? |
|---|---|---|
| `tau_b_permutation.py` | §4.3 coverage–inversion permutation test (τ_b = −0.481 curated, −0.587 extended) | yes |
| `aggregator_extended/` | Extended-tier aggregator sweep (15.8% → 7.2%) | yes |
| `cross_category/` | Cross-category audit — classification, STS, reranking | after `01_download.py` |
| `fork_concordance/` | Concordance with language-specific MTEB forks | after `01_download.py` |
| `fresh_snapshot/` | Replication on a later leaderboard snapshot | needs a fresh clone; see its `NOTES.md` |

The last three need a local clone of the public MTEB results repository, which
is third-party and too large to vendor here. `mteb-ranking-audit/code/01_download.py`
produces exactly that clone. `fresh_snapshot/` additionally needs a *second*
clone taken at a later date, supplied via the `FRESH_RESULTS_DIR` environment
variable — it tests snapshot stability, so it cannot be reproduced from a single
point in time.

---

### `mteb-language-gap/` — own-run MMTEB sweep (GPU)

The evaluation harness behind the paper's own-run Italian/Japanese/Hindi (and
Swahili) results — ~17 open-weight encoders plus BM25, with a hardware-aware
batch-size policy in `config/hardware.yaml`.

```bash
cd mteb-language-gap
pip install -r requirements.txt
python scripts/cache_datasets.py     # pre-download MTEB datasets
python scripts/eval_lb.py            # the main per-language sweep  (GPU, hours)
python scripts/eval_bm25.py          # BM25 baselines               (CPU)
python scripts/make_lb_table.py      # per-language comparison tables
```

Raw per-model JSON lands in `results/{lang}_lb/` and `results/bm25/`. The
`results/ita_lb_maxseq{128,256,512,1024}/` directories hold the chunker /
max-sequence-length sensitivity sweep.

---

### `rag-dataset/` — multilingual RAG benchmark (GPU + paid API)

A five-stage synthetic-corpus generator (`src/pipeline/`) producing six
configurations — Italian/Japanese/Hindi × finance/law — followed by retrieval
and downstream answer-quality evaluation.

**The generated corpus is not distributed with this repository** (see
[What is not included](#what-is-not-included)). Only the code and the per-model
evaluation *results* under `output/evaluation/` ship here.

```bash
python -m src.pipeline.orchestrator      # regenerate the corpus (paid API, budget-capped $25)
python scripts/eval_rag.py               # retrieval eval        (GPU)
python scripts/eval_bm25.py              # BM25 baseline         (CPU)
python scripts/eval_downstream_rag.py    # answer-quality eval   (GPU + paid API)
python scripts/eval_hybrid_rrf.py        # RRF hybrid fusion
```

`config.yaml` sets `docs_per_combo: 250`, per-query-type targets
(factual 175 / multi-hop 175 / summarization 100 / unanswerable 50) and
`budget_usd: 25.0`. Note the 500-query figure is the *target*; realised counts
after validation gating are 375–472, mean 444.

> **`judge_llm_sdk` is a stub, not a real package.** `eval_R4b_judge.py` and
> `eval_R5c_open_weight_propagation.py` import a placeholder judge client.
> Replace the import and instantiation with the SDK of whatever LLM you use as
> a judge before running them; they cannot execute as shipped.

---

### `rag-deployment-benchmark/` — deployment cost (GPU)

Measures index-build throughput, query throughput, single-query latency
(p50/p99) and peak VRAM for each model on one GPU.

```bash
python download_models.py
python benchmark.py --all
```

Released results were produced on a single **NVIDIA RTX 3090** (25.3 GB usable,
CUDA 12.8, PyTorch 2.8), 3 runs per model, 1,000 docs and 1,744 queries.
`Qwen3-Embedding-8B` and `llama-embed-nemotron-8b` are absent from the results:
both OOM at this VRAM tier, which is itself a reported finding.

---

## Artifact → paper map

The camera-ready `main.tex` carries 172 inline `% PROV:` provenance comments
recording which released artifact produces which printed number. The most
important mappings:

| Paper element | Artifact |
|---|---|
| **Table 1** — 17 languages × 3 tiers (τ, inv%, n) | `mteb-ranking-audit/results/sensitivity_table.csv` |
| Abstract — 33% median inversion | `mteb-ranking-audit/results/summary_curated.csv`, median of 17 rows |
| Abstract — extended-tier τ = 0.69 | `mteb-ranking-audit/analysis/resample_summary.json` |
| Abstract / §4.3 — 33% → 14.5% mitigation | `extended-experiments/outputs/exp7_summary_by_aggregator.csv` (`lang_macro_18lang_loo`) |
| **Table 3** — hidden-failure cluster | `mteb-ranking-audit/results/hidden_failure_cluster.csv`, `hidden_failures.csv` |
| **Table 4** / App. — English-vs-multilingual profile | `mteb-ranking-audit/analysis/eng_vs_multi_profile.csv` |
| §4.3 — coverage predicts inversion (ρ = −0.59 / −0.67) | `summary_curated.csv`, `summary_extended.csv` |
| Task taxonomy — 11 English-derived / 7 multilingual | `mteb-ranking-audit/analysis/task_taxonomy.csv` |
| Roster tiers — n = 7 / 25 / 28–57 | `mteb-ranking-audit/analysis/roster_membership.csv` |
| `tab:roster-sensitivity` — 4 roster variants | `extended-experiments/outputs/exp8_roster_summary.csv` |
| `tab:probes` — task-fixed paired bootstrap | `extended-experiments/outputs/exp10_paired_bootstrap.csv` |
| App. — Borda replication | `analysis/robustness_analyses/task1_borda/borda_replication.csv` |
| **Table 6** — RAG results, 17 models × 6 configs | `rag-dataset/output/evaluation/<model>/*.json` |
| §5 — closed-book baseline, $3.92 / 4,838 calls | `extended-experiments/outputs/exp11_closed_book_summary.json` |
| §5 — MTEB-to-RAG propagation ρ | `extended-experiments/outputs/exp14_mteb_to_rag_propagation.csv` |
| App. — deployment benchmark, 126 cells | `rag-deployment-benchmark/results/deployment_results_merged.csv` |
| App. — index-build cost (≈2.5× overhead) | `analysis/analysis_output/stats/rev_S12_index_build.json` |
| App. — API deployment latency/pricing, n=50 | `analysis/analysis_output/stats/rev_S1_api_deployment.json` |

---

## Cost and hardware

| Track | GPU | Paid API | Rough cost |
|---|---|---|---|
| `mteb-ranking-audit/` | — | — | free, ~30 min |
| `extended-experiments/` (except `exp11`) | — | — | free, seconds–minutes |
| `extended-experiments/exp11` | — | OpenAI + judge | $3.92 actual, $10 hard ceiling |
| `analysis/` | — (except `task_c_per_query_bootstrap.py`) | — | free |
| `mteb-language-gap/` | **yes** | — | hours of GPU time |
| `rag-dataset/` generation | — | OpenAI | budget-capped $25 |
| `rag-dataset/` evaluation | **yes** | varies by script | — |
| `rag-deployment-benchmark/` | **yes** | — | hours of GPU time |

Total compute reported in the paper is ~35 GPU-hours; the per-run timers that
ship in this release sum to ~8 GPU-hours (the MMTEB sweeps, index builds and
model downloads were not individually timed).

Released runs used a single CUDA GPU (24 GB-class; 25.3 GB usable) with ~270 GB
system RAM and 128 CPU cores — see `mteb-language-gap/config/hardware.yaml` for
the exact batch-size tier policy.

### Environment

Python **3.11+** (`rag-dataset/pyproject.toml`). Four `requirements.txt` files
cover the sub-projects independently; install only what you intend to run.
`mteb-ranking-audit/` and `analysis/` have no requirements file — they need
`mteb numpy pandas scipy statsmodels matplotlib`.

```bash
cp .env.example .env    # then fill in only the keys you need
```

| Variable | Needed by |
|---|---|
| `OPENAI_API_KEY` | RAG corpus generation, all OpenAI embedding evals, downstream RAG, `exp11` |
| `CHATGPT_API_KEY` | alias used by some scripts; usually equal to `OPENAI_API_KEY` |
| `JUDGE_API_KEY` | the judge-LLM stub scripts (also need a real SDK substituted) |
| `COHERE_API_KEY` | `rag-dataset/scripts/eval_api_embed.py`, `analysis/integrate_api_results.py` |
| `GOOGLE_API_KEY` | Gemini embedding API |
| `VOYAGE_API_KEY` | optional; only to re-run the Voyage rows |

---

## Tests

34 unit-test files across four sub-projects. `mteb-language-gap/tests/` mocks
`SentenceTransformer` with a seeded 8-dim fixture and provides an ephemeral
ChromaDB, so most tests run **without a GPU or any model download**.

```bash
cd rag-dataset      && pytest      # ships a pytest.ini (asyncio_mode=auto)
cd mteb-language-gap && pytest tests/
```

`mteb-ranking-audit/` and `rag-deployment-benchmark/` have no test suite.

---

## What is not included

Stated plainly, so you know what you cannot reproduce from this tree alone:

- **The generated RAG corpus.** Only the generation code and the per-model
  evaluation results ship. The corpus is available from the author on request;
  `src/pipeline/orchestrator.py` regenerates it for ~$25 in API spend.
- **A working judge-LLM client.** `judge_llm_sdk` is a stub (see above).
- **The native-speaker annotation artifacts.** The annotation files and
  spot-check logs behind the paper's annotation appendix are not part of this
  release.
- **A local clone of the public MTEB results repository**, needed by the
  cross-category and fork-concordance analyses. It is third-party and too large
  to vendor; `mteb-ranking-audit/code/01_download.py` fetches it for you.
- **A second, later leaderboard snapshot**, needed by the fresh-snapshot
  replication. That analysis measures stability *across* snapshots, so it
  cannot be reproduced from one point in time.
- **Italian leaderboard CSVs.** See Known issues.

---

## Known issues

These are recorded rather than hidden. Most were found by the author's own
provenance audit during camera-ready preparation.

1. **Two Italian leaderboard CSVs are wrong.**
   `italian_performance_per_task.csv` and `italian_summary.csv` are
   byte-identical to their Japanese counterparts — their task columns are
   Japanese's task set, and MIRACL has no Italian split at all. The correct
   Italian per-task and summary extracts are **not present in this release**.
   Both files are quarantined in `analysis/mteb_csvs/KNOWN_BAD/` rather than
   deleted, so the defect stays inspectable; see that directory's README. The
   own-run Italian results in `mteb-language-gap/results/ita_lb/` are a
   separate pipeline and are unaffected.

   Note that `italian_performance_per_langauge.csv` is **not** part of this
   defect, despite having only an `eng-Latn` column. That absence is the
   paper's §4.5 finding — the MMTEB Italian filter view exposes no per-Italian
   score — and the file is correct evidence for it.
2. **`mteb-ranking-audit/code/11`–`14` are legacy near-duplicates** of scripts
   `07`–`10`, written against a different directory layout. They are not in the
   sub-project's canonical `01`→`10` run order and are not runnable as shipped.
   Use `07`–`10`.
3. **`exp13` and `exp14` reference paths outside this tree** and are not
   runnable as pathed. `exp14`'s output CSV ships, so the paper's numbers
   remain inspectable.
4. **Hardcoded `/workspace` paths** appear in `mteb-language-gap/scripts/`
   (`cache_datasets.py`, `model_pipeline.py`, `utils.py`) and
   `rag-dataset/scripts/`. These assume a Linux container mount and will need
   editing on any other machine.
5. **One appendix table row has no producing script.** The Belebele-only rows
   were computed from the per-language CSVs, but the script that did it was not
   located during the provenance audit.
6. **`extended-experiments/tests/` is empty** — the sub-project that produces
   the headline mitigation number has no unit tests.
7. **A fresh run writes its outputs one directory above the released copies.**
   `03_analyze.py`, `05_audit_artifacts.py` and `06_followup_analyses.py` write
   `summary*.csv`, `sensitivity_table.csv` and `hidden_failures.csv` to the
   `mteb-ranking-audit/` root; the released copies of those same artifacts live
   in `mteb-ranking-audit/results/`, which is where this README's artifact map
   points. The scripts agree with each other, so the pipeline runs end to end —
   but compare a fresh run against `results/` rather than expecting it to
   overwrite that directory.
8. **`exp5_threshold_sensitivity.py` reports a 4-model cluster, not 5.** This is
   correct, not a failed reproduction. `exp5` sweeps the **curated tier (n=25)**
   only, and `granite-97m-r2` is not in the curated roster — it has 0
   curated-tier affected languages and 5 extended-tier ones
   (`hidden_failure_cluster.csv`). The paper's five-model cluster is four
   curated-tier models plus that one extended-tier model. Note that
   `extended-experiments/outputs/SUMMARY.md` prints
   "Headline cluster composition differs from manuscript: missing
   {'granite-97m-r2'}" without this context — the tier scope is the explanation.
9. **`extended-experiments/outputs/SUMMARY.md` is an append log, not a report.**
   `run_all.py` appends on every run, so the file holds several stacked copies
   of each experiment block from different points in the project's history, and
   some values drift between blocks (Mean(18) median inversion appears as both
   0.327 and 0.330; the 6-task strict-taxonomy p as 0.0049, 0.0205 and 0.0243).
   **The CSVs in `outputs/` are authoritative**; `SUMMARY.md` is kept only as a
   run trace. The paper's cited values come from the CSVs.

---

## Citation

```bibtex
@inproceedings{rastogi2026reliability,
  title     = {A Reliability Audit of Aggregated {MTEB} Retrieval Scores:
               Coverage Asymmetry and Language-Specific Rank Inversions
               Across 17 Languages},
  author    = {Rastogi, Archit},
  booktitle = {Proceedings of the 2026 Joint Conference of the Asia-Pacific
               Chapter of the Association for Computational Linguistics and the
               International Joint Conference on Natural Language Processing
               (AACL-IJCNLP 2026)},
  year      = {2026}
}
```

## License

[MIT](LICENSE) © 2026 Archit Rastogi.

The MIT licence covers the code and the analysis outputs produced by it.

It does **not** relicense third-party material that those outputs are derived
from. The released CSV/JSON results contain scores computed over public
benchmark datasets — including BelebeleRetrieval, MIRACL, MLQA and
WikipediaRetrievalMultilingual, reached through the `mteb` package — and scores
attributed to named third-party embedding models. Those datasets and model
weights remain under their own licences and terms, held by their respective
publishers; consult each dataset card and model card before redistributing or
building on the corresponding rows. No third-party source code or model weights
are vendored in this repository.
