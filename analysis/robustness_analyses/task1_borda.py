#!/usr/bin/env python3
"""Task 1: Borda count replication of headline tau/inversion finding."""
import logging
import time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

SEED = 20260506
N_BOOT = 10_000
SCRIPT_DIR = Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent          # analysis/
REPO_ROOT = SCRIPT_DIR.parent.parent      # repo root
OUT_DIR = SCRIPT_DIR / "task1_borda"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(OUT_DIR / "task1_borda.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# model_id -> Model column name in all_performance_per_task.csv
MODEL_NAME_MAP = {
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
    "intfloat/e5-small-v2": "e5-small-v2",
    "ibm-granite/granite-embedding-107m-multilingual": "granite-embedding-107m-multilingual",
    "jinaai/jina-embeddings-v5-text-nano": "jina-embeddings-v5-text-nano",
    "intfloat/multilingual-e5-base": "multilingual-e5-base",
    "intfloat/multilingual-e5-large": "multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct": "multilingual-e5-large-instruct",
    "BAAI/bge-m3": "bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0",
    "intfloat/e5-large-v2": "e5-large-v2",
    "microsoft/harrier-oss-v1-0.6b": "harrier-oss-v1-0.6b",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
    "Salesforce/SFR-Embedding-Mistral": "SFR-Embedding-Mistral",
    "intfloat/e5-mistral-7b-instruct": "e5-mistral-7b-instruct",
    "nvidia/llama-embed-nemotron-8b": "llama-embed-nemotron-8b",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
}

ENGLISH_ONLY = {"intfloat/e5-small-v2", "intfloat/e5-large-v2"}
IT_EXCLUDE = {"nvidia/llama-embed-nemotron-8b"}
LANG_MAP = {"italian": "IT", "japanese": "JA", "hindi": "HI"}


def rank_desc(values):
    return stats.rankdata(-np.array(values), method="average")


def inversion_count(pred_ranks, lang_ranks):
    n = len(pred_ranks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pred_ranks[i] - pred_ranks[j]) * (lang_ranks[i] - lang_ranks[j]) < 0:
                count += 1
    return count


def bootstrap_tau_ci(pred_scores, out_scores, rng, n_boot=N_BOOT):
    n = len(pred_scores)
    mat = np.column_stack([pred_scores, out_scores])
    boot_taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = mat[idx]
        rx = rank_desc(s[:, 0])
        ry = rank_desc(s[:, 1])
        boot_taus.append(stats.kendalltau(rx, ry).correlation)
    return np.percentile(boot_taus, [2.5, 97.5])


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Load 18-task public MMTEB data
    all_perf = pd.read_csv(
        ANALYSIS_DIR / "mteb_csvs" / "all_mteb" / "all_performance_per_task.csv"
    )
    task_cols = [c for c in all_perf.columns if c not in ("Unnamed: 0", "Model")]
    log.info(f"all_performance_per_task.csv: {len(all_perf)} rows, {len(task_cols)} tasks")
    log.info(f"Tasks: {task_cols}")

    all_perf = all_perf.set_index("Model")

    # Filter to the 17 evaluated models
    paper_rows = {}
    missing = []
    for mid, mname in MODEL_NAME_MAP.items():
        if mname in all_perf.index:
            paper_rows[mid] = all_perf.loc[mname, task_cols]
        else:
            missing.append((mid, mname))
            log.warning(f"Missing from all_perf: {mname}")

    log.info(f"Found {len(paper_rows)}/17 models; missing: {missing}")
    perf_df = pd.DataFrame(paper_rows).T  # model_id x task
    perf_df.index.name = "model_id"

    # Report any missing values
    for t in task_cols:
        n_miss = perf_df[t].isna().sum()
        if n_miss > 0:
            bad = perf_df[perf_df[t].isna()].index.tolist()
            log.warning(f"Task {t}: {n_miss} missing -> {bad}")

    complete_tasks = [t for t in task_cols if perf_df[t].notna().all()]
    dropped_tasks = [t for t in task_cols if t not in complete_tasks]
    log.info(f"Complete tasks ({len(complete_tasks)}): {complete_tasks}")
    if dropped_tasks:
        log.warning(f"Dropped incomplete tasks ({len(dropped_tasks)}): {dropped_tasks}")

    perf_complete = perf_df[complete_tasks].astype(float)
    n_all = len(perf_complete)

    # Borda scores: rank within our model set (highest score -> rank 1 -> Borda = n_all-1)
    borda_scores = pd.DataFrame(index=perf_complete.index)
    for t in complete_tasks:
        ranks = stats.rankdata(-perf_complete[t].values, method="average")
        borda_scores[t] = n_all - ranks

    borda_total = borda_scores.sum(axis=1)
    log.info(f"Borda total: min={borda_total.min():.1f} max={borda_total.max():.1f}")
    log.info(f"Top-5 by Borda: {borda_total.nlargest(5).index.tolist()}")

    borda_reset = borda_total.rename("borda_total").reset_index()
    borda_reset.columns = ["model_id", "borda_total"]

    unified = pd.read_csv(ANALYSIS_DIR / "analysis_output" / "unified_results.csv")

    rows = []
    for lang, lang_code in LANG_MAP.items():
        sub = unified[unified["language"] == lang].copy()
        sub = sub[~sub["model_id"].isin(ENGLISH_ONLY)]
        if lang == "italian":
            sub = sub[~sub["model_id"].isin(IT_EXCLUDE)]
        n = len(sub)
        total_pairs = n * (n - 1) // 2
        log.info(f"[{lang_code}] n={n}, models: {sub['short_name'].tolist()}")

        sub = sub.merge(borda_reset, on="model_id", how="left")
        n_miss = sub["borda_total"].isna().sum()
        if n_miss > 0:
            bad = sub[sub["borda_total"].isna()]["short_name"].tolist()
            log.warning(f"[{lang_code}] {n_miss} missing Borda -> dropping {bad}")
            sub = sub.dropna(subset=["borda_total"])
            n = len(sub)
            total_pairs = n * (n - 1) // 2

        for pred_name, pred_col in [
            ("MMTEB_Mean_Task", "mteb_agg_score"),
            ("MMTEB_Borda", "borda_total"),
        ]:
            pred_scores = sub[pred_col].values.astype(float)
            out_scores = sub["lang_specific_avg"].values.astype(float)

            pred_ranks = rank_desc(pred_scores)
            lang_ranks = rank_desc(out_scores)

            tau = stats.kendalltau(pred_ranks, lang_ranks).correlation
            ci_lo, ci_hi = bootstrap_tau_ci(pred_scores, out_scores, rng)
            inv_count = inversion_count(pred_ranks, lang_ranks)
            inv_pct = 100.0 * inv_count / total_pairs

            log.info(
                f"[{lang_code}] {pred_name}: tau={tau:.4f} [{ci_lo:.4f},{ci_hi:.4f}] "
                f"inv={inv_count}/{total_pairs} ({inv_pct:.1f}%)"
            )
            rows.append(
                dict(
                    language=lang_code,
                    n_models=n,
                    predictor=pred_name,
                    tau=round(tau, 4),
                    tau_ci_lo=round(float(ci_lo), 4),
                    tau_ci_hi=round(float(ci_hi), 4),
                    inversion_count=inv_count,
                    inversion_total=total_pairs,
                    inversion_pct=round(inv_pct, 2),
                )
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "borda_replication.csv", index=False)
    log.info("Saved borda_replication.csv")

    def get(lang, pred):
        return out_df[(out_df.language == lang) & (out_df.predictor == pred)].iloc[0]

    it_mean = get("IT", "MMTEB_Mean_Task")
    ja_mean = get("JA", "MMTEB_Mean_Task")
    hi_mean = get("HI", "MMTEB_Mean_Task")
    it_borda = get("IT", "MMTEB_Borda")
    ja_borda = get("JA", "MMTEB_Borda")
    hi_borda = get("HI", "MMTEB_Borda")

    borda_inv_avg = (
        it_borda.inversion_pct + ja_borda.inversion_pct + hi_borda.inversion_pct
    ) / 3
    persists = borda_inv_avg > 15.0

    if persists:
        conclusion = (
            "qualitatively matching the Mean(Task) result; the rank-inversion phenomenon "
            "is robust to the choice between MMTEB's two published aggregators"
        )
    else:
        conclusion = (
            "showing materially lower inversion rates than the Mean(Task) baseline; "
            "the finding may be partly aggregator-specific"
        )

    md = (
        "## Borda Count Replication (Section 4.2 / App. F)\n\n"
        "To verify the headline finding is not specific to arithmetic aggregation, "
        f"we recompute MTEB rank using Borda count over the same {len(complete_tasks)} "
        "retrieval tasks and re-run the rank-correlation analysis. "
        f"Borda gives Kendall tau of {it_borda.tau:.2f} / {ja_borda.tau:.2f} / {hi_borda.tau:.2f} "
        f"for IT/JA/HI (95% CI: [{it_borda.tau_ci_lo:.2f}, {it_borda.tau_ci_hi:.2f}] / "
        f"[{ja_borda.tau_ci_lo:.2f}, {ja_borda.tau_ci_hi:.2f}] / "
        f"[{hi_borda.tau_ci_lo:.2f}, {hi_borda.tau_ci_hi:.2f}]) "
        f"with {it_borda.inversion_pct:.1f}% / {ja_borda.inversion_pct:.1f}% / {hi_borda.inversion_pct:.1f}% "
        f"inversions, {conclusion}. "
        f"For reference, Mean(Task) gives tau = {it_mean.tau:.2f} / {ja_mean.tau:.2f} / {hi_mean.tau:.2f} "
        f"with {it_mean.inversion_pct:.1f}% / {ja_mean.inversion_pct:.1f}% / {hi_mean.inversion_pct:.1f}% inversions.\n"
    )
    (OUT_DIR / "borda_replication.md").write_text(md, encoding="utf-8")
    log.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
