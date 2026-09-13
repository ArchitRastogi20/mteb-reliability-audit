"""
Per-language analysis: Kendall τ + bootstrap CI, inversion rate, top-k regret,
top-1 loss. Produces per-language analysis.json and master summary.csv.

Usage:
  python 03_analyze.py            # restricted 15-model roster
  python 03_analyze.py --extended # extended roster (all MTEB models with 4+ tasks)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests

EXTENDED = "--extended" in sys.argv
CURATED  = "--curated"  in sys.argv

_suffix = "_extended" if EXTENDED else ("_curated" if CURATED else "")

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "code" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

RNG_SEED = 42
N_BOOTSTRAP = 10_000
TOP_K_VALUES = [1, 3, 5]

LANG_META = {
    "ita": ("Italian",    "case_studies"),
    "jpn": ("Japanese",   "case_studies"),
    "hin": ("Hindi",      "case_studies"),
    "ara": ("Arabic",     "audit_languages"),
    "zho": ("Chinese",    "audit_languages"),
    "deu": ("German",     "audit_languages"),
    "spa": ("Spanish",    "audit_languages"),
    "rus": ("Russian",    "audit_languages"),
    "fra": ("French",     "audit_languages"),
    "kor": ("Korean",     "audit_languages"),
    "ben": ("Bengali",    "audit_languages"),
    "vie": ("Vietnamese", "audit_languages"),
    "ind": ("Indonesian", "audit_languages"),
    "fas": ("Persian",    "audit_languages"),
    "swa": ("Swahili",    "audit_languages"),
    "tha": ("Thai",       "audit_languages"),
    "tel": ("Telugu",     "audit_languages"),
    "tam": ("Tamil",      "audit_languages"),
}


def lang_folder(iso3: str) -> Path:
    name, subdir = LANG_META[iso3]
    return BASE / subdir / iso3


def load_data(iso3: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    folder = lang_folder(iso3)
    lang_path   = folder / f"lang_avg{_suffix}.csv"
    global_path = BASE / "global" / f"mteb_agg{_suffix}.csv"
    if not lang_path.exists() or not global_path.exists():
        return None, None
    lang_df   = pd.read_csv(lang_path,   encoding="utf-8")
    global_df = pd.read_csv(global_path, encoding="utf-8")
    merged = lang_df.merge(global_df[["model", "mteb_agg"]], on="model", how="inner")
    return merged, lang_df


def midpoint_rank(series: pd.Series) -> np.ndarray:
    return series.rank(method="average").values


def bootstrap_tau(mteb_ranks: np.ndarray, lang_ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(mteb_ranks)
    taus = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        t, _ = kendalltau(mteb_ranks[idx], lang_ranks[idx])
        taus[i] = t
    return taus


def compute_inversions(df: pd.DataFrame) -> pd.DataFrame:
    models = df["model"].tolist()
    mteb_vals = df["mteb_agg"].values
    lang_vals = df["lang_avg"].values
    rows = []
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            mteb_order = mteb_vals[i] > mteb_vals[j]  # True if i > j in MTEB
            lang_order  = lang_vals[i]  > lang_vals[j]   # True if i > j in lang
            is_inv = mteb_order != lang_order
            rows.append({
                "model_a": models[i],
                "model_b": models[j],
                "mteb_a":  float(mteb_vals[i]),
                "mteb_b":  float(mteb_vals[j]),
                "lang_a":  float(lang_vals[i]),
                "lang_b":  float(lang_vals[j]),
                "is_inversion": int(is_inv),
            })
    return pd.DataFrame(rows)


def top_k_regret(df: pd.DataFrame, k: int) -> dict:
    sorted_by_lang  = df.sort_values("lang_avg", ascending=False)
    sorted_by_mteb  = df.sort_values("mteb_agg", ascending=False)

    oracle_top_k    = sorted_by_lang.head(k)
    mteb_top_k      = sorted_by_mteb.head(k)

    oracle_best     = float(oracle_top_k["lang_avg"].max())
    mteb_best       = float(mteb_top_k["lang_avg"].max())
    regret          = oracle_best - mteb_best

    oracle_expected = float(oracle_top_k["lang_avg"].mean())
    mteb_expected   = float(mteb_top_k["lang_avg"].mean())
    expected_loss   = oracle_expected - mteb_expected

    return {
        "regret":        regret,
        "expected_loss": expected_loss,
        "oracle_best":   oracle_best,
        "mteb_best":     mteb_best,
    }


def analyze_language(iso3: str) -> dict | None:
    merged, lang_df = load_data(iso3)
    if merged is None or len(merged) < 4:
        log.info("%s: insufficient data (n=%s) — skipped", iso3, len(merged) if merged is not None else 0)
        return None

    n_models = len(merged)
    n_tasks  = int(merged["n_tasks"].iloc[0]) if "n_tasks" in merged.columns else 0

    # Threshold: ≥6 models, ≥2 tasks. The paper-roster threshold of 12 can't be met since
    # only ~7 of the 15 paper models have submitted multilingual retrieval results to MTEB.
    if n_models < 6 or n_tasks < 2:
        log.info("%s: n_models=%d n_tasks=%d — below threshold, skipped", iso3, n_models, n_tasks)
        return None

    folder = lang_folder(iso3)

    mteb_ranks = midpoint_rank(merged["mteb_agg"])
    lang_ranks = midpoint_rank(merged["lang_avg"])

    tau, pval = kendalltau(mteb_ranks, lang_ranks)
    tau, pval = float(tau), float(pval)

    rng = np.random.default_rng(RNG_SEED)
    boot_taus = bootstrap_tau(mteb_ranks, lang_ranks, rng)
    ci_low, ci_high = float(np.percentile(boot_taus, 2.5)), float(np.percentile(boot_taus, 97.5))
    np.save(folder / "tau_bootstrap.npy", boot_taus)

    inv_df = compute_inversions(merged)
    inv_df.to_csv(folder / "inversions.csv", index=False, encoding="utf-8")
    n_pairs = len(inv_df)
    n_inv   = int(inv_df["is_inversion"].sum())
    inversion_rate = n_inv / n_pairs if n_pairs > 0 else float("nan")

    regret_results = {}
    for k in TOP_K_VALUES:
        regret_results[k] = top_k_regret(merged, k)

    top1_global_model = merged.loc[merged["mteb_agg"].idxmax(), "model"]
    top1_global_score = float(merged.loc[merged["mteb_agg"].idxmax(), "lang_avg"])
    top1_oracle_score = float(merged["lang_avg"].max())
    top1_loss         = top1_oracle_score - top1_global_score

    result = {
        "iso3":           iso3,
        "lang_name":      LANG_META[iso3][0],
        "n_models":       n_models,
        "n_tasks":        n_tasks,
        "kendall_tau":    tau,
        "tau_ci_low":     ci_low,
        "tau_ci_high":    ci_high,
        "p_value":        pval,
        "inversion_rate": inversion_rate,
        "n_inversions":   n_inv,
        "n_pairs":        n_pairs,
        "regret_1":       regret_results[1]["regret"],
        "regret_3":       regret_results[3]["regret"],
        "regret_5":       regret_results[5]["regret"],
        "expected_loss_1": regret_results[1]["expected_loss"],
        "expected_loss_3": regret_results[3]["expected_loss"],
        "expected_loss_5": regret_results[5]["expected_loss"],
        "top1_loss":      top1_loss,
        "top1_global_model": top1_global_model,
        "top1_global_score": top1_global_score,
        "top1_oracle_score": top1_oracle_score,
    }

    with open(folder / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("%s: τ=%.3f [%.3f,%.3f] inv_rate=%.3f regret_1=%.3f top1_loss=%.3f",
             iso3, tau, ci_low, ci_high, inversion_rate,
             regret_results[1]["regret"], top1_loss)
    return result


def main():
    results = []
    for iso3 in LANG_META:
        log.info("Analyzing %s", iso3)
        r = analyze_language(iso3)
        if r:
            results.append(r)

    if not results:
        log.error("No languages passed threshold — nothing to summarize")
        return

    df = pd.DataFrame(results)

    # FDR correction across all kept languages
    reject, p_fdr, _, _ = multipletests(df["p_value"].fillna(1.0).values, alpha=0.05, method="fdr_bh")
    df["p_value_fdr"] = p_fdr
    df["reject_h0_fdr"] = reject

    cols = [
        "iso3", "lang_name", "n_models", "n_tasks",
        "kendall_tau", "tau_ci_low", "tau_ci_high",
        "p_value", "p_value_fdr", "reject_h0_fdr",
        "inversion_rate", "regret_1", "regret_3", "regret_5", "top1_loss",
    ]
    out_name = f"summary{_suffix}.csv"
    summary = df[cols].sort_values("kendall_tau")
    summary.to_csv(BASE / out_name, index=False, encoding="utf-8")
    log.info("%s saved (%d rows)", out_name, len(summary))

    n_audit_kept = sum(1 for r in results if LANG_META[r["iso3"]][1] == "audit_languages")
    median_tau   = float(df["kendall_tau"].median())
    median_inv   = float(df["inversion_rate"].median())

    low_tau = df[df["kendall_tau"] < 0.30]["iso3"].tolist()
    high_inv = df[df["inversion_rate"] > 0.40]["iso3"].tolist()
    failures = sorted(set(low_tau + high_inv))

    print(f"\nAudit complete: {n_audit_kept}/15 audit languages kept.")
    print(f"Median Kendall tau across audit = {median_tau:.3f}.")
    print(f"Median inversion rate = {median_inv * 100:.1f}%.")
    print(f"Hidden failure cases identified: {failures if failures else 'none'}.")
    print(f"\nResults at: {BASE / out_name}")


if __name__ == "__main__":
    main()
