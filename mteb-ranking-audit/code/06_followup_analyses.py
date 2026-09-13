"""
Three follow-up analyses on existing audit data. No new downloads.

Produces:
  snowflake_audit.csv
  task_asymmetry_table.csv
  plots/inversion_vs_task_count.png
  hidden_failure_cluster.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

AUDIT_DIR  = BASE / "audit_languages"
CASE_DIR   = BASE / "case_studies"
CASE_LANGS = {"ita", "jpn", "hin"}

TIERS       = ["restricted", "curated", "extended"]
TIER_SUFFIX = {"restricted": "", "curated": "_curated", "extended": "_extended"}

# Canonical language name map (covers Tamil which is absent from summary files)
LANG_NAMES = {
    "ara": "Arabic",    "ben": "Bengali",   "deu": "German",
    "fas": "Persian",   "fra": "French",    "hin": "Hindi",
    "ind": "Indonesian","ita": "Italian",   "jpn": "Japanese",
    "kor": "Korean",    "rus": "Russian",   "spa": "Spanish",
    "swa": "Swahili",   "tam": "Tamil",     "tel": "Telugu",
    "tha": "Thai",      "vie": "Vietnamese","zho": "Chinese",
}


def _load_lang_avg(iso3: str, suffix: str) -> pd.DataFrame | None:
    fname = f"lang_avg{suffix}.csv"
    for base in [CASE_DIR, AUDIT_DIR]:
        p = base / iso3 / fname
        if p.exists():
            return pd.read_csv(p, encoding="utf-8")
    return None


def _load_global(suffix: str) -> pd.DataFrame | None:
    p = BASE / "global" / f"mteb_agg{suffix}.csv"
    return pd.read_csv(p, encoding="utf-8") if p.exists() else None


def _all_iso3() -> list[str]:
    langs: set[str] = set()
    for d in AUDIT_DIR.iterdir():
        if d.is_dir():
            langs.add(d.name)
    langs.update(CASE_LANGS)
    return sorted(langs)


# ── Analysis 1: Snowflake audit ────────────────────────────────────────

def snowflake_audit() -> pd.DataFrame:
    """
    Trace snowflake-arctic-embed across all tiers and languages.
    Paper uses snowflake-arctic-embed-l-v2.0 (large); only -m-v2.0 (medium)
    is present in the extended data.
    """
    all_iso3 = _all_iso3()
    records = []

    for tier in TIERS:
        suffix = TIER_SUFFIX[tier]
        global_df = _load_global(suffix)
        if global_df is None:
            for iso3 in all_iso3:
                records.append(_snowflake_missing_row(tier, iso3))
            continue

        snow_in_global = global_df[
            global_df["model"].str.contains("snowflake|arctic-embed", case=False, na=False)
        ]
        if snow_in_global.empty:
            for iso3 in all_iso3:
                records.append(_snowflake_missing_row(tier, iso3))
            log.info("Snowflake not found in %s global (%s)", tier, "global/mteb_agg" + suffix + ".csv")
            continue

        # Global rank across all models in this tier
        global_df = global_df.copy()
        global_df["global_rank"] = global_df["mteb_agg"].rank(
            ascending=False, method="min"
        ).astype(int)
        n_global = len(global_df)
        top50_thresh = n_global * 0.50

        # Pick first matching snowflake model (only one expected)
        snow_row = global_df[
            global_df["model"].str.contains("snowflake|arctic-embed", case=False, na=False)
        ].iloc[0]
        snow_model = snow_row["model"]
        snow_mteb   = float(snow_row["mteb_agg"])
        snow_grank  = int(snow_row["global_rank"])

        log.info("Snowflake in %s global: %s  mteb_agg=%.4f  global_rank=%d/%d",
                 tier, snow_model, snow_mteb, snow_grank, n_global)

        for iso3 in all_iso3:
            lang_df = _load_lang_avg(iso3, suffix)
            if lang_df is None:
                records.append(_snowflake_missing_row(tier, iso3))
                continue

            merged = lang_df.merge(
                global_df[["model", "mteb_agg", "global_rank"]], on="model", how="inner"
            )
            if merged.empty:
                records.append(_snowflake_missing_row(tier, iso3))
                continue

            n_local = len(merged)
            bot25_thresh = n_local * 0.75

            merged["local_rank"] = merged["lang_avg"].rank(
                ascending=False, method="min"
            ).astype(int)

            snow_local = merged[
                merged["model"].str.contains("snowflake|arctic-embed", case=False, na=False)
            ]
            if snow_local.empty:
                # Model not present for this language
                rec = {
                    "tier": tier, "iso3": iso3, "lang_name": LANG_NAMES.get(iso3, iso3),
                    "n_models": n_local, "global_rank": snow_grank, "local_rank": None,
                    "mteb_agg": round(snow_mteb, 5), "lang_avg": None,
                    "is_hidden_failure": False,
                    "reason_if_not": "missing from language evaluation",
                }
            else:
                sr    = snow_local.iloc[0]
                lrank = int(sr["local_rank"])
                lavg  = float(sr["lang_avg"])
                is_hf = (snow_grank <= top50_thresh) and (lrank > bot25_thresh)

                if is_hf:
                    reason = ""
                elif snow_grank > top50_thresh:
                    reason = "global rank not top-50% (openly bad, not hidden)"
                else:
                    reason = "local rank not bottom-25% (no failure here)"

                rec = {
                    "tier": tier, "iso3": iso3, "lang_name": LANG_NAMES.get(iso3, iso3),
                    "n_models": n_local, "global_rank": snow_grank, "local_rank": lrank,
                    "mteb_agg": round(snow_mteb, 5), "lang_avg": round(lavg, 5),
                    "is_hidden_failure": is_hf,
                    "reason_if_not": reason,
                }
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(BASE / "snowflake_audit.csv", index=False)
    log.info("Saved snowflake_audit.csv (%d rows)", len(df))
    hf_count = int(df["is_hidden_failure"].sum())
    log.info("  snowflake hidden failures: %d", hf_count)
    return df


def _snowflake_missing_row(tier: str, iso3: str) -> dict:
    return {
        "tier": tier, "iso3": iso3, "lang_name": LANG_NAMES.get(iso3, iso3),
        "n_models": None, "global_rank": None, "local_rank": None,
        "mteb_agg": None, "lang_avg": None,
        "is_hidden_failure": False, "reason_if_not": "missing from tier",
    }


# ── Analysis 2: Task-asymmetry test ───────────────────────────────────

def _spearman_bootstrap(x: np.ndarray, y: np.ndarray,
                        rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (rho, ci_lo, ci_hi) via 10k bootstrap; skips NaN samples (constant x)."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rho_obs, _ = spearmanr(x, y)
    import warnings
    n = len(x)
    rhos: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rhos.append(r)
    if len(rhos) < 100:
        return float(rho_obs), float("nan"), float("nan")
    rhos_arr = np.array(rhos)
    return float(rho_obs), float(np.percentile(rhos_arr, 2.5)), float(np.percentile(rhos_arr, 97.5))


def task_asymmetry() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    summaries = {t: pd.read_csv(BASE / f"summary{TIER_SUFFIX[t]}.csv") for t in TIERS}

    # ── scatter plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    tier_colors = {"restricted": "#2171b5", "curated": "#e6550d", "extended": "#31a354"}

    table_rows = []
    spearman_results = {}

    for ax, tier in zip(axes, TIERS):
        df = summaries[tier].copy()
        if df.empty:
            ax.set_title(tier.capitalize())
            continue

        x = df["n_tasks"].values.astype(float)
        y = df["inversion_rate"].values.astype(float)
        median_inv = float(np.median(y))

        rho, ci_lo, ci_hi = _spearman_bootstrap(x, y, rng)
        spearman_results[tier] = (rho, ci_lo, ci_hi)
        log.info("%s task-asymmetry: Spearman rho=%.3f [%.3f, %.3f]",
                 tier, rho, ci_lo, ci_hi)

        # Jitter x so discrete values don't stack
        jitter = rng.uniform(-0.07, 0.07, size=len(x))
        ax.scatter(x + jitter, y, color=tier_colors[tier], alpha=0.8, s=60, zorder=3)

        # ISO labels
        for xi, yi, iso3 in zip(x + jitter, y, df["iso3"]):
            ax.annotate(iso3, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="black", zorder=4)

        ax.axhline(median_inv, color="gray", linestyle="--", linewidth=1.0,
                   label=f"Median inv={median_inv:.3f}", zorder=2)

        rho_str = f"ρ={rho:.3f} [{ci_lo:.3f},{ci_hi:.3f}]" if not np.isnan(rho) else "ρ=n/a"
        ax.set_title(f"{tier.capitalize()}\n{rho_str}", fontsize=10)
        ax.set_xlabel("n_tasks", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Inversion rate", fontsize=10)
        ax.set_xticks(sorted(df["n_tasks"].unique()))
        ax.legend(fontsize=8)

        # Build table rows
        df["n_tasks_bin"] = df["n_tasks"].apply(
            lambda t: "2" if t == 2 else ("3" if t == 3 else "4+")
        )
        for bin_label, grp in df.groupby("n_tasks_bin"):
            table_rows.append({
                "tier": tier,
                "n_tasks_bin": bin_label,
                "n_languages": len(grp),
                "mean_inversion_rate": round(grp["inversion_rate"].mean(), 4),
                "median_inversion_rate": round(grp["inversion_rate"].median(), 4),
                "mean_tau": round(grp["kendall_tau"].mean(), 4),
            })

    fig.suptitle("Inversion rate vs number of language tasks per tier", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = BASE / "plots" / "inversion_vs_task_count.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)

    # ── table ─────────────────────────────────────────────────────────
    tbl = pd.DataFrame(table_rows).sort_values(
        ["tier", "n_tasks_bin"],
        key=lambda s: s.map({"restricted": 0, "curated": 1, "extended": 2,
                              "2": 0, "3": 1, "4+": 2})
    ).reset_index(drop=True)
    tbl.to_csv(BASE / "task_asymmetry_table.csv", index=False)
    log.info("Saved task_asymmetry_table.csv")
    return tbl


# ── Analysis 3: Hidden-failure cluster ────────────────────────────────

def hidden_failure_cluster() -> pd.DataFrame:
    hf = pd.read_csv(BASE / "hidden_failures.csv")
    if hf.empty:
        log.warning("hidden_failures.csv is empty — nothing to cluster")
        pd.DataFrame().to_csv(BASE / "hidden_failure_cluster.csv", index=False)
        return pd.DataFrame()

    # Load curated global for mteb_agg_curated lookup
    gcur = _load_global("_curated")
    if gcur is not None:
        gcur_map = dict(zip(gcur["model"], gcur["mteb_agg"]))
    else:
        gcur_map = {}

    # Count (tier, iso3) cells per model
    cell_counts = (
        hf.groupby("model")[["tier", "iso3"]]
        .apply(lambda g: len(g.drop_duplicates()))
        .rename("n_cells")
    )
    qualifying = cell_counts[cell_counts >= 3].index.tolist()
    if not qualifying:
        log.warning("No model has >= 3 (tier, iso3) cells in hidden_failures.csv")
        pd.DataFrame().to_csv(BASE / "hidden_failure_cluster.csv", index=False)
        return pd.DataFrame()

    rows = []
    for model in qualifying:
        sub = hf[hf["model"] == model]

        best_global_rank   = int(sub["global_rank"].min())
        median_local_rank  = float(sub["local_rank"].median())

        n_cur  = int(sub[sub["tier"] == "curated"]["iso3"].nunique())
        n_ext  = int(sub[sub["tier"] == "extended"]["iso3"].nunique())
        n_rest = int(sub[sub["tier"] == "restricted"]["iso3"].nunique())

        # mteb_agg_curated: look up by model name in curated global
        mteb_cur = gcur_map.get(model)
        if mteb_cur is None:
            # Try substring match for extended full-name models
            for k, v in gcur_map.items():
                if k in model or model in k:
                    mteb_cur = v
                    break

        rows.append({
            "model":                          model,
            "best_global_rank":               best_global_rank,
            "median_local_rank":              round(median_local_rank, 1),
            "n_languages_affected_curated":   n_cur,
            "n_languages_affected_extended":  n_ext,
            "n_languages_affected_restricted": n_rest,
            "mteb_agg_curated":               round(float(mteb_cur), 5) if mteb_cur is not None else None,
        })

    cluster = (
        pd.DataFrame(rows)
        .sort_values("n_languages_affected_curated", ascending=False)
        .reset_index(drop=True)
    )
    cluster.to_csv(BASE / "hidden_failure_cluster.csv", index=False)
    log.info("Saved hidden_failure_cluster.csv (%d models)", len(cluster))
    for _, r in cluster.iterrows():
        log.info("  %s: global_rank=%s, median_local=%.1f, "
                 "cur=%d lang, ext=%d lang, rest=%d lang",
                 r["model"], r["best_global_rank"], r["median_local_rank"],
                 r["n_languages_affected_curated"],
                 r["n_languages_affected_extended"],
                 r["n_languages_affected_restricted"])
    return cluster


def main() -> None:
    snow = snowflake_audit()
    tbl  = task_asymmetry()
    clus = hidden_failure_cluster()

    log.info("=== follow-up analysis summary ===")
    log.info("snowflake_audit: %d rows, %d hidden failures",
             len(snow), int(snow["is_hidden_failure"].sum()))
    log.info("task_asymmetry_table: %d rows", len(tbl))
    log.info("hidden_failure_cluster: %d qualifying models", len(clus))


if __name__ == "__main__":
    main()
