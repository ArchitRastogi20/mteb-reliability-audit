"""
Generates four audit artifacts on top of existing pipeline outputs:
  sensitivity_table.csv / sensitivity_table.md
  plots/tau_vs_roster_tier.png
  hidden_failures.csv
No new downloads required.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "code" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

AUDIT_DIR = BASE / "audit_languages"
CASE_DIR = BASE / "case_studies"
CASE_LANGS = {"ita", "jpn", "hin"}

TIERS = ["restricted", "curated", "extended"]
TIER_SUFFIX = {"restricted": "", "curated": "_curated", "extended": "_extended"}
TIER_LABEL = {"restricted": "Restricted", "curated": "Curated", "extended": "Extended"}


def _lang_name_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for f in ["summary_extended.csv", "summary_curated.csv", "summary.csv"]:
        p = BASE / f
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                m.setdefault(str(row["iso3"]), str(row["lang_name"]))
    return m


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
    for iso3 in CASE_LANGS:
        langs.add(iso3)
    return sorted(langs)


# ── 1. Sensitivity table ───────────────────────────────────────────────

def build_sensitivity_table() -> pd.DataFrame:
    summaries = {t: pd.read_csv(BASE / f"summary{TIER_SUFFIX[t]}.csv") for t in TIERS}
    lname = _lang_name_map()

    all_iso3 = set()
    for df in summaries.values():
        all_iso3.update(df["iso3"].astype(str))

    rows = []
    for iso3 in sorted(all_iso3):
        row: dict = {"iso3": iso3, "lang_name": lname.get(iso3, iso3)}
        for tier in TIERS:
            df = summaries[tier]
            sub = df[df["iso3"] == iso3]
            t = tier[0]  # r / c / e
            if sub.empty:
                row[f"n_{tier}"] = None
                row[f"tau_{tier}"] = None
                row[f"ci_low_{t}"] = None
                row[f"ci_high_{t}"] = None
                row[f"inv_{t}"] = None
                row[f"fdr_{t}"] = False
            else:
                r = sub.iloc[0]
                row[f"n_{tier}"] = int(r["n_models"])
                row[f"tau_{tier}"] = round(float(r["kendall_tau"]), 4)
                lo = r.get("tau_ci_low")
                hi = r.get("tau_ci_high")
                row[f"ci_low_{t}"] = round(float(lo), 4) if pd.notna(lo) else None
                row[f"ci_high_{t}"] = round(float(hi), 4) if pd.notna(hi) else None
                row[f"inv_{t}"] = round(float(r["inversion_rate"]), 4)
                row[f"fdr_{t}"] = bool(r["reject_h0_fdr"])
        rows.append(row)

    sens = pd.DataFrame(rows)
    sens = sens.sort_values("tau_curated", ascending=True).reset_index(drop=True)

    csv_cols = [
        "iso3", "lang_name",
        "n_restricted", "tau_restricted", "ci_low_r", "ci_high_r", "inv_r",
        "n_curated",    "tau_curated",    "ci_low_c", "ci_high_c", "inv_c",
        "n_extended",   "tau_extended",   "ci_low_e", "ci_high_e", "inv_e",
    ]
    sens[csv_cols].to_csv(BASE / "sensitivity_table.csv", index=False)
    log.info("Saved sensitivity_table.csv (%d languages)", len(sens))

    _write_sensitivity_md(sens)
    return sens


def _fmt_tau(val, fdr: bool) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = f"{val:.3f}"
    return f"**{s}**" if fdr else s


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.3f}"


def _ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "—"
    if (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _write_sensitivity_md(sens: pd.DataFrame) -> None:
    header = (
        "| iso3 | Language "
        "| n_r | τ_r | CI_r | inv_r "
        "| n_c | τ_c | CI_c | inv_c "
        "| n_e | τ_e | CI_e | inv_e |"
    )
    sep = "|---|---|" + "---|---|---|---|" * 3

    lines = [
        "# Sensitivity Table — Three-Tier Audit\n\n",
        "_τ values in **bold** = FDR-significant (BH q<0.05). "
        "Tiers: R=Restricted (~7 paper models), C=Curated (~25 deployment-realistic), "
        "E=Extended (all MMTEB-eligible)._\n\n",
        header + "\n",
        sep + "\n",
    ]

    for _, row in sens.iterrows():
        line = (
            f"| {row['iso3']} | {row['lang_name']} "
            f"| {row['n_restricted'] or '—'} "
            f"| {_fmt_tau(row['tau_restricted'], row['fdr_r'])} "
            f"| {_ci(row['ci_low_r'], row['ci_high_r'])} "
            f"| {_fmt(row['inv_r'])} "
            f"| {row['n_curated'] or '—'} "
            f"| {_fmt_tau(row['tau_curated'], row['fdr_c'])} "
            f"| {_ci(row['ci_low_c'], row['ci_high_c'])} "
            f"| {_fmt(row['inv_c'])} "
            f"| {row['n_extended'] or '—'} "
            f"| {_fmt_tau(row['tau_extended'], row['fdr_e'])} "
            f"| {_ci(row['ci_low_e'], row['ci_high_e'])} "
            f"| {_fmt(row['inv_e'])} |"
        )
        lines.append(line + "\n")

    # Median footer
    med_r = sens["tau_restricted"].median()
    med_c = sens["tau_curated"].median()
    med_e = sens["tau_extended"].median()
    med_ir = sens["inv_r"].median()
    med_ic = sens["inv_c"].median()
    med_ie = sens["inv_e"].median()
    lines.append(
        f"| — | **Median** "
        f"| — | {med_r:.3f} | — | {med_ir:.3f} "
        f"| — | {med_c:.3f} | — | {med_ic:.3f} "
        f"| — | {med_e:.3f} | — | {med_ie:.3f} |\n"
    )

    (BASE / "sensitivity_table.md").write_text("".join(lines), encoding="utf-8")
    log.info("Saved sensitivity_table.md")


# ── 2. Sensitivity plot ────────────────────────────────────────────────

def plot_tier_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    tier_x = {"restricted": 0, "curated": 1, "extended": 2}
    n_langs = len(sens)
    cmap = cm.get_cmap("tab20", max(n_langs, 1))

    # Per-language faint lines
    for i, (_, row) in enumerate(sens.iterrows()):
        color = cmap(i % 20)
        xs, ys = [], []
        for tier in TIERS:
            v = row[f"tau_{tier}"]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                xs.append(tier_x[tier])
                ys.append(float(v))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, alpha=0.30, linewidth=1.0, zorder=2)

        # CI bars
        for tier in TIERS:
            v = row[f"tau_{tier}"]
            t = tier[0]
            lo, hi = row[f"ci_low_{t}"], row[f"ci_high_{t}"]
            if v is not None and lo is not None and hi is not None:
                if not any(isinstance(x, float) and np.isnan(x) for x in [v, lo, hi]):
                    err_lo = max(0.0, float(v) - float(lo))
                    err_hi = max(0.0, float(hi) - float(v))
                    ax.errorbar(tier_x[tier], float(v),
                                yerr=[[err_lo], [err_hi]],
                                fmt="none", color=color, alpha=0.18,
                                capsize=2, zorder=1)

    # Bold median line
    medians = [
        float(sens["tau_restricted"].median()),
        float(sens["tau_curated"].median()),
        float(sens["tau_extended"].median()),
    ]
    ax.plot([0, 1, 2], medians, color="black", linewidth=2.5, zorder=5,
            label=(f"Median  R={medians[0]:.3f}  C={medians[1]:.3f}  E={medians[2]:.3f}"))
    ax.scatter([0, 1, 2], medians, color="black", s=60, zorder=6)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Restricted\n(~7 models)", "Curated\n(~25 models)", "Extended\n(12–57 models)"],
                       fontsize=11)
    ax.set_ylabel("Kendall τ", fontsize=12)
    ax.set_title("Kendall τ stability across model-roster tiers", fontsize=13)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.5, 1.05)

    plt.tight_layout()
    out = BASE / "plots" / "tau_vs_roster_tier.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("Saved %s", out)


# ── 3. Hidden failures ─────────────────────────────────────────────────

def build_hidden_failures() -> pd.DataFrame:
    lname = _lang_name_map()
    all_iso3 = _all_iso3()
    records: list[pd.DataFrame] = []

    for tier in TIERS:
        suffix = TIER_SUFFIX[tier]
        global_df = _load_global(suffix)
        if global_df is None:
            log.warning("No global file for tier %s — skipping", tier)
            continue

        # Global rank across ALL models in this tier's global file
        global_df = global_df.copy()
        global_df["global_rank"] = (
            global_df["mteb_agg"].rank(ascending=False, method="min").astype(int)
        )
        n_global = len(global_df)
        top50_thresh = n_global * 0.50

        for iso3 in all_iso3:
            lang_df = _load_lang_avg(iso3, suffix)
            if lang_df is None or lang_df.empty:
                continue

            merged = lang_df.merge(
                global_df[["model", "mteb_agg", "global_rank"]], on="model", how="inner"
            )
            if merged.empty:
                continue

            n_local = len(merged)
            bot25_thresh = n_local * 0.75

            merged["local_rank"] = (
                merged["lang_avg"].rank(ascending=False, method="min").astype(int)
            )
            merged["rank_drop"] = merged["local_rank"] - merged["global_rank"]

            fails = merged[
                (merged["global_rank"] <= top50_thresh) &
                (merged["local_rank"] > bot25_thresh)
            ].copy()

            if fails.empty:
                continue

            fails["tier"] = tier
            fails["iso3"] = iso3
            fails["lang_name"] = lname.get(iso3, iso3)
            records.append(
                fails[["tier", "iso3", "lang_name", "model",
                        "global_rank", "local_rank", "rank_drop",
                        "mteb_agg", "lang_avg"]]
            )

    if not records:
        log.warning("No hidden failures found")
        out = BASE / "hidden_failures.csv"
        pd.DataFrame(columns=["tier", "iso3", "lang_name", "model",
                               "global_rank", "local_rank", "rank_drop",
                               "mteb_agg", "lang_avg"]).to_csv(out, index=False)
        return pd.DataFrame()

    hf = pd.concat(records, ignore_index=True)
    hf = hf.sort_values("rank_drop", ascending=False).reset_index(drop=True)
    hf.to_csv(BASE / "hidden_failures.csv", index=False)
    log.info("Saved hidden_failures.csv (%d rows)", len(hf))

    # Diagnostic summary for models of interest
    for pattern in ["harrier", "KaLM-Gemma3", "granite-311m", "granite-embedding-311m"]:
        sub = hf[hf["model"].str.contains(pattern, case=False, na=False)]
        if not sub.empty:
            counts = sub.groupby("tier")["iso3"].nunique().to_dict()
            log.info("  %s hidden failures per tier: %s", pattern, counts)

    return hf


def main() -> None:
    sens = build_sensitivity_table()
    plot_tier_sensitivity(sens)
    hf = build_hidden_failures()

    log.info("=== audit artifact summary ===")
    log.info("sensitivity_table: %d languages × 3 tiers", len(sens))
    if not hf.empty:
        log.info("hidden_failures: %d rows across %d (tier, lang) pairs",
                 len(hf), hf.groupby(["tier", "iso3"]).ngroups)
        by_tier = hf.groupby("tier")["model"].count()
        log.info("  per tier: %s", dict(by_tier))


if __name__ == "__main__":
    main()
