"""
Generate Table: hidden-failure cluster (global rank vs. per-language rank).

Extended (n=28–57, language-dependent). All publicly indexed submissions to
the 18 MMTEB(Multilingual) retrieval tasks at the 2026-05 snapshot,
intersected with each language's available evaluations. granite-97m-r2
(extended global rank 28) is the cluster member that appears only in the
extended tier; per-language inclusion flags are released in
analysis/extended_roster.csv.

Cluster models and their data sources:
  - granite-311m, harrier-0.6b, Seed1.6-embed, inf-retriever-v1  → curated tier (n=25)
  - granite-97m-r2                                                  → extended tier (n=28–57, *)

Columns in the output table:
  best_global   -- best (lowest) MTEB rank across tiers (from roster_membership.csv)
  median_local  -- median per-language rank across 17 audit languages
  n_affected    -- languages where the model falls in the bottom quartile
  mteb_agg      -- aggregate MTEB score (×100) from global/mteb_agg_curated.csv

For curated-tier models, bottom-quartile threshold = local rank ≥ 19 (of 25).
For granite-97m-r2 (*), global rank, local ranks, and bottom-quartile threshold
(> n_lang × 0.75) are relative to the extended pool.

Produces:
  analysis/extended_roster.csv  -- per-language inclusion flags for granite-97m-r2
  paper/table_hf_cluster.tex    -- populated LaTeX table

Data sources (pre-computed by earlier steps):
  analysis/curated_local_ranks.csv
  analysis/roster_membership.csv
  global/mteb_agg_curated.csv
  global/mteb_agg_extended.csv
  <lang>/lang_avg_extended.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent   # mteb-ranking-audit/
ANALYSIS = BASE / "analysis"
PAPER    = BASE / "paper"
ANALYSIS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

# ── Cluster model definitions ─────────────────────────────────────────────────
CLUSTER_CURATED_ORDER = [
    "harrier-0.6b",
    "inf-retriever-v1",
    "Seed1.6-embed",
    "granite-311m",
]

# Extended-tier cluster model: uses __ separator in mteb_agg_extended.csv
GRANITE_97M_EXTENDED_ID = "ibm-granite__granite-embedding-97m-multilingual-r2"
GRANITE_97M_DISPLAY     = "granite-97m-r2"

# 17 audit languages (Tamil excluded: n_tasks < 2 in curated/extended tiers)
LANG_META = {
    "ita": "case_studies",
    "jpn": "case_studies",
    "hin": "case_studies",
    "ara": "audit_languages",
    "zho": "audit_languages",
    "deu": "audit_languages",
    "spa": "audit_languages",
    "rus": "audit_languages",
    "fra": "audit_languages",
    "kor": "audit_languages",
    "ben": "audit_languages",
    "vie": "audit_languages",
    "ind": "audit_languages",
    "fas": "audit_languages",
    "swa": "audit_languages",
    "tha": "audit_languages",
    "tel": "audit_languages",
}

N_AFFECTED_THRESH = 19   # bottom 25% of n=25 curated tier


# ── Data loading ──────────────────────────────────────────────────────────────

def load_curated_data():
    local_ranks  = pd.read_csv(ANALYSIS / "curated_local_ranks.csv")
    roster       = pd.read_csv(ANALYSIS / "roster_membership.csv")
    mteb_curated = pd.read_csv(BASE / "global" / "mteb_agg_curated.csv")
    return local_ranks, roster, mteb_curated


def best_global_rank(row) -> int:
    """Return the best (lowest) rank across tiers."""
    r_curated    = row["global_rank_curated"]
    r_restricted = row["global_rank_restricted"]
    if pd.notna(r_restricted) and r_restricted > 0:
        return int(min(r_curated, r_restricted))
    return int(r_curated)


# ── Curated-tier stats ────────────────────────────────────────────────────────

def curated_stats(local_ranks: pd.DataFrame, roster: pd.DataFrame,
                  mteb_curated: pd.DataFrame, display_name: str) -> dict:
    df = local_ranks[local_ranks["display_name"] == display_name]
    if df.empty:
        raise ValueError(f"No rows for {display_name} in curated_local_ranks.csv")

    median_local = float(df["local_rank"].median())
    n_affected   = int((df["local_rank"] >= N_AFFECTED_THRESH).sum())

    roster_row   = roster[roster["display_name"] == display_name]
    if roster_row.empty:
        raise ValueError(f"{display_name} not found in roster_membership.csv")
    best_global  = best_global_rank(roster_row.iloc[0])

    mteb_row     = mteb_curated[mteb_curated["model"] == display_name]
    if mteb_row.empty:
        raise ValueError(f"{display_name} not found in mteb_agg_curated.csv")
    mteb_agg     = float(mteb_row.iloc[0]["mteb_agg"])

    return {
        "display_name": display_name,
        "best_global":  best_global,
        "median_local": median_local,
        "n_affected":   n_affected,
        "mteb_agg":     mteb_agg,
        "tier":         "curated",
    }


# ── Extended-tier stats for granite-97m-r2 ───────────────────────────────────

def extended_stats_granite97m(mteb_extended: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Returns (stats_dict, extended_roster_df).
    extended_roster_df has per-language inclusion flags for granite-97m-r2.
    """
    ext_sorted = mteb_extended.sort_values("mteb_agg", ascending=False).reset_index(drop=True)
    mask       = ext_sorted["model"] == GRANITE_97M_EXTENDED_ID
    if not mask.any():
        raise ValueError(f"{GRANITE_97M_EXTENDED_ID} not found in mteb_agg_extended.csv")
    global_rank = int(mask.idxmax()) + 1
    mteb_agg    = float(ext_sorted.loc[mask.idxmax(), "mteb_agg"])

    local_ranks_ext = []
    n_affected_ext  = 0
    roster_rows     = []

    for iso3, subdir in LANG_META.items():
        lang_path = BASE / subdir / iso3 / "lang_avg_extended.csv"
        if not lang_path.exists():
            roster_rows.append({
                "language_iso": iso3,
                "n_extended":   0,
                "included":     False,
                "local_rank":   None,
            })
            continue
        lang_df = pd.read_csv(lang_path)
        included = GRANITE_97M_EXTENDED_ID in lang_df["model"].values
        n_lang   = len(lang_df)
        roster_rows.append({
            "language_iso": iso3,
            "n_extended":   n_lang,
            "included":     included,
            "local_rank":   None,
        })
        if not included:
            continue
        lang_df_s   = lang_df.sort_values("lang_avg", ascending=False).reset_index(drop=True)
        rank_0based = lang_df_s[lang_df_s["model"] == GRANITE_97M_EXTENDED_ID].index[0]
        local_rank  = int(rank_0based) + 1
        roster_rows[-1]["local_rank"] = local_rank
        local_ranks_ext.append(local_rank)
        if local_rank > n_lang * 0.75:
            n_affected_ext += 1

    if not local_ranks_ext:
        raise ValueError("No extended-tier language data found for granite-97m-r2")

    stats = {
        "display_name": GRANITE_97M_DISPLAY,
        "best_global":  global_rank,
        "median_local": float(np.median(local_ranks_ext)),
        "n_affected":   n_affected_ext,
        "mteb_agg":     mteb_agg,
        "tier":         "extended",
    }
    return stats, pd.DataFrame(roster_rows)


# ── LaTeX writer ──────────────────────────────────────────────────────────────

def format_row(stats: dict, extended: bool = False) -> str:
    name   = stats["display_name"] + (r"$^*$" if extended else "")
    g_rank = stats["best_global"]
    med    = int(stats["median_local"])
    n_aff  = stats["n_affected"]
    mteb   = stats["mteb_agg"] * 100
    return f"{name} & {g_rank} & {med} & {n_aff} & {mteb:.1f} \\\\"


def write_table(rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Global rank} & \textbf{Median local} "
        r"& \textbf{Langs.\ affected} & \textbf{MTEB} \\",
        r"\midrule",
    ]
    for stats in rows:
        if stats["tier"] == "curated":
            lines.append(format_row(stats, extended=False))
    lines.append(r"\midrule")
    for stats in rows:
        if stats["tier"] == "extended":
            lines.append(format_row(stats, extended=True))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Hidden-failure cluster: global MTEB rank vs.\ per-language rank "
        r"across 17 audit languages. \emph{Global rank}: position in the 25-model curated "
        r"tier by MTEB-agg (best rank across tiers shown). \emph{Median local}: median "
        r"rank within each language's curated pool. \emph{Langs.\ affected}: count of "
        r"the 17 languages where the model falls in the bottom quartile "
        r"(local rank $\geq 19$ of 25). \emph{MTEB}: aggregate MTEB score ($\times 100$). "
        r"$^*$granite-97m-r2 is in the extended roster ($n{\approx}57$); "
        r"its global rank, local ranks, and bottom-quartile threshold "
        r"($>\!n_{\text{lang}}{\times}0.75$) are relative to that pool "
        r"(see analysis/extended\_roster.csv).}",
        r"\label{tab:hf-cluster}",
        r"\end{table}",
    ]
    out = PAPER / "table_hf_cluster.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    local_ranks, roster, mteb_curated = load_curated_data()
    mteb_extended = pd.read_csv(BASE / "global" / "mteb_agg_extended.csv")

    rows = []

    print("Curated-tier cluster models:")
    for name in CLUSTER_CURATED_ORDER:
        s = curated_stats(local_ranks, roster, mteb_curated, name)
        rows.append(s)
        print(
            f"  {s['display_name']:20s}  global={s['best_global']:2d}  "
            f"median_local={s['median_local']:.1f}  n_affected={s['n_affected']}  "
            f"mteb={s['mteb_agg']*100:.1f}"
        )

    print("\nExtended-tier cluster model (granite-97m-r2):")
    s, roster_df = extended_stats_granite97m(mteb_extended)
    rows.append(s)
    print(
        f"  {s['display_name']:20s}  global={s['best_global']:2d}  "
        f"median_local={s['median_local']:.1f}  n_affected={s['n_affected']}  "
        f"mteb={s['mteb_agg']*100:.1f}"
    )

    roster_df.to_csv(ANALYSIS / "extended_roster.csv", index=False)
    print(f"Wrote analysis/extended_roster.csv ({len(roster_df)} rows)")

    write_table(rows)


if __name__ == "__main__":
    main()
