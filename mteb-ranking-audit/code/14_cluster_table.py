"""
Generate paper/table_hf_cluster.tex: the hidden-failure cluster table.

Cluster models and their data sources:
  - granite-311m, harrier-0.6b, Seed1.6-embed, inf-retriever-v1  → curated tier
  - granite-97m-r2                                                  → extended tier (*)

Columns:
  best_global      -- best (lowest) MTEB rank across tiers from roster_membership.csv
  median_local     -- median local rank across 17 languages from curated_local_ranks.csv
  n_affected       -- count of languages where local_rank >= 19 (bottom 25% of n=25 curated)
  mteb_agg         -- MTEB aggregate score (× 100) from global/mteb_agg_curated.csv

For granite-97m-r2 (*): global rank and mteb_agg from mteb_agg_extended.csv;
  median_local and n_affected from per-language lang_avg_extended.csv files
  (n_affected uses bottom-25% threshold per language: rank > n_lang × 0.75).

Data sources:
  analysis/curated_local_ranks.csv
  analysis/roster_membership.csv
  global/mteb_agg_curated.csv
  global/mteb_agg_extended.csv
  <lang>/lang_avg_extended.csv  (one per language, 17 total)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent          # more_exp_claude_10_may/
DATA = HERE.parent / "mteb_audit_data"

ANALYSIS = HERE / "analysis"
PAPER    = HERE / "paper"
PAPER.mkdir(exist_ok=True)

# ── Cluster model definitions ────────────────────────────────────────────────
# display_name → model_id (HF short name in curated_local_ranks.csv)
CLUSTER_CURATED_ORDER = [
    "harrier-0.6b",
    "inf-retriever-v1",
    "Seed1.6-embed",
    "granite-311m",
]

# Extended-tier cluster model: uses __ format in mteb_agg_extended.csv
GRANITE_97M_EXTENDED_ID = "ibm-granite__granite-embedding-97m-multilingual-r2"
GRANITE_97M_DISPLAY     = "granite-97m-r2"

# 17 languages from the audit pipeline
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

N_AFFECTED_THRESH = 19  # bottom-25% of n=25 curated tier


# ── Data loading ────────────────────────────────────────────────────────────

def load_curated_data():
    local_ranks  = pd.read_csv(ANALYSIS / "curated_local_ranks.csv")
    roster       = pd.read_csv(ANALYSIS / "roster_membership.csv")
    mteb_curated = pd.read_csv(DATA / "global" / "mteb_agg_curated.csv")
    return local_ranks, roster, mteb_curated


def best_global_rank(row) -> int:
    """Return the best (lowest) rank across tiers."""
    r_curated    = row["global_rank_curated"]
    r_restricted = row["global_rank_restricted"]
    if pd.notna(r_restricted) and r_restricted > 0:
        return int(min(r_curated, r_restricted))
    return int(r_curated)


# ── Curated-tier stats ───────────────────────────────────────────────────────

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


# ── Extended-tier stats for granite-97m-r2 ──────────────────────────────────

def extended_stats_granite97m(mteb_extended: pd.DataFrame) -> dict:
    # Global rank in extended tier (sorted descending by mteb_agg).
    ext_sorted  = mteb_extended.sort_values("mteb_agg", ascending=False).reset_index(drop=True)
    mask        = ext_sorted["model"] == GRANITE_97M_EXTENDED_ID
    if not mask.any():
        raise ValueError(f"{GRANITE_97M_EXTENDED_ID} not found in mteb_agg_extended.csv")
    global_rank = int(mask.idxmax()) + 1          # 1-based rank
    mteb_agg    = float(ext_sorted.loc[mask.idxmax(), "mteb_agg"])

    # Per-language local rank.
    local_ranks_ext = []
    n_affected_ext  = 0

    for iso3, subdir in LANG_META.items():
        lang_path = DATA / subdir / iso3 / "lang_avg_extended.csv"
        if not lang_path.exists():
            continue
        lang_df = pd.read_csv(lang_path)
        if GRANITE_97M_EXTENDED_ID not in lang_df["model"].values:
            continue
        lang_df_s    = lang_df.sort_values("lang_avg", ascending=False).reset_index(drop=True)
        n_lang       = len(lang_df_s)
        rank_0based  = lang_df_s[lang_df_s["model"] == GRANITE_97M_EXTENDED_ID].index[0]
        local_rank   = int(rank_0based) + 1
        local_ranks_ext.append(local_rank)
        thresh = n_lang * 0.75   # bottom-25% of extended pool
        if local_rank > thresh:
            n_affected_ext += 1

    if not local_ranks_ext:
        raise ValueError("No extended-tier language data found for granite-97m-r2")

    return {
        "display_name": GRANITE_97M_DISPLAY,
        "best_global":  global_rank,
        "median_local": float(np.median(local_ranks_ext)),
        "n_affected":   n_affected_ext,
        "mteb_agg":     mteb_agg,
        "tier":         "extended",
    }


# ── LaTeX writer ─────────────────────────────────────────────────────────────

def format_row(stats: dict, extended: bool = False) -> str:
    name     = stats["display_name"]
    if extended:
        name = name + r"$^*$"
    g_rank   = stats["best_global"]
    med      = int(stats["median_local"])
    n_aff    = stats["n_affected"]
    mteb     = stats["mteb_agg"] * 100

    return f"{name} & {g_rank} & {med} & {n_aff} & {mteb:.1f} \\\\"


def write_table(rows: list[dict]) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Global rank} & \textbf{Median local} "
        r"& \textbf{Langs.\ affected} & \textbf{MTEB} \\"
    )
    lines.append(r"\midrule")

    for stats in rows:
        if stats["tier"] == "curated":
            lines.append(format_row(stats, extended=False))

    lines.append(r"\midrule")

    for stats in rows:
        if stats["tier"] == "extended":
            lines.append(format_row(stats, extended=True))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Hidden-failure cluster: global MTEB rank vs.\ per-language rank "
        r"across 17 languages. \emph{Global rank}: position in the 25-model curated "
        r"tier by MTEB-agg (best rank across tiers shown). \emph{Median local}: median "
        r"rank within each language's curated pool. \emph{Langs.\ affected}: number of "
        r"the 17 languages where the model falls in the bottom quartile "
        r"(local rank $\geq 19$ of 25). \emph{MTEB}: aggregate MTEB score ($\times 100$). "
        r"$^*$granite-97m-r2 is in the extended roster ($n{\approx}57$); "
        r"its global rank, local ranks, and bottom-quartile threshold "
        r"($>\!n_{\text{lang}}{\times}0.75$) are relative to that pool.}"
    )
    lines.append(r"\label{tab:hf-cluster}")
    lines.append(r"\end{table}")

    out = PAPER / "table_hf_cluster.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    local_ranks, roster, mteb_curated = load_curated_data()
    mteb_extended = pd.read_csv(DATA / "global" / "mteb_agg_extended.csv")

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
    s = extended_stats_granite97m(mteb_extended)
    rows.append(s)
    print(
        f"  {s['display_name']:20s}  global={s['best_global']:2d}  "
        f"median_local={s['median_local']:.1f}  n_affected={s['n_affected']}  "
        f"mteb={s['mteb_agg']*100:.1f}"
    )

    write_table(rows)


if __name__ == "__main__":
    main()
