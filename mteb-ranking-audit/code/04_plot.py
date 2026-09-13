"""
Generate three summary plots from summary.csv (or summary_extended.csv with --extended).

Usage:
  python 04_plot.py            # plots from restricted roster summary.csv
  python 04_plot.py --extended # plots from extended roster summary_extended.csv
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXTENDED = "--extended" in sys.argv
CURATED  = "--curated"  in sys.argv

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "code" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

_suffix = "_extended" if EXTENDED else ("_curated" if CURATED else "")


def load_summary() -> pd.DataFrame:
    fname = f"summary{_suffix}.csv"
    path = BASE / fname
    if not path.exists():
        raise FileNotFoundError(f"{fname} not found at {path}")
    return pd.read_csv(path, encoding="utf-8")


def plot_tau(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("kendall_tau")
    labels = df_sorted["lang_name"].tolist()
    taus   = df_sorted["kendall_tau"].values
    ci_low = taus - df_sorted["tau_ci_low"].values
    ci_hi  = df_sorted["tau_ci_high"].values - taus
    yerr   = np.array([ci_low, ci_hi])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    ax.bar(x, taus, yerr=yerr, capsize=4, color="steelblue", ecolor="black", alpha=0.8)
    ax.axhline(0, color="red", linewidth=1.0, linestyle="--", label="τ = 0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Kendall τ")
    ax.set_title("Kendall τ: global vs language ranking (with 95% CI)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS / f"tau_per_language{_suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_inversion(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("inversion_rate")
    labels = df_sorted["lang_name"].tolist()
    rates  = df_sorted["inversion_rate"].values * 100  # percent

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    ax.bar(x, rates, color="coral", alpha=0.8)
    ax.axhline(50.0, color="gray",  linewidth=1.0, linestyle="--", label="50% (random)")
    ax.axhline(20.0, color="green", linewidth=1.0, linestyle=":",  label="20% (paper headline)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Inversion rate (%)")
    ax.set_title("Pairwise inversion rate per language")
    ax.legend()
    fig.tight_layout()
    out = PLOTS / f"inversion_per_language{_suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_regret_top1(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("regret_1", ascending=False)
    labels  = df_sorted["lang_name"].tolist()
    regrets = df_sorted["regret_1"].values

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    ax.bar(x, regrets, color="mediumpurple", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Top-1 regret (NDCG@10 points)")
    ax.set_title("Top-1 regret: NDCG lost by selecting global top-1 vs oracle")
    fig.tight_layout()
    out = PLOTS / f"regret_top1{_suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_tau_vs_roster_size() -> None:
    candidates = [
        ("", "restricted"),
        ("_curated", "curated"),
        ("_extended", "extended"),
    ]
    lang_data: dict[str, list[tuple[int, float, float, float]]] = {}
    for sfx, label in candidates:
        path = BASE / f"summary{sfx}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8")
        for _, row in df.iterrows():
            lang = row["lang_name"]
            if lang not in lang_data:
                lang_data[lang] = []
            lang_data[lang].append((
                int(row["n_models"]),
                float(row["kendall_tau"]),
                float(row["tau_ci_low"]),
                float(row["tau_ci_high"]),
            ))

    if not lang_data:
        log.warning("plot_tau_vs_roster_size: no summary files found")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for lang, points in sorted(lang_data.items()):
        points_sorted = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        ci_lo = [p[1] - p[2] for p in points_sorted]
        ci_hi = [p[3] - p[1] for p in points_sorted]
        line, = ax.plot(xs, ys, marker="o", label=lang)
        ax.errorbar(xs, ys,
                    yerr=[ci_lo, ci_hi],
                    fmt="none", ecolor=line.get_color(), capsize=3, alpha=0.5)

    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Number of models in roster")
    ax.set_ylabel("Kendall τ")
    ax.set_title("Kendall τ stability across roster sizes")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    out = PLOTS / "tau_vs_roster_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def main():
    df = load_summary()
    plot_tau(df)
    plot_inversion(df)
    plot_regret_top1(df)
    plot_tau_vs_roster_size()
    log.info("All plots saved to %s", PLOTS)


if __name__ == "__main__":
    main()
