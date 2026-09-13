"""Paired bootstrap test for dependent Spearman correlation differences.

For each of three contrasts (full lang-avg gap vs single-task gap):
  - IT-HI, full vs Wikipedia
  - IT-JA, full vs Belebele
  - JA-HI, full vs MIRACL
resample model indices with replacement B=10000 times, compute the difference
of paired Spearman rho per resample, report observed delta, 95% percentile CI,
and two-sided p-value (twice the smaller tail mass).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import UNIFIED_RESULTS, STANDARD_14, OUTPUTS, OUTPUTS

CONTRASTS = [
    # (label, lang_a, lang_b, single_task_column)
    ("IT-HI full vs Wikipedia", "italian",  "hindi",    "task_WikipediaRetrievalMultilingual"),
    ("IT-JA full vs Belebele",  "italian",  "japanese", "task_BelebeleRetrieval"),
    ("JA-HI full vs MIRACL",    "japanese", "hindi",    "task_MIRACLRetrievalHardNegatives"),
]


def _gap_vector(df: pd.DataFrame, lang: str, gap_col: str) -> pd.Series:
    sub = df[df["language"] == lang].copy()
    if gap_col == "lang_avg":
        sub["gap"] = sub["lang_specific_avg"] - sub["mteb_agg_score"]
    else:
        sub["gap"] = sub[gap_col] - sub["mteb_agg_score"]
    return sub.set_index("model_id")["gap"].dropna()


def _spearman_paired(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def paired_bootstrap_diff(x: np.ndarray, yA: np.ndarray, yB: np.ndarray,
                          B: int = 10000, seed: int = 0) -> dict:
    """rho(x, yA) - rho(x, yB) bootstrap on shared model indices.

    yA and yB are paired with x on the same sample (model indices). Resamples
    indices with replacement; computes both rhos under the same resampled
    sample; returns delta = rho_A - rho_B.
    """
    x = np.asarray(x); yA = np.asarray(yA); yB = np.asarray(yB)
    n = len(x)
    assert len(yA) == n == len(yB)
    delta_obs = _spearman_paired(x, yA) - _spearman_paired(x, yB)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _spearman_paired(x[idx], yA[idx]) - _spearman_paired(x[idx], yB[idx])
    deltas = deltas[~np.isnan(deltas)]
    ci = np.quantile(deltas, [0.025, 0.975])
    p_low  = float((deltas <= 0).mean())
    p_high = float((deltas >= 0).mean())
    p_two  = float(min(1.0, 2 * min(p_low, p_high)))
    return {
        "delta_obs": float(delta_obs),
        "ci95": (float(ci[0]), float(ci[1])),
        "p_two_sided": p_two,
        "B_effective": int(len(deltas)),
    }


def main(B: int = 10000, seed: int = 0) -> None:
    df = pd.read_csv(UNIFIED_RESULTS)
    df = df[df["model_id"].isin(STANDARD_14)].copy()

    rows = []
    full_results = {}
    for label, la, lb, single_col in CONTRASTS:
        gap_full_a = _gap_vector(df, la, "lang_avg")
        gap_full_b = _gap_vector(df, lb, "lang_avg")
        gap_sing_a = _gap_vector(df, la, single_col)
        gap_sing_b = _gap_vector(df, lb, single_col)
        common = sorted(set(gap_full_a.index) & set(gap_full_b.index)
                        & set(gap_sing_a.index) & set(gap_sing_b.index))
        x_full  = gap_full_a.loc[common].values
        y_full  = gap_full_b.loc[common].values
        x_sing  = gap_sing_a.loc[common].values
        y_sing  = gap_sing_b.loc[common].values
        rho_full = _spearman_paired(x_full, y_full)
        rho_sing = _spearman_paired(x_sing, y_sing)
        rng = np.random.default_rng(seed)
        n = len(common); deltas = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, size=n)
            r1 = _spearman_paired(x_full[idx], y_full[idx])
            r2 = _spearman_paired(x_sing[idx], y_sing[idx])
            deltas[i] = r1 - r2
        deltas = deltas[~np.isnan(deltas)]
        ci = np.quantile(deltas, [0.025, 0.975])
        delta_obs = rho_full - rho_sing
        p_low  = float((deltas <= 0).mean())
        p_high = float((deltas >= 0).mean())
        p_two  = float(min(1.0, 2 * min(p_low, p_high)))
        # Store raw floats; rounding happens once at display time so we
        # don't introduce a double-rounding artifact (e.g. 0.9384615 ->
        # round(_,4)=0.9385 -> :.3f="0.939" vs raw :.3f="0.938").
        row = {
            "contrast":    label,
            "n_models":    int(n),
            "rho_full":    float(rho_full),
            "rho_single":  float(rho_sing),
            "delta":       float(delta_obs),
            "ci95_lo":     float(ci[0]),
            "ci95_hi":     float(ci[1]),
            "p_two_sided": float(p_two),
        }
        rows.append(row)
        full_results[label] = {**row, "B": int(len(deltas)), "models": common}

    out_csv  = OUTPUTS  / "exp10_paired_bootstrap.csv"
    out_json = OUTPUTS / "exp10_paired_bootstrap.json"
    df_out = pd.DataFrame(rows)
    # Apply display rounding once, at write time.
    df_out.to_csv(out_csv, index=False, float_format="%.4f")
    out_json.write_text(json.dumps(full_results, indent=2, default=str))
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    print()
    # 3-dp display for the console table (matches manuscript convention).
    print(df_out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
