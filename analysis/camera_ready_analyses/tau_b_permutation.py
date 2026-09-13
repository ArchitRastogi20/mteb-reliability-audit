"""Tie-aware Kendall tau-b + permutation test: n_tasks vs per-language inversion rate.

Inputs: the released per-language summary CSVs (17 audit languages each).
Output: tau_b_permutation.csv with tau-b and 10k-permutation p per tier.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "curated": ROOT / "mteb-ranking-audit" / "results" / "summary_curated.csv",
    "extended": ROOT / "mteb-ranking-audit" / "results" / "summary_extended.csv",
}
N_PERM = 10_000
SEED = 12345


def perm_test(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    obs = kendalltau(x, y, variant="b").statistic
    count = 0
    for _ in range(N_PERM):
        t = kendalltau(rng.permutation(x), y, variant="b").statistic
        if abs(t) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1) / (N_PERM + 1)  # add-one permutation p
    return obs, p


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for tier, path in INPUTS.items():
        df = pd.read_csv(path)
        x = df["n_tasks"].to_numpy(float)
        y = df["inversion_rate"].to_numpy(float)
        tau_b, p = perm_test(x, y, rng)
        rows.append({"tier": tier, "n_languages": len(df), "tau_b": round(tau_b, 4),
                     "perm_p": round(p, 4), "n_perm": N_PERM, "seed": SEED})
        print(f"{tier}: n={len(df)} tau_b={tau_b:.4f} perm_p={p:.4f}")
    out = Path(__file__).with_name("tau_b_permutation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
