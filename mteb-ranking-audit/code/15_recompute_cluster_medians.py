"""
Recompute the `median_local_rank` column of results/hidden_failure_cluster.csv
directly from results/hidden_failures.csv.

Why this script exists
-----------------------
The original generator (code/06_followup_analyses.py, hidden_failure_cluster())
computes, per model:

    sub = hf[hf["model"] == model]
    median_local_rank = float(sub["local_rank"].median())

i.e. it pools every row for that model across ALL tiers (curated, extended,
restricted) and ALL languages -- including Tamil ("tam") -- into one median.
This is not a valid pooling: "restricted" has only 7 models, so a restricted
local_rank of 5 or 6 is near the BOTTOM of that tier, but numerically looks
like a good (small) rank next to curated-tier values that range 1-25. Mixing
the two scales pulls the pooled median toward whichever tier happens to have
more rows. Tamil is also not one of the 17 audit languages in the
camera-ready paper (it was dropped for having only one available task), so
it should not contribute to a language-count-normalized statistic.

The camera-ready paper corrects this: median_local_rank is computed (a) only
from Tamil-excluded rows, and (b) only from the single tier in which the
model has the most affected languages (its "primary" tier -- curated for
four of the five cluster models, extended for the one, granite-97m-r2, that
has no curated-tier rows at all). Concretely, per model:

    sub = hidden_failures[(hidden_failures.model == model) &
                           (hidden_failures.iso3 != "tam")]
    primary_tier = sub.groupby("tier")["iso3"].nunique().idxmax()
    median_local_rank = sub[sub.tier == primary_tier]["local_rank"].median()

Running this against the released results/hidden_failures.csv reproduces
exactly the four camera-ready values (21 / 20 / 20 / 40.5) and leaves
granite-311m's value unchanged at 24 -- all recomputed straight from
released data, no numbers invented or hand-edited.

No other column of hidden_failure_cluster.csv is touched: best_global_rank,
n_languages_affected_{curated,extended,restricted}, and mteb_agg_curated are
read from the existing file and passed through unchanged, and row order is
preserved.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

HIDDEN_FAILURES = RESULTS / "hidden_failures.csv"
CLUSTER_TABLE = RESULTS / "hidden_failure_cluster.csv"

NOT_AN_AUDIT_LANGUAGE = {"tam"}  # Tamil: dropped from the 17 audit languages (n_tasks=1)


def recompute_median_local_rank(hf: pd.DataFrame, model: str) -> float:
    sub = hf[(hf["model"] == model) & (~hf["iso3"].isin(NOT_AN_AUDIT_LANGUAGE))]
    if sub.empty:
        raise ValueError(f"No non-Tamil rows for model {model!r} in hidden_failures.csv")
    primary_tier = sub.groupby("tier")["iso3"].nunique().idxmax()
    tier_sub = sub[sub["tier"] == primary_tier]
    return float(tier_sub["local_rank"].median())


def main() -> None:
    hf = pd.read_csv(HIDDEN_FAILURES)
    cluster = pd.read_csv(CLUSTER_TABLE)

    corrected = cluster.copy()
    changes = []
    for i, row in cluster.iterrows():
        model = row["model"]
        old = row["median_local_rank"]
        new = round(recompute_median_local_rank(hf, model), 1)
        corrected.loc[i, "median_local_rank"] = new
        if old != new:
            changes.append((model, old, new))

    corrected.to_csv(CLUSTER_TABLE, index=False)

    print(f"Rewrote {CLUSTER_TABLE} ({len(corrected)} rows).")
    if changes:
        print("Changed median_local_rank for:")
        for model, old, new in changes:
            print(f"  {model}: {old} -> {new}")
    else:
        print("No changes -- file was already consistent with the corrected method.")


if __name__ == "__main__":
    main()
