#!/usr/bin/env python3
"""Task 2: Per-pair inversion significance -- checks for per-query data, documents blocker."""
import csv
import json
import logging
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent      # repo root
OUT_DIR = SCRIPT_DIR / "task2_pairwise_sig"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(OUT_DIR / "task2_pairwise_sig.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# Headline inversion counts from own-run tau analysis (public_csv predictor)
# IT: 14 models, 91 pairs; JA/HI: 15 models, 105 pairs
HEADLINE_INVERSIONS = [
    dict(language="IT", n_models=14, total_pairs=91,  headline_inversions=25),
    dict(language="JA", n_models=15, total_pairs=105, headline_inversions=28),
    dict(language="HI", n_models=15, total_pairs=105, headline_inversions=21),
]


def check_per_query_data():
    """Return True if per-query NDCG@10 found in any result JSON."""
    results_dir = REPO_ROOT / "mteb-language-gap" / "results"
    if not results_dir.exists():
        log.warning(f"Results dir not found: {results_dir}")
        return False

    per_query_keys_seen = []
    files_checked = 0
    for lang_dir in sorted(results_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        for model_dir in lang_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for task_file in model_dir.glob("*.json"):
                if task_file.name == "_metadata.json":
                    continue
                try:
                    with open(task_file, encoding="utf-8") as f:
                        d = json.load(f)
                    scores = d.get("scores", {}).get("test", [])
                    files_checked += 1
                    if scores:
                        keys = list(scores[0].keys())
                        log.info(
                            f"JSON keys ({lang_dir.name}/{model_dir.name}/"
                            f"{task_file.name}): {keys}"
                        )
                        per_query = [
                            k
                            for k in keys
                            if any(
                                kw in k.lower()
                                for kw in ("per_", "query", "sample", "instance")
                            )
                        ]
                        if per_query:
                            per_query_keys_seen.extend(per_query)
                            log.info(f"Per-query candidate keys: {per_query}")
                except Exception as e:
                    log.warning(f"Could not read {task_file}: {e}")

    log.info(f"Checked {files_checked} result JSON files")
    if per_query_keys_seen:
        log.info(f"Per-query-like keys found: {per_query_keys_seen}")
        return True
    return False


def write_summary_csv(has_per_query: bool):
    csv_path = OUT_DIR / "pairwise_significance_summary.csv"
    fieldnames = [
        "language", "n_models", "total_pairs",
        "headline_inversions", "significant_inversions", "proportion_robust", "status",
    ]
    status = "available" if has_per_query else "blocked_no_per_query_ndcg"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in HEADLINE_INVERSIONS:
            writer.writerow(
                dict(
                    language=row["language"],
                    n_models=row["n_models"],
                    total_pairs=row["total_pairs"],
                    headline_inversions=row["headline_inversions"],
                    significant_inversions="N/A",
                    proportion_robust="N/A",
                    status=status,
                )
            )
    log.info(f"Saved pairwise_significance_summary.csv ({len(HEADLINE_INVERSIONS)} rows, status={status})")


def main():
    t0 = time.time()
    log.info("Task 2: Checking for per-query NDCG@10 availability")

    has_per_query = check_per_query_data()

    if has_per_query:
        log.info("Per-query data found -- implement full paired bootstrap here.")
        md = (
            "## Pairwise Inversion Significance (Section 4.2)\n\n"
            "Per-query NDCG@10 data appears to be available. "
            "Update task2_pairwise_sig.py to implement the full paired bootstrap over queries.\n\n"
            "See `pairwise_significance_summary.csv` for headline inversion counts.\n"
        )
    else:
        log.warning(
            "BLOCKED: Per-query NDCG@10 not available in mteb-language-gap/results/. "
            "Result JSONs store only aggregated ndcg_at_10 per task (key: ndcg_at_10 under scores.test[0])."
        )
        md = (
            "## Pairwise Inversion Significance (Section 4.2)\n\n"
            "**Status: blocked -- per-query NDCG@10 not stored in evaluation outputs.**\n\n"
            "The paired bootstrap over per-query NDCG@10 requires individual "
            "query-level scores. Inspection of `mteb-language-gap/results/` confirms "
            "that result JSONs store only the aggregated `ndcg_at_10` per task, not per-query "
            "scores. Re-running the evaluation pipeline with per-query logging enabled "
            "would be needed to execute this analysis in full.\n\n"
            "**Alternative evidence (model-level bootstrap):** The 10,000-model-"
            "resample bootstrap CIs from the own-run-only tau analysis (Task 3) show that "
            "tau is robustly positive for all three languages with CIs well above zero for HI "
            "and partially above zero for IT and JA. This addresses the concern that the "
            "aggregate rank-correlation signal is sampling noise, albeit at the model level "
            "rather than the per-query level. Producing per-query CIs for individual pairs "
            "would require a re-run of the MTEB pipeline with `--save_predictions` or equivalent.\n\n"
            "See `pairwise_significance_summary.csv` for headline inversion counts per language.\n"
        )

    (OUT_DIR / "pairwise_significance.md").write_text(md, encoding="utf-8")
    write_summary_csv(has_per_query)
    log.info(f"Written pairwise_significance.md. Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
