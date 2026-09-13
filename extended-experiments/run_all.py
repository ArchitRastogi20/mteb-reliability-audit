"""
Run all experiments in priority order. Each experiment is independent;
failure in one does not block the others.

Usage:
    python run_all.py
"""

from __future__ import annotations

import importlib
import time
import traceback

EXPERIMENTS = [
    "exp1_multilingual_only_aggregator",
    "exp3_stricter_english_taxonomy",
    "exp2_cluster_boxplot",
    "exp4_null_baseline",
    "exp5_threshold_sensitivity",
    "exp6_per_language_regret",
    "exp7_alt_aggregators",
    "exp8_roster_sensitivity",
]


def main() -> None:
    print("Running all experiments in priority order.\n")
    results = []
    for name in EXPERIMENTS:
        print(f"--- {name} ---")
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            mod.main()
            elapsed = time.time() - t0
            print(f"    ok ({elapsed:.1f}s)\n")
            results.append((name, "ok", elapsed))
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            results.append((name, f"failed: {e}", time.time() - t0))
            print()

    print("\nSummary:")
    for name, status, elapsed in results:
        print(f"  {name}: {status} ({elapsed:.1f}s)")
    print("\nSee outputs/SUMMARY.md for headline results.")


if __name__ == "__main__":
    main()
