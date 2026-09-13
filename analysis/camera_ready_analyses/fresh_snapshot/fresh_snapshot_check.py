"""Fresh-snapshot robustness check for the curated tier.

Re-extracts curated-25 scores from a fresh sparse clone of the public results
repository (commit 044b132, 2026-07-09) using the frozen pipeline's
definitions (18-task global aggregate; per-language consensus tasks;
full-coverage lang-avg), then compares per-language Kendall tau, pairwise
inversion rate, and the hidden-failure cluster against the frozen snapshot
(`mteb_audit_data/summary_curated.csv`, extracted from the 2026-05-08 clone).

Roster and the 18-task global definition are held fixed at the frozen
audit's values for comparability; only the scores are fresh.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[3]
# This analysis requires re-cloning the MTEB results repo at a later snapshot
# date (see NOTES.md); point FRESH_RESULTS_DIR at that clone's `results/` dir.
_FRESH_ENV = "FRESH_RESULTS_DIR"
if _FRESH_ENV not in os.environ:
    raise RuntimeError(
        f"Environment variable {_FRESH_ENV} is not set. This analysis requires "
        "a fresh sparse clone of the public MTEB results repository "
        "(https://github.com/embeddings-benchmark/results) taken at a later "
        "snapshot date than the frozen audit clone -- see "
        "analysis/camera_ready_analyses/fresh_snapshot/NOTES.md for the exact "
        "commit/date used previously. Set FRESH_RESULTS_DIR to that clone's "
        "'results/' directory before running this script."
    )
FRESH = Path(os.environ[_FRESH_ENV])
FRESH_COMMIT = "044b132709ad4f080ecb46c7289336b88b1412a6 (2026-07-09)"
OUT = Path(__file__).parent

spec = importlib.util.spec_from_file_location(
    "_common", ROOT / "extended-experiments" / "_common.py")
_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_common)
inversion_rate = _common.inversion_rate

LANG_BCP47 = {
    "ita": ["ita-Latn"], "jpn": ["jpn-Jpan"], "hin": ["hin-Deva"],
    "ara": ["ara-Arab", "arb-Arab"], "zho": ["zho-Hans"], "deu": ["deu-Latn"],
    "spa": ["spa-Latn"], "rus": ["rus-Cyrl"], "fra": ["fra-Latn"],
    "kor": ["kor-Hang", "kor-Kore"], "ben": ["ben-Beng"], "vie": ["vie-Latn"],
    "ind": ["ind-Latn"], "fas": ["pes-Arab", "fas-Arab"],
    "swa": ["swh-Latn", "swa-Latn"], "tha": ["tha-Thai"], "tel": ["tel-Telu"],
}
LANG_TASKS = ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives",
              "MLQARetrieval", "WikipediaRetrievalMultilingual",
              "MrTidyRetrieval", "XPQARetrieval"]
MMTEB_RETRIEVAL_TASKS = [
    "AILAStatutes", "ArguAna", "BelebeleRetrieval", "CovidRetrieval",
    "HagridRetrieval", "LEMBPasskeyRetrieval", "LegalBenchCorporateLobbying",
    "MIRACLRetrievalHardNegatives", "MLQARetrieval", "SCIDOCS", "SpartQA",
    "StackOverflowQA", "StatcanDialogueDatasetRetrieval", "TRECCOVID",
    "TempReasonL1", "TwitterHjerneRetrieval", "WikipediaRetrievalMultilingual",
    "WinoGrande",
]
ALL_BCP = {c for codes in LANG_BCP47.values() for c in codes}
HIDDEN_FAILURE_MODELS = ["granite-311m", "harrier-0.6b", "Seed1.6-embed",
                         "inf-retriever-v1"]


def collect_model(rev_dir: Path):
    """(task -> {bcp47: ndcg}) for LANG_TASKS and (task -> mean ndcg) for the 18."""
    lang_scores, global_scores = {}, {}
    for task in set(LANG_TASKS) | set(MMTEB_RETRIEVAL_TASKS):
        p = rev_dir / f"{task}.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        scores_obj = data.get("scores", {})
        test_block = scores_obj.get("test") or scores_obj.get("dev") or []
        per_lang = {}
        if isinstance(test_block, list):
            for e in test_block:
                s = e.get("ndcg_at_10")
                if s is None:
                    continue
                for lang in e.get("languages", []):
                    if lang in ALL_BCP:
                        per_lang[lang] = float(s)
        if task in MMTEB_RETRIEVAL_TASKS:
            vals = []
            for split in scores_obj.values():
                if isinstance(split, list):
                    for e in split:
                        s = e.get("ndcg_at_10", e.get("main_score"))
                        if s is not None:
                            vals.append(float(s))
            if vals:
                global_scores[task] = float(np.mean(vals))
        if task in LANG_TASKS and per_lang:
            lang_scores[task] = per_lang
    return lang_scores, global_scores


def main():
    mapping = pd.read_csv(ROOT / "analysis" / "camera_ready_analyses" / "fork_concordance"
                          / "curated_model_mapping.csv")
    per_model_lang, per_model_global, revs = {}, {}, {}
    for _, r in mapping.iterrows():
        mdir = FRESH / r.results_dir
        rev_dirs = sorted(d for d in mdir.iterdir()
                          if d.is_dir() and d.name != "external")
        if not rev_dirs:
            rev_dirs = sorted(d for d in mdir.iterdir() if d.is_dir())
        rev = rev_dirs[-1]  # lexicographically last, frozen-pipeline convention
        revs[r.display_name] = rev.name
        ls, gs = collect_model(rev)
        per_model_lang[r.display_name] = ls
        per_model_global[r.display_name] = gs

    gagg = {m: np.mean([v for v in g.values()])
            for m, g in per_model_global.items() if g}
    n_gtasks = {m: len(g) for m, g in per_model_global.items()}
    print(f"fresh snapshot {FRESH_COMMIT}: global agg for {len(gagg)}/25 models "
          f"(tasks in agg: min {min(n_gtasks.values())}, max {max(n_gtasks.values())})")

    frozen = pd.read_csv(ROOT / "mteb-ranking-audit" / "results" / "summary_curated.csv")
    frozen = frozen.set_index("iso3")

    rows, hf_cells = [], {m: 0 for m in HIDDEN_FAILURE_MODELS}
    for iso, codes in LANG_BCP47.items():
        scores = {}
        for m, tasks in per_model_lang.items():
            row = {}
            for t, langs in tasks.items():
                for c in codes:
                    if c in langs:
                        row[t] = langs[c]
                        break
            scores[m] = row
        tasks_seen = sorted({t for r_ in scores.values() for t in r_})
        if not tasks_seen:
            continue
        n_any = sum(1 for r_ in scores.values() if r_)
        min_m = max(1, int(np.ceil(0.5 * n_any)))
        consensus = [t for t in tasks_seen
                     if sum(1 for r_ in scores.values() if t in r_) >= min_m]
        pool = {m: np.mean([r_[t] for t in consensus])
                for m, r_ in scores.items()
                if all(t in r_ for t in consensus) and m in gagg}
        if len(pool) < 5:
            continue
        models = list(pool)
        lang_v = np.array([pool[m] for m in models])
        glob_v = np.array([gagg[m] for m in models])
        tau = kendalltau(glob_v, lang_v).statistic
        inv = inversion_rate(glob_v, lang_v)
        n = len(models)
        # hidden failures: top-half global, bottom-quartile local
        grank = pd.Series(glob_v, index=models).rank(ascending=False)
        lrank = pd.Series(lang_v, index=models).rank(ascending=False)
        for m in HIDDEN_FAILURE_MODELS:
            if m in pool and grank[m] <= n / 2 and lrank[m] >= 0.75 * n:
                hf_cells[m] += 1
        fz = frozen.loc[iso] if iso in frozen.index else None
        rows.append({
            "iso3": iso, "n_models": n, "n_tasks": len(consensus),
            "tau_fresh": round(tau, 3), "inv_fresh": round(inv, 3),
            "tau_frozen": round(float(fz["kendall_tau"]), 3) if fz is not None else None,
            "inv_frozen": round(float(fz["inversion_rate"]), 3) if fz is not None else None,
        })

    df = pd.DataFrame(rows).sort_values("iso3")
    df.to_csv(OUT / "fresh_snapshot_summary.csv", index=False)
    med_inv_f, med_tau_f = df.inv_fresh.median(), df.tau_fresh.median()
    rho, rho_p = spearmanr(df.n_tasks, df.inv_fresh)
    print(df.to_string(index=False))
    print(f"\nMEDIANS fresh: inversion {med_inv_f:.3f}, tau {med_tau_f:.3f} "
          f"(frozen: {df.inv_frozen.median():.3f} / {df.tau_frozen.median():.3f})")
    print(f"coverage rho(n_tasks, inv) fresh: {rho:.3f} (p={rho_p:.3f}); frozen: -0.59")
    print("hidden-failure cells (fresh):",
          {m: c for m, c in hf_cells.items()})
    pd.DataFrame([{"model": m, "cells_fresh": c} for m, c in hf_cells.items()]
                 ).to_csv(OUT / "fresh_hidden_failures.csv", index=False)
    (OUT / "fresh_revisions.csv").write_text(
        "model,revision\n" + "\n".join(f"{m},{r}" for m, r in sorted(revs.items())),
        encoding="utf-8")


if __name__ == "__main__":
    main()
