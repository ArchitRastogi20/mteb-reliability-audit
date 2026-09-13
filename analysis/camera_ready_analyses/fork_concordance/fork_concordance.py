"""Concordance analysis for dedicated per-language benchmark forks.

For each dedicated per-language benchmark available in the locally installed
`mteb` registry (French, Persian, Russian, Chinese, Arabic, Vietnamese forks),
this script quantifies, using ONLY local data (no network access):

  (a) How much of a deployment-relevant 25-model curated roster the fork
      covers in the public raw-results clone (voluntary-submission sparsity).
  (b) Whether the fork's model ranking agrees more with our per-language
      retrieval ranking than with the global multilingual aggregate ranking
      (Kendall tau on the overlap set).

Inputs (all local):
  - `mteb` benchmark registry (task lists, task types, eval splits, subsets)
  - raw results clone: results/<org>__<model>/<revision>/<TaskName>.json
  - curated roster:            analysis/roster_membership.csv
  - per-language scores:       extended-experiments/data/lang_avg_per_model.csv
  - global aggregate scores:   extended-experiments/data/mteb_agg_per_model.csv

Outputs (written next to this script):
  - fork_concordance.csv       one row per fork
  - curated_model_mapping.csv  roster model id -> results-clone directory
  - fork_model_coverage.csv    per (fork, model) coverage fractions and scores
  - NOTES.md                   headlines and caveats
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

import mteb

# ---------------------------------------------------------------------------
# Paths (derived from this file's location; no hard-coded machine paths)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# RESULTS_ROOT is produced by running mteb-ranking-audit/code/01_download.py
# first (it clones the public MTEB results repo into mteb-ranking-audit/mteb_results_repo).
RESULTS_ROOT = REPO / "mteb-ranking-audit" / "mteb_results_repo" / "results"
ROSTER_CSV = REPO / "analysis" / "roster_membership.csv"
LANG_AVG_CSV = REPO / "extended-experiments" / "data" / "lang_avg_per_model.csv"
MTEB_AGG_CSV = REPO / "extended-experiments" / "data" / "mteb_agg_per_model.csv"

COVERAGE_THRESHOLD = 0.5  # model must have >= 50% of the fork's tasks
MIN_OVERLAP_FOR_TAU = 6   # below this, rank agreement is marked underpowered

# ---------------------------------------------------------------------------
# Fork definitions: (fork label, language iso639-3 used in our per-language
# data, registry-name candidates in priority order, language code prefixes
# used as a fallback subset filter)
# ---------------------------------------------------------------------------
FORKS = [
    ("MTEB-French", "fra", ["MTEB(fra, v1)"], {"fra", "fr"}),
    ("FaMTEB",      "fas", ["MTEB(fas, v2)", "MTEB(fas, v1)"], {"fas", "pes", "fa"}),
    ("ruMTEB",      "rus", ["MTEB(rus, v1.1)", "MTEB(rus, v1)"], {"rus", "ru"}),
    ("C-MTEB",      "zho", ["MTEB(cmn, v1)", "C-MTEB"], {"zho", "cmn", "zh"}),
    ("ArabicMTEB",  "ara", ["ArabicMTEB", "MTEB(ara, v1)", "MTEB(ara, v2)"], {"ara", "arb", "ar"}),
    ("VN-MTEB",     "vie", ["VN-MTEB (vie, v1)"], {"vie", "vi"}),
]

# Hidden-failure models of interest (matched by case-insensitive substring
# against display name / short model id / results directory name).
HIDDEN_FAILURE_SUBSTRINGS = [
    "granite-311m",
    "harrier-0.6b",
    "seed1.6-embed",
    "inf-retriever-v1",
    "granite-97m-r2",
]

# Extra (non-roster) model tracked only for hidden-failure coverage.
EXTRA_MODELS = [
    ("ibm-granite/granite-embedding-97m-multilingual-r2", "granite-97m-r2"),
]


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------
def resolve_forks():
    """Return ({fork_label: benchmark_or_None}, registry_names_by_fork)."""
    registry = {b.name: b for b in mteb.get_benchmarks()}
    lower = {k.lower(): k for k in registry}
    resolved, reg_names = {}, {}
    for fork, _iso, candidates, _pref in FORKS:
        found = None
        for cand in candidates:
            if cand.lower() in lower:
                found = registry[lower[cand.lower()]]
                break
        if found is None:
            # last resort: substring scan, case-insensitive
            probes = [c.lower() for c in candidates] + [f"({_iso}", fork.lower()]
            for name_l, name in lower.items():
                if any(p in name_l for p in probes):
                    found = registry[name]
                    break
        resolved[fork] = found
        reg_names[fork] = found.name if found is not None else "NOT_IN_LOCAL_REGISTRY"
    return resolved, reg_names


def benchmark_task_meta(bench):
    """Extract per-task metadata needed for scoring from a benchmark object."""
    metas = []
    for t in bench.tasks:
        m = t.metadata
        hf_subsets = list(getattr(t, "hf_subsets", None) or ["default"])
        metas.append({
            "name": m.name,
            "type": m.type,
            "eval_splits": list(m.eval_splits),
            "hf_subsets": hf_subsets,
        })
    return metas


# ---------------------------------------------------------------------------
# Model-id normalisation / mapping to results-clone directories
# ---------------------------------------------------------------------------
def build_model_mapping(roster_df, results_dirs):
    """Map each roster model_id to a results-clone directory.

    Strategy: (1) exact `<org>__<name>` match, case-insensitive;
              (2) unique match on the short name (part after the org prefix).
    """
    dir_by_lower = {d.lower(): d for d in results_dirs}
    short_index = {}
    for d in results_dirs:
        short = d.split("__", 1)[-1].lower()
        short_index.setdefault(short, []).append(d)

    rows = []
    for _, r in roster_df.iterrows():
        model_id, display = r["model_id"], r["display_name"]
        cand = model_id.replace("/", "__").lower()
        short = model_id.split("/")[-1].lower()
        results_dir, method = None, "UNMATCHED"
        if cand in dir_by_lower:
            results_dir, method = dir_by_lower[cand], "exact_org__name"
        elif short in short_index and len(short_index[short]) == 1:
            results_dir, method = short_index[short][0], "unique_short_name"
        rows.append({
            "model_id": model_id,
            "display_name": display,
            "results_dir": results_dir or "",
            "match_method": method,
        })
    return pd.DataFrame(rows)


def latest_rev_dir(model_dir: Path) -> Path:
    revs = [p for p in model_dir.iterdir() if p.is_dir()]
    return sorted(revs)[-1] if revs else model_dir


# ---------------------------------------------------------------------------
# Score extraction from raw result JSONs
# ---------------------------------------------------------------------------
def _finite(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def extract_task_score(json_path: Path, task_meta: dict, lang_prefixes: set):
    """Score for one (model, task): average subsets within a split, then splits.

    Split policy: prefer the 'test' split when present; otherwise use the
    task's declared eval splits that are present in the file (handles
    dev-only tasks); otherwise whatever splits exist.
    Subset policy: restrict to the benchmark's declared hf_subsets; if none
    match (naming drift in old files), fall back to language-prefix matching
    on the per-subset `languages` field; if that also fails, use all subsets.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    scores = data.get("scores") or {}
    if not scores:
        return None

    if "test" in scores:
        splits = ["test"]
    else:
        splits = [s for s in task_meta["eval_splits"] if s in scores] or list(scores)

    allowed = set(task_meta["hf_subsets"])
    split_means = []
    for sp in splits:
        entries = scores.get(sp) or []
        sel = [e for e in entries if e.get("hf_subset") in allowed]
        if not sel:
            sel = [
                e for e in entries
                if any(str(l).split("-")[0].lower() in lang_prefixes
                       for l in (e.get("languages") or []))
            ]
        if not sel:
            sel = entries
        vals = [e.get("main_score") for e in sel if _finite(e.get("main_score"))]
        if vals:
            split_means.append(sum(vals) / len(vals))
    if not split_means:
        return None
    return sum(split_means) / len(split_means)


def model_task_scores(rev_dir: Path, task_metas: list, lang_prefixes: set) -> dict:
    """Return {task_name: score} for tasks of the benchmark present for this model."""
    out = {}
    for tm in task_metas:
        fp = rev_dir / f"{tm['name']}.json"
        if fp.exists():
            s = extract_task_score(fp, tm, lang_prefixes)
            if s is not None:
                out[tm["name"]] = s
    return out


def summarize_model(per_task: dict, task_metas: list):
    """Coverage fractions and mean scores, full-benchmark and retrieval-only."""
    n_all = len(task_metas)
    retr_names = [tm["name"] for tm in task_metas if tm["type"] == "Retrieval"]
    n_retr = len(retr_names)

    cov_full = len(per_task) / n_all if n_all else 0.0
    retr_scores = [per_task[n] for n in retr_names if n in per_task]
    cov_retr = len(retr_scores) / n_retr if n_retr else 0.0

    score_full = (sum(per_task.values()) / len(per_task)
                  if per_task and cov_full >= COVERAGE_THRESHOLD else None)
    score_retr = (sum(retr_scores) / len(retr_scores)
                  if retr_scores and cov_retr >= COVERAGE_THRESHOLD else None)
    return cov_full, cov_retr, score_full, score_retr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for p in (RESULTS_ROOT, ROSTER_CSV, LANG_AVG_CSV, MTEB_AGG_CSV):
        if not p.exists():
            sys.exit(f"Missing required input: {p}")

    roster = pd.read_csv(ROSTER_CSV)
    lang_avg = pd.read_csv(LANG_AVG_CSV)
    lang_avg = lang_avg[lang_avg["tier"] == "curated"]
    agg = pd.read_csv(MTEB_AGG_CSV)
    agg = agg[agg["tier"] == "curated"].set_index("model_id")["mteb_agg"]

    results_dirs = sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())
    print(f"results clone: {len(results_dirs)} model directories")

    # -- curated model -> results dir mapping ------------------------------
    mapping = build_model_mapping(roster, results_dirs)
    print("\n=== Curated 25-model roster -> results-clone directory mapping ===")
    for _, r in mapping.iterrows():
        print(f"  {r['model_id']:<55} -> {r['results_dir'] or '(NO MATCH)':<55} [{r['match_method']}]")
    mapping.to_csv(HERE / "curated_model_mapping.csv", index=False)
    unmatched = mapping[mapping["match_method"] == "UNMATCHED"]
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} curated models unmatched: "
              f"{unmatched['model_id'].tolist()}")

    # evaluation set: curated 25 + extra hidden-failure model(s)
    eval_models = []  # (display_name, results_dir, in_curated_roster)
    for _, r in mapping.iterrows():
        if r["results_dir"]:
            eval_models.append((r["display_name"], r["results_dir"], True))
    dir_by_lower = {d.lower(): d for d in results_dirs}
    for mid, disp in EXTRA_MODELS:
        d = dir_by_lower.get(mid.replace("/", "__").lower())
        if d and disp not in {m[0] for m in eval_models}:
            eval_models.append((disp, d, False))

    # -- resolve fork benchmarks in the local registry ---------------------
    resolved, reg_names = resolve_forks()
    print("\n=== Fork benchmarks in local registry ===")
    for fork, _iso, _c, _p in FORKS:
        b = resolved[fork]
        print(f"  {fork:<12} -> {reg_names[fork]}"
              + (f"  ({len(list(b.tasks))} tasks)" if b is not None else ""))

    sanity_triples = []
    fork_rows, cov_rows = [], []

    for fork, iso, _cands, lang_prefixes in FORKS:
        bench = resolved[fork]
        if bench is None:
            fork_rows.append({
                "fork": fork, "language": iso,
                "registry_name": "NOT_IN_LOCAL_REGISTRY", "status": "not_in_local_registry",
                "n_tasks": 0, "n_tasks_retrieval": 0, "n_models_total": None,
                "n_curated_covered": None, "hidden_failure_models_covered": "",
                "hidden_failure_models_covered_retrieval": "",
                "tau_fork_vs_global": None, "tau_fork_vs_langavg": None, "n_overlap": None,
                "tau_retrieval_vs_global": None, "tau_retrieval_vs_langavg": None,
                "n_overlap_retrieval": None,
            })
            continue

        task_metas = benchmark_task_meta(bench)
        n_tasks = len(task_metas)
        n_retr = sum(1 for tm in task_metas if tm["type"] == "Retrieval")
        print(f"\n=== {fork} [{reg_names[fork]}]: {n_tasks} tasks "
              f"({n_retr} retrieval) ===")

        # ---- curated roster (+ extra) scoring ----
        per_model = {}
        for disp, dname, in_roster in eval_models:
            rev = latest_rev_dir(RESULTS_ROOT / dname)
            pt = model_task_scores(rev, task_metas, lang_prefixes)
            cov_f, cov_r, sc_f, sc_r = summarize_model(pt, task_metas)
            per_model[disp] = {"cov_full": cov_f, "cov_retr": cov_r,
                               "score_full": sc_f, "score_retr": sc_r,
                               "in_roster": in_roster, "dir": dname}
            cov_rows.append({"fork": fork, "model": disp, "in_curated_roster": in_roster,
                             "n_tasks_scored": len(pt),
                             "coverage_full": round(cov_f, 4),
                             "coverage_retrieval": round(cov_r, 4),
                             "fork_score_full": sc_f, "fork_score_retrieval": sc_r})
            if len(sanity_triples) < 3 and pt:
                tname = sorted(pt)[0]
                sanity_triples.append((disp, dname, tname, pt[tname]))

        covered = {d: v for d, v in per_model.items()
                   if v["in_roster"] and v["score_full"] is not None}
        n_curated_covered = len(covered)

        # hidden-failure coverage (full-benchmark and retrieval-only thresholds)
        hf_covered, hf_covered_retr = [], []
        for disp, v in per_model.items():
            aliases = f"{disp} {v['dir']}".lower()
            if any(s in aliases for s in HIDDEN_FAILURE_SUBSTRINGS):
                if v["score_full"] is not None:
                    hf_covered.append(disp)
                if v["score_retr"] is not None:
                    hf_covered_retr.append(disp)
        hf_covered = sorted(set(hf_covered))
        hf_covered_retr = sorted(set(hf_covered_retr))

        # ---- rank agreement on the overlap set ----
        lang_scores = lang_avg[lang_avg["language_iso"] == iso] \
            .set_index("model_id")["lang_avg_ndcg"]

        def tau_pair(score_key):
            ok = [d for d, v in per_model.items()
                  if v["in_roster"] and v[score_key] is not None
                  and d in agg.index and d in lang_scores.index]
            n = len(ok)
            if n < MIN_OVERLAP_FOR_TAU:
                return None, None, n
            fs = [per_model[d][score_key] for d in ok]
            t_g, _ = kendalltau(fs, [agg[d] for d in ok])
            t_l, _ = kendalltau(fs, [lang_scores[d] for d in ok])
            return t_g, t_l, n

        tau_g, tau_l, n_ov = tau_pair("score_full")
        tau_g_r, tau_l_r, n_ov_r = tau_pair("score_retr")

        # ---- whole-clone roster size at threshold ----
        n_total = 0
        for dname in results_dirs:
            rev = latest_rev_dir(RESULTS_ROOT / dname)
            # cheap existence pre-check before parsing
            present = [tm for tm in task_metas if (rev / f"{tm['name']}.json").exists()]
            if len(present) / n_tasks < COVERAGE_THRESHOLD:
                continue
            n_scored = 0
            for tm in present:
                if extract_task_score(rev / f"{tm['name']}.json", tm, lang_prefixes) is not None:
                    n_scored += 1
            if n_scored / n_tasks >= COVERAGE_THRESHOLD:
                n_total += 1

        print(f"  models in whole clone meeting >={COVERAGE_THRESHOLD:.0%} coverage: "
              f"{n_total} / {len(results_dirs)}")
        print(f"  curated-25 covered: {n_curated_covered} / 25 ; "
              f"hidden-failure covered (full): {hf_covered} ; "
              f"(retrieval-only): {hf_covered_retr}")
        print(f"  tau(full) fork-vs-global={tau_g} fork-vs-langavg={tau_l} n={n_ov}"
              + ("  [UNDERPOWERED]" if n_ov is not None and n_ov < MIN_OVERLAP_FOR_TAU else ""))
        print(f"  tau(retr) fork-vs-global={tau_g_r} fork-vs-langavg={tau_l_r} n={n_ov_r}")

        fork_rows.append({
            "fork": fork, "language": iso, "registry_name": reg_names[fork],
            "status": "ok" if n_ov >= MIN_OVERLAP_FOR_TAU else "underpowered",
            "n_tasks": n_tasks, "n_tasks_retrieval": n_retr,
            "n_models_total": n_total, "n_curated_covered": n_curated_covered,
            "hidden_failure_models_covered": ";".join(hf_covered),
            "hidden_failure_models_covered_retrieval": ";".join(hf_covered_retr),
            "tau_fork_vs_global": tau_g, "tau_fork_vs_langavg": tau_l, "n_overlap": n_ov,
            "tau_retrieval_vs_global": tau_g_r, "tau_retrieval_vs_langavg": tau_l_r,
            "n_overlap_retrieval": n_ov_r,
        })

    out = pd.DataFrame(fork_rows)
    col_order = ["fork", "language", "registry_name", "n_tasks", "n_tasks_retrieval",
                 "n_models_total", "n_curated_covered", "hidden_failure_models_covered",
                 "hidden_failure_models_covered_retrieval",
                 "tau_fork_vs_global", "tau_fork_vs_langavg", "n_overlap",
                 "tau_retrieval_vs_global", "tau_retrieval_vs_langavg",
                 "n_overlap_retrieval", "status"]
    out = out[col_order]
    out.to_csv(HERE / "fork_concordance.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(HERE / "fork_model_coverage.csv", index=False)

    # ---- sanity check: print triples and re-verify one against raw JSON ----
    print("\n=== SANITY: example (model, task, score) triples ===")
    for disp, dname, tname, sc in sanity_triples:
        print(f"  ({disp}, {tname}, {sc:.6f})")
    if sanity_triples:
        disp, dname, tname, sc = sanity_triples[0]
        fp = latest_rev_dir(RESULTS_ROOT / dname) / f"{tname}.json"
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        print(f"  re-verify against raw file: {fp}")
        for sp, entries in raw["scores"].items():
            for e in entries:
                print(f"    split={sp} hf_subset={e.get('hf_subset')} "
                      f"main_score={e.get('main_score')}")
        print(f"    computed task score: {sc}")

    print("\nWrote:")
    for f in ("fork_concordance.csv", "curated_model_mapping.csv", "fork_model_coverage.csv"):
        print(f"  {HERE / f}")


if __name__ == "__main__":
    main()
