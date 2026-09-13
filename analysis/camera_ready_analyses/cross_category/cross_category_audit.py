"""Cross-category rank-agreement audit on public MTEB leaderboard results.

Replicates the retrieval-audit methodology (global-aggregate ranking vs
per-language ranking, per language) for the Classification, STS and
Reranking task categories of the multilingual benchmark, using ONLY the
locally cloned public results repository. No network access, no own runs.

Method per category C:
  1. T_C = multilingual-benchmark tasks with metadata.type == C.
  2. Per model: task score = mean(main_score) over the task's eval-split
     subsets (all languages); global C-aggregate = mean over tasks in T_C,
     requiring coverage >= 80% of |T_C|.
  3. Per-language score for L: mean over L's subsets within a task first,
     then across tasks exposing L. A subset counts for L only if it lists
     <= 2 languages (monolingual or crosslingual pair; pooled language-ID
     style subsets with many languages are excluded).
  4. Rosters: (a) "extended" = all covered models; (b) "curated25" =
     intersection with the curated 25-model roster.
  5. Per language (mirroring the retrieval audit): consensus task set =
     tasks exposing L covered by >= 50% of roster models with any L data;
     keep models with full coverage of the consensus set; require
     n_tasks >= 2 and n_models >= 6. Kendall tau-b (+p), pairwise
     inversion rate (ties = 0.5), BH-FDR across languages within
     (category, roster).
  6. Spearman rho between n_tasks(C, L) and inversion rate.

Sanity checks (run first):
  A. Faithful re-derivation of the retrieval extended-tier median
     inversion rate from the same raw path (anchor: 15.4%).
  B. Hand spot-check of two task JSONs with parsed values printed.

Outputs (written next to this script):
  summary_<category>_<roster>.csv, headline.csv,
  threshold_sensitivity.csv, run.log (stdout tee'd by the caller).

Usage:
  python cross_category_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# RESULTS_ROOT is produced by running mteb-ranking-audit/code/01_download.py
# first (it clones the public MTEB results repo into mteb-ranking-audit/mteb_results_repo).
RESULTS_ROOT = ROOT / "mteb-ranking-audit" / "mteb_results_repo" / "results"
ROSTER_CSV = ROOT / "analysis" / "roster_membership.csv"

BENCHMARK_NAME = "MTEB(Multilingual, v2)"  # task lists for the three
# categories below are identical between v1 and v2 of the benchmark.
CATEGORIES = ["Classification", "STS", "Reranking"]

COVERAGE_THRESHOLD = 0.80      # fraction of |T_C| a model must have
THRESHOLD_GRID = [0.70, 0.80, 0.90, 1.00]
CONSENSUS_FRAC = 0.50          # per-language consensus task threshold
MIN_TASKS_PER_LANG = 2
MIN_MODELS = 6
UNDERPOWERED_N = 8             # curated intersection below this => flag
MAX_LANGS_PER_SUBSET = 2       # mono or crosslingual pair only

EXCLUDE_DIRS = {"mteb__baseline-bm25s", "mteb__baseline-random-encoder"}

AUDIT_LANGS = [
    "fra", "ind", "tel", "kor", "swa", "fas", "tha", "jpn", "rus",
    "ita", "zho", "deu", "ben", "ara", "hin", "spa", "vie",
]
LANG_NAMES = {
    "fra": "French", "ind": "Indonesian", "tel": "Telugu", "kor": "Korean",
    "swa": "Swahili", "fas": "Persian", "tha": "Thai", "jpn": "Japanese",
    "rus": "Russian", "ita": "Italian", "zho": "Chinese", "deu": "German",
    "ben": "Bengali", "ara": "Arabic", "hin": "Hindi", "spa": "Spanish",
    "vie": "Vietnamese",
}
# ISO-639-3 subtag aliases observed in result JSONs / task metadata.
# cmn = Mandarin, cmo = nonstandard code used by some tasks for Chinese,
# arb = Modern Standard Arabic, pes = Iranian Persian, swh = coastal Swahili.
SUBTAG_ALIAS = {"swh": "swa", "pes": "fas", "cmn": "zho", "cmo": "zho",
                "arb": "ara"}
AUDIT_SET = set(AUDIT_LANGS)

RNG_SEED = 42


def canon_subtag(bcp47: str) -> str:
    sub = bcp47.split("-")[0]
    return SUBTAG_ALIAS.get(sub, sub)


def inversion_rate(a, b) -> float:
    """Fraction of unordered pairs whose order disagrees between a and b.

    Ties (in either vector) contribute 0.5. Sign-based, so raw scores and
    ranks give identical results.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")
    pairs = n * (n - 1) / 2
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    iu = np.triu_indices(n, k=1)
    sa = np.sign(da[iu])
    sb = np.sign(db[iu])
    discordant = np.sum(sa * sb < 0)
    ties = np.sum((sa == 0) | (sb == 0))
    return float((discordant + 0.5 * ties) / pairs)


def pick_revision(model_dir: Path) -> Path:
    """Mirror the retrieval audit: alphabetically last revision subdir."""
    revs = [p for p in model_dir.iterdir() if p.is_dir()]
    return sorted(revs)[-1] if revs else model_dir


# ---------------------------------------------------------------------------
# Benchmark task metadata
# ---------------------------------------------------------------------------

def load_benchmark_tasks() -> dict[str, list[dict]]:
    import mteb  # local package; no network needed for metadata

    bench = mteb.get_benchmark(BENCHMARK_NAME)
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for task in bench.tasks:
        cat = task.metadata.type
        if cat not in by_cat:
            continue
        el = task.metadata.eval_langs
        if isinstance(el, dict):
            subset_langs = list(el.values())
        else:
            subset_langs = [list(el)]
        exposed: set[str] = set()
        for codes in subset_langs:
            if len(codes) > MAX_LANGS_PER_SUBSET:
                continue
            for c in codes:
                cc = canon_subtag(c)
                if cc in AUDIT_SET:
                    exposed.add(cc)
        by_cat[cat].append({
            "name": task.metadata.name,
            "eval_splits": list(task.metadata.eval_splits),
            "main_score": task.metadata.main_score,
            "exposed_langs_meta": exposed,
        })
    for c in CATEGORIES:
        by_cat[c].sort(key=lambda d: d["name"])
    return by_cat


# ---------------------------------------------------------------------------
# Result-JSON parsing (generic, main_score based)
# ---------------------------------------------------------------------------

def parse_task_json(path: Path, eval_splits: list[str]):
    """Return (task_score, {iso3: per-language mean}) or None.

    task_score = mean(main_score) over all subset entries of the task's
    eval splits (falling back to test/validation/dev, then any split).
    Per-language values use only subsets listing <= MAX_LANGS_PER_SUBSET
    languages.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # unreadable file
        print(f"  [warn] could not read {path.name}: {exc}")
        return None

    scores = data.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        return None
    splits = [s for s in eval_splits if s in scores]
    if not splits:
        splits = [s for s in ("test", "validation", "dev") if s in scores]
    if not splits:
        splits = list(scores.keys())

    all_vals: list[float] = []
    lang_vals: dict[str, list[float]] = {}
    for split in splits:
        block = scores.get(split)
        if not isinstance(block, list):
            continue
        for entry in block:
            v = entry.get("main_score")
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            all_vals.append(v)
            langs = entry.get("languages", [])
            if not langs or len(langs) > MAX_LANGS_PER_SUBSET:
                continue
            hits = {canon_subtag(c) for c in langs} & AUDIT_SET
            for L in hits:
                lang_vals.setdefault(L, []).append(v)
    if not all_vals:
        return None
    return (
        float(np.mean(all_vals)),
        {L: float(np.mean(vs)) for L, vs in lang_vals.items()},
    )


def scan_all_models(by_cat: dict[str, list[dict]]):
    """One pass over the results repo.

    Returns {category: {model_dir_name: {task_name: (score, lang_scores)}}}.
    """
    data: dict[str, dict[str, dict[str, tuple]]] = {c: {} for c in CATEGORIES}
    n_dirs = 0
    for mdir in sorted(RESULTS_ROOT.iterdir()):
        if not mdir.is_dir() or mdir.name in EXCLUDE_DIRS:
            continue
        n_dirs += 1
        rev = pick_revision(mdir)
        for cat, tasks in by_cat.items():
            per_task: dict[str, tuple] = {}
            for t in tasks:
                p = rev / f"{t['name']}.json"
                if not p.exists():
                    continue
                parsed = parse_task_json(p, t["eval_splits"])
                if parsed is not None:
                    per_task[t["name"]] = parsed
            if per_task:
                data[cat][mdir.name] = per_task
    print(f"Scanned {n_dirs} model directories under {RESULTS_ROOT}")
    return data


# ---------------------------------------------------------------------------
# Per-category / per-roster analysis
# ---------------------------------------------------------------------------

def analyze(cat: str, tasks: list[dict], cat_data: dict, roster: set[str] | None,
            roster_label: str, coverage: float = COVERAGE_THRESHOLD,
            verbose: bool = True):
    """Return (summary_df, info dict)."""
    n_T = len(tasks)
    min_tasks = coverage * n_T
    covered = {
        m: d for m, d in cat_data.items()
        if len(d) >= min_tasks and (roster is None or m in roster)
    }
    global_agg = {
        m: float(np.mean([ts for ts, _ in d.values()])) for m, d in covered.items()
    }
    meta_expose = {L: sum(1 for t in tasks if L in t["exposed_langs_meta"])
                   for L in AUDIT_LANGS}

    rows = []
    skipped = {}
    for L in AUDIT_LANGS:
        task_counts: Counter = Counter()
        models_any = set()
        for m, d in covered.items():
            for tname, (_, lv) in d.items():
                if L in lv:
                    task_counts[tname] += 1
                    models_any.add(m)
        n_candidates = len(task_counts)
        if not models_any:
            skipped[L] = "no data"
            continue
        min_models_for_task = max(1, math.ceil(CONSENSUS_FRAC * len(models_any)))
        consensus = sorted(
            t for t, c in task_counts.items() if c >= min_models_for_task
        )
        if len(consensus) < MIN_TASKS_PER_LANG:
            skipped[L] = (f"n_tasks={len(consensus)} (<{MIN_TASKS_PER_LANG}; "
                          f"candidates={n_candidates}, meta={meta_expose[L]})")
            continue
        kept = [
            m for m, d in covered.items()
            if all(t in d and L in d[t][1] for t in consensus)
        ]
        if len(kept) < MIN_MODELS:
            skipped[L] = f"n_models={len(kept)} (<{MIN_MODELS})"
            continue
        lang_avg = {
            m: float(np.mean([covered[m][t][1][L] for t in consensus]))
            for m in kept
        }
        g = np.array([global_agg[m] for m in kept])
        l = np.array([lang_avg[m] for m in kept])
        g_ranks = rankdata(g, method="average")
        l_ranks = rankdata(l, method="average")
        tau, p = kendalltau(g_ranks, l_ranks)
        rows.append({
            "language": L,
            "lang_name": LANG_NAMES[L],
            "n_models": len(kept),
            "n_tasks": len(consensus),
            "n_tasks_candidates": n_candidates,
            "n_tasks_meta": meta_expose[L],
            "tau": float(tau),
            "p": float(p),
            "inversion_rate": inversion_rate(g, l),
            "consensus_tasks": ";".join(consensus),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        reject, p_fdr, _, _ = multipletests(
            df["p"].fillna(1.0).values, alpha=0.05, method="fdr_bh")
        df["p_fdr"] = p_fdr
        df["fdr_reject"] = reject
        df = df.sort_values("inversion_rate", ascending=False).reset_index(drop=True)
        cols = ["language", "lang_name", "n_models", "n_tasks", "tau", "p",
                "p_fdr", "fdr_reject", "inversion_rate",
                "n_tasks_candidates", "n_tasks_meta", "consensus_tasks"]
        df = df[cols]

    if len(df) >= 3 and df["n_tasks"].nunique() > 1:
        rho, rho_p = spearmanr(df["n_tasks"].values, df["inversion_rate"].values)
    else:
        rho, rho_p = float("nan"), float("nan")

    info = {
        "category": cat,
        "roster": roster_label,
        "coverage_threshold": coverage,
        "n_T": n_T,
        "n_models_roster": len(covered),
        "n_langs": len(df),
        "median_inversion": float(df["inversion_rate"].median()) if len(df) else float("nan"),
        "median_tau": float(df["tau"].median()) if len(df) else float("nan"),
        "n_FDR_reject": int(df["fdr_reject"].sum()) if len(df) else 0,
        "rho_nt_inversion": float(rho),
        "rho_p": float(rho_p),
        "median_n_models": float(df["n_models"].median()) if len(df) else float("nan"),
        "underpowered": bool(len(df) and df["n_models"].median() < UNDERPOWERED_N),
        "skipped": skipped,
    }
    if verbose:
        print(f"\n--- {cat} / {roster_label} (|T_C|={n_T}, coverage>={coverage:.0%})")
        print(f"    roster after coverage filter: {info['n_models_roster']} models")
        print(f"    languages kept: {info['n_langs']}  "
              f"median_inv={info['median_inversion']:.3f}  "
              f"median_tau={info['median_tau']:.3f}  "
              f"FDR rejects: {info['n_FDR_reject']}  "
              f"rho(n_t, inv)={info['rho_nt_inversion']:.3f} (p={info['rho_p']:.3f})")
        for L, why in skipped.items():
            print(f"    skipped {L}: {why}")
    return df, info


# ---------------------------------------------------------------------------
# Sanity check A: faithful retrieval extended-tier replication
# ---------------------------------------------------------------------------

RETR_MMTEB18 = [
    "AILAStatutes", "ArguAna", "BelebeleRetrieval", "CovidRetrieval",
    "HagridRetrieval", "LEMBPasskeyRetrieval", "LegalBenchCorporateLobbying",
    "MIRACLRetrievalHardNegatives", "MLQARetrieval", "SCIDOCS", "SpartQA",
    "StackOverflowQA", "StatcanDialogueDatasetRetrieval", "TRECCOVID",
    "TempReasonL1", "TwitterHjerneRetrieval", "WikipediaRetrievalMultilingual",
    "WinoGrande",
]
RETR_LANG_TASKS = [
    "BelebeleRetrieval", "MIRACLRetrievalHardNegatives", "MLQARetrieval",
    "WikipediaRetrievalMultilingual", "MrTidyRetrieval", "XPQARetrieval",
]
RETR_LANG_MAP = {
    "ita": "ita-Latn", "jpn": "jpn-Jpan", "hin": "hin-Deva", "ara": "ara-Arab",
    "zho": "zho-Hans", "deu": "deu-Latn", "spa": "spa-Latn", "rus": "rus-Cyrl",
    "fra": "fra-Latn", "kor": "kor-Hang", "ben": "ben-Beng", "vie": "vie-Latn",
    "ind": "ind-Latn", "fas": "pes-Arab", "swa": "swh-Latn", "tha": "tha-Thai",
    "tel": "tel-Telu", "tam": "tam-Taml",
}
RETR_LANG_ALT = {
    "ara": ["ara-Arab", "arb-Arab"],
    "fas": ["pes-Arab", "fas-Arab"],
    "kor": ["kor-Hang", "kor-Kore"],
    "swa": ["swh-Latn", "swa-Latn"],
}


def _retr_collect(rev: Path, all_bcp47: set[str]):
    """Faithful port of the retrieval audit's per-model JSON collection."""
    lang_set = set(RETR_LANG_TASKS)
    mmteb_set = set(RETR_MMTEB18)
    task_lang: dict[str, dict[str, float]] = {}
    global_scores: dict[str, float] = {}
    for task_name in lang_set | mmteb_set:
        p = rev / f"{task_name}.json"
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        scores_obj = data.get("scores", {})
        test_block = scores_obj.get("test") or scores_obj.get("dev") or \
            data.get("test") or data.get("dev") or []

        lang_scores_for_task: dict[str, float] = {}
        if isinstance(test_block, list):
            for entry in test_block:
                score = entry.get("ndcg_at_10")
                if score is None:
                    continue
                for lang in entry.get("languages", []):
                    if lang in all_bcp47:
                        lang_scores_for_task[lang] = float(score)  # last wins

        if task_name in mmteb_set:
            vals: list[float] = []
            for split_block in scores_obj.values():
                if isinstance(split_block, list):
                    for entry in split_block:
                        s = entry.get("ndcg_at_10")
                        if s is None:
                            s = entry.get("main_score")
                        if s is not None:
                            vals.append(float(s))
            if vals:
                global_scores[task_name] = float(np.mean(vals))

        if task_name in lang_set and lang_scores_for_task:
            task_lang[task_name] = lang_scores_for_task
    return task_lang, global_scores


def sanity_retrieval(anchor: float = 0.154, tol: float = 0.02):
    print("\n=== SANITY A: retrieval extended-tier replication ===")
    all_bcp47 = set(RETR_LANG_MAP.values())
    for alts in RETR_LANG_ALT.values():
        all_bcp47.update(alts)

    ext_lang: dict[str, dict[str, dict[str, float]]] = {}
    ext_global: dict[str, float] = {}
    for mdir in sorted(RESULTS_ROOT.iterdir()):
        if not mdir.is_dir() or mdir.name in EXCLUDE_DIRS:
            continue
        rev = pick_revision(mdir)
        if not all((rev / f"{t}.json").exists() for t in RETR_MMTEB18):
            continue
        task_lang, g_scores = _retr_collect(rev, all_bcp47)
        if task_lang:
            ext_lang[mdir.name] = task_lang
            if g_scores:
                ext_global[mdir.name] = float(np.mean(list(g_scores.values())))
    print(f"Extended roster (all 18 retrieval task files present): {len(ext_lang)} models")

    inv_rates = {}
    for iso3, primary in RETR_LANG_MAP.items():
        codes = RETR_LANG_ALT.get(iso3, [primary])
        if primary not in codes:
            codes = [primary] + codes
        lang_scores: dict[str, dict[str, float]] = {}
        tasks_seen: set[str] = set()
        for m, task_dict in ext_lang.items():
            row = {}
            for task, inner in task_dict.items():
                score = None
                for code in codes:
                    score = inner.get(code)
                    if score is not None:
                        break
                if score is not None:
                    row[task] = score
                    tasks_seen.add(task)
            lang_scores[m] = row
        if not tasks_seen:
            continue
        all_cols = sorted(tasks_seen)
        n_any = sum(1 for r in lang_scores.values() if r)
        min_m = max(1, math.ceil(0.50 * n_any))
        consensus = [t for t in all_cols
                     if sum(1 for r in lang_scores.values() if t in r) >= min_m]
        if not consensus:
            consensus = all_cols
        kept = [m for m, r in lang_scores.items()
                if all(t in r for t in consensus) and m in ext_global]
        if len(kept) < MIN_MODELS or len(consensus) < MIN_TASKS_PER_LANG:
            continue
        g = np.array([ext_global[m] for m in kept])
        l = np.array([float(np.mean([lang_scores[m][t] for t in consensus]))
                      for m in kept])
        # strict pairwise inversion count, as in the original analysis
        n = len(kept)
        n_inv = 0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                n_pairs += 1
                if (g[i] > g[j]) != (l[i] > l[j]):
                    n_inv += 1
        inv_rates[iso3] = n_inv / n_pairs
        print(f"  {iso3}: n_models={len(kept)} n_tasks={len(consensus)} "
              f"inv={inv_rates[iso3]:.4f}")

    med = float(np.median(list(inv_rates.values())))
    ok = abs(med - anchor) <= tol
    print(f"Replicated retrieval extended median inversion = {med:.4f} "
          f"({med * 100:.1f}%) over {len(inv_rates)} languages")
    print(f"Anchor = {anchor * 100:.1f}%  ->  {'PASS' if ok else 'FAIL (>2 pts off)'}")
    return med, ok, len(inv_rates)


# ---------------------------------------------------------------------------
# Sanity check B: spot-check two task JSONs by hand
# ---------------------------------------------------------------------------

def sanity_spotcheck(by_cat):
    print("\n=== SANITY B: spot-check of two task JSONs ===")
    model = "intfloat__multilingual-e5-small"
    rev = pick_revision(RESULTS_ROOT / model)
    checks = [("STS", "STS22.v2"), ("Classification", "MassiveIntentClassification")]
    for cat, tname in checks:
        p = rev / f"{tname}.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        tmeta = next(t for t in by_cat[cat] if t["name"] == tname)
        print(f"[{model} / {tname}] eval_splits={tmeta['eval_splits']}")
        for split, block in data["scores"].items():
            shown = 0
            for e in block:
                if shown >= 3:
                    break
                print(f"  raw {split:12s} subset={e.get('hf_subset'):8s} "
                      f"languages={e.get('languages')} main_score={e.get('main_score')}")
                shown += 1
        parsed = parse_task_json(p, tmeta["eval_splits"])
        task_score, lang_scores = parsed
        print(f"  parsed task_score (pooled mean) = {task_score:.6f}")
        for L in sorted(lang_scores):
            print(f"  parsed per-language {L} = {lang_scores[L]:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Benchmark: {BENCHMARK_NAME}")
    by_cat = load_benchmark_tasks()
    for c in CATEGORIES:
        exposed = {L: n for L in AUDIT_LANGS
                   if (n := sum(1 for t in by_cat[c] if L in t["exposed_langs_meta"]))}
        print(f"{c}: |T_C|={len(by_cat[c])}; metadata language exposure "
              f"(subsets with <= {MAX_LANGS_PER_SUBSET} languages): {exposed}")

    sanity_spotcheck(by_cat)
    retr_median, retr_ok, retr_nlang = sanity_retrieval()

    print("\nScanning results repository for the three categories ...")
    cat_data = scan_all_models(by_cat)
    for c in CATEGORIES:
        print(f"{c}: {len(cat_data[c])} models with >=1 task file")

    curated = set(
        pd.read_csv(ROSTER_CSV)["model_id"].str.replace("/", "__", regex=False)
    )
    print(f"Curated roster: {len(curated)} models")

    headline_rows = []
    threshold_rows = []
    for cat in CATEGORIES:
        tasks = by_cat[cat]
        # threshold effect on the extended roster
        for cov in THRESHOLD_GRID:
            _, info = analyze(cat, tasks, cat_data[cat], None, "extended",
                              coverage=cov, verbose=False)
            threshold_rows.append({
                "category": cat, "coverage_threshold": cov,
                "n_models_roster": info["n_models_roster"],
                "n_langs": info["n_langs"],
                "median_inversion": info["median_inversion"],
                "median_tau": info["median_tau"],
            })
        for roster, label in [(None, "extended"), (curated, "curated25")]:
            df, info = analyze(cat, tasks, cat_data[cat], roster, label)
            out = HERE / f"summary_{cat.lower()}_{label}.csv"
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"    wrote {out.name} ({len(df)} rows)")
            headline_rows.append({
                "category": cat,
                "roster": label,
                "n_models_roster": info["n_models_roster"],
                "n_langs": info["n_langs"],
                "median_inversion": round(info["median_inversion"], 4)
                    if not math.isnan(info["median_inversion"]) else float("nan"),
                "median_tau": round(info["median_tau"], 4)
                    if not math.isnan(info["median_tau"]) else float("nan"),
                "n_FDR_reject": info["n_FDR_reject"],
                "rho_nt_inversion": round(info["rho_nt_inversion"], 4)
                    if not math.isnan(info["rho_nt_inversion"]) else float("nan"),
                "rho_p": round(info["rho_p"], 4)
                    if not math.isnan(info["rho_p"]) else float("nan"),
                "median_n_models": info["median_n_models"],
                "underpowered": info["underpowered"],
            })

    headline = pd.DataFrame(headline_rows)
    headline.to_csv(HERE / "headline.csv", index=False, encoding="utf-8")
    pd.DataFrame(threshold_rows).to_csv(
        HERE / "threshold_sensitivity.csv", index=False, encoding="utf-8")

    print("\n=== HEADLINE ===")
    print(headline.to_string(index=False))
    print(f"\nRetrieval sanity: replicated extended median inversion = "
          f"{retr_median * 100:.1f}% over {retr_nlang} languages "
          f"({'PASS' if retr_ok else 'FAIL'} vs 15.4% anchor)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
