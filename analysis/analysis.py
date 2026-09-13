#!/usr/bin/env python3
"""
Analysis pipeline: multi-language embedding model evaluation.
Produces analysis_output/ artifacts. Run: python analysis.py
"""

import os, sys, json, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr, kendalltau

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT      = os.path.join(ROOT, "mteb_csvs")
LANG_DIRS     = {
    # NOTE: italian_mteb/ is intentionally absent. The Italian leaderboard CSVs
    # are known-bad (they duplicate the Japanese ones) and are quarantined in
    # mteb_csvs/KNOWN_BAD/ -- see that directory's README. Loading them here
    # would silently produce Japanese numbers labelled Italian, so this path is
    # left pointing at a directory that does not exist and will fail loudly.
    "italian":  os.path.join(CSV_ROOT, "italian_mteb"),
    "japanese": os.path.join(CSV_ROOT, "japnese_mteb"),   # folder has typo
    "hindi":    os.path.join(CSV_ROOT, "hindi_mteb"),
}
OWN_RUNS_BASE = os.path.join(ROOT, "mteb-language-gap", "results")
OWN_RUNS_LANG = {"italian": "ita_lb", "japanese": "jpn_lb", "hindi": "hin_lb"}
MTEB_AGG_CSV  = os.path.join(ROOT, "mteb-language-gap", "config", "mteb_agg_scores_subset.csv")
RAG_EVAL_DIR  = os.path.join(ROOT, "rag-dataset", "output", "evaluation")
OUT_DIR       = os.path.join(ROOT, "analysis_output")
FIG_DIR       = os.path.join(OUT_DIR, "figures")
TAB_DIR       = os.path.join(OUT_DIR, "tables")
STAT_DIR      = os.path.join(OUT_DIR, "stats")

# ── Model Registry ─────────────────────────────────────────────────────────────
def _p(s):
    s = s.strip()
    if s.endswith("B"): return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"): return int(float(s[:-1]) * 1_000_000)
    raise ValueError(s)

MODEL_REGISTRY = {
    "intfloat/multilingual-e5-small":                 {"params_numeric": _p("118M"), "dim": 384,  "tier": "small",      "english_only": False},
    "intfloat/e5-small-v2":                           {"params_numeric": _p("33M"),  "dim": 384,  "tier": "small",      "english_only": True},
    "ibm-granite/granite-embedding-107m-multilingual": {"params_numeric": _p("107M"), "dim": 768,  "tier": "small",      "english_only": False},
    "jinaai/jina-embeddings-v5-text-nano":            {"params_numeric": _p("212M"), "dim": 1024, "tier": "small",      "english_only": False},
    "intfloat/multilingual-e5-base":                  {"params_numeric": _p("278M"), "dim": 768,  "tier": "medium",     "english_only": False},
    "intfloat/multilingual-e5-large":                 {"params_numeric": _p("560M"), "dim": 1024, "tier": "medium",     "english_only": False},
    "intfloat/multilingual-e5-large-instruct":        {"params_numeric": _p("560M"), "dim": 1024, "tier": "medium",     "english_only": False},
    "BAAI/bge-m3":                                    {"params_numeric": _p("568M"), "dim": 1024, "tier": "medium",     "english_only": False},
    "Snowflake/snowflake-arctic-embed-l-v2.0":        {"params_numeric": _p("568M"), "dim": 1024, "tier": "medium",     "english_only": False},
    "intfloat/e5-large-v2":                           {"params_numeric": _p("335M"), "dim": 1024, "tier": "medium",     "english_only": True},
    "microsoft/harrier-oss-v1-0.6b":                  {"params_numeric": _p("596M"), "dim": 1024, "tier": "medium",     "english_only": False},
    "Qwen/Qwen3-Embedding-0.6B":                      {"params_numeric": _p("600M"), "dim": 1024, "tier": "qwen_sub1b", "english_only": False},
    "Qwen/Qwen3-Embedding-4B":                        {"params_numeric": _p("4B"),   "dim": 2560, "tier": "large",      "english_only": False},
    "Salesforce/SFR-Embedding-Mistral":               {"params_numeric": _p("7B"),   "dim": 4096, "tier": "large",      "english_only": False},
    "intfloat/e5-mistral-7b-instruct":                {"params_numeric": _p("7B"),   "dim": 4096, "tier": "large",      "english_only": False},
    "nvidia/llama-embed-nemotron-8b":                 {"params_numeric": _p("7.5B"), "dim": 4096, "tier": "xlarge",     "english_only": False},
    "Qwen/Qwen3-Embedding-8B":                        {"params_numeric": _p("8B"),   "dim": 4096, "tier": "xlarge",     "english_only": False},
}
WORKING_SET    = list(MODEL_REGISTRY.keys())   # 17 models
ENGLISH_ONLY   = {"intfloat/e5-small-v2", "intfloat/e5-large-v2"}
EXCLUDE_ITALIAN = {"nvidia/llama-embed-nemotron-8b"}

def sanitize_model_id(model_id: str) -> str:
    name = model_id.split("/")[-1]
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name)[:20]
    h = hashlib.md5(model_id.encode()).hexdigest()[:6]
    return f"{safe}_{h}"

SAN_TO_HF = {sanitize_model_id(m): m for m in WORKING_SET}

LB_TO_HF = {
    "multilingual-e5-small":               "intfloat/multilingual-e5-small",
    "e5-small-v2":                         "intfloat/e5-small-v2",
    "granite-embedding-107m-multilingual":  "ibm-granite/granite-embedding-107m-multilingual",
    "jina-embeddings-v5-text-nano":         "jinaai/jina-embeddings-v5-text-nano",
    "multilingual-e5-base":                 "intfloat/multilingual-e5-base",
    "multilingual-e5-large-instruct":       "intfloat/multilingual-e5-large-instruct",
    "multilingual-e5-large":               "intfloat/multilingual-e5-large",
    "bge-m3":                              "BAAI/bge-m3",
    "snowflake-arctic-embed-l-v2.0":       "Snowflake/snowflake-arctic-embed-l-v2.0",
    "e5-large-v2":                         "intfloat/e5-large-v2",
    "harrier-oss-v1-0.6b":                "microsoft/harrier-oss-v1-0.6b",
    "Qwen3-Embedding-0.6B":               "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B":                 "Qwen/Qwen3-Embedding-4B",
    "SFR-Embedding-Mistral":              "Salesforce/SFR-Embedding-Mistral",
    "e5-mistral-7b-instruct":             "intfloat/e5-mistral-7b-instruct",
    "llama-embed-nemotron-8b":            "nvidia/llama-embed-nemotron-8b",
    "Qwen3-Embedding-8B":                "Qwen/Qwen3-Embedding-8B",
}
HF_TO_LB = {v: k for k, v in LB_TO_HF.items()}

LANG_TASKS = {
    "italian":  ["BelebeleRetrieval", "WikipediaRetrievalMultilingual"],
    "japanese": ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives"],
    "hindi":    ["BelebeleRetrieval", "MIRACLRetrievalHardNegatives", "MLQARetrieval", "WikipediaRetrievalMultilingual"],
}
LB_VALID_ITA_TASKS = {"BelebeleRetrieval"}

BM25_BASELINES = {
    "ja_finance": {"ndcg_at_10": 50.33, "recall_at_10": 63.03, "mrr": 46.27},
    "ja_law":     {"ndcg_at_10": 30.96, "recall_at_10": 47.07, "mrr": 26.05},
    "hi_finance": {"ndcg_at_10": 92.31, "recall_at_10": 96.62, "mrr": 90.93},
    "hi_law":     {"ndcg_at_10": 78.98, "recall_at_10": 89.39, "mrr": 75.68},
}

ANCHOR_PAIRS = [
    ("intfloat/multilingual-e5-small", "Qwen/Qwen3-Embedding-8B"),
    ("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"),
    ("intfloat/multilingual-e5-small", "BAAI/bge-m3"),
    ("ibm-granite/granite-embedding-107m-multilingual", "Qwen/Qwen3-Embedding-4B"),
]

LANG_COLORS  = {"italian": "#2ca02c", "japanese": "#1f77b4", "hindi": "#ff7f0e"}
LANG_MARKERS = {"italian": "o", "japanese": "s", "hindi": "^"}
SHORT_NAMES  = {
    "intfloat/multilingual-e5-small":                 "ml-e5-small",
    "intfloat/e5-small-v2":                           "e5-small-v2",
    "ibm-granite/granite-embedding-107m-multilingual": "granite-107m",
    "jinaai/jina-embeddings-v5-text-nano":            "jina-v5-nano",
    "intfloat/multilingual-e5-base":                  "ml-e5-base",
    "intfloat/multilingual-e5-large":                 "ml-e5-large",
    "intfloat/multilingual-e5-large-instruct":        "ml-e5-large-i",
    "BAAI/bge-m3":                                    "bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0":        "snowflake-l-v2",
    "intfloat/e5-large-v2":                           "e5-large-v2",
    "microsoft/harrier-oss-v1-0.6b":                  "harrier-0.6b",
    "Qwen/Qwen3-Embedding-0.6B":                      "Qwen3-0.6B",
    "Qwen/Qwen3-Embedding-4B":                        "Qwen3-4B",
    "Salesforce/SFR-Embedding-Mistral":               "SFR-Mistral",
    "intfloat/e5-mistral-7b-instruct":                "e5-mistral-7b",
    "nvidia/llama-embed-nemotron-8b":                 "nemotron-8b",
    "Qwen/Qwen3-Embedding-8B":                        "Qwen3-8B",
}
LABEL_MODELS = {"intfloat/multilingual-e5-small", "Qwen/Qwen3-Embedding-8B", "BAAI/bge-m3"}

_TIER_COLORS_RANK = {
    "small":      "#1f77b4",
    "qwen_sub1b": "#1f77b4",   # grouped with small in scatter legend
    "medium":     "#ff7f0e",
    "large":      "#d62728",
    "xlarge":     "#d62728",   # grouped with large in scatter legend
}

# Inverse-frequency weights: tasks covering more language subsets are down-weighted.
# BelebeleRetrieval (3 langs) → 1/3; Wikipedia+MIRACL (2 langs each) → 1/2; MLQA (1 lang) → 1.
_TASK_WEIGHTS = {
    "task_BelebeleRetrieval":             1 / 3,
    "task_WikipediaRetrievalMultilingual": 1 / 2,
    "task_MIRACLRetrievalHardNegatives":  1 / 2,
    "task_MLQARetrieval":                 1,
}

# ── Section 1: Load MTEB_agg (authoritative) ──────────────────────────────────

def load_mteb_agg() -> dict:
    df = pd.read_csv(MTEB_AGG_CSV)
    return df.set_index("model_id")["mteb_agg"].to_dict()


# ── Section 2: Load own MTEB runs ─────────────────────────────────────────────

def load_own_runs(language: str) -> dict:
    base = os.path.join(OWN_RUNS_BASE, OWN_RUNS_LANG[language])
    results = {}
    seen = set()
    unmatched = []

    for folder in sorted(os.listdir(base)):
        folder_path = os.path.join(base, folder)
        if not os.path.isdir(folder_path):
            continue
        model_id = SAN_TO_HF.get(folder)
        if model_id is None:
            unmatched.append(folder)
            continue
        if model_id in seen:
            continue
        seen.add(model_id)

        task_scores = {}
        for task in LANG_TASKS[language]:
            task_file = os.path.join(folder_path, f"{task}.json")
            if not os.path.exists(task_file):
                continue
            with open(task_file) as f:
                data = json.load(f)
            score = data["scores"]["test"][0]["ndcg_at_10"]
            task_scores[task] = round(score * 100.0, 4)

        if task_scores:
            results[model_id] = task_scores

    print(f"[own_runs/{language}] matched {len(results)} models; "
          f"{len(unmatched)} unrecognised folders: {unmatched or 'none'}")
    if len(results) < 12:
        print(f"[WARNING] Only {len(results)} own-run models found for {language} "
              f"— expected >=12. Check SAN_TO_HF mapping vs actual folder names.")
    return results


# ── Section 3: Load leaderboard per-task scores ────────────────────────────────

def load_lb_lang_tasks(language: str) -> dict:
    lang_dir = LANG_DIRS[language]
    task_csv = next(
        (os.path.join(lang_dir, f) for f in os.listdir(lang_dir) if "per_task" in f),
        None
    )
    if task_csv is None:
        raise FileNotFoundError(f"No per_task CSV found in {lang_dir}")
    df = pd.read_csv(task_csv, index_col=0)

    valid_tasks = (
        LB_VALID_ITA_TASKS if language == "italian"
        else set(LANG_TASKS[language])
    )
    result = {}
    for _, row in df.iterrows():
        lb_name = str(row.get("Model", ""))
        if not lb_name:
            continue
        scores = {}
        for task in valid_tasks:
            if task in df.columns:
                val = row.get(task)
                if val is not None and not pd.isna(val):
                    scores[task] = float(val)
        if scores:
            result[lb_name] = scores
    return result


# ── Section 4: Build unified_results.csv ──────────────────────────────────────

def build_unified_results() -> pd.DataFrame:
    mteb_agg_map = load_mteb_agg()
    rows = []
    own_runs_by_lang = {}
    lb_by_lang = {}

    for language in ["italian", "japanese", "hindi"]:
        own_runs = load_own_runs(language)
        lb_lang  = load_lb_lang_tasks(language)
        own_runs_by_lang[language] = own_runs
        lb_by_lang[language]       = lb_lang

        for model_id in WORKING_SET:
            reg      = MODEL_REGISTRY[model_id]
            lb_name  = HF_TO_LB.get(model_id)
            mteb_agg = mteb_agg_map.get(model_id, float("nan"))

            lb_task_scores  = lb_lang.get(lb_name, {}) if lb_name else {}
            own_task_scores = own_runs.get(model_id, {})

            task_sources = {}
            task_scores  = {}
            for task in LANG_TASKS[language]:
                if task in own_task_scores:
                    task_scores[task]  = own_task_scores[task]
                    task_sources[task] = "own"
                elif task in lb_task_scores:
                    task_scores[task]  = lb_task_scores[task]
                    task_sources[task] = "lb"

            if not task_scores:
                source = "missing"
            elif all(v == "own" for v in task_sources.values()):
                source = "own_run"
            elif all(v == "lb"  for v in task_sources.values()):
                source = "leaderboard"
            else:
                source = "merged"

            valid_scores = [v for v in task_scores.values() if not np.isnan(v)]
            lang_avg = np.mean(valid_scores) if valid_scores else float("nan")
            gap = (lang_avg - mteb_agg) if not (np.isnan(lang_avg) or np.isnan(mteb_agg)) else float("nan")
            gap_pct = (gap / mteb_agg * 100) if (not np.isnan(gap) and mteb_agg != 0) else float("nan")

            row = {
                "model_id":          model_id,
                "short_name":        SHORT_NAMES[model_id],
                "params_numeric":    reg["params_numeric"],
                "dim":               reg["dim"],
                "tier":              reg["tier"],
                "english_only":      reg["english_only"],
                "language":          language,
                "mteb_agg_score":    round(mteb_agg, 4)  if not np.isnan(mteb_agg)  else float("nan"),
                "lang_specific_avg": round(lang_avg, 4)  if not np.isnan(lang_avg)  else float("nan"),
                "gap":               round(gap, 4)        if not np.isnan(gap)        else float("nan"),
                "gap_pct":           round(gap_pct, 4)    if not np.isnan(gap_pct)    else float("nan"),
                "n_tasks":           len(valid_scores),
                "source":            source,
            }
            for task in LANG_TASKS[language]:
                row[f"task_{task}"] = round(task_scores.get(task, float("nan")), 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "unified_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[unified_results] {len(df)} rows -> {out_path}")

    _cross_check(own_runs_by_lang, lb_by_lang)
    return df


def _cross_check(own_runs_by_lang, lb_by_lang):
    flags = []
    for language in ["italian", "japanese", "hindi"]:
        own = own_runs_by_lang[language]
        lb  = lb_by_lang[language]
        for model_id, own_tasks in own.items():
            lb_name  = HF_TO_LB.get(model_id)
            lb_tasks = lb.get(lb_name, {}) if lb_name else {}
            for task, own_score in own_tasks.items():
                lb_score = lb_tasks.get(task)
                if lb_score is None:
                    continue
                delta = abs(own_score - lb_score)
                if delta > 5.0:
                    flags.append((model_id, language, task, own_score, lb_score, delta))
    if flags:
        print("[WARNING] Own vs leaderboard delta > 5 pts:")
        for f in flags:
            print(f"  {f[0].split('/')[-1]} | {f[1]} | {f[2]}: own={f[3]:.1f} lb={f[4]:.1f} delta={f[5]:.1f}")
    else:
        print("[cross-check] All own-run vs leaderboard deltas <= 5 pts.")


def _validate_unified_results(df: pd.DataFrame):
    assert len(df) == 51, f"Expected 51 rows (17x3), got {len(df)}"

    row = df[(df.model_id == "intfloat/multilingual-e5-small") & (df.language == "italian")].iloc[0]
    assert row.source == "own_run"
    assert row.mteb_agg_score > 70, f"MTEB_agg should be ~77, got {row.mteb_agg_score}"
    assert row.gap > 0, f"Expected positive gap for ml-e5-small/italian, got {row.gap}"

    row = df[(df.model_id == "Qwen/Qwen3-Embedding-4B") & (df.language == "japanese")].iloc[0]
    assert row.source == "merged", f"Qwen3-4B JPN should be merged, got {row.source}"

    row = df[(df.model_id == "Salesforce/SFR-Embedding-Mistral") & (df.language == "hindi")].iloc[0]
    assert row.source == "leaderboard"

    nan_agg = df[df.mteb_agg_score.isna()]
    assert len(nan_agg) == 0, f"Models with NaN mteb_agg: {nan_agg.model_id.unique()}"

    print("[validate] unified_results.csv passed all assertions")


# ── Section 5: Build unified_rag_results.csv ──────────────────────────────────

def load_rag_summary() -> dict:
    with open(os.path.join(RAG_EVAL_DIR, "summary.json")) as f:
        return json.load(f)


def build_unified_rag_results() -> pd.DataFrame:
    raw = load_rag_summary()
    datasets = ["ja_finance", "ja_law", "hi_finance", "hi_law"]
    rows = []

    for model_id, dataset_results in raw.items():
        if model_id not in MODEL_REGISTRY:
            continue
        reg = MODEL_REGISTRY[model_id]
        for ds_key, res in dataset_results.items():
            if ds_key not in datasets:
                continue
            overall = res.get("overall", {})
            rows.append({
                "model_id":      model_id,
                "short_name":    SHORT_NAMES[model_id],
                "params_numeric":reg["params_numeric"],
                "dim":           reg["dim"],
                "tier":          reg["tier"],
                "dataset":       ds_key,
                "ndcg_at_10":    round(overall.get("ndcg_at_10", float("nan")) * 100, 2),
                "recall_at_10":  round(overall.get("recall_at_10", float("nan")) * 100, 2),
                "mrr":           round(overall.get("mrr", float("nan")) * 100, 2),
                "source":        "own_eval",
            })

    for ds_key, metrics in BM25_BASELINES.items():
        rows.append({
            "model_id":      "BM25",
            "short_name":    "BM25",
            "params_numeric":0,
            "dim":           0,
            "tier":          "baseline",
            "dataset":       ds_key,
            "ndcg_at_10":    round(metrics["ndcg_at_10"], 2),
            "recall_at_10":  round(metrics["recall_at_10"], 2),
            "mrr":           round(metrics["mrr"], 2),
            "source":        "bm25_baseline",
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "unified_rag_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[unified_rag] {len(df)} rows -> {out_path}")
    return df


def _validate_rag_results(df: pd.DataFrame):
    bm25 = df[df.model_id == "BM25"]
    assert len(bm25) == 4, f"Expected 4 BM25 rows, got {len(bm25)}"
    ja_fin = bm25[bm25.dataset == "ja_finance"].iloc[0]
    assert abs(ja_fin.ndcg_at_10 - 50.33) < 0.5
    assert "perplexity-ai/pplx-embed-v1-0.6b" not in df.model_id.values
    print("[validate] unified_rag_results.csv passed assertions")


# ── Section 6: Gap vs Parameters regression ───────────────────────────────────

def analysis_31_gap_regression(df: pd.DataFrame) -> dict:
    results = {}
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (~df.gap.isna()) &
            (df.source != "missing")
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        sub["log_params"] = np.log10(sub.params_numeric)
        if len(sub) < 3:
            continue
        slope, intercept, r, p, _ = linregress(sub.log_params, sub.gap)
        results[lang] = {
            "slope": round(slope, 4), "intercept": round(intercept, 4),
            "r2": round(r**2, 4), "pvalue": round(p, 6), "n": len(sub),
            "data": sub,
        }
    return results

def save_regression_results(reg: dict):
    lines = ["# Gap vs Parameters Regression Results\n",
             "# gap = slope * log10(params) + intercept\n"]
    for lang, res in reg.items():
        lines += [
            f"## {lang.capitalize()}  (n={res['n']})",
            f"  slope     = {res['slope']:.4f}  (NDCG pts per decade of params)",
            f"  intercept = {res['intercept']:.4f}",
            f"  R2        = {res['r2']:.4f}",
            f"  p-value   = {res['pvalue']:.6f}",
            "",
        ]
    with open(os.path.join(STAT_DIR, "regression_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print("[stats] regression_results.txt saved")


# ── Section 7: Gap Inflation Ratios ───────────────────────────────────────────

PAIR_LABELS = {
    ("intfloat/multilingual-e5-small",                 "Qwen/Qwen3-Embedding-8B"):     "ml-e5-small vs Qwen3-8B",
    ("intfloat/multilingual-e5-small",                 "intfloat/multilingual-e5-large"): "ml-e5-small vs ml-e5-large",
    ("intfloat/multilingual-e5-small",                 "BAAI/bge-m3"):                  "ml-e5-small vs bge-m3",
    ("ibm-granite/granite-embedding-107m-multilingual", "Qwen/Qwen3-Embedding-4B"):      "granite-107m vs Qwen3-4B",
}

def analysis_32_gap_inflation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[df.language == lang].set_index("model_id")
        for (sm_id, lg_id), label in PAIR_LABELS.items():
            if sm_id not in sub.index or lg_id not in sub.index:
                continue
            sm, lg = sub.loc[sm_id], sub.loc[lg_id]
            perceived = lg.mteb_agg_score - sm.mteb_agg_score
            actual    = lg.lang_specific_avg - sm.lang_specific_avg
            inflation = (perceived / actual) if (not np.isnan(actual) and actual != 0) else float("nan")
            rows.append({
                "pair": label, "language": lang,
                "perceived_gap":    round(perceived, 2),
                "actual_gap":       round(actual, 2),
                "inflation_ratio":  round(inflation, 2) if not np.isnan(inflation) else float("nan"),
            })
    return pd.DataFrame(rows)


# ── Section 8: Decision Error Simulation ──────────────────────────────────────

def analysis_33_decision_error(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = [80, 82, 85, 87, 90, 92, 95]
    rows = []
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (~df.model_id.isin(EXCLUDE_ITALIAN if lang == "italian" else set())) &
            (df.source != "missing")
        ].copy()
        sub = sub.sort_values(["params_numeric", "lang_specific_avg"],
                               ascending=[True, False])

        for thr in thresholds:
            mteb_above  = sub[sub.mteb_agg_score >= thr]
            lang_above  = sub[sub.lang_specific_avg >= thr]

            mteb_model  = mteb_above.iloc[0].model_id  if len(mteb_above) else None
            mteb_params = mteb_above.iloc[0].params_numeric if len(mteb_above) else float("nan")
            lang_model  = lang_above.iloc[0].model_id  if len(lang_above) else None
            lang_params = lang_above.iloc[0].params_numeric if len(lang_above) else float("nan")

            ratio = (mteb_params / lang_params
                     if (not np.isnan(mteb_params) and not np.isnan(lang_params) and lang_params > 0)
                     else float("nan"))
            rows.append({
                "language":    lang,
                "threshold":   thr,
                "mteb_model":  SHORT_NAMES.get(mteb_model, mteb_model),
                "mteb_params": mteb_params,
                "lang_model":  SHORT_NAMES.get(lang_model, lang_model),
                "lang_params": lang_params,
                "param_ratio": round(ratio, 1) if not np.isnan(ratio) else float("nan"),
            })
    return pd.DataFrame(rows)


# ── Section 9: Cross-Language Gap Correlations + Summary Statistics ────────────

def analysis_35_spearman_gaps(df: pd.DataFrame):
    multi_models = [m for m in WORKING_SET if m not in ENGLISH_ONLY]
    langs = ["italian", "japanese", "hindi"]
    gaps_by_lang = {}
    for lang in langs:
        sub = df[
            (df.language == lang) &
            (df.model_id.isin(multi_models)) &
            (~df.model_id.isin(EXCLUDE_ITALIAN if lang == "italian" else set())) &
            (~df.gap.isna())
        ]
        gaps_by_lang[lang] = dict(zip(sub.model_id, sub.gap))

    common = sorted(
        set(gaps_by_lang["italian"]) &
        set(gaps_by_lang["japanese"]) &
        set(gaps_by_lang["hindi"])
    )
    matrix = {}
    for l1 in langs:
        matrix[l1] = {}
        v1 = [gaps_by_lang[l1][m] for m in common]
        for l2 in langs:
            v2 = [gaps_by_lang[l2][m] for m in common]
            rho, _ = spearmanr(v1, v2)
            matrix[l1][l2] = round(rho, 3)
    return pd.DataFrame(matrix, index=langs), common

def save_spearman_results(corr_df: pd.DataFrame, common_models: list):
    lines = [
        "# Cross-Language Gap Correlations (Spearman rho)",
        f"# n = {len(common_models)} multilingual models",
        "",
        corr_df.to_string(),
        "",
        "# Models included:",
        *[f"#   {m}" for m in common_models],
        "",
        "# Include in final analysis only if correlations > 0.6",
    ]
    with open(os.path.join(STAT_DIR, "spearman_gaps.txt"), "w") as f:
        f.write("\n".join(lines))
    print("[stats] spearman_gaps.txt saved")

def save_summary_statistics(df, reg, inflation_df):
    small_tiers = {"small", "qwen_sub1b"}
    large_tiers = {"large", "xlarge"}
    multi = df[~df.english_only]

    small_gaps = multi[multi.tier.isin(small_tiers)].gap.dropna()
    large_gaps = multi[multi.tier.isin(large_tiers)].gap.dropna()

    lines = [
        "# Key Numbers",
        "",
        f"Gap range (small multilingual): {small_gaps.min():.1f} to {small_gaps.max():.1f} pts",
        f"Gap range (large multilingual): {large_gaps.min():.1f} to {large_gaps.max():.1f} pts",
        "",
    ]

    for lang in ["italian", "japanese", "hindi"]:
        sub = inflation_df[inflation_df.language == lang].inflation_ratio.dropna()
        if len(sub):
            lines.append(f"Avg inflation ratio ({lang}): {sub.mean():.2f}x (range {sub.min():.2f}-{sub.max():.2f}x)")
    lines.append("")

    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) & (~df.english_only) &
            (~df.lang_specific_avg.isna()) &
            (~df.model_id.isin(EXCLUDE_ITALIAN if lang == "italian" else set()))
        ]
        bs = sub[sub.tier.isin(small_tiers)].sort_values("lang_specific_avg", ascending=False)
        bl = sub[sub.tier.isin(large_tiers)].sort_values("lang_specific_avg", ascending=False)
        if len(bs) and len(bl):
            best_small, best_large = bs.iloc[0], bl.iloc[0]
            pct = best_small.lang_specific_avg / best_large.lang_specific_avg * 100
            ratio = best_large.params_numeric / best_small.params_numeric
            lines.append(
                f"{lang}: best small={SHORT_NAMES[best_small.model_id]} "
                f"({best_small.lang_specific_avg:.1f}), "
                f"best large={SHORT_NAMES[best_large.model_id]} "
                f"({best_large.lang_specific_avg:.1f}), "
                f"small achieves {pct:.1f}%, param ratio {ratio:.0f}x"
            )
    lines += [
        "",
        "Languages: 3 (Italian, Japanese, Hindi)",
        f"Models (working set): {len(WORKING_SET)}",
        "Excluded: 3 (jina-v3 eval failure, gte-Qwen2-7B eval failure, pplx-embed near-random)",
        "MTEB tasks: Italian=2, Japanese=2, Hindi=4",
        "RAG datasets: 4 (ja_finance, ja_law, hi_finance, hi_law)",
    ]
    with open(os.path.join(STAT_DIR, "summary_statistics.txt"), "w") as f:
        f.write("\n".join(lines))
    print("[stats] summary_statistics.txt saved")


# ── Section 10: Figures ────────────────────────────────────────────────────────

def _pub_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

def _save_fig(fig, name: str):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    print(f"[fig] {name}.pdf/png saved")
    plt.close(fig)

def figure1_gap_vs_params(df: pd.DataFrame, reg: dict):
    _pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for lang in ["italian", "japanese", "hindi"]:
        res = reg.get(lang)
        if not res:
            continue
        sub = res["data"]
        xs, ys = sub.log_params.values, sub.gap.values

        ax.scatter(xs, ys, color=LANG_COLORS[lang], marker=LANG_MARKERS[lang],
                   s=60, zorder=3, alpha=0.85,
                   label=f"{lang.capitalize()} (R2={res['r2']:.2f})")
        x_range = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_range, res["slope"] * x_range + res["intercept"],
                color=LANG_COLORS[lang], lw=1.5, ls="--", alpha=0.7)

        for _, row in sub.iterrows():
            if row.model_id in LABEL_MODELS:
                ax.annotate(SHORT_NAMES[row.model_id],
                            (np.log10(row.params_numeric), row.gap),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(4, 3), textcoords="offset points")

    eng = df[(df.english_only) & (~df.gap.isna())]
    if not eng.empty:
        ax.scatter(np.log10(eng.params_numeric), eng.gap,
                   color="gray", marker="x", s=60, lw=1.5, zorder=2,
                   label="English-only (excl. from regression)")

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("Gap (lang-specific NDCG@10 - MTEB-agg)")
    ax.set_title("Aggregation Gap vs Model Size")
    ax.set_xticks([7.5, 8, 8.5, 9, 9.5, 10])
    ax.set_xticklabels(["33M", "100M", "316M", "1B", "3B", "10B"])
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "fig1_gap_vs_params")

def figure2_decision_error(decision_df: pd.DataFrame):
    _pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    thresholds = sorted(decision_df.threshold.unique())

    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = decision_df[decision_df.language == lang].sort_values("threshold")
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.isna(row.param_ratio):
                ax.text(i, 0.2, "N/A", ha="center", va="bottom",
                        fontsize=7, color="gray", style="italic")
            else:
                ax.bar(i, row.param_ratio,
                       color=LANG_COLORS[lang], edgecolor="white", lw=0.5, alpha=0.85)
                if row.param_ratio > 2 and row.mteb_model:
                    ax.text(i, row.param_ratio + 0.2, row.mteb_model,
                            ha="center", fontsize=6.5, rotation=45)
        ax.axhline(1, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([str(t) for t in thresholds])
        ax.set_xlabel("Performance Threshold (%)")
        ax.set_ylabel("Param Ratio (MTEB / Lang choice)")
        ax.set_title(lang.capitalize())

    fig.suptitle("Decision Error: MTEB-guided vs Language-guided Model Selection", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig2_decision_error")

def _pareto_mask(xs, ys):
    n = len(xs)
    mask = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and xs[j] <= xs[i] and ys[j] >= ys[i]:
                if xs[j] < xs[i] or ys[j] > ys[i]:
                    mask[i] = False
                    break
    return mask

def figure3_pareto(df: pd.DataFrame):
    _pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = df[
            (df.language == lang) & (~df.english_only) &
            (~df.lang_specific_avg.isna()) & (~df.mteb_agg_score.isna()) &
            (~df.model_id.isin(EXCLUDE_ITALIAN if lang == "italian" else set()))
        ].copy()
        sub["lp"] = np.log10(sub.params_numeric)
        xs = sub.lp.values
        lang_ys = sub.lang_specific_avg.values
        mteb_ys = sub.mteb_agg_score.values

        pareto_lang = _pareto_mask(xs, lang_ys)
        pareto_mteb = _pareto_mask(xs, mteb_ys)

        ax.scatter(xs, lang_ys, color=LANG_COLORS[lang], s=50, zorder=3,
                   alpha=0.85, label="Lang-specific")
        ax.scatter(xs, mteb_ys, color="gray", s=50, zorder=2,
                   alpha=0.5, marker="D", label="MTEB-agg")

        pf = np.where(pareto_lang)[0]
        pf_order = pf[np.argsort(xs[pf])]
        ax.plot(xs[pf_order], lang_ys[pf_order],
                color=LANG_COLORS[lang], lw=1.5, zorder=4)

        diff = pareto_lang & ~pareto_mteb
        if diff.any():
            ax.scatter(xs[diff], lang_ys[diff], color="red", s=100,
                       marker="*", zorder=5, label="Lang-Pareto, not MTEB-Pareto")

        for _, row in sub.iterrows():
            if row.model_id in LABEL_MODELS:
                ax.annotate(SHORT_NAMES[row.model_id],
                            (row.lp, row.lang_specific_avg),
                            fontsize=7, ha="left",
                            xytext=(3, 2), textcoords="offset points")

        ax.set_xlabel("log10(Parameters)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(lang.capitalize())
        ax.set_xticks([7.5, 8.5, 9.5])
        ax.set_xticklabels(["33M", "316M", "3B"])
        if lang == "italian":
            ax.legend(fontsize=7)

    fig.suptitle("Cost-Performance Pareto Frontiers", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig3_pareto")


# ── Section 11: LaTeX Tables ───────────────────────────────────────────────────

def _f(x, d=1):
    return "--" if pd.isna(x) else f"{x:.{d}f}"

def table1_main_results(df: pd.DataFrame):
    langs = ["italian", "japanese", "hindi"]
    pivot = df.pivot_table(index="model_id", columns="language",
                           values=["mteb_agg_score", "lang_specific_avg", "gap"],
                           aggfunc="first")
    models = sorted(WORKING_SET, key=lambda m: MODEL_REGISTRY[m]["params_numeric"])

    col_spec = "l r | rrr | rrr | rrr"
    lines = [
        r"\begin{table*}[htbp]", r"\centering",
        r"\caption{MTEB-agg score, language-specific NDCG@10, and gap "
        r"(lang$-$MTEB) for all 17 models. "
        r"$\dagger$~English-only. $\ddagger$~excluded from Italian.}",
        r"\label{tab:main-results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r" & & \multicolumn{3}{c|}{Italian} & \multicolumn{3}{c|}{Japanese} & \multicolumn{3}{c}{Hindi} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Model & Params & MTEB & Lang & Gap & MTEB & Lang & Gap & MTEB & Lang & Gap \\",
        r"\midrule",
    ]
    for mid in models:
        reg = MODEL_REGISTRY[mid]
        p = reg["params_numeric"]
        p_str = f"{p/1e9:.1f}B" if p >= 1e9 else f"{int(p/1e6)}M"
        name = SHORT_NAMES[mid]
        if reg["english_only"]:    name += r"$^\dagger$"
        if mid in EXCLUDE_ITALIAN: name += r"$^\ddagger$"
        cells = [name, p_str]
        for lang in langs:
            try:
                cells += [
                    _f(pivot[("mteb_agg_score", lang)][mid]),
                    _f(pivot[("lang_specific_avg", lang)][mid]),
                    _f(pivot[("gap", lang)][mid]),
                ]
            except KeyError:
                cells += ["--", "--", "--"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(TAB_DIR, "table1_main_results.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table1_main_results.tex saved")

def table2_rag_results(rag_df: pd.DataFrame):
    datasets = ["ja_finance", "ja_law", "hi_finance", "hi_law"]
    ds_labels = {"ja_finance": "JA/Fin", "ja_law": "JA/Law",
                 "hi_finance": "HI/Fin", "hi_law": "HI/Law"}
    pivot = rag_df.pivot_table(index="model_id", columns="dataset",
                                values="ndcg_at_10", aggfunc="first")
    neural = rag_df[rag_df.model_id != "BM25"]
    best_neural  = {d: neural[neural.dataset==d].ndcg_at_10.max() for d in datasets}
    best_overall = {d: rag_df[rag_df.dataset==d].ndcg_at_10.max() for d in datasets}

    model_order = sorted(
        [m for m in WORKING_SET if m in pivot.index],
        key=lambda m: MODEL_REGISTRY[m]["params_numeric"]
    ) + (["BM25"] if "BM25" in pivot.index else [])

    col_spec = "l r " + "r" * len(datasets)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{RAG benchmark NDCG@10. Bold = best neural per column. "
        r"Underline = best overall (incl. BM25).}",
        r"\label{tab:rag-results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & Params & " + " & ".join(ds_labels[d] for d in datasets) + r" \\",
        r"\midrule",
    ]
    for mid in model_order:
        if mid == "BM25":
            lines.append(r"\midrule")
            name, p_str = "BM25", "--"
        else:
            reg = MODEL_REGISTRY[mid]
            p = reg["params_numeric"]
            p_str = f"{p/1e9:.1f}B" if p >= 1e9 else f"{int(p/1e6)}M"
            name = SHORT_NAMES[mid]
        cells = [name, p_str]
        for d in datasets:
            val = pivot.at[mid, d] if (mid in pivot.index and d in pivot.columns) else float("nan")
            s = _f(val)
            if not pd.isna(val):
                if val >= best_overall[d] - 0.05:
                    s = r"\underline{" + s + "}"
                elif val >= best_neural[d] - 0.05 and mid != "BM25":
                    s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "table2_rag_results.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table2_rag_results.tex saved")

def table3_gap_inflation(inflation_df: pd.DataFrame):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Gap inflation ratios. Perceived = MTEB-agg(large) $-$ MTEB-agg(small). "
        r"Actual = lang-avg(large) $-$ lang-avg(small). "
        r"Inflation = Perceived / Actual.}",
        r"\label{tab:gap-inflation}",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"Pair & Language & Perceived & Actual & Ratio \\",
        r"\midrule",
    ]
    prev = None
    for _, row in inflation_df.sort_values(["pair", "language"]).iterrows():
        p = row.pair if row.pair != prev else ""
        prev = row.pair
        ratio_str = (_f(row.inflation_ratio, 2) + r"$\times$"
                     if not pd.isna(row.inflation_ratio) else "--")
        lines.append(
            f"{p} & {row.language.capitalize()} & "
            f"{_f(row.perceived_gap)} & {_f(row.actual_gap)} & {ratio_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "table3_gap_inflation.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table3_gap_inflation.tex saved")

def table4_decision_error(decision_df: pd.DataFrame):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Smallest model above each threshold under MTEB-agg vs language-specific "
        r"scores. Ratio = params(MTEB choice) / params(Lang choice).}",
        r"\label{tab:decision-error}",
        r"\begin{tabular}{r l l r}",
        r"\toprule",
        r"Threshold & MTEB Choice & Lang Choice & Ratio \\",
        r"\midrule",
    ]
    langs = ["italian", "japanese", "hindi"]
    for idx, lang in enumerate(langs):
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{lang.capitalize()}}}}}\\ \midrule")
        for _, row in decision_df[decision_df.language == lang].sort_values("threshold").iterrows():
            r_str = (_f(row.param_ratio) + r"$\times$"
                     if not pd.isna(row.param_ratio) else "--")
            lines.append(
                f"${row.threshold}\\%$ & {row.mteb_model or '--'} & "
                f"{row.lang_model or '--'} & {r_str} \\\\"
            )
        if idx < len(langs) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "table4_decision_error.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table4_decision_error.tex saved")


# ── Section 12: Rank Inversion Analysis ───────────────────────────────────────

def _draw_rank_inversions_fig(rank_data: dict, hidden_failures: list | None):
    _pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    hf_set = set(hidden_failures) if hidden_failures else set()
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = rank_data.get(lang)
        if sub is None:
            continue
        n = len(sub)
        for _, row in sub.iterrows():
            color = _TIER_COLORS_RANK.get(row.tier, "#888888")
            ax.scatter(row.mteb_rank, row.lang_rank, color=color, s=60, zorder=3, alpha=0.85)
            if abs(row.mteb_rank - row.lang_rank) >= 4:
                ax.annotate(row.short_name, (row.mteb_rank, row.lang_rank),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(4, 3), textcoords="offset points")
            if row.model_id in hf_set:
                ax.scatter(row.mteb_rank, row.lang_rank, marker="x", s=120,
                           color="red", linewidths=2, zorder=5)
        lim = n + 1
        ax.plot([1, lim], [1, lim], "k--", lw=0.8, alpha=0.6, label="y=x")
        ax.set_xlabel("MTEB rank")
        ax.set_ylabel(f"{lang.capitalize()} rank")
        ax.set_title(lang.capitalize())
        ax.set_xlim(0.5, lim + 0.5)
        ax.set_ylim(0.5, lim + 0.5)
        if lang == "italian":  # legend only on first panel to avoid clutter
            handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=8, label=t)
                for t, c in [("small/qwen_sub1b", "#1f77b4"),
                              ("medium", "#ff7f0e"),
                              ("large/xlarge", "#d62728")]
            ]
            if hf_set:
                handles.append(Line2D([0], [0], marker="x", color="red",
                                      markersize=8, linestyle="None",
                                      markeredgewidth=2, label="hidden failure"))
            ax.legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle("MTEB Rank vs Language-Specific Rank", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig_rank_inversions")


def _table_rank_inversions(rank_data: dict):
    lines = [
        r"\begin{table*}[htbp]", r"\centering",
        r"\caption{Models where MTEB rank differs from language-specific rank by $\geq 4$ positions.}",
        r"\label{tab:rank-inversions}",
        r"\begin{tabular}{l r r r l}",
        r"\toprule",
        r"Model & Params & MTEB rank & Lang rank & Language \\",
        r"\midrule",
    ]
    for lang in ["italian", "japanese", "hindi"]:
        sub = rank_data.get(lang)
        if sub is None:
            continue
        notable = sub[abs(sub["mteb_rank"] - sub["lang_rank"]) >= 4].sort_values("mteb_rank")
        if len(notable):
            lines.append(
                rf"\multicolumn{{5}}{{l}}{{\textbf{{{lang.capitalize()}}}}}\\ \midrule"
            )
            for _, row in notable.iterrows():
                p = row.params_numeric
                p_str = f"{p/1e9:.1f}B" if p >= 1e9 else f"{int(p/1e6)}M"
                lines.append(
                    f"{row.short_name} & {p_str} & {int(row.mteb_rank)} & "
                    f"{int(row.lang_rank)} & {lang.capitalize()} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(TAB_DIR, "table_rank_inversions.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table_rank_inversions.tex saved")


def analysis_rank_inversions(df: pd.DataFrame) -> dict:
    rank_data = {}
    lines = []
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.lang_specific_avg.isna())
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        sub["mteb_rank"] = sub["mteb_agg_score"].rank(ascending=False, method="min").astype(int)
        sub["lang_rank"] = sub["lang_specific_avg"].rank(ascending=False, method="min").astype(int)
        n = len(sub)
        total_pairs = n * (n - 1) // 2
        recs = sub[["mteb_rank", "lang_rank"]].to_dict("records")
        inv_count = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if (recs[i]["mteb_rank"] < recs[j]["mteb_rank"]) !=
               (recs[i]["lang_rank"] < recs[j]["lang_rank"])
        )
        tau, p = kendalltau(sub["mteb_rank"].values, sub["lang_rank"].values)
        notable = sub[abs(sub["mteb_rank"] - sub["lang_rank"]) >= 4].sort_values("mteb_rank")
        lines += [
            f"## {lang.capitalize()} (n={n})",
            f"  Kendall tau = {tau:.3f}  (p={p:.3f})",
            f"  Rank inversions: {inv_count} / {total_pairs} pairs "
            f"({100 * inv_count / total_pairs:.1f}%)",
            "  Notable inversions:",
            *[
                f"    {row.short_name}: MTEB rank {int(row.mteb_rank)} "
                f"→ {lang.capitalize()} rank {int(row.lang_rank)}"
                for _, row in notable.iterrows()
            ],
            "",
        ]
        rank_data[lang] = sub
    with open(os.path.join(STAT_DIR, "rank_inversions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] rank_inversions.txt saved")
    _draw_rank_inversions_fig(rank_data, hidden_failures=None)
    _table_rank_inversions(rank_data)
    return rank_data


# ── Section 13: Per-Query-Type RAG Breakdown ──────────────────────────────────

def _fig_rag_bytype(df: pd.DataFrame):
    _pub_style()
    datasets = ["ja_finance", "ja_law", "hi_finance", "hi_law"]
    ds_labels = {"ja_finance": "JA/Finance", "ja_law": "JA/Law",
                 "hi_finance": "HI/Finance", "hi_law": "HI/Law"}
    query_types = ["factual", "multi_hop", "summarization", "unanswerable"]
    qt_colors = {
        "factual": "#1f77b4", "multi_hop": "#ff7f0e",
        "summarization": "#2ca02c", "unanswerable": "#d62728",
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds].copy()
        model_order = sorted(
            sub.model_id.unique(),
            key=lambda m: MODEL_REGISTRY.get(m, {}).get("params_numeric", 0)
        )
        n_models = len(model_order)
        n_qt = len(query_types)
        bar_width = 0.8 / n_qt
        for qt_idx, qt in enumerate(query_types):
            qt_sub = sub[sub.query_type == qt].set_index("model_id")
            xs = np.arange(n_models) + (qt_idx - n_qt / 2 + 0.5) * bar_width
            ys = [
                qt_sub.loc[m, "ndcg_at_10"] if m in qt_sub.index else 0.0
                for m in model_order
            ]
            ax.bar(xs, ys, width=bar_width, label=qt,
                   color=qt_colors[qt], alpha=0.85)
        bm25_val = BM25_BASELINES[ds]["ndcg_at_10"]
        ax.axhline(bm25_val, color="black", lw=1.2, ls="--", alpha=0.8,
                   label=f"BM25 ({bm25_val:.1f})")
        ax.set_xticks(np.arange(n_models))
        ax.set_xticklabels(
            [SHORT_NAMES.get(m, m) for m in model_order],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_title(ds_labels[ds])
        ax.set_ylabel("NDCG@10 (%)")
        if ds == "ja_finance":
            ax.legend(fontsize=7)
    fig.suptitle("RAG Performance by Query Type", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig_rag_bytype")


def _table_rag_bytype(df: pd.DataFrame):
    datasets = ["ja_finance", "ja_law", "hi_finance", "hi_law"]
    ds_labels = {"ja_finance": "JA/Finance", "ja_law": "JA/Law",
                 "hi_finance": "HI/Finance", "hi_law": "HI/Law"}
    query_types = ["factual", "multi_hop", "summarization"]
    all_lines = []
    for ds in datasets:
        sub = df[(df.dataset == ds) & (df.query_type.isin(query_types))].copy()
        pivot = sub.pivot_table(
            index="model_id", columns="query_type",
            values="ndcg_at_10", aggfunc="first",
        )
        model_order = sorted(
            [m for m in pivot.index if m in MODEL_REGISTRY],
            key=lambda m: MODEL_REGISTRY[m]["params_numeric"],
        )
        best = {qt: pivot[qt].max() for qt in query_types if qt in pivot.columns}
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            rf"\caption{{RAG NDCG@10 by query type: {ds_labels[ds]}. "
            r"Bold = best per column. Unanswerable excluded (all models score 0).}",
            rf"\label{{tab:rag-bytype-{ds}}}",
            r"\begin{tabular}{l r r r}",
            r"\toprule",
            r"Model & Factual & Multi-hop & Summarization \\",
            r"\midrule",
        ]
        for mid in model_order:
            cells = [SHORT_NAMES[mid]]
            for qt in query_types:
                val = (pivot.at[mid, qt]
                       if (mid in pivot.index and qt in pivot.columns)
                       else float("nan"))
                s = _f(val, 1)
                if not pd.isna(val) and val >= best.get(qt, float("nan")) - 0.5:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        all_lines.extend(lines)
    with open(os.path.join(TAB_DIR, "table_rag_bytype.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print("[table] table_rag_bytype.tex saved")


def _stats_rag_bytype(df: pd.DataFrame):
    lines = [
        "# RAG By-Type: Spearman rho (log10(params) vs NDCG) per query type", "",
    ]
    for ds in ["ja_finance", "ja_law", "hi_finance", "hi_law"]:
        lines.append(f"## {ds}")
        # summarization excluded: smaller spread makes size correlation less interpretable
        # unanswerable excluded: all models score 0 — Spearman undefined on constant column
        for qt in ["factual", "multi_hop"]:
            sub = df[
                (df.dataset == ds) & (df.query_type == qt)
            ].dropna(subset=["ndcg_at_10"]).copy()
            sub = sub[sub.model_id.isin(MODEL_REGISTRY)]
            if len(sub) >= 3:
                rho, p = spearmanr(np.log10(sub.params_numeric), sub.ndcg_at_10)
                lines.append(
                    f"  Spearman(log_params, {qt}_NDCG): "
                    f"rho={rho:.3f}, p={p:.3f} (n={len(sub)})"
                )
        lines.append("")
    with open(os.path.join(STAT_DIR, "rag_bytype_stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] rag_bytype_stats.txt saved")


def build_rag_bytype_results() -> pd.DataFrame:
    query_types = ["factual", "multi_hop", "summarization", "unanswerable"]
    rows = []
    for folder in sorted(os.listdir(RAG_EVAL_DIR)):
        folder_path = os.path.join(RAG_EVAL_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        model_id = SAN_TO_HF.get(folder)
        if model_id is None:
            continue
        reg = MODEL_REGISTRY[model_id]
        for ds_key in ["ja_finance", "ja_law", "hi_finance", "hi_law"]:
            lang_code, domain = ds_key.split("_", 1)
            json_path = os.path.join(folder_path, f"{lang_code}_{domain}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            by_type = data.get("by_type", {})
            for qt in query_types:
                qt_data = by_type.get(qt, {})
                rows.append({
                    "model_id":      model_id,
                    "short_name":    SHORT_NAMES[model_id],
                    "params_numeric":reg["params_numeric"],
                    "tier":          reg["tier"],
                    "dataset":       ds_key,
                    "query_type":    qt,
                    "ndcg_at_10":    round(qt_data.get("ndcg_at_10", float("nan")) * 100, 2),
                    "recall_at_10":  round(qt_data.get("recall_at_10", float("nan")) * 100, 2),
                    "mrr":           round(qt_data.get("mrr", float("nan")) * 100, 2),
                })
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "unified_rag_bytype_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[rag_bytype] {len(df)} rows -> {out_path}")
    _fig_rag_bytype(df)
    _table_rag_bytype(df)
    _stats_rag_bytype(df)
    return df


# ── Section 14: Alternative Aggregation Proposal ──────────────────────────────

def _fig_aggregation_comparison(agg_df: pd.DataFrame):
    _pub_style()
    small_id, large_id = ANCHOR_PAIRS[0]
    strategies = ["mteb_agg", "lang_filtered", "min_score_agg", "task_coverage_weighted"]
    strat_labels = {
        "mteb_agg":               "MTEB-agg",
        "lang_filtered":          "Lang-filtered",
        "min_score_agg":          "Min-score",
        "task_coverage_weighted": "Coverage-wtd",
    }
    strat_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    langs = ["italian", "japanese", "hindi"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(langs))
    n_s = len(strategies)
    bw = 0.8 / n_s
    for si, strat in enumerate(strategies):
        vals = []
        for lang in langs:
            sub = agg_df[agg_df.language == lang].set_index("model_id")
            if small_id not in sub.index or large_id not in sub.index:
                vals.append(float("nan"))
                continue
            sm, lg = sub.loc[small_id, strat], sub.loc[large_id, strat]
            sm_l = sub.loc[small_id, "lang_specific_avg"]
            lg_l = sub.loc[large_id, "lang_specific_avg"]
            denom = lg_l - sm_l
            vals.append((lg - sm) / denom
                        if (not pd.isna(denom) and denom != 0)
                        else float("nan"))
        xpos = x + (si - n_s / 2 + 0.5) * bw
        ax.bar(xpos, vals, width=bw, label=strat_labels[strat],
               color=strat_colors[si], alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in langs])
    ax.set_ylabel("Inflation Ratio")
    ax.set_title("Inflation Ratio by Aggregation Strategy (anchor: ml-e5-small vs Qwen3-8B)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "fig_aggregation_comparison")


def _table_aggregation(agg_df: pd.DataFrame):
    small_id, large_id = ANCHOR_PAIRS[0]
    strategies = ["mteb_agg", "lang_filtered", "min_score_agg", "task_coverage_weighted"]
    strat_labels = {
        "mteb_agg":               "MTEB-agg",
        "lang_filtered":          "Lang-filtered",
        "min_score_agg":          "Min-score",
        "task_coverage_weighted": "Coverage-wtd",
    }
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Inflation ratios for alternative aggregation strategies. "
        r"Anchor: ml-e5-small vs Qwen3-8B. "
        r"Lang-filtered $\equiv$ 1.00$\times$ by construction.}",
        r"\label{tab:aggregation}",
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"Strategy & Italian & Japanese & Hindi \\",
        r"\midrule",
    ]
    for strat in strategies:
        cells = [strat_labels[strat]]
        for lang in ["italian", "japanese", "hindi"]:
            sub = agg_df[agg_df.language == lang].set_index("model_id")
            if small_id not in sub.index or large_id not in sub.index:
                cells.append("--")
                continue
            sm, lg = sub.loc[small_id, strat], sub.loc[large_id, strat]
            sm_l = sub.loc[small_id, "lang_specific_avg"]
            lg_l = sub.loc[large_id, "lang_specific_avg"]
            denom = lg_l - sm_l
            if pd.isna(denom) or denom == 0:
                cells.append("--")
            else:
                cells.append(_f((lg - sm) / denom, 2) + r"$\times$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "table_aggregation.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table_aggregation.tex saved")


def compute_alternative_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    task_cols = [c for c in df.columns if c.startswith("task_")]
    # min_score_agg: global minimum over all task scores for this model across all languages.
    # Produces a conservative single score per model used as the "perceived" aggregate.
    min_score_map = {}
    for mid in df.model_id.unique():
        model_rows = df[df.model_id == mid]
        all_scores = []
        for col in task_cols:
            all_scores.extend(model_rows[col].dropna().tolist())
        min_score_map[mid] = min(all_scores) if all_scores else float("nan")

    rows = []
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[df.language == lang].copy()
        for _, row in sub.iterrows():
            mid = row.model_id
            avail = [
                (col, _TASK_WEIGHTS[col]) for col in task_cols
                if col in _TASK_WEIGHTS and not pd.isna(row[col])
            ]
            if avail:
                wsum = sum(w * row[col] for col, w in avail)
                wdenom = sum(w for _, w in avail)
                cov_w = wsum / wdenom
            else:
                cov_w = float("nan")
            rows.append({
                "model_id":               mid,
                "short_name":             row.short_name,
                "params_numeric":         row.params_numeric,
                "language":               lang,
                "mteb_agg":               row.mteb_agg_score,
                "lang_filtered":          row.lang_specific_avg,
                "min_score_agg":          round(min_score_map.get(mid, float("nan")), 4),
                "task_coverage_weighted": round(cov_w, 4) if not pd.isna(cov_w) else float("nan"),
                "lang_specific_avg":      row.lang_specific_avg,
            })

    agg_df = pd.DataFrame(rows)

    small_id, large_id = ANCHOR_PAIRS[0]
    strategies = ["mteb_agg", "lang_filtered", "min_score_agg", "task_coverage_weighted"]
    header = f"{'Strategy':<25} {'Italian':>10} {'Japanese':>10} {'Hindi':>10}"
    lines = [
        "# Inflation Ratios by Aggregation Strategy",
        "# Anchor pair: ml-e5-small vs Qwen3-8B",
        "# lang_filtered is always 1.00 by construction (it IS the language score)",
        "# min_score_agg is constant across languages (global model minimum) — inflation varies via denominator",
        "",
        header,
        "-" * len(header),
    ]
    for strat in strategies:
        row_vals = []
        for lang in ["italian", "japanese", "hindi"]:
            sub = agg_df[agg_df.language == lang].set_index("model_id")
            if small_id not in sub.index or large_id not in sub.index:
                row_vals.append("N/A")
                continue
            sm = sub.loc[small_id, strat]
            lg = sub.loc[large_id, strat]
            lg_l = sub.loc[large_id, "lang_specific_avg"]
            sm_l = sub.loc[small_id, "lang_specific_avg"]
            denom = lg_l - sm_l
            if pd.isna(denom) or denom == 0:
                row_vals.append("N/A")
            else:
                row_vals.append(f"{(lg - sm) / denom:.2f}x")
        lines.append(f"{strat:<25} {row_vals[0]:>10} {row_vals[1]:>10} {row_vals[2]:>10}")

    with open(os.path.join(STAT_DIR, "alternative_aggregation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] alternative_aggregation.txt saved")

    _fig_aggregation_comparison(agg_df)
    _table_aggregation(agg_df)
    return agg_df


# ── Section 15: Hidden Failure Mode Analysis ──────────────────────────────────

def analysis_failure_modes(df: pd.DataFrame, rank_data: dict) -> list:
    FAILURE_LANG_THRESH = 50.0
    FAILURE_MTEB_THRESH = 75.0
    lines = [
        "# Hidden Failure Analysis",
        "# Definition: lang_specific_avg < 50 AND mteb_agg_score > 75",
        "",
    ]
    hidden_failures = []
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.lang_specific_avg.isna())
        ].copy()
        hf = sub[
            (sub.lang_specific_avg < FAILURE_LANG_THRESH) &
            (sub.mteb_agg_score > FAILURE_MTEB_THRESH)
        ]
        if len(hf):
            lines.append(f"## {lang.capitalize()}")
            for _, row in hf.iterrows():
                ratio = row.mteb_agg_score / row.lang_specific_avg
                lines.append(
                    f"  {row.short_name}: MTEB={row.mteb_agg_score:.1f}, "
                    f"lang_avg={row.lang_specific_avg:.1f}, "
                    f"ratio={ratio:.2f}x (MTEB score is {ratio:.1f}x misleading)"
                )
                if row.model_id not in hidden_failures:
                    hidden_failures.append(row.model_id)
            lines.append("")
    lines += [
        "## Model Notes (training corpus)",
        "  Snowflake/snowflake-arctic-embed-l-v2.0:",
        "    TODO: verify from model card — known to be trained on English/Nordic corpora",
        "  nvidia/llama-embed-nemotron-8b:",
        "    TODO: verify from model card — English-centric instruction-tuned LLM backbone",
    ]
    with open(os.path.join(STAT_DIR, "hidden_failures.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] hidden_failures.txt saved")
    # Regenerate rank inversions figure with hidden failure markers overlaid
    _draw_rank_inversions_fig(rank_data, hidden_failures=hidden_failures)
    return hidden_failures


# ── Section 16: BelebeleRetrieval-Only Controlled Comparison ─────────────────

def _fig_belebele_controlled(df: pd.DataFrame):
    _pub_style()
    tier_colors = {
        "small": "#1f77b4", "qwen_sub1b": "#1f77b4",
        "medium": "#ff7f0e", "large": "#d62728", "xlarge": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.task_BelebeleRetrieval.isna())
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        sub = sub.sort_values("params_numeric", ascending=True).reset_index(drop=True)
        sub["belebele_gap"] = sub["task_BelebeleRetrieval"] - sub["mteb_agg_score"]
        for i, row in sub.iterrows():
            color = tier_colors.get(row.tier, "#888888")
            ax.scatter(row.belebele_gap, i, color=color, s=60, zorder=3, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.short_name.tolist(), fontsize=7)
        ax.set_xlabel("BelebeleRetrieval gap (task − MTEB-agg)")
        ax.set_title(lang.capitalize())
    fig.suptitle("BelebeleRetrieval-Only Controlled Gap Comparison", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig_belebele_controlled")


def _table_belebele_controlled(bel_gaps: dict):
    # Include nemotron for Japanese/Hindi; Italian column will be "--" (excluded from that eval)
    multi_models = [m for m in WORKING_SET if m not in ENGLISH_ONLY]
    ita_gaps = {m: bel_gaps["italian"].get(m, float("nan")) for m in multi_models}
    sorted_models = sorted(
        multi_models,
        key=lambda m: ita_gaps.get(m, float("-inf")),
        reverse=True,
    )
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{BelebeleRetrieval gap (BelebeleRetrieval NDCG@10 $-$ MTEB-agg) "
        r"per language. Sorted by Italian gap descending. "
        r"Excludes English-only models. Nemotron Italian column is -- (not evaluated).}",
        r"\label{tab:belebele-controlled}",
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"Model & Italian gap & Japanese gap & Hindi gap \\",
        r"\midrule",
    ]
    for mid in sorted_models:
        cells = [SHORT_NAMES[mid]]
        for lang in ["italian", "japanese", "hindi"]:
            gap = bel_gaps[lang].get(mid, float("nan"))
            cells.append(_f(gap, 1))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "table_belebele_controlled.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[table] table_belebele_controlled.tex saved")


def analysis_belebele_controlled(df: pd.DataFrame):
    FULL_AVG_SPEARMAN = {
        ("italian", "japanese"): 0.512,
        ("japanese", "hindi"):   0.829,
        ("italian", "hindi"):    0.459,
    }
    bel_gaps = {}
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.task_BelebeleRetrieval.isna())
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        sub["belebele_gap"] = sub["task_BelebeleRetrieval"] - sub["mteb_agg_score"]
        bel_gaps[lang] = sub.set_index("model_id")["belebele_gap"]

    common_models = sorted(
        set(bel_gaps["italian"].index) &
        set(bel_gaps["japanese"].index) &
        set(bel_gaps["hindi"].index)
    )
    langs = ["italian", "japanese", "hindi"]
    lines = ["# BelebeleRetrieval-Only Controlled Comparison", ""]
    for lang in langs:
        g = bel_gaps[lang].dropna()
        lines += [
            f"## {lang.capitalize()}",
            f"  mean={g.mean():.2f}, std={g.std():.2f}, "
            f"min={g.min():.2f}, max={g.max():.2f}",
            "",
        ]
    lines.append("## Spearman rho matrix (belebele_gap, common models only)")
    for l1 in langs:
        for l2 in langs:
            v1 = [bel_gaps[l1].get(m, float("nan")) for m in common_models]
            v2 = [bel_gaps[l2].get(m, float("nan")) for m in common_models]
            rho, _ = spearmanr(v1, v2)
            lines.append(f"  {l1}-{l2}: rho={rho:.3f}")
    lines.append("")
    lines.append("## Comparison with full-average Spearman")
    for (l1, l2), full_rho in FULL_AVG_SPEARMAN.items():
        v1 = [bel_gaps[l1].get(m, float("nan")) for m in common_models]
        v2 = [bel_gaps[l2].get(m, float("nan")) for m in common_models]
        bel_rho, _ = spearmanr(v1, v2)
        direction = "higher" if bel_rho > full_rho else "lower"
        drives = (
            "may drive" if abs(bel_rho - full_rho) > 0.1
            else "does not appear to drive"
        )
        lines.append(
            f"  {l1}-{l2}: belebele={bel_rho:.3f} vs full-avg={full_rho:.3f} "
            f"({direction}) — task composition {drives} "
            f"the cross-language correlation pattern"
        )
    with open(os.path.join(STAT_DIR, "belebele_controlled.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] belebele_controlled.txt saved")
    _fig_belebele_controlled(df)
    _table_belebele_controlled(bel_gaps)


# ── Section 17: Bootstrap Confidence Intervals on Inflation Ratios ────────────

def _fig_bootstrap_ci(ci_data: dict):
    _pub_style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)
    for ax, lang in zip(axes, ["italian", "japanese", "hindi"]):
        med, lo, hi = ci_data[lang]["anchor"]
        ax.barh([0], [hi - lo], left=[lo], height=0.35,
                color=LANG_COLORS[lang], alpha=0.55, zorder=2)
        ax.scatter([med], [0], color=LANG_COLORS[lang], s=80, zorder=3)
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_yticks([])
        ax.set_ylabel(lang.capitalize(), rotation=0, labelpad=60, va="center")
        ax.set_ylim(-0.5, 0.5)
    axes[-1].set_xlabel("Inflation ratio")
    fig.suptitle("Bootstrap 95% CI — Anchor-Pair Inflation Ratio", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig_bootstrap_ci")


def analysis_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int = 10000):
    rng = np.random.default_rng(42)
    ci_data = {}
    lines = ["# Bootstrap Confidence Intervals — Inflation Ratios", ""]

    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.lang_specific_avg.isna()) &
            (~df.mteb_agg_score.isna())
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        n = len(sub)
        mteb_v  = sub["mteb_agg_score"].values
        lang_v  = sub["lang_specific_avg"].values
        param_v = sub["params_numeric"].values

        anchor_inflations, mean_inflations = [], []
        for _ in range(n_bootstrap):
            idx   = rng.integers(0, n, size=n)
            s_m   = mteb_v[idx]
            s_l   = lang_v[idx]
            s_p   = param_v[idx]
            si    = np.argmin(s_p)
            li    = np.argmax(s_p)
            denom = s_l[li] - s_l[si]
            if denom != 0 and not np.isnan(denom):
                anchor_inflations.append((s_m[li] - s_m[si]) / denom)
            ml, ll = np.mean(s_m), np.mean(s_l)
            if ll != 0:
                mean_inflations.append(ml / ll)

        a_arr = np.array(anchor_inflations)
        m_arr = np.array(mean_inflations)
        med_a = np.median(a_arr)
        lo_a, hi_a = np.percentile(a_arr, [2.5, 97.5])
        med_m = np.median(m_arr)
        lo_m, hi_m = np.percentile(m_arr, [2.5, 97.5])
        p_a = np.mean(a_arr > 1.0)
        p_m = np.mean(m_arr > 1.0)
        p_str_a = "< 0.001" if p_a > 0.999 else ("> 0.999" if p_a < 0.001 else f"= {1 - p_a:.3f}")
        p_str_m = "< 0.001" if p_m > 0.999 else ("> 0.999" if p_m < 0.001 else f"= {1 - p_m:.3f}")
        ci_data[lang] = {"anchor": (med_a, lo_a, hi_a), "mean": (med_m, lo_m, hi_m)}
        lines += [
            f"{lang.capitalize()}:",
            f"  Anchor-pair: inflation = {med_a:.2f} "
            f"[95% CI: {lo_a:.2f} – {hi_a:.2f}]  (p {p_str_a})",
            f"  Mean-score:  inflation = {med_m:.2f} "
            f"[95% CI: {lo_m:.2f} – {hi_m:.2f}]  (p {p_str_m})",
            "",
        ]

    with open(os.path.join(STAT_DIR, "bootstrap_ci.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] bootstrap_ci.txt saved")
    _fig_bootstrap_ci(ci_data)


# ── Section 18: Embedding Dimension vs Gap Regression ─────────────────────────

def _fig_dim_vs_gap(plot_data: dict):
    _pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lang in ["italian", "japanese", "hindi"]:
        d = plot_data[lang]
        sub = d["sub"]
        xs_p = sub.log_params.values
        xs_d = sub.log_dim.values
        ys   = sub.gap.values
        axes[0].scatter(xs_p, ys, color=LANG_COLORS[lang],
                        marker=LANG_MARKERS[lang], s=50, alpha=0.8,
                        label=lang.capitalize())
        sl_p, ic_p, p_p = d["params_reg"]
        if p_p < 0.1:
            xr = np.linspace(xs_p.min(), xs_p.max(), 50)
            axes[0].plot(xr, sl_p * xr + ic_p,
                         color=LANG_COLORS[lang], lw=1.2, ls="--")
        axes[1].scatter(xs_d, ys, color=LANG_COLORS[lang],
                        marker=LANG_MARKERS[lang], s=50, alpha=0.8,
                        label=lang.capitalize())
        sl_d, ic_d, p_d = d["dim_reg"]
        if p_d < 0.1:
            xr = np.linspace(xs_d.min(), xs_d.max(), 50)
            axes[1].plot(xr, sl_d * xr + ic_d,
                         color=LANG_COLORS[lang], lw=1.2, ls="--")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_ylabel("Gap (lang − MTEB-agg)")
    axes[0].set_xlabel("log10(Parameters)")
    axes[0].set_title("Params Regression")
    axes[0].set_xticks([7.5, 8.0, 8.5, 9.0, 9.5, 9.9])
    axes[0].set_xticklabels(["33M", "100M", "316M", "1B", "3B", "8B"])
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("log2(Embedding Dimension)")
    axes[1].set_title("Dim Regression")
    axes[1].set_xticks([8, 9, 10, 11, 12])
    axes[1].set_xticklabels(["256", "512", "1024", "2048", "4096"])
    axes[1].legend(fontsize=8)
    fig.suptitle("Gap vs Model Size: Parameters vs Embedding Dimension", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig_dim_vs_gap")


def analysis_dim_regression(df: pd.DataFrame):
    lines = ["# Embedding Dimension vs Gap Regression", ""]
    plot_data = {}
    for lang in ["italian", "japanese", "hindi"]:
        sub = df[
            (df.language == lang) &
            (~df.english_only) &
            (df.source != "missing") &
            (~df.gap.isna()) &
            (~df.dim.isna())
        ].copy()
        if lang == "italian":
            sub = sub[~sub.model_id.isin(EXCLUDE_ITALIAN)]
        sub["log_params"] = np.log10(sub.params_numeric.astype(float))
        sub["log_dim"]    = np.log2(sub.dim.astype(float))
        n = len(sub)

        sl_p, ic_p, r_p, p_p, _ = linregress(sub.log_params, sub.gap)
        sl_d, ic_d, r_d, p_d, _ = linregress(sub.log_dim,    sub.gap)

        X = np.column_stack([sub.log_params.values,
                              sub.log_dim.values,
                              np.ones(n)])
        y = sub.gap.values
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_mv  = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        key = " *** dim R2 > params R2 ***" if r_d ** 2 > r_p ** 2 else ""
        lines += [
            f"## {lang.capitalize()}  (n={n})",
            f"  log10(params): slope={sl_p:.3f}, R2={r_p**2:.4f}, p={p_p:.4f}",
            f"  log2(dim):     slope={sl_d:.3f}, R2={r_d**2:.4f}, p={p_d:.4f}{key}",
            f"  multivariate:  R2={r2_mv:.4f}  "
            f"(log_params coeff={coeffs[0]:.3f}, log_dim coeff={coeffs[1]:.3f})",
            "",
        ]
        plot_data[lang] = {
            "sub":        sub,
            "params_reg": (sl_p, ic_p, p_p),
            "dim_reg":    (sl_d, ic_d, p_d),
        }

    with open(os.path.join(STAT_DIR, "dim_regression.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[stats] dim_regression.txt saved")
    _fig_dim_vs_gap(plot_data)


def _smoke_test_loaders():
    agg = load_mteb_agg()
    assert "intfloat/multilingual-e5-small" in agg
    assert abs(agg["intfloat/multilingual-e5-small"] - 77.0) < 0.5, \
        f"MTEB_agg for ml-e5-small: expected ~77.0, got {agg['intfloat/multilingual-e5-small']}"
    assert "Qwen/Qwen3-Embedding-8B" in agg
    assert abs(agg["Qwen/Qwen3-Embedding-8B"] - 90.4) < 0.5

    own_ita = load_own_runs("italian")
    assert "BAAI/bge-m3" in own_ita
    assert "BelebeleRetrieval" in own_ita["BAAI/bge-m3"]
    assert "WikipediaRetrievalMultilingual" in own_ita["BAAI/bge-m3"]
    assert "intfloat/e5-mistral-7b-instruct" not in own_ita
    assert "Salesforce/SFR-Embedding-Mistral" in own_ita
    assert "WikipediaRetrievalMultilingual" not in own_ita["Salesforce/SFR-Embedding-Mistral"]

    own_jpn = load_own_runs("japanese")
    assert "BAAI/bge-m3" in own_jpn
    assert "Qwen/Qwen3-Embedding-4B" in own_jpn
    assert "BelebeleRetrieval" in own_jpn["Qwen/Qwen3-Embedding-4B"]
    assert "MIRACLRetrievalHardNegatives" not in own_jpn["Qwen/Qwen3-Embedding-4B"]
    assert "Salesforce/SFR-Embedding-Mistral" not in own_jpn

    lb_ita = load_lb_lang_tasks("italian")
    for lb_name, scores in lb_ita.items():
        assert "MIRACLRetrievalHardNegatives" not in scores, \
            f"MIRACL should be excluded for Italian but found in {lb_name}"

    print("[smoke] All data loaders OK")


def setup_dirs():
    for d in [OUT_DIR, FIG_DIR, TAB_DIR, STAT_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[setup] Output directories created.")

def main():
    print("=" * 60)
    print("Analysis pipeline starting")
    print("=" * 60)
    setup_dirs()

    df = build_unified_results()
    _validate_unified_results(df)
    rag_df = build_unified_rag_results()
    _validate_rag_results(rag_df)

    reg_results  = analysis_31_gap_regression(df)
    save_regression_results(reg_results)
    inflation_df = analysis_32_gap_inflation(df)
    decision_df  = analysis_33_decision_error(df)
    corr_df, common = analysis_35_spearman_gaps(df)
    save_spearman_results(corr_df, common)
    save_summary_statistics(df, reg_results, inflation_df)

    figure1_gap_vs_params(df, reg_results)
    figure2_decision_error(decision_df)
    figure3_pareto(df)

    table1_main_results(df)
    table2_rag_results(rag_df)
    table3_gap_inflation(inflation_df)
    table4_decision_error(decision_df)

    # ── Supplementary analyses ────────────────────────────────────────────────
    supp_df = pd.read_csv(os.path.join(OUT_DIR, "unified_results.csv"))

    rank_data = analysis_rank_inversions(supp_df)       # Task A
    build_rag_bytype_results()                          # Task B
    compute_alternative_aggregations(supp_df)           # Task C
    analysis_failure_modes(supp_df, rank_data)          # Task D
    analysis_belebele_controlled(supp_df)               # Task E
    analysis_bootstrap_ci(supp_df)                      # Task F
    analysis_dim_regression(supp_df)                    # Task G

    print("[main] All supplementary analyses complete.")

    import shutil
    shutil.copy(__file__, os.path.join(OUT_DIR, "analysis.py"))

    print("=" * 60)
    print(f"Done. All outputs in: {OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    setup_dirs()
    print(f"[setup] Working set: {len(WORKING_SET)} models")
    assert len(SAN_TO_HF) == 17, f"Expected 17 sanitized IDs, got {len(SAN_TO_HF)}"
    print("[setup] SAN_TO_HF verified (17 entries)")
    _smoke_test_loaders()
    main()
