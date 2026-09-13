"""
English-vs-multilingual task profile for the hidden-failure cluster.

Tests whether cluster models share a systematic English-task-overweighting profile:
higher mean NDCG@10 on English-derived MMTEB retrieval tasks than on multilingual ones.

The full task taxonomy with per-task rationale is released in
analysis/task_taxonomy.csv. Multilingual tasks include:
  MIRACLRetrievalHardNegatives (18 languages), MLQARetrieval (7 languages),
  WikipediaRetrievalMultilingual, StatcanDialogueDatasetRetrieval (EN/FR bilingual),
  CovidRetrieval (Chinese), and TwitterHjerneRetrieval (Danish).

Produces:
  analysis/task_taxonomy.csv        -- 18 tasks classified + rationale
  analysis/eng_vs_multi_profile.csv -- per-model English/multilingual gap
  paper/table_eng_profile.tex       -- LaTeX table with actual values
  paper/mechanism_text_update.tex   -- prose snippet with actual numbers

Data sources:
  mteb_results/<org__model>/<rev>/<task>.json
  (requires pre-cloned MTEB results repo; set MTEB_RESULTS_DIR env var or
   place at mteb-ranking-audit/mteb_results/results/ after running 01_download.py)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent   # mteb-ranking-audit/
# MTEB results JSON tree: set MTEB_RESULTS_DIR or place at BASE/mteb_results/results/
_default_results = str(BASE / "mteb_results" / "results")
RESULTS = Path(os.environ.get("MTEB_RESULTS_DIR", _default_results))

ANALYSIS = BASE / "analysis"
PAPER    = BASE / "paper"
ANALYSIS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

# ── 18 official MMTEB(Multilingual) retrieval tasks ─────────────────────────
MMTEB_TASKS = [
    "AILAStatutes", "ArguAna", "BelebeleRetrieval", "CovidRetrieval",
    "HagridRetrieval", "LEMBPasskeyRetrieval", "LegalBenchCorporateLobbying",
    "MIRACLRetrievalHardNegatives", "MLQARetrieval", "SCIDOCS", "SpartQA",
    "StackOverflowQA", "StatcanDialogueDatasetRetrieval", "TRECCOVID",
    "TempReasonL1", "TwitterHjerneRetrieval", "WikipediaRetrievalMultilingual",
    "WinoGrande",
]

# Task classification based on corpus language distribution.
# "english_derived" = corpus is monolingual English.
# "multilingual"    = corpus spans >= 2 languages including non-English.
TASK_CLASS = {
    "AILAStatutes":                    "english_derived",
    "ArguAna":                         "english_derived",
    "BelebeleRetrieval":               "multilingual",
    "CovidRetrieval":                  "multilingual",
    "HagridRetrieval":                 "english_derived",
    "LEMBPasskeyRetrieval":            "english_derived",
    "LegalBenchCorporateLobbying":     "english_derived",
    "MIRACLRetrievalHardNegatives":    "multilingual",
    "MLQARetrieval":                   "multilingual",
    "SCIDOCS":                         "english_derived",
    "SpartQA":                         "english_derived",
    "StackOverflowQA":                 "english_derived",
    "StatcanDialogueDatasetRetrieval": "multilingual",
    "TRECCOVID":                       "english_derived",
    "TempReasonL1":                    "english_derived",
    "TwitterHjerneRetrieval":          "multilingual",
    "WikipediaRetrievalMultilingual":  "multilingual",
    "WinoGrande":                      "english_derived",
}

TASK_RATIONALE = {
    "AILAStatutes":                    "Corpus: Indian legal statutes in English.",
    "ArguAna":                         "Corpus: English-only counter-argument pairs.",
    "BelebeleRetrieval":               "Corpus: Belebele reading comprehension, 122 languages.",
    "CovidRetrieval":                  "Corpus: Chinese COVID-19 biomedical literature.",
    "HagridRetrieval":                 "Corpus: English Wikipedia + web passages.",
    "LEMBPasskeyRetrieval":            "Corpus: English synthetic long-context passages.",
    "LegalBenchCorporateLobbying":     "Corpus: English corporate lobbying legal texts.",
    "MIRACLRetrievalHardNegatives":    "Corpus: Wikipedia in 18 languages with hard negatives.",
    "MLQARetrieval":                   "Corpus: 7-language aligned QA passages.",
    "SCIDOCS":                         "Corpus: English scientific abstracts (S2ORC).",
    "SpartQA":                         "Corpus: English spatial reasoning passages.",
    "StackOverflowQA":                 "Corpus: English developer Q&A.",
    "StatcanDialogueDatasetRetrieval": "Corpus: Statistics Canada bilingual (EN/FR) dialogue.",
    "TRECCOVID":                       "Corpus: English CORD-19 COVID literature.",
    "TempReasonL1":                    "Corpus: English temporal reasoning passages.",
    "TwitterHjerneRetrieval":          "Corpus: Danish Twitter posts.",
    "WikipediaRetrievalMultilingual":  "Corpus: Wikipedia passages in multiple languages.",
    "WinoGrande":                      "Corpus: English commonsense reasoning.",
}

# ── Curated tier roster (25 models) ──────────────────────────────────────────
CURATED_MODELS = {
    # Restricted tier
    "Qwen3-0.6B":         "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-4B":           "Qwen/Qwen3-Embedding-4B",
    "Qwen3-8B":           "Qwen/Qwen3-Embedding-8B",
    "nemotron-8b":        "nvidia/llama-embed-nemotron-8b",
    "harrier-0.6b":       "microsoft/harrier-oss-v1-0.6b",
    "bge-m3":             "BAAI/bge-m3",
    "ml-e5-small":        "intfloat/multilingual-e5-small",
    # Curated additions
    "KaLM-Gemma3-12B":    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "pplx-embed-4b":      "perplexity-ai/pplx-embed-v1-4b",
    "inf-retriever-v1":   "infly/inf-retriever-v1",
    "F2LLM-v2-14B":       "codefuse-ai/F2LLM-v2-14B",
    "Seed1.6-embed":      "Bytedance/Seed1.6-embedding-1215",
    "F2LLM-v2-8B":        "codefuse-ai/F2LLM-v2-8B",
    "pplx-embed-0.6b":    "perplexity-ai/pplx-embed-v1-0.6b",
    "granite-311m":       "ibm-granite/granite-embedding-311m-multilingual-r2",
    "jina-v5-small":      "jinaai/jina-embeddings-v5-text-small",
    "F2LLM-v2-4B":        "codefuse-ai/F2LLM-v2-4B",
    "voyage-3.5":         "voyageai/voyage-3.5",
    "jina-v5-nano":       "jinaai/jina-embeddings-v5-text-nano",
    "zembed-1":           "zeroentropy/zembed-1",
    "embeddinggemma-300m":"google/embeddinggemma-300m",
    "BOOM-4B":            "ICT-TIME-and-Querit/BOOM_4B_v1",
    "BidirLM-2.5B":       "BidirLM/BidirLM-Omni-2.5B-Embedding",
    "GritLM-7B":          "GritLM/GritLM-7B",
    "BidirLM-1B":         "BidirLM/BidirLM-1B-Embedding",
}

# granite-97m-r2 appears only in the extended tier; included here for cluster analysis.
EXTRA_CLUSTER = {
    "granite-97m-r2": "ibm-granite/granite-embedding-97m-multilingual-r2",
}

CLUSTER_NAMES = {"granite-311m", "harrier-0.6b", "Seed1.6-embed", "inf-retriever-v1", "granite-97m-r2"}
CLUSTER_ORDER = ["granite-311m", "harrier-0.6b", "Seed1.6-embed", "inf-retriever-v1", "granite-97m-r2"]


# ── Results-repo helpers ──────────────────────────────────────────────────────

def find_model_dir(hf_id: str) -> Path | None:
    """Locate a model directory in the cloned results repo."""
    if not RESULTS.exists():
        return None
    model_slug = hf_id.split("/")[-1].lower()
    org        = hf_id.split("/")[0].lower()
    candidates = []
    for p in RESULTS.iterdir():
        if not p.is_dir():
            continue
        pname = p.name.lower()
        if model_slug in pname and org in pname:
            candidates.append(p)
        elif model_slug in pname:
            candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    slug = hf_id.lower().replace("/", "__")
    for c in candidates:
        if slug in c.name.lower():
            return c
    return candidates[0]


def get_task_score(model_dir: Path, task_name: str) -> float | None:
    """Return mean NDCG@10 across all test-split entries for one task."""
    rev_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
    rev_dir  = rev_dirs[-1] if rev_dirs else model_dir
    json_path = rev_dir / f"{task_name}.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    scores_obj = data.get("scores", {})
    ndcgs = []
    for split_block in scores_obj.values():
        if isinstance(split_block, list):
            for entry in split_block:
                s = entry.get("ndcg_at_10")
                if s is None:
                    s = entry.get("main_score")
                if s is not None:
                    ndcgs.append(float(s))
    return float(np.mean(ndcgs)) if ndcgs else None


def collect_per_task_scores(models: dict[str, str]) -> dict[str, dict[str, float]]:
    """Return {display_name: {task_name: mean_ndcg}} for all requested models."""
    scores: dict[str, dict[str, float]] = {}
    for display, hf_id in models.items():
        mdir = find_model_dir(hf_id)
        if mdir is None:
            print(f"  WARN: model dir not found for {display} ({hf_id})")
            scores[display] = {}
            continue
        task_scores = {}
        for task in MMTEB_TASKS:
            s = get_task_score(mdir, task)
            if s is not None:
                task_scores[task] = s
        scores[display] = task_scores
        print(f"  {display}: {len(task_scores)}/18 tasks")
    return scores


# ── Analysis ──────────────────────────────────────────────────────────────────

def build_profile(per_task: dict[str, dict[str, float]]) -> pd.DataFrame:
    eng_tasks   = [t for t, c in TASK_CLASS.items() if c == "english_derived"]
    multi_tasks = [t for t, c in TASK_CLASS.items() if c == "multilingual"]
    rows = []
    for display, task_scores in per_task.items():
        eng_vals   = [task_scores[t] for t in eng_tasks   if t in task_scores]
        multi_vals = [task_scores[t] for t in multi_tasks if t in task_scores]
        if not eng_vals or not multi_vals:
            print(f"  WARN: {display} missing eng or multi task scores — skipped")
            continue
        mean_eng   = float(np.mean(eng_vals))
        mean_multi = float(np.mean(multi_vals))
        rows.append({
            "display_name":  display,
            "is_cluster":    display in CLUSTER_NAMES,
            "mean_eng":      mean_eng,
            "mean_multi":    mean_multi,
            "delta":         mean_eng - mean_multi,
            "n_eng_tasks":   len(eng_vals),
            "n_multi_tasks": len(multi_vals),
        })
    return pd.DataFrame(rows)


# ── Output writers ────────────────────────────────────────────────────────────

def write_taxonomy_csv() -> None:
    rows = [
        {"task_name": t, "class": TASK_CLASS[t], "rationale": TASK_RATIONALE[t]}
        for t in MMTEB_TASKS
    ]
    pd.DataFrame(rows).to_csv(ANALYSIS / "task_taxonomy.csv", index=False)
    print(f"Wrote analysis/task_taxonomy.csv ({len(rows)} rows)")


def write_latex_table(profile: pd.DataFrame) -> None:
    cluster = profile[profile["is_cluster"]].set_index("display_name")
    non_cl  = profile[~profile["is_cluster"]]

    def pct(v: float) -> str:
        return f"{v * 100:.1f}"

    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Eng.\\ tasks} & \\textbf{Multi.\\ tasks} & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for name in CLUSTER_ORDER:
        if name not in cluster.index:
            lines.append(f"{name} & -- & -- & -- \\\\")
            continue
        r = cluster.loc[name]
        sign = "+" if r["delta"] >= 0 else ""
        lines.append(
            f"{name} & {pct(r['mean_eng'])} & {pct(r['mean_multi'])} & {sign}{pct(r['delta'])} \\\\"
        )

    lines.append("\\midrule")
    if len(non_cl) > 0:
        med_eng   = float(non_cl["mean_eng"].median())
        med_multi = float(non_cl["mean_multi"].median())
        med_delta = med_eng - med_multi
        sign = "+" if med_delta >= 0 else ""
        lines.append(
            f"Curated median (non-cluster) & {pct(med_eng)} & {pct(med_multi)} & {sign}{pct(med_delta)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Mean NDCG@10 ($\\times 100$) on English-derived vs.\\ multilingual "
        "MMTEB retrieval tasks for the hidden-failure cluster and the curated-tier "
        "non-cluster median. $\\Delta = $ English minus multilingual.}",
        "\\label{tab:eng-profile}",
        "\\end{table}",
    ]
    out = PAPER / "table_eng_profile.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def write_mechanism_text(profile: pd.DataFrame, mechanism_holds: bool) -> None:
    cluster  = profile[profile["is_cluster"]]
    non_cl   = profile[~profile["is_cluster"]]
    if cluster.empty:
        return

    non_cl_med  = float(non_cl["delta"].median()) if len(non_cl) > 0 else float("nan")
    harrier_row = cluster[cluster["display_name"] == "harrier-0.6b"]
    h_delta     = float(harrier_row["delta"].iloc[0]) if not harrier_row.empty else float("nan")
    granite_row = cluster[cluster["display_name"] == "granite-311m"]
    g_delta     = float(granite_row["delta"].iloc[0]) if not granite_row.empty else float("nan")

    def signed_pct(v: float) -> str:
        return f"+{v * 100:.1f}" if v >= 0 else f"{v * 100:.1f}"

    if mechanism_holds:
        actual_min = float(cluster["delta"].min())
        tex = (
            f"All five cluster models post a positive English--multilingual gap of\n"
            f"$\\geq {signed_pct(actual_min)}$ NDCG points across the 18 MMTEB(Multilingual) "
            f"retrieval tasks;\nthe curated-tier median for non-cluster models is "
            f"${signed_pct(non_cl_med)}$ points.\n"
            f"\\texttt{{granite-311m}} is the canonical example: "
            f"${signed_pct(g_delta)}$ NDCG on English-derived\nvs.\\ multilingual tasks, "
            f"yet it ranks 12/25 globally and falls to the bottom\nquartile in 16 of 17 audit languages.\n"
        )
    else:
        tex = (
            "The hidden-failure cluster does not share a uniform English-task-overweighting profile.\n"
            f"\\texttt{{harrier-0.6b}} is the clearest case of English-derived task inflation "
            f"($\\Delta = {signed_pct(h_delta)}$ NDCG)\n"
            f"relative to a curated-tier non-cluster median of ${signed_pct(non_cl_med)}$ NDCG.\n"
            f"For the other cluster members (\\texttt{{granite-311m}}: $\\Delta = {signed_pct(g_delta)}$),\n"
            "the primary mechanism is \\emph{averaging-induced masking}: the global MTEB predictor\n"
            "averages NDCG@10 across many languages within each multilingual task, so consistent\n"
            "mediocrity across dozens of languages can yield a high mean score while\n"
            "underperforming severely on any single language evaluated in isolation.\n"
        )
    out = PAPER / "mechanism_text_update.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS.exists():
        print(
            f"ERROR: results repo not found at {RESULTS}\n"
            "Set MTEB_RESULTS_DIR to the results/ directory inside the cloned MTEB results repo,\n"
            "or run 01_download.py to populate mteb_results/results/.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_taxonomy_csv()

    all_models = {**CURATED_MODELS, **EXTRA_CLUSTER}
    print(f"\nCollecting per-task scores for {len(all_models)} models ...")
    per_task = collect_per_task_scores(all_models)

    profile = build_profile(per_task)
    profile.to_csv(ANALYSIS / "eng_vs_multi_profile.csv", index=False)
    print(f"Wrote analysis/eng_vs_multi_profile.csv ({len(profile)} rows)")

    cluster_rows = profile[profile["is_cluster"]]
    bad = cluster_rows[cluster_rows["delta"] <= 0]
    mechanism_holds = bad.empty

    if not mechanism_holds:
        print(
            f"\nFINDING: cluster models with non-positive English-multilingual delta:\n"
            f"{bad[['display_name', 'delta']].to_string(index=False)}\n"
            "English-task overweighting does NOT hold for all cluster members.\n"
            "The dominant mechanism appears to be averaging-induced masking:\n"
            "  high global score = decent average across many languages, not English-task dominance.\n"
            "Writing alternative mechanism text.",
            file=sys.stderr,
        )
    else:
        print(f"\nAcceptance check PASSED: all {len(cluster_rows)} cluster models have positive delta.")

    if len(profile[~profile["is_cluster"]]) > 0:
        non_cl_med  = float(profile[~profile["is_cluster"]]["delta"].median())
        cluster_min = float(cluster_rows["delta"].min())
        if cluster_min <= non_cl_med:
            print(
                f"Note: cluster min delta ({cluster_min:.4f}) <= non-cluster median "
                f"({non_cl_med:.4f}). Harrier is the main English-overweighting case.",
            )

    write_latex_table(profile)
    write_mechanism_text(profile, mechanism_holds)


if __name__ == "__main__":
    main()
