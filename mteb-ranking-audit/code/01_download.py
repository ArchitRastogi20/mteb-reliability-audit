"""
Download MTEB multilingual results for 18 languages.
Benchmark: MTEB(Multilingual) — pinned to this version throughout.
Strategy: MTEB Python SDK first, then fall back to cloning the results repo.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "code" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

BENCHMARK_NAME = "MTEB(Multilingual)"
log.info("Benchmark version pinned to: %s", BENCHMARK_NAME)

# ── ISO 639-3 → BCP-47 mapping ────────────────────────────────────────────────
# MTEB result JSONs use hyphen-separated BCP-47 codes (e.g. "ita-Latn", "ara-Arab")
LANG_MAP = {
    "ita": {"bcp47": "ita-Latn", "name": "Italian",    "folder": BASE / "case_studies" / "ita"},
    "jpn": {"bcp47": "jpn-Jpan", "name": "Japanese",   "folder": BASE / "case_studies" / "jpn"},
    "hin": {"bcp47": "hin-Deva", "name": "Hindi",      "folder": BASE / "case_studies" / "hin"},
    "ara": {"bcp47": "ara-Arab", "name": "Arabic",     "folder": BASE / "audit_languages" / "ara"},
    "zho": {"bcp47": "zho-Hans", "name": "Chinese",    "folder": BASE / "audit_languages" / "zho"},
    "deu": {"bcp47": "deu-Latn", "name": "German",     "folder": BASE / "audit_languages" / "deu"},
    "spa": {"bcp47": "spa-Latn", "name": "Spanish",    "folder": BASE / "audit_languages" / "spa"},
    "rus": {"bcp47": "rus-Cyrl", "name": "Russian",    "folder": BASE / "audit_languages" / "rus"},
    "fra": {"bcp47": "fra-Latn", "name": "French",     "folder": BASE / "audit_languages" / "fra"},
    "kor": {"bcp47": "kor-Hang", "name": "Korean",     "folder": BASE / "audit_languages" / "kor"},
    "ben": {"bcp47": "ben-Beng", "name": "Bengali",    "folder": BASE / "audit_languages" / "ben"},
    "vie": {"bcp47": "vie-Latn", "name": "Vietnamese", "folder": BASE / "audit_languages" / "vie"},
    "ind": {"bcp47": "ind-Latn", "name": "Indonesian", "folder": BASE / "audit_languages" / "ind"},
    "fas": {"bcp47": "pes-Arab", "name": "Persian",    "folder": BASE / "audit_languages" / "fas"},
    "swa": {"bcp47": "swh-Latn", "name": "Swahili",    "folder": BASE / "audit_languages" / "swa"},
    "tha": {"bcp47": "tha-Thai", "name": "Thai",       "folder": BASE / "audit_languages" / "tha"},
    "tel": {"bcp47": "tel-Telu", "name": "Telugu",     "folder": BASE / "audit_languages" / "tel"},
    "tam": {"bcp47": "tam-Taml", "name": "Tamil",      "folder": BASE / "audit_languages" / "tam"},
}

# Alternative BCP-47 codes to try if primary not found (hyphen format).
# Some tasks use different codes for the same language (e.g. MIRACL vs Belebele).
LANG_ALT = {
    "ara": ["ara-Arab", "arb-Arab"],   # MIRACL uses ara-Arab; Belebele uses arb-Arab (MSA)
    "fas": ["pes-Arab", "fas-Arab"],   # Belebele uses pes-Arab, MIRACL uses fas-Arab
    "kor": ["kor-Hang", "kor-Kore"],   # Belebele uses kor-Hang, MIRACL uses kor-Kore
    "swa": ["swh-Latn", "swa-Latn"],   # Belebele uses swh-Latn, MIRACL uses swa-Latn
}

MODELS = {
    "granite-107m":          "ibm-granite/granite-embedding-107m-multilingual",
    "ml-e5-small":           "intfloat/multilingual-e5-small",
    "jina-v3":               "jinaai/jina-embeddings-v3",
    "ml-e5-base":            "intfloat/multilingual-e5-base",
    "ml-e5-large":           "intfloat/multilingual-e5-large",
    "ml-e5-large-instruct":  "intfloat/multilingual-e5-large-instruct",
    "bge-m3":                "BAAI/bge-m3",
    "snowflake-arctic-l":    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen3-0.6B":            "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-4B":              "Qwen/Qwen3-Embedding-4B",
    "Qwen3-8B":              "Qwen/Qwen3-Embedding-8B",
    "SFR-Embedding-Mistral": "Salesforce/SFR-Embedding-Mistral",
    "e5-mistral-7b":         "intfloat/e5-mistral-7b-instruct",
    "nemotron-8b":           "nvidia/llama-embed-nemotron-8b",
    "harrier-0.6b":          "microsoft/harrier-oss-v1-0.6b",
}

MODELS_CURATED = {
    **MODELS,
    "voyage-3.5":        "voyageai/voyage-3.5",
    "KaLM-Gemma3-12B":   "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "embeddinggemma-300m":"google/embeddinggemma-300m",
    "inf-retriever-v1":  "infly/inf-retriever-v1",
    "F2LLM-v2-4B":       "codefuse-ai/F2LLM-v2-4B",
    "F2LLM-v2-8B":       "codefuse-ai/F2LLM-v2-8B",
    "F2LLM-v2-14B":      "codefuse-ai/F2LLM-v2-14B",
    "BOOM-4B":           "ICT-TIME-and-Querit/BOOM_4B_v1",
    "jina-v5-small":     "jinaai/jina-embeddings-v5-text-small",
    "pplx-embed-0.6b":   "perplexity-ai/pplx-embed-v1-0.6b",
    "pplx-embed-4b":     "perplexity-ai/pplx-embed-v1-4b",
    "zembed-1":          "zeroentropy/zembed-1",
    "GritLM-7B":         "GritLM/GritLM-7B",
    "Seed1.6-embed":     "Bytedance/Seed1.6-embedding-1215",
    "BidirLM-1B":        "BidirLM/BidirLM-1B-Embedding",
    "BidirLM-2.5B":      "BidirLM/BidirLM-Omni-2.5B-Embedding",
    "granite-311m":      "ibm-granite/granite-embedding-311m-multilingual-r2",
    "jina-v5-nano":      "jinaai/jina-embeddings-v5-text-nano",
}

# Tasks used for per-language scoring (multilingual retrieval benchmarks).
LANG_TASKS = [
    "BelebeleRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MLQARetrieval",
    "WikipediaRetrievalMultilingual",
    "MrTidyRetrieval",
    "XPQARetrieval",
]

# Official MMTEB(Multilingual) retrieval tasks — used to compute the global predictor
# (Mean(Task, Type=Retrieval) as published on the leaderboard, snapshot 2026-05).
# Source: mteb.get_benchmark("MTEB(Multilingual)").tasks filtered by type==Retrieval.
MMTEB_RETRIEVAL_TASKS = [
    "AILAStatutes", "ArguAna", "BelebeleRetrieval", "CovidRetrieval",
    "HagridRetrieval", "LEMBPasskeyRetrieval", "LegalBenchCorporateLobbying",
    "MIRACLRetrievalHardNegatives", "MLQARetrieval", "SCIDOCS", "SpartQA",
    "StackOverflowQA", "StatcanDialogueDatasetRetrieval", "TRECCOVID",
    "TempReasonL1", "TwitterHjerneRetrieval", "WikipediaRetrievalMultilingual",
    "WinoGrande",
]

RESULTS_REPO = BASE / "mteb_results_repo"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_results_repo():
    if not RESULTS_REPO.exists():
        log.info("Cloning MTEB results repo …")
        result = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/embeddings-benchmark/results",
             str(RESULTS_REPO)],
        )
        if result.returncode != 0:
            # Partial checkout (e.g. Windows long-path) — tolerate if results/ exists
            results_dir = RESULTS_REPO / "results"
            if results_dir.exists() and any(results_dir.iterdir()):
                log.warning(
                    "git clone exited %d (likely long-path checkout failure on Windows) "
                    "but results/ is populated — continuing",
                    result.returncode,
                )
            else:
                raise RuntimeError(f"git clone failed with exit {result.returncode} and results/ is empty")
    else:
        log.info("Results repo already present at %s", RESULTS_REPO)


def find_model_dir(short_name: str, hf_id: str | None) -> Path | None:
    """Locate a model directory inside the cloned results repo."""
    if not RESULTS_REPO.exists():
        return None
    models_root = RESULTS_REPO / "results"
    if not models_root.exists():
        return None

    candidates = []
    for p in models_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if hf_id and hf_id.lower().split("/")[-1] in name:
            candidates.append(p)
        elif short_name.lower() in name:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # prefer exact HF-id match
        for c in candidates:
            if hf_id and hf_id.lower().replace("/", "__") in c.name.lower():
                return c
        return candidates[0]
    return None


def score_from_json(json_path: Path, task_name: str, bcp47_codes: list[str]) -> float | None:
    """Extract NDCG@10 for a given language from a task result JSON."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Could not read %s: %s", json_path, e)
        return None

    # Navigate: test -> <lang_code> -> ndcg_at_10
    test_block = data.get("scores", {}).get("test", [])
    if not test_block:
        # some files use "test" as a list directly
        test_block = data.get("test", [])

    # test_block may be a list of dicts or a dict keyed by lang
    if isinstance(test_block, list):
        for entry in test_block:
            langs = entry.get("languages", [])
            if any(c in langs for c in bcp47_codes):
                v = entry.get("ndcg_at_10")
                if v is not None:
                    return float(v)
    elif isinstance(test_block, dict):
        for code in bcp47_codes:
            if code in test_block:
                v = test_block[code].get("ndcg_at_10")
                if v is not None:
                    return float(v)
    return None


def resolve_harrier(models_root: Path) -> str | None:
    for p in models_root.iterdir():
        if "harrier" in p.name.lower():
            return p.name
    return None


def _collect_model_scores(
    short_name: str,
    rev_dir: Path,
    all_bcp47: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Parse all relevant task JSONs for one model revision directory.

    Returns (task_lang_scores, global_scores) where:
    - task_lang_scores[task][bcp47_code] = ndcg score  (LANG_TASKS only)
    - global_scores[task] = mean-over-all-langs score  (MMTEB_RETRIEVAL_TASKS only)
    """
    _lang_set = set(LANG_TASKS)
    _mmteb_set = set(MMTEB_RETRIEVAL_TASKS)
    task_lang_scores: dict[str, dict[str, float]] = {}
    global_scores: dict[str, float] = {}

    for task_name in _lang_set | _mmteb_set:
        json_path = rev_dir / f"{task_name}.json"
        if not json_path.exists():
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning("%s/%s: read error %s", short_name, task_name, e)
            continue

        scores_obj = data.get("scores", {})
        test_block = scores_obj.get("test") or scores_obj.get("dev") or \
                     data.get("test") or data.get("dev") or []

        lang_scores_for_task: dict[str, float] = {}
        all_task_scores: list[float] = []

        if isinstance(test_block, list):
            for entry in test_block:
                langs = entry.get("languages", [])
                score = entry.get("ndcg_at_10")
                if score is None:
                    continue
                all_task_scores.append(float(score))
                for lang in langs:
                    if lang in all_bcp47:
                        lang_scores_for_task[lang] = float(score)
        elif isinstance(test_block, dict):
            for lang_code, metrics in test_block.items():
                score = metrics.get("ndcg_at_10") if isinstance(metrics, dict) else None
                if score is None:
                    continue
                all_task_scores.append(float(score))
                if lang_code in all_bcp47:
                    lang_scores_for_task[lang_code] = float(score)

        if task_name in _mmteb_set:
            mmteb_scores: list[float] = []
            for split_block in scores_obj.values():
                if isinstance(split_block, list):
                    for entry in split_block:
                        s = entry.get("ndcg_at_10") if entry.get("ndcg_at_10") is not None \
                            else entry.get("main_score")
                        if s is not None:
                            mmteb_scores.append(float(s))
            if mmteb_scores:
                global_scores[task_name] = float(np.mean(mmteb_scores))

        if task_name in _lang_set and lang_scores_for_task:
            task_lang_scores[task_name] = lang_scores_for_task

    return task_lang_scores, global_scores


# ── Global aggregate ──────────────────────────────────────────────────────────

def build_global_agg(results: dict[str, dict]) -> pd.DataFrame:
    """
    results: {short_name: {task_name: score}}
    Global agg = mean of all retrieval task scores available for that model,
    restricted to TARGET_TASKS to stay consistent with the benchmark scope.
    """
    rows = []
    for short, task_scores in results.items():
        vals = [v for v in task_scores.values() if v is not None]
        if not vals:
            continue
        rows.append({
            "model": short,
            "mteb_agg": float(np.mean(vals)),
            "n_tasks_in_agg": len(vals),
        })
    return pd.DataFrame(rows).sort_values("mteb_agg", ascending=False).reset_index(drop=True)


# ── Per-language extraction ───────────────────────────────────────────────────

def extract_lang_data(
    iso3: str,
    bcp47_primary: str,
    folder: Path,
    all_model_task_scores: dict,
    *,
    suffix: str = "",
) -> None:
    """Build per_task.csv and lang_avg{suffix}.csv for one language."""
    folder.mkdir(parents=True, exist_ok=True)
    bcp47_alts = LANG_ALT.get(iso3, [bcp47_primary])
    if bcp47_primary not in bcp47_alts:
        bcp47_alts = [bcp47_primary] + bcp47_alts

    # Collect scores: {short_name: {task: score}}
    lang_scores: dict[str, dict[str, float]] = {}
    tasks_seen: set[str] = set()

    for short_name, task_dict in all_model_task_scores.items():
        row = {}
        for task, lang_map_inner in task_dict.items():
            score = None
            for code in bcp47_alts:
                score = lang_map_inner.get(code)
                if score is not None:
                    break
            if score is not None:
                row[task] = score
                tasks_seen.add(task)
        lang_scores[short_name] = row

    if not tasks_seen:
        log.info("  %s: no tasks found — skipped", iso3)
        return

    # Build per_task.csv (all tasks seen, all models)
    all_task_cols = sorted(tasks_seen)
    rows = []
    for short_name, row in lang_scores.items():
        entry = {"model": short_name}
        for t in all_task_cols:
            entry[f"{t}_ndcg"] = row.get(t)
        entry["n_tasks"] = sum(1 for t in all_task_cols if row.get(t) is not None)
        rows.append(entry)

    per_task_df = pd.DataFrame(rows)
    per_task_df.to_csv(folder / "per_task.csv", index=False, encoding="utf-8")

    # Consensus task set: tasks where ≥ 50% of models with any data have results.
    # 50% threshold keeps only tasks that most of the restricted roster covers.
    n_models_any = (per_task_df["n_tasks"] > 0).sum()
    min_models_for_task = max(1, int(np.ceil(0.50 * n_models_any)))
    consensus_tasks = [
        t for t in all_task_cols
        if per_task_df[f"{t}_ndcg"].notna().sum() >= min_models_for_task
    ]

    if not consensus_tasks:
        consensus_tasks = all_task_cols  # fallback: use all tasks

    log.info("  %s: consensus tasks (≥%d/%d models): %s",
             iso3, min_models_for_task, n_models_any, consensus_tasks)

    # Build lang_avg.csv: models with full coverage on consensus task set
    full_coverage = [
        short_name for short_name, row in lang_scores.items()
        if all(row.get(t) is not None for t in consensus_tasks)
    ]
    dropped = [m for m in lang_scores if m not in full_coverage]
    if dropped:
        log.info("  %s: dropped models (missing consensus tasks): %s", iso3, dropped)

    lang_avg_rows = []
    for short_name in full_coverage:
        row_data = lang_scores[short_name]
        vals = [row_data[t] for t in consensus_tasks]
        lang_avg_rows.append({
            "model": short_name,
            "lang_avg": float(np.mean(vals)),
            "n_tasks": len(vals),
        })

    lang_avg_df = pd.DataFrame(lang_avg_rows)
    lang_avg_df.to_csv(folder / f"lang_avg{suffix}.csv", index=False, encoding="utf-8")

    n_models = len(full_coverage)
    n_tasks  = len(consensus_tasks)
    log.info("  %s: n_models=%d, n_tasks=%d, kept=%s", iso3, n_models, n_tasks,
             "YES" if n_models >= 7 and n_tasks >= 2 else "NO (insufficient coverage)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_results_repo()

    models_root = RESULTS_REPO / "results"
    if not models_root.exists():
        log.error("results/ not found inside repo — check clone")
        sys.exit(1)

    # all_model_task_scores: {short_name: {task_name: {bcp47: score}}}
    all_model_task_scores: dict[str, dict[str, dict[str, float]]] = {}
    # global_task_scores: {short_name: {task_name: score}}  (any lang, for global agg)
    global_task_scores: dict[str, dict[str, float]] = {}

    all_bcp47 = set()
    for v in LANG_MAP.values():
        all_bcp47.add(v["bcp47"])
    for alts in LANG_ALT.values():
        all_bcp47.update(alts)

    for short_name, hf_id in MODELS.items():
        if hf_id is None:
            log.warning("Skipping %s — HF ID not resolved", short_name)
            continue

        model_dir = find_model_dir(short_name, hf_id)
        if model_dir is None:
            log.warning("Model dir not found for %s (%s)", short_name, hf_id)
            continue

        # Pick latest revision subdir (or use model_dir directly if JSONs are there)
        rev_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if rev_dirs:
            rev_dir = sorted(rev_dirs)[-1]
        else:
            rev_dir = model_dir

        task_lang_scores, g_scores = _collect_model_scores(short_name, rev_dir, all_bcp47)
        all_model_task_scores[short_name] = task_lang_scores
        global_task_scores[short_name] = g_scores
        log.info("Loaded %s: %d lang tasks, %d MMTEB tasks", short_name, len(task_lang_scores), len(g_scores))

    # Global aggregate
    global_df = build_global_agg(global_task_scores)
    global_out = BASE / "global" / "mteb_agg.csv"
    global_df.to_csv(global_out, index=False, encoding="utf-8")
    log.info("Saved global/mteb_agg.csv (%d models)", len(global_df))

    for iso3, meta in LANG_MAP.items():
        log.info("Processing language: %s (%s)", iso3, meta["name"])
        extract_lang_data(
            iso3=iso3,
            bcp47_primary=meta["bcp47"],
            folder=meta["folder"],
            all_model_task_scores=all_model_task_scores,
            suffix="",
        )

    log.info("Phase 2 (restricted roster) complete.")

    # ── Phase 2b: Extended roster ─────────────────────────────────────────────
    # Discover models with all 18 official MMTEB retrieval tasks, giving a
    # global predictor that matches the public leaderboard Mean(Task) score.
    EXCLUDE_DIRS = {"mteb__baseline-bm25s", "mteb__baseline-random-encoder"}

    log.info("Scanning for extended roster (models with all 18 MMTEB retrieval tasks)...")
    ext_model_task_scores: dict[str, dict[str, dict[str, float]]] = {}
    ext_global_scores: dict[str, dict[str, float]] = {}

    for mdir in sorted(models_root.iterdir()):
        if not mdir.is_dir() or mdir.name in EXCLUDE_DIRS:
            continue
        rev_dirs = [p for p in mdir.iterdir() if p.is_dir()]
        rev_dir = sorted(rev_dirs)[-1] if rev_dirs else mdir
        has_all = all((rev_dir / f"{t}.json").exists() for t in MMTEB_RETRIEVAL_TASKS)
        if not has_all:
            continue

        short = mdir.name
        task_lang, g_scores = _collect_model_scores(short, rev_dir, all_bcp47)

        if task_lang:
            ext_model_task_scores[short] = task_lang
            ext_global_scores[short] = g_scores

    log.info("Extended roster: %d models with multilingual data", len(ext_model_task_scores))

    # Global agg for extended roster
    ext_global_df = build_global_agg(ext_global_scores)
    ext_global_df.to_csv(BASE / "global" / "mteb_agg_extended.csv", index=False, encoding="utf-8")
    log.info("Saved global/mteb_agg_extended.csv (%d models)", len(ext_global_df))

    for iso3, meta in LANG_MAP.items():
        extract_lang_data(
            iso3=iso3,
            bcp47_primary=meta["bcp47"],
            folder=meta["folder"],
            all_model_task_scores=ext_model_task_scores,
            suffix="_extended",
        )

    log.info("Phase 2b (extended roster) complete.")

    # ── Phase 2c: Curated mid-tier roster ─────────────────────────────────────
    log.info("Phase 2c: curated mid-tier roster (%d model entries)...", len(MODELS_CURATED))
    cur_model_task_scores: dict[str, dict[str, dict[str, float]]] = {}
    cur_global_scores: dict[str, dict[str, float]] = {}

    for short_name, hf_id in MODELS_CURATED.items():
        if hf_id is None:
            continue
        model_dir = find_model_dir(short_name, hf_id)
        if model_dir is None:
            log.warning("Curated: model dir not found for %s (%s)", short_name, hf_id)
            continue
        rev_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        rev_dir = sorted(rev_dirs)[-1] if rev_dirs else model_dir

        task_lang_scores, g_scores = _collect_model_scores(short_name, rev_dir, all_bcp47)
        cur_model_task_scores[short_name] = task_lang_scores
        cur_global_scores[short_name] = g_scores

    cur_global_df = build_global_agg(cur_global_scores)
    cur_global_df.to_csv(BASE / "global" / "mteb_agg_curated.csv", index=False, encoding="utf-8")
    log.info("Saved global/mteb_agg_curated.csv (%d models)", len(cur_global_df))

    for iso3, meta in LANG_MAP.items():
        log.info("Processing curated language: %s (%s)", iso3, meta["name"])
        extract_lang_data(
            iso3=iso3,
            bcp47_primary=meta["bcp47"],
            folder=meta["folder"],
            all_model_task_scores=cur_model_task_scores,
            suffix="_curated",
        )

    log.info("Phase 2c (curated mid-tier) complete.")


if __name__ == "__main__":
    main()
