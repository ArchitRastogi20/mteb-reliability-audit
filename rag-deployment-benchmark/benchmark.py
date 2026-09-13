#!/usr/bin/env python3
"""
Deployment benchmark for embedding models on the RAG corpus.

Measures corpus encoding throughput, query throughput, single-query latency
(p50/p99), and peak VRAM usage for corpus vs. query batches separately.
Runs each model N_RUNS times and reports mean ± std. Merges with RAG NDCG
scores to compute per-language efficiency ratios.

Usage:
    python benchmark.py --all
    python benchmark.py --all --resume
    python benchmark.py --model BAAI/bge-m3
    python benchmark.py --all --dry-run
    python benchmark.py --all --skip-model Qwen/Qwen3-Embedding-8B
    python benchmark.py --all --batch-size 64
    python benchmark.py --all --runs 5
    python benchmark.py --all --dataset /path/to/rag/output
    python benchmark.py --all --ndcg-file /path/to/summary.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

import hardware as hw_module

# ── Compatibility patches ───────────────────────────────────────────────────────

def _patch_transformers_tied_weights() -> None:
    """
    Newer transformers calls self.all_tied_weights_keys in
    _move_missing_keys_from_meta_to_device without a getattr fallback.
    Custom models that skip post_init() (e.g. jinaai/jina-embeddings-v3's
    XLMRobertaLoRA) never populate this instance attribute, causing an
    AttributeError routed through torch.nn.Module.__getattr__.
    Add the missing attribute before delegating to the original method.
    """
    try:
        import transformers.modeling_utils as _mu
        _orig = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device

        def _patched(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return _orig(self, *args, **kwargs)

        _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched
    except Exception:
        pass


_patch_transformers_tied_weights()

# ── Constants ──────────────────────────────────────────────────────────────────

BATCH_SIZE  = 32    # fixed for fair cross-model comparison
WARMUP_RUNS = 100
TIMED_RUNS  = 500
N_RUNS      = 3

REPO_ROOT   = Path(__file__).parent
CONFIG_FILE = REPO_ROOT / "config" / "models.yaml"
RESULTS_DIR = REPO_ROOT / "results"
STATE_FILE  = RESULTS_DIR / "benchmark_state.json"
RESULTS_CSV = RESULTS_DIR / "deployment_results.csv"
MERGED_CSV  = RESULTS_DIR / "deployment_results_merged.csv"

RAG_DATASET_DEFAULT = Path("/workspace/rag-dataset/output")
NDCG_FILE_DEFAULT   = RAG_DATASET_DEFAULT / "evaluation" / "summary.json"

# Evaluation combos — must match the NDCG evaluation
COMBOS = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]

logger = logging.getLogger("benchmark")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rag_texts(rag_dir: Path) -> tuple[list[str], list[str]]:
    """Return (corpus_docs, queries) from all benchmark combos."""
    docs: list[str] = []
    queries: list[str] = []

    for lang, domain in COMBOS:
        corpus_path = rag_dir / lang / domain / "corpus.jsonl"
        query_path  = rag_dir / lang / domain / "queries.jsonl"

        if not corpus_path.exists():
            logger.warning("Missing corpus: %s — skipping", corpus_path)
            continue
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line).get("text", ""))

        if not query_path.exists():
            logger.warning("Missing queries: %s — skipping", query_path)
            continue
        with query_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line).get("text", ""))

    return docs, queries


# ── Model helpers ──────────────────────────────────────────────────────────────

def _get_local_path(model_id: str, cache_dir: Path) -> Path:
    try:
        local = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        return Path(local)
    except Exception:
        raise FileNotFoundError(
            f"Model '{model_id}' not found in cache '{cache_dir}'.\n"
            f"Run: python download_models.py --model {model_id}"
        )


def _apply_jina_patch() -> None:
    # Workaround for JinaEmbeddingsV5Model missing config_class on repeated loads
    for mod in list(sys.modules.values()):
        jina_cls = getattr(mod, "__dict__", {}).get("JinaEmbeddingsV5Model")
        if jina_cls is not None and not hasattr(jina_cls, "config_class"):
            try:
                jina_cls.config_class = None
            except Exception:
                pass


def _load_model(local_path: Path, model_config: dict) -> SentenceTransformer:
    _apply_jina_patch()
    hf_kwargs = model_config.get("model_kwargs") or {}
    model = SentenceTransformer(
        str(local_path),
        device="cuda",
        trust_remote_code=True,
        **({"model_kwargs": hf_kwargs} if hf_kwargs else {}),
    )
    model.eval()
    return model


def _encode(
    model: SentenceTransformer,
    texts: list[str],
    prefix: str,
    batch_size: int,
    encode_kwargs: dict | None = None,
) -> np.ndarray:
    prefixed = [prefix + t for t in texts] if prefix else texts
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        **(encode_kwargs or {}),
    )


# ── Trial ──────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    vram_corpus_mb: float
    vram_query_mb: float
    corpus_encode_sec: float
    query_throughput_qps: float
    latency_p50_ms: float
    latency_p99_ms: float
    embedding_dim: int


def _run_trial(
    model: SentenceTransformer,
    docs: list[str],
    queries: list[str],
    model_config: dict,
    batch_size: int,
) -> TrialResult:
    p_prefix        = model_config.get("passage_prefix", "")
    q_prefix        = model_config.get("query_prefix", "")
    q_encode_kwargs = model_config.get("query_encode_kwargs") or {}
    warmup_runs     = model_config.get("warmup_runs", WARMUP_RUNS)
    timed_runs      = model_config.get("timed_runs",  TIMED_RUNS)
    # Corpus encoding intentionally uses no encode_kwargs — passage is always plain.

    # Corpus encoding — reset inside trial so mean VRAM is meaningful
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _encode(model, docs, p_prefix, batch_size)
    torch.cuda.synchronize()
    corpus_sec = time.perf_counter() - t0
    vram_corpus_mb = torch.cuda.max_memory_allocated() / 1e6

    # Query throughput — separate reset to isolate query VRAM from corpus VRAM
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _encode(model, queries, q_prefix, batch_size, q_encode_kwargs)
    torch.cuda.synchronize()
    query_sec = time.perf_counter() - t0
    vram_query_mb = torch.cuda.max_memory_allocated() / 1e6
    qps = len(queries) / query_sec

    # Single-query latency
    single = [queries[0]]
    for _ in range(warmup_runs):
        _encode(model, single, q_prefix, 1, q_encode_kwargs)

    latencies_ms: list[float] = []
    for _ in range(timed_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _encode(model, single, q_prefix, 1, q_encode_kwargs)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return TrialResult(
        vram_corpus_mb=round(vram_corpus_mb, 1),
        vram_query_mb=round(vram_query_mb, 1),
        corpus_encode_sec=round(corpus_sec, 3),
        query_throughput_qps=round(qps, 1),
        latency_p50_ms=round(float(np.percentile(latencies_ms, 50)), 3),
        latency_p99_ms=round(float(np.percentile(latencies_ms, 99)), 3),
        embedding_dim=model.get_sentence_embedding_dimension() or 0,
    )


# ── Benchmark one model ────────────────────────────────────────────────────────

def benchmark_model(
    model_id: str,
    model_config: dict,
    cache_dir: Path,
    docs: list[str],
    queries: list[str],
    batch_size: int,
    n_runs: int,
) -> dict:
    local_path = _get_local_path(model_id, cache_dir)
    model = _load_model(local_path, model_config)

    trials: list[TrialResult] = []
    for run_idx in range(n_runs):
        logger.info("  run %d/%d …", run_idx + 1, n_runs)
        t = _run_trial(model, docs, queries, model_config, batch_size)
        trials.append(t)
        logger.info(
            "    corpus=%.2fs  qps=%.0f  p50=%.1fms  "
            "vram_corpus=%.0fMB  vram_query=%.0fMB",
            t.corpus_encode_sec, t.query_throughput_qps, t.latency_p50_ms,
            t.vram_corpus_mb, t.vram_query_mb,
        )

    def _agg(vals: list[float]) -> tuple[float, float]:
        return round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)

    m_vc, s_vc = _agg([t.vram_corpus_mb for t in trials])
    m_vq, s_vq = _agg([t.vram_query_mb for t in trials])
    m_cs, s_cs = _agg([t.corpus_encode_sec for t in trials])
    m_qp, s_qp = _agg([t.query_throughput_qps for t in trials])
    m_p5, s_p5 = _agg([t.latency_p50_ms for t in trials])
    m_p9, s_p9 = _agg([t.latency_p99_ms for t in trials])

    return {
        "model_id":                   model_id,
        "params":                     model_config["params"],
        "dim":                        model_config["dim"],
        "embedding_dim":              trials[0].embedding_dim,
        "batch_size":                 batch_size,
        "n_runs":                     n_runs,
        "n_docs":                     len(docs),
        "n_queries":                  len(queries),
        "vram_corpus_mb_mean":        m_vc,
        "vram_corpus_mb_std":         s_vc,
        "vram_query_mb_mean":         m_vq,
        "vram_query_mb_std":          s_vq,
        "corpus_encode_sec_mean":     m_cs,
        "corpus_encode_sec_std":      s_cs,
        "query_throughput_qps_mean":  m_qp,
        "query_throughput_qps_std":   s_qp,
        "latency_p50_ms_mean":        m_p5,
        "latency_p50_ms_std":         s_p5,
        "latency_p99_ms_mean":        m_p9,
        "latency_p99_ms_std":         s_p9,
    }


# ── CSV output ─────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "model_id", "params", "dim", "embedding_dim", "batch_size", "n_runs",
    "n_docs", "n_queries",
    "vram_corpus_mb_mean", "vram_corpus_mb_std",
    "vram_query_mb_mean",  "vram_query_mb_std",
    "corpus_encode_sec_mean", "corpus_encode_sec_std",
    "query_throughput_qps_mean", "query_throughput_qps_std",
    "latency_p50_ms_mean", "latency_p50_ms_std",
    "latency_p99_ms_mean", "latency_p99_ms_std",
]


def _hw_comments(hw: hw_module.HardwareInfo) -> list[str]:
    return [
        f"# GPU:      {hw.gpu_name}",
        f"# VRAM:     {hw.vram_gb:.1f} GB",
        f"# CUDA:     {hw.cuda_version}",
        f"# Torch:    {hw.torch_version}",
        f"# Platform: {hw.platform}",
    ]


def save_results(results: list[dict], path: Path, hw: hw_module.HardwareInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        for line in _hw_comments(hw):
            f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved %d rows → %s", len(results), path)


# ── NDCG merge & efficiency ────────────────────────────────────────────────────

def _parse_params(s: str) -> float:
    """'107M' → 107e6, '4B' → 4e9."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    return float(s)


def merge_with_ndcg(
    results: list[dict],
    ndcg_file: Path,
    merged_path: Path,
    hw: hw_module.HardwareInfo,
) -> None:
    if not ndcg_file.exists():
        logger.warning("NDCG file not found: %s — skipping merge.", ndcg_file)
        return

    # summary.json: {model_id: {combo_key: {"overall": {"ndcg_at_10": ...}}}}
    summary: dict = json.loads(ndcg_file.read_text())

    def _avg_ndcg(model_id: str, combo_keys: list[str]) -> float | None:
        model_data = summary.get(model_id)
        if not model_data:
            return None
        vals = [
            model_data[k]["overall"]["ndcg_at_10"]
            for k in combo_keys
            if k in model_data
        ]
        return round(float(np.mean(vals)), 4) if vals else None

    merged_cols = _CSV_COLS + [
        "ndcg_hi", "ndcg_ja",
        "efficiency_hi", "efficiency_ja",
    ]

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        for line in _hw_comments(hw):
            f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=merged_cols, extrasaction="ignore")
        writer.writeheader()

        for row in results:
            mid        = row["model_id"]
            params_n   = _parse_params(str(row["params"]))
            log_params = math.log10(params_n)

            ndcg_hi = _avg_ndcg(mid, ["hi_finance", "hi_law"])
            ndcg_ja = _avg_ndcg(mid, ["ja_finance", "ja_law"])

            merged_row = dict(row)
            merged_row["ndcg_hi"] = ndcg_hi
            merged_row["ndcg_ja"] = ndcg_ja
            merged_row["efficiency_hi"] = (
                round(ndcg_hi / log_params, 5) if ndcg_hi is not None else None
            )
            merged_row["efficiency_ja"] = (
                round(ndcg_ja / log_params, 5) if ndcg_ja is not None else None
            )
            writer.writerow(merged_row)

    logger.info("Merged results → %s", merged_path)


# ── Pretty table ───────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    if not results:
        return
    cols = [
        ("Model",           "model_id",                   44),
        ("Params",          "params",                      7),
        ("VRAM_C(MB)",      "vram_corpus_mb_mean",         11),
        ("VRAM_Q(MB)",      "vram_query_mb_mean",          11),
        ("Corpus(s)",       "corpus_encode_sec_mean",      10),
        ("Q/s",             "query_throughput_qps_mean",    8),
        ("p50(ms)",         "latency_p50_ms_mean",          9),
        ("p99(ms)",         "latency_p99_ms_mean",          9),
    ]
    header = "  ".join(h.ljust(w) for h, _, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print("\n" + header)
    print(sep)
    for r in results:
        row = "  ".join(str(r.get(k, "—")).ljust(w) for _, k, w in cols)
        print(row)
    print()


# ── State ──────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": [], "failed": [], "in_progress": None}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _sanitize(model_id: str) -> str:
    name = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]", "_", name)[:40]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment benchmark for embedding models"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",   action="store_true", help="Benchmark all models in config")
    group.add_argument("--model", metavar="HF_ID",     help="Benchmark a single model")
    parser.add_argument("--resume",     action="store_true", help="Skip completed models")
    parser.add_argument("--dry-run",    action="store_true", help="Print plan and data counts, no GPU work")
    parser.add_argument("--skip-model", metavar="HF_ID",     help="Exclude one model by ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--runs",       type=int, default=N_RUNS)
    parser.add_argument("--dataset",    type=Path, default=RAG_DATASET_DEFAULT)
    parser.add_argument("--ndcg-file",  type=Path, default=NDCG_FILE_DEFAULT)
    args = parser.parse_args()

    _setup_logging()

    cfg_data  = yaml.safe_load(CONFIG_FILE.read_text())
    model_map = {m["id"]: m for m in cfg_data["models"]}
    cache_dir = Path(cfg_data.get("model_cache", "/workspace/model_cache"))

    if args.model:
        if args.model not in model_map:
            logger.error("Model %r not in config. Available: %s", args.model, list(model_map))
            sys.exit(1)
        model_ids = [args.model]
    else:
        model_ids = [m["id"] for m in cfg_data["models"]]

    if args.skip_model:
        model_ids = [m for m in model_ids if m != args.skip_model]

    state = _load_state()
    if args.resume:
        before    = len(model_ids)
        model_ids = [m for m in model_ids if m not in state["completed"]]
        logger.info("[Resume] Skipped %d completed. %d remaining.",
                    before - len(model_ids), len(model_ids))

    # Always load data — dry-run prints counts, live run uses the texts
    logger.info("Loading RAG texts from %s …", args.dataset)
    docs, queries = load_rag_texts(args.dataset)
    logger.info("Loaded %d corpus docs, %d queries across %d combos.",
                len(docs), len(queries), len(COMBOS))

    if args.dry_run:
        print(f"\nDry run — {len(model_ids)} model(s) queued")
        print(f"Dataset : {args.dataset}")
        print(f"Combos  : {COMBOS}")
        print(f"Docs    : {len(docs)}")
        print(f"Queries : {len(queries)}")
        print(f"Batch   : {args.batch_size}   Runs: {args.runs}")
        print()
        for mid in model_ids:
            mc = model_map[mid]
            print(f"  {mid:60s}  {mc['params']:6s}  tier={mc['tier']}")
        return

    hw = hw_module.detect()
    hw_module.print_summary(hw)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Reload previously completed rows so the final CSV is complete on --resume
    all_results: list[dict] = []
    if args.resume and RESULTS_CSV.exists():
        with RESULTS_CSV.open(encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        for row in csv.DictReader(iter(lines)):
            if row.get("model_id") in state["completed"]:
                all_results.append(row)

    for model_id in model_ids:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", model_id)
        logger.info("=" * 60)

        state["in_progress"] = model_id
        _save_state(state)

        batch_size = model_map[model_id].get("batch_size", args.batch_size)
        result     = None
        _MAX_OOM   = 4

        for attempt in range(_MAX_OOM):
            _oom = False
            try:
                result = benchmark_model(
                    model_id, model_map[model_id], cache_dir,
                    docs, queries, batch_size, args.runs,
                )
                break
            except torch.cuda.OutOfMemoryError:
                _oom = True
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                break
            except Exception as exc:
                logger.error("[FAIL] %s: %s", model_id, exc, exc_info=True)
                break

            # Clear VRAM outside the except block so the traceback is dropped first
            if _oom:
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                if attempt < _MAX_OOM - 1:
                    logger.warning(
                        "[OOM] Retrying %s at batch_size=%d (attempt %d/%d)",
                        model_id, batch_size, attempt + 2, _MAX_OOM,
                    )
                else:
                    logger.error("[FAIL] %s OOM on all %d attempts.", model_id, _MAX_OOM)

        gc.collect()
        torch.cuda.empty_cache()

        if result is not None:
            all_results.append(result)
            json_path = RESULTS_DIR / f"{_sanitize(model_id)}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(result, indent=2))
            state["completed"].append(model_id)
            logger.info(
                "[OK] %s  qps=%.0f  p50=%.1fms  vram_corpus=%.0fMB  vram_query=%.0fMB",
                model_id,
                result["query_throughput_qps_mean"],
                result["latency_p50_ms_mean"],
                result["vram_corpus_mb_mean"],
                result["vram_query_mb_mean"],
            )
        else:
            if not any(f["model_id"] == model_id for f in state["failed"]):
                state["failed"].append({"model_id": model_id})

        state["in_progress"] = None
        _save_state(state)

    if all_results:
        save_results(all_results, RESULTS_CSV, hw)
        print_table(all_results)
        merge_with_ndcg(all_results, args.ndcg_file, MERGED_CSV, hw)

    logger.info(
        "Done.  Completed: %d   Failed: %d",
        len(state["completed"]), len(state["failed"]),
    )
    if state["failed"]:
        logger.warning("Failed models: %s", [f["model_id"] for f in state["failed"]])


if __name__ == "__main__":
    main()
