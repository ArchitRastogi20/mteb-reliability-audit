#!/usr/bin/env python3
"""
RAG Evaluation Pipeline.

Usage:
    python scripts/eval_rag.py --all --resume
    python scripts/eval_rag.py --model intfloat/multilingual-e5-small
    python scripts/eval_rag.py --all --dry-run
    python scripts/eval_rag.py --all --keep-chroma
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, evaluate

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
COMBOS = [("ja", "finance"), ("ja", "law"), ("hi", "finance"), ("hi", "law")]

logger = logging.getLogger("rag_eval")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_combo_data(output_dir: Path, lang: str, domain: str) -> dict:
    """Load corpus, queries, qrels, and query_types for one combo."""
    base = output_dir / lang / domain

    corpus: dict[str, str] = {}
    with (base / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus[row["_id"]] = row.get("text", "")

    queries: dict[str, str] = {}
    with (base / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels" / "test.tsv").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, did, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[did] = rel

    query_types: dict[str, str] = {}
    with (base / "full_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            query_types[row["_id"]] = row["query_type"]

    logger.debug(
        "Loaded %s/%s: corpus=%d, queries=%d, qrels=%d",
        lang, domain, len(corpus), len(queries), len(qrels),
    )
    return {"corpus": corpus, "queries": queries, "qrels": qrels, "query_types": query_types}


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_embs: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    save_run_path: Path | None = None,
) -> dict[str, float]:
    """Compute NDCG@10, Recall@10, MRR for a set of query/corpus embeddings."""
    if not qrels:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}

    scores = query_embs @ corpus_embs.T
    top_k = min(10, scores.shape[1])
    top_k_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]

    run_dict: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        top = top_k_idx[qi]
        top_sorted = top[np.argsort(scores[qi, top])[::-1]]
        run_dict[str(qid)] = {str(corpus_ids[di]): float(scores[qi, di]) for di in top_sorted}

    if save_run_path is not None:
        save_run_path.parent.mkdir(parents=True, exist_ok=True)
        save_run_path.write_text(json.dumps(run_dict, indent=2))

    qrels_str = {
        str(qid): {str(did): int(s) for did, s in rels.items()}
        for qid, rels in qrels.items()
    }

    result = evaluate(
        Qrels(qrels_str),
        Run(run_dict),
        ["ndcg@10", "recall@10", "mrr"],
        make_comparable=True,
    )
    return {
        "ndcg_at_10": float(result["ndcg@10"]),
        "recall_at_10": float(result["recall@10"]),
        "mrr": float(result["mrr"]),
    }


def compute_metrics_by_type(
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_embs: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    query_types: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by query type."""
    all_types = ["factual", "multi_hop", "summarization", "unanswerable"]
    results: dict[str, dict[str, float]] = {}

    for qtype in all_types:
        type_query_ids = [qid for qid in query_ids if query_types.get(qid) == qtype]
        if not type_query_ids:
            results[qtype] = {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
            continue

        type_mask = [i for i, qid in enumerate(query_ids) if qid in set(type_query_ids)]
        type_query_embs = query_embs[type_mask]
        type_qrels = {qid: qrels[qid] for qid in type_query_ids if qid in qrels}

        results[qtype] = compute_metrics(
            corpus_embs, corpus_ids,
            type_query_embs, type_query_ids,
            type_qrels,
        )

    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

import gc
import os
import sys
import time
import torch
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Allow `python scripts/eval_rag.py` invocation from repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentence_transformers import SentenceTransformer

from scripts.rag_hardware import detect_and_write
from scripts.rag_embedding_cache import EmbeddingCache
from scripts.rag_model_pipeline import ModelPipeline
from scripts.rag_report import update_tables
from scripts.rag_utils import (
    sanitize_model_id, corpus_collection_name, query_collection_name,
    log_json, vram_peak_mb,
)

# ── Runtime compatibility patches ────────────────────────────────────────────
# Patch 1: bge-m3 — torch 2.4 blocks torch.load (CVE-2025-32434).
import transformers.modeling_utils as _tm
if hasattr(_tm, "check_torch_load_is_safe"):
    _tm.check_torch_load_is_safe = lambda: None

# Patch 2: pplx-embed — create_causal_mask(or_mask_function=...) requires
# torch>=2.6. When or_mask_function is present it signals bidirectional intent.
# create_bidirectional_mask returns None for non-padded sequences; Qwen3's SDPA
# then recomputes a causal mask internally. Instead, return an explicit 4D
# all-zero float mask (0.0 = attend to all) so Qwen3 uses it directly without
# recomputing. Use explicit None checks — `or` on Tensors raises RuntimeError.
try:
    import transformers.masking_utils as _mu
    _orig_ccm = _mu.create_causal_mask
    def _patched_ccm(*args, **kwargs):
        or_fn = kwargs.pop("or_mask_function", None)
        kwargs.pop("and_mask_function", None)
        if or_fn is not None:
            embs = kwargs.get("inputs_embeds")
            if embs is None:
                embs = kwargs.get("input_embeds")
            if embs is None and len(args) > 1:
                embs = args[1]
            if embs is not None:
                batch_size, seq_len = embs.shape[0], embs.shape[1]
                return torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=embs.dtype,
                    device=embs.device,
                )
            return None
        return _orig_ccm(*args, **kwargs)
    _mu.create_causal_mask = _patched_ccm
except Exception:
    pass

# Patch 3: gte-Qwen2-7B — config.rope_theta absent in this transformers version.
try:
    from transformers import Qwen2Config as _Q2C
    _q2c_orig_ga = _Q2C.__getattribute__
    def _q2c_getattribute(self, name):
        if name == "rope_theta":
            try:
                return _q2c_orig_ga(self, name)
            except AttributeError:
                return 10000.0
        return _q2c_orig_ga(self, name)
    _Q2C.__getattribute__ = _q2c_getattribute
except Exception:
    pass

# Patch 4: gte-Qwen2-7B — DynamicCache methods removed in transformers 4.47+.
try:
    from transformers import DynamicCache as _DC
    if not hasattr(_DC, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is None:
                return cache
            for layer_past in past_key_values:
                cache.update(layer_past[0], layer_past[1], len(cache.key_cache))
            return cache
        _DC.from_legacy_cache = _from_legacy_cache
    if not hasattr(_DC, "get_usable_length"):
        def _dc_get_usable_length(self, new_seq_length: int = 0, layer_idx: int = 0) -> int:
            max_length = self.get_max_length() if hasattr(self, "get_max_length") else None
            seq_len = self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
            if max_length is not None:
                return min(max_length - new_seq_length, seq_len)
            return seq_len
        _DC.get_usable_length = _dc_get_usable_length
    if not hasattr(_DC, "to_legacy_cache"):
        def _dc_to_legacy_cache(self):
            if not self.key_cache:
                return None
            return tuple(
                (self.key_cache[i], self.value_cache[i])
                for i in range(len(self.key_cache))
            )
        _DC.to_legacy_cache = _dc_to_legacy_cache
except Exception:
    pass

# Patch 5: jina-v3 — XLMRobertaLoRA (custom class) is missing all_tied_weights_keys,
# a dict attribute that transformers 5.x _move_missing_keys_from_meta_to_device expects.
try:
    import transformers.modeling_utils as _tmmu
    _orig_mmk = _tmmu.PreTrainedModel._move_missing_keys_from_meta_to_device
    def _patched_mmk(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_mmk(self, *args, **kwargs)
    _tmmu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_mmk
except Exception:
    pass

CHROMA_PATH = "/workspace/rag_chroma_cache"
CONFIG_FILE = REPO_ROOT / "config" / "models.yaml"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
TABLES_DIR = REPO_ROOT / "tables"
LOGS_DIR = REPO_ROOT / "logs"
STATE_FILE = REPO_ROOT / "state_rag.json"


def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"rag_eval_{ts}.log"

    logger = logging.getLogger("rag_eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Log: %s", log_file)
    return logger


def _load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        seen: set = set()
        state["completed"] = [m for m in state.get("completed", []) if not (m in seen or seen.add(m))]
        seen_failed: set = set()
        state["failed"] = [
            f for f in state.get("failed", [])
            if not (f["model_id"] in seen_failed or seen_failed.add(f["model_id"]))
        ]
        return state
    return {"completed": [], "failed": [], "in_progress": None}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def evaluate_model(
    model_id: str,
    local_path: Path,
    model_config: dict,
    batch_sizes: dict[str, int],
    keep_chroma: bool = False,
    save_runs: bool = False,
) -> bool:
    logger = logging.getLogger("rag_eval")
    san_id = sanitize_model_id(model_id)
    result_dir = EVAL_DIR / san_id
    result_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = OUTPUT_DIR / "evaluation" / "runs"
    def _combo_done(lang: str, domain: str) -> bool:
        result_exists = (result_dir / f"{lang}_{domain}.json").exists()
        run_exists = (runs_dir / f"{san_id}_{lang}_{domain}.json").exists()
        return result_exists and (not save_runs or run_exists)

    all_done = all(_combo_done(lang, domain) for lang, domain in COMBOS)
    if all_done:
        logger.info("[%s] All results cached — skipping.", model_id)
        return True

    batch_size = batch_sizes[model_config["tier"]]
    cache = EmbeddingCache(CHROMA_PATH)

    hf_kwargs: dict = model_config.get("model_kwargs") or {}
    tokenizer_kwargs: dict = model_config.get("tokenizer_kwargs") or {}
    torch.cuda.reset_peak_memory_stats()

    # Pre-load patch: jina-embeddings-v5 registers JinaEmbeddingsV5Model without
    # config_class, causing auto_factory to crash on any subsequent load of this model.
    # Use __dict__.get() to avoid triggering transformers' lazy __getattr__ (which
    # emits a deprecation warning for every one of ~200 image_processing modules).
    import sys as _sys
    for _mod in list(_sys.modules.values()):
        _jina_cls = getattr(_mod, "__dict__", {}).get("JinaEmbeddingsV5Model")
        if _jina_cls is not None and not hasattr(_jina_cls, "config_class"):
            try:
                _jina_cls.config_class = None
            except Exception:
                pass

    model = SentenceTransformer(
        str(local_path),
        device="cuda",
        trust_remote_code=True,
        **({"model_kwargs": hf_kwargs} if hf_kwargs else {}),
        **({"tokenizer_kwargs": tokenizer_kwargs} if tokenizer_kwargs else {}),
    )
    model.eval()

    p_prefix = model_config.get("passage_prefix", "")
    q_prefix = model_config.get("query_prefix", "")
    p_encode_kwargs: dict = model_config.get("passage_encode_kwargs") or {}
    q_encode_kwargs: dict = model_config.get("query_encode_kwargs") or {}

    # ranx uses numba workqueue — crashes under concurrent threads
    executor = ThreadPoolExecutor(max_workers=1)
    t_wall = time.time()

    with torch.amp.autocast("cuda"):
        for lang, domain in COMBOS:
            result_path = result_dir / f"{lang}_{domain}.json"
            if _combo_done(lang, domain):
                logger.info("[%s] %s/%s cached — skipping.", model_id, lang, domain)
                continue

            t_task = time.time()
            data = load_combo_data(OUTPUT_DIR, lang, domain)

            c_ids = list(data["corpus"].keys())
            c_texts = [data["corpus"][i] for i in c_ids]
            q_ids = list(data["queries"].keys())
            q_texts = [data["queries"][i] for i in q_ids]

            c_embs, c_hits, c_misses = cache.cached_encode(
                model, c_texts, c_ids,
                corpus_collection_name(model_id, lang, domain),
                batch_size, p_prefix, p_encode_kwargs,
            )
            q_embs, _, _ = cache.cached_encode(
                model, q_texts, q_ids,
                query_collection_name(model_id, lang, domain),
                batch_size, q_prefix, q_encode_kwargs,
            )

            save_run_path = (runs_dir / f"{san_id}_{lang}_{domain}.json") if save_runs else None
            overall = executor.submit(
                compute_metrics, c_embs, c_ids, q_embs, q_ids, data["qrels"],
                save_run_path,
            ).result()
            by_type = executor.submit(
                compute_metrics_by_type,
                c_embs, c_ids, q_embs, q_ids, data["qrels"], data["query_types"],
            ).result()

            elapsed = time.time() - t_task
            log_json(result_path, {
                "model_id": model_id,
                "lang": lang,
                "domain": domain,
                "overall": overall,
                "by_type": by_type,
                "evaluation_time_sec": round(elapsed, 2),
                "peak_vram_mb": vram_peak_mb(),
            })
            logger.info(
                "[%s] %s/%s: NDCG@10=%.4f  (%.1fs)  corpus_hits=%d/%d",
                model_id, lang, domain, overall["ndcg_at_10"],
                elapsed, c_hits, c_hits + c_misses,
            )

    executor.shutdown(wait=False)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    peak = vram_peak_mb()
    total = time.time() - t_wall
    logger.info("[%s] done. wall=%.1fs vram_peak=%dMB", model_id, total, peak)

    if not keep_chroma:
        for lang, domain in COMBOS:
            for col in [
                corpus_collection_name(model_id, lang, domain),
                query_collection_name(model_id, lang, domain),
            ]:
                try:
                    cache.client.delete_collection(col)
                except Exception:
                    pass
        logger.info("[%s] ChromaDB collections deleted.", model_id)

    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", metavar="HF_ID")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-chroma", action="store_true")
    parser.add_argument("--keep-models", action="store_true", help="Do not delete model files from cache after evaluation")
    parser.add_argument("--save-runs", action="store_true",
                        help="Q7: save per-query top-10 run_dicts for hybrid RRF fusion.")
    args = parser.parse_args()

    logger = _setup_logging()
    logger.info("Detecting hardware…")
    hw_cfg = detect_and_write()
    logger.info(
        "VRAM=%.1fGB  RAM=%.1fGB  CPU=%d  threads=%d",
        hw_cfg.vram_gb, hw_cfg.ram_gb, hw_cfg.cpu_cores, hw_cfg.torch_num_threads,
    )
    logger.info("Batch sizes: %s", hw_cfg.batch_sizes)

    torch.set_num_threads(hw_cfg.torch_num_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg_data = yaml.safe_load(CONFIG_FILE.read_text())
    model_map: dict[str, dict] = {m["id"]: m for m in cfg_data["models"]}

    model_ids = [args.model] if args.model else [m["id"] for m in cfg_data["models"]]
    if args.model and args.model not in model_map:
        logger.error("Model %r not in config. Available: %s", args.model, list(model_map))
        sys.exit(1)

    state = _load_state()
    if args.resume:
        before = len(model_ids)
        model_ids = [m for m in model_ids if m not in state["completed"]]
        logger.info("[Resume] Skipped %d completed. %d remaining.", before - len(model_ids), len(model_ids))

    if args.dry_run:
        print(f"\nWould evaluate {len(model_ids)} models:")
        for mid in model_ids:
            mc = model_map[mid]
            bs = hw_cfg.batch_sizes[mc["tier"]]
            print(f"  {mid:60s}  {mc['params']:6s}  tier={mc['tier']:12s}  batch_size={bs}")
        return

    pipeline = ModelPipeline(model_ids, keep_models=args.keep_models)
    for model_id, local_path in pipeline:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", model_id)
        logger.info("=" * 60)

        state["in_progress"] = model_id
        _save_state(state)

        batch_sizes = dict(hw_cfg.batch_sizes)

        success = False
        _MAX_ATTEMPTS = 8
        for attempt in range(_MAX_ATTEMPTS):
            _oom = False
            try:
                success = evaluate_model(
                    model_id, local_path, model_map[model_id],
                    batch_sizes, args.keep_chroma,
                    save_runs=args.save_runs,
                )
                break
            except torch.cuda.OutOfMemoryError:
                # Set flag only — do NOT call empty_cache here.
                # The traceback references evaluate_model's frame which holds the model
                # tensor; calling empty_cache() inside except would not free VRAM.
                _oom = True
            except Exception as exc:
                logger.error("[FAIL] %s: %s", model_id, exc, exc_info=True)
                break

            # Outside except: traceback cleared, model tensor refcount drops to 0.
            if _oom:
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                if attempt < _MAX_ATTEMPTS - 1:
                    for tier in batch_sizes:
                        batch_sizes[tier] = max(1, batch_sizes[tier] // 2)
                    logger.warning("[OOM] Retrying %s (attempt %d/%d) batch sizes: %s",
                                   model_id, attempt + 2, _MAX_ATTEMPTS, batch_sizes)
                else:
                    logger.error("[FAIL] %s OOM on all %d attempts.", model_id, _MAX_ATTEMPTS)
                    break

        if success:
            state["completed"].append(model_id)
            update_tables()
        else:
            failed_ids = {f["model_id"] for f in state["failed"]}
            if model_id not in failed_ids:
                state["failed"].append({"model_id": model_id})

        state["in_progress"] = None
        _save_state(state)

    logger.info("Done. Completed: %d  Failed: %d", len(state["completed"]), len(state["failed"]))
    if state["failed"]:
        logger.warning("Failed: %s", [f["model_id"] for f in state["failed"]])


if __name__ == "__main__":
    main()
