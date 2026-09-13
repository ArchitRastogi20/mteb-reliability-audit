#!/usr/bin/env python3
"""
Italian RAG evaluation using pre-cached models in /workspace/model_cache.

Usage:
    python scripts/eval_italian.py --all --resume
    python scripts/eval_italian.py --model intfloat/multilingual-e5-small
    python scripts/eval_italian.py --all --dry-run
    python scripts/eval_italian.py --all --keep-chroma
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).parent.parent
MODEL_CACHE_DIR = Path("/workspace/model_cache")
CHROMA_PATH = "/workspace/rag_chroma_cache"
CONFIG_FILE = REPO_ROOT / "config" / "models.yaml"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
LOGS_DIR = REPO_ROOT / "logs"
STATE_FILE = REPO_ROOT / "state_italian.json"
OUTPUT_DIR = REPO_ROOT / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Runtime compatibility patches (mirrors eval_rag.py) ──────────────────────

import transformers.modeling_utils as _tm
if hasattr(_tm, "check_torch_load_is_safe"):
    _tm.check_torch_load_is_safe = lambda: None

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
                b, s = embs.shape[0], embs.shape[1]
                return torch.zeros((b, 1, s, s), dtype=embs.dtype, device=embs.device)
            return None
        return _orig_ccm(*args, **kwargs)
    _mu.create_causal_mask = _patched_ccm
except Exception:
    pass

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

try:
    from transformers import DynamicCache as _DC
    if not hasattr(_DC, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is None:
                return cache
            for lp in past_key_values:
                cache.update(lp[0], lp[1], len(cache.key_cache))
            return cache
        _DC.from_legacy_cache = _from_legacy_cache
    if not hasattr(_DC, "get_usable_length"):
        def _dc_get_usable_length(self, new_seq_length=0, layer_idx=0):
            ml = self.get_max_length() if hasattr(self, "get_max_length") else None
            sl = self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
            return min(ml - new_seq_length, sl) if ml is not None else sl
        _DC.get_usable_length = _dc_get_usable_length
    if not hasattr(_DC, "to_legacy_cache"):
        def _dc_to_legacy_cache(self):
            if not self.key_cache:
                return None
            return tuple((self.key_cache[i], self.value_cache[i]) for i in range(len(self.key_cache)))
        _DC.to_legacy_cache = _dc_to_legacy_cache
except Exception:
    pass

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

from sentence_transformers import SentenceTransformer

from scripts.rag_hardware import detect_and_write
from scripts.rag_embedding_cache import EmbeddingCache
from scripts.rag_report import update_tables
from scripts.rag_utils import (
    corpus_collection_name, log_json, query_collection_name,
    sanitize_model_id, vram_peak_mb,
)
from scripts.eval_rag import compute_metrics, compute_metrics_by_type, load_combo_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_italian_combos() -> list[tuple[str, str]]:
    """Return Italian (lang, domain) pairs that have corpus data on disk."""
    found = []
    for domain in ("finance", "law"):
        if (OUTPUT_DIR / "it" / domain / "corpus.jsonl").exists():
            found.append(("it", domain))
    return found


def _resolve_local_path(model_id: str) -> Path:
    """Return the cached snapshot path without downloading."""
    try:
        local = snapshot_download(
            repo_id=model_id,
            cache_dir=str(MODEL_CACHE_DIR),
            local_files_only=True,
        )
        return Path(local)
    except Exception as exc:
        raise FileNotFoundError(
            f"Model {model_id!r} not in {MODEL_CACHE_DIR}. "
            f"Run /workspace/download_models.py first. ({exc})"
        ) from exc


def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"italian_eval_{ts}.log"

    log = logging.getLogger("italian_eval")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info("Log: %s", log_file)
    return log


def _load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        seen: set = set()
        state["completed"] = [m for m in state.get("completed", []) if not (m in seen or seen.add(m))]
        return state
    return {"completed": [], "failed": [], "in_progress": None}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_model_italian(
    model_id: str,
    local_path: Path,
    model_config: dict,
    batch_sizes: dict[str, int],
    combos: list[tuple[str, str]],
    keep_chroma: bool = False,
) -> bool:
    log = logging.getLogger("italian_eval")
    result_dir = EVAL_DIR / sanitize_model_id(model_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    if all((result_dir / f"{lang}_{domain}.json").exists() for lang, domain in combos):
        log.info("[%s] All Italian results cached — skipping.", model_id)
        return True

    batch_size = batch_sizes[model_config["tier"]]
    cache = EmbeddingCache(CHROMA_PATH)
    hf_kwargs: dict = model_config.get("model_kwargs") or {}
    tokenizer_kwargs: dict = model_config.get("tokenizer_kwargs") or {}
    torch.cuda.reset_peak_memory_stats()

    # jina-v5 pre-load guard
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

    # Match autocast dtype to model dtype so task-specific index-puts in
    # jina-v3's custom modules (embedding.py, mha.py) stay consistent.
    _model_dtype = next(model.parameters()).dtype
    _autocast_dtype = _model_dtype if _model_dtype in (torch.bfloat16, torch.float16) else torch.float16

    with torch.amp.autocast("cuda", dtype=_autocast_dtype):
        for lang, domain in combos:
            result_path = result_dir / f"{lang}_{domain}.json"
            if result_path.exists():
                log.info("[%s] %s/%s cached — skipping.", model_id, lang, domain)
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

            overall = executor.submit(
                compute_metrics, c_embs, c_ids, q_embs, q_ids, data["qrels"]
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
            log.info(
                "[%s] %s/%s: NDCG@10=%.4f  (%.1fs)  corpus_hits=%d/%d",
                model_id, lang, domain, overall["ndcg_at_10"],
                elapsed, c_hits, c_hits + c_misses,
            )

    executor.shutdown(wait=False)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    total = time.time() - t_wall
    log.info("[%s] done. wall=%.1fs  vram_peak=%dMB", model_id, total, vram_peak_mb())

    if not keep_chroma:
        for lang, domain in combos:
            for col in [
                corpus_collection_name(model_id, lang, domain),
                query_collection_name(model_id, lang, domain),
            ]:
                try:
                    cache.client.delete_collection(col)
                except Exception:
                    pass
        log.info("[%s] ChromaDB collections deleted.", model_id)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Italian RAG Evaluation (pre-cached models)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", metavar="HF_ID", help="Evaluate a single model by HuggingFace ID")
    group.add_argument("--all", action="store_true", help="Evaluate all models in config")
    parser.add_argument("--resume", action="store_true", help="Skip models with all results already on disk")
    parser.add_argument("--domain", choices=["finance", "law"], help="Restrict to one domain (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="List models without running")
    parser.add_argument("--keep-chroma", action="store_true", help="Keep ChromaDB collections after evaluation")
    args = parser.parse_args()

    log = _setup_logging()

    combos = _detect_italian_combos()
    if args.domain:
        combos = [(l, d) for l, d in combos if d == args.domain]
    if not combos:
        log.error("No Italian data found for the requested domain(s). Run the generation pipeline first.")
        sys.exit(1)
    log.info("Italian combos: %s", combos)

    log.info("Detecting hardware…")
    hw_cfg = detect_and_write()
    log.info(
        "VRAM=%.1fGB  RAM=%.1fGB  CPU=%d  threads=%d",
        hw_cfg.vram_gb, hw_cfg.ram_gb, hw_cfg.cpu_cores, hw_cfg.torch_num_threads,
    )
    log.info("Batch sizes by tier: %s", hw_cfg.batch_sizes)

    torch.set_num_threads(hw_cfg.torch_num_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg_data = yaml.safe_load(CONFIG_FILE.read_text())
    model_map: dict[str, dict] = {m["id"]: m for m in cfg_data["models"]}

    model_ids = [args.model] if args.model else [m["id"] for m in cfg_data["models"]]
    if args.model and args.model not in model_map:
        log.error("Model %r not in config. Available: %s", args.model, list(model_map))
        sys.exit(1)

    state = _load_state()
    if args.resume:
        before = len(model_ids)
        def _all_results_exist(mid: str) -> bool:
            d = EVAL_DIR / sanitize_model_id(mid)
            return all((d / f"{lang}_{domain}.json").exists() for lang, domain in combos)
        model_ids = [m for m in model_ids if not _all_results_exist(m)]
        log.info("[Resume] Skipped %d fully-complete. %d remaining.", before - len(model_ids), len(model_ids))

    if args.dry_run:
        print(f"\nWould evaluate {len(model_ids)} models on Italian combos {combos}:")
        for mid in model_ids:
            mc = model_map[mid]
            bs = hw_cfg.batch_sizes[mc["tier"]]
            cache_dir = MODEL_CACHE_DIR / f"models--{mid.replace('/', '--')}"
            status = "CACHED" if cache_dir.exists() else "MISSING"
            print(f"  [{status}] {mid:60s}  {mc['params']:6s}  tier={mc['tier']:12s}  batch={bs}")
        return

    for model_id in model_ids:
        log.info("=" * 60)
        log.info("Evaluating: %s", model_id)
        log.info("=" * 60)

        try:
            local_path = _resolve_local_path(model_id)
        except FileNotFoundError as exc:
            log.error("[SKIP] %s", exc)
            failed_ids = {f["model_id"] for f in state.get("failed", [])}
            if model_id not in failed_ids:
                state.setdefault("failed", []).append({"model_id": model_id, "reason": "not_cached"})
            _save_state(state)
            continue

        state["in_progress"] = model_id
        _save_state(state)

        batch_sizes = dict(hw_cfg.batch_sizes)
        success = False
        _MAX_ATTEMPTS = 8

        for attempt in range(_MAX_ATTEMPTS):
            _oom = False
            try:
                success = evaluate_model_italian(
                    model_id, local_path, model_map[model_id],
                    batch_sizes, combos, args.keep_chroma,
                )
                break
            except torch.cuda.OutOfMemoryError:
                _oom = True
            except Exception as exc:
                log.error("[FAIL] %s: %s", model_id, exc, exc_info=True)
                break

            if _oom:
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                if attempt < _MAX_ATTEMPTS - 1:
                    for tier in batch_sizes:
                        batch_sizes[tier] = max(1, batch_sizes[tier] // 2)
                    log.warning(
                        "[OOM] Retrying %s (attempt %d/%d) batch=%s",
                        model_id, attempt + 2, _MAX_ATTEMPTS, batch_sizes,
                    )
                else:
                    log.error("[FAIL] %s OOM on all %d attempts.", model_id, _MAX_ATTEMPTS)
                    break

        if success:
            state["completed"].append(model_id)
            update_tables()
        else:
            failed_ids = {f["model_id"] for f in state.get("failed", [])}
            if model_id not in failed_ids:
                state.setdefault("failed", []).append({"model_id": model_id})

        state["in_progress"] = None
        _save_state(state)

    log.info(
        "Done. Completed: %d  Failed: %d",
        len(state["completed"]), len(state.get("failed", [])),
    )
    if state.get("failed"):
        log.warning("Failed: %s", [f["model_id"] for f in state["failed"]])


if __name__ == "__main__":
    main()
