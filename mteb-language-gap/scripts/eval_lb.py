#!/usr/bin/env python3
"""
Unified leaderboard evaluation CLI.

Usage:
    python scripts/eval_lb.py --lang jpn --all --resume
    python scripts/eval_lb.py --lang hin --model intfloat/multilingual-e5-small
    python scripts/eval_lb.py --lang ita --all --dry-run
    python scripts/eval_lb.py --lang jpn --all --keep-chroma
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from scripts.detect_hardware import detect_and_write
from scripts.lb_evaluator import LANG_CONFIGS, LeaderboardEvaluator
from scripts.make_lb_table import update_table
from scripts.model_pipeline import ModelPipeline

REPO_ROOT = Path(__file__).parent.parent
CONFIG_FILE = REPO_ROOT / "config" / "models.yaml"
LOGS_DIR = REPO_ROOT / "logs"


def _setup_logging(lang_code: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{lang_code}_lb_{ts}.log"

    logger = logging.getLogger("mteb_lb")
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

    logger.info("Log file: %s", log_file)
    return logger


def _state_path(lang_code: str) -> Path:
    return REPO_ROOT / f"state_{lang_code}_lb.json"


def _load_state(lang_code: str) -> dict:
    p = _state_path(lang_code)
    if p.exists():
        state = json.loads(p.read_text())
        # Deduplicate completed list (preserve order, keep first occurrence)
        seen: set = set()
        state["completed"] = [m for m in state.get("completed", []) if not (m in seen or seen.add(m))]
        # Deduplicate failed list by model_id
        seen_failed: set = set()
        state["failed"] = [
            f for f in state.get("failed", [])
            if not (f["model_id"] in seen_failed or seen_failed.add(f["model_id"]))
        ]
        return state
    return {"completed": [], "failed": [], "in_progress": None}


def _save_state(lang_code: str, state: dict) -> None:
    _state_path(lang_code).write_text(json.dumps(state, indent=2))


def evaluate_one(
    model_id: str,
    local_path: Path,
    model_config: dict,
    lang_code: str,
    batch_sizes: dict[str, int],
    keep_chroma: bool,
    max_seq_length: int | None = None,
    results_suffix: str | None = None,
) -> bool:
    logger = logging.getLogger("mteb_lb")
    lang_cfg = LANG_CONFIGS[lang_code]

    _MAX_ATTEMPTS = 8
    for attempt in range(_MAX_ATTEMPTS):
        _oom = False
        ev = None
        try:
            ev = LeaderboardEvaluator(
                model_id=model_id,
                local_model_path=str(local_path),
                model_config=model_config,
                lang_config=lang_cfg,
                batch_sizes=batch_sizes,
                keep_chroma=keep_chroma,
                max_seq_length=max_seq_length,
                results_suffix=results_suffix,
            )
            ev.run()
            return True
        except torch.cuda.OutOfMemoryError:
            # Set flag only — do NOT call empty_cache here.
            # The exception traceback references the run() stack frame which holds the
            # model tensor; calling empty_cache() inside except would not free it.
            # Cleanup happens below, after the except block exits and the traceback clears.
            _oom = True
        except Exception as exc:
            logger.error("[FAIL] %s: %s", model_id, exc, exc_info=True)
            return False

        # Runs OUTSIDE the except block: traceback is cleared, model tensor refcount drops
        # to 0, allowing gc + empty_cache to actually reclaim the VRAM.
        if _oom:
            del ev
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
                return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="MTEB Leaderboard Evaluation")
    parser.add_argument("--lang", required=True, choices=["jpn", "hin", "ita", "swa"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", metavar="HF_ID", help="Evaluate single model")
    group.add_argument("--all", action="store_true", help="Evaluate all 20 models")
    parser.add_argument("--resume", action="store_true", help="Skip completed models")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--keep-chroma", action="store_true", help="Do not delete ChromaDB after model")
    parser.add_argument("--keep-models", action="store_true", help="Do not delete model files from cache after evaluation")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override SentenceTransformer.max_seq_length (Q3 chunker sweep). "
             "If None, use the model's default.",
    )
    parser.add_argument(
        "--results-suffix",
        type=str,
        default=None,
        help="Suffix for results dir; e.g. --results-suffix _maxseq256 writes to "
             "results/{lang}_lb_maxseq256/ instead of results/{lang}_lb/.",
    )
    args = parser.parse_args()

    logger = _setup_logging(args.lang)

    logger.info("Detecting hardware…")
    hw_cfg = detect_and_write()
    logger.info(
        "Hardware: VRAM=%.1fGB  RAM=%.1fGB  CPU=%d cores  threads=%d",
        hw_cfg.vram_gb, hw_cfg.ram_gb, hw_cfg.cpu_cores, hw_cfg.torch_num_threads,
    )
    logger.info("Batch sizes: %s", hw_cfg.batch_sizes)

    torch.set_num_threads(hw_cfg.torch_num_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg_data = yaml.safe_load(CONFIG_FILE.read_text())
    model_map: dict[str, dict] = {m["id"]: m for m in cfg_data["models"]}

    if args.model:
        if args.model not in model_map:
            logger.error("Model %r not in config. Available: %s", args.model, list(model_map))
            sys.exit(1)
        model_ids = [args.model]
    else:
        model_ids = [m["id"] for m in cfg_data["models"]]

    state = _load_state(args.lang)

    if args.resume:
        skipped = [m for m in model_ids if m in state["completed"]]
        model_ids = [m for m in model_ids if m not in state["completed"]]
        if skipped:
            logger.info("[Resume] Skipping %d completed models. %d remaining.", len(skipped), len(model_ids))

    if args.dry_run:
        print(f"\nWould evaluate {len(model_ids)} models for lang={args.lang}:")
        for mid in model_ids:
            mc = model_map[mid]
            bs = hw_cfg.batch_sizes[mc["tier"]]
            print(f"  {mid:60s}  {mc['params']:6s}  tier={mc['tier']:12s}  batch_size={bs}")
        return

    pipeline = ModelPipeline(model_ids, keep_models=args.keep_models)
    for model_id, local_path in pipeline:
        logger.info("=" * 60)
        logger.info("Evaluating [%s]: %s", args.lang.upper(), model_id)
        logger.info("=" * 60)

        state["in_progress"] = model_id
        _save_state(args.lang, state)

        batch_sizes = dict(hw_cfg.batch_sizes)  # mutable copy per model (OOM retry may halve)
        success = evaluate_one(
            model_id, local_path, model_map[model_id],
            args.lang, batch_sizes, args.keep_chroma,
            max_seq_length=args.max_seq_length,
            results_suffix=args.results_suffix,
        )

        if success:
            state["completed"].append(model_id)
            update_table(args.lang)
        else:
            failed_ids = {f["model_id"] for f in state["failed"]}
            if model_id not in failed_ids:
                state["failed"].append({"model_id": model_id})

        state["in_progress"] = None
        _save_state(args.lang, state)

    logger.info("=" * 60)
    logger.info("Done. Completed: %d  Failed: %d", len(state["completed"]), len(state["failed"]))
    if state["failed"]:
        logger.warning("Failed models: %s", [f["model_id"] for f in state["failed"]])


if __name__ == "__main__":
    main()
