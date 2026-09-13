"""S12 — Index-build cost (wall-time + peak GPU VRAM) for the ja_finance corpus.

For each open-weight model in the model roster, encode the existing 1k-doc
ja_finance corpus once on a single GPU and record:
  - encode_wall_time_sec : wall-clock for the encode call
  - peak_vram_mb_during_encode : torch.cuda.max_memory_allocated() (encode-only window)
  - throughput_docs_per_sec
  - corpus_size, params, model_id

Output: analysis_output/stats/rev_S12_index_build.json

Practitioner motivation: index-build is the binding deployment constraint
(must fit in VRAM AND finish in reasonable wall time before any query
serves), but the existing Tab. 11 in the upstream paper reports only
query-time QPS/VRAM/latency. S12 closes that gap.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Must be set before any HF library import so HF Hub resolves the right cache dir.
import os
os.environ.setdefault("HF_HUB_CACHE", "/workspace/model_cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_S12")

OUT = REPO_ROOT / "analysis_output" / "stats" / "rev_S12_index_build.json"
OUTPUT_DIR = REPO_ROOT / "output"

# (model_id, params_str) — same roster as the this paper Tab. 11
# Paired with the prefix the model wants for "passage" role.
MODELS_TO_TIME: list[tuple[str, str]] = [
    ("ibm-granite/granite-embedding-107m-multilingual", "107M"),
    ("intfloat/multilingual-e5-small", "118M"),
    ("intfloat/e5-small-v2", "33M"),
    ("intfloat/multilingual-e5-base", "278M"),
    ("jinaai/jina-embeddings-v5-text-nano", "212M"),
    ("intfloat/multilingual-e5-large-instruct", "560M"),
    ("intfloat/multilingual-e5-large", "560M"),
    ("BAAI/bge-m3", "568M"),
    ("Snowflake/snowflake-arctic-embed-l-v2.0", "568M"),
    ("microsoft/harrier-oss-v1-0.6b", "596M"),
    ("Qwen/Qwen3-Embedding-0.6B", "600M"),
    ("Qwen/Qwen3-Embedding-4B", "4B"),
    ("Salesforce/SFR-Embedding-Mistral", "7B"),
    ("intfloat/e5-mistral-7b-instruct", "7B"),
    ("Qwen/Qwen3-Embedding-8B", "8B"),
]

# Models whose passage role wants a prefix (look up from config/models.yaml)
PASSAGE_PREFIX: dict[str, str] = {
    # Empty string means "no prefix" — the key being present signals explicit registration.
    "Snowflake/snowflake-arctic-embed-l-v2.0": "",
    # Add others here only if the model card requires a passage prefix
}

# Extra kwargs passed to model.encode() for models that require a task hint.
ENCODE_EXTRA_KWARGS: dict[str, dict] = {
    "jinaai/jina-embeddings-v5-text-nano": {"task": "retrieval"},
}


def time_one_model(model_id: str, params: str, corpus_texts: list[str], batch_size: int = 32) -> dict[str, Any]:
    """Encode `corpus_texts` once with `model_id`; return timing + VRAM dict.

    Loads the model fresh, calls `model.encode(...)` once, measures wall-time and
    peak GPU VRAM (`torch.cuda.max_memory_allocated`). Releases the model after.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    logger.info("[S12] %s (%s) — loading...", model_id, params)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t_load_start = time.time()
    try:
        model = SentenceTransformer(model_id, device="cuda", trust_remote_code=True)
        model.eval()
    except Exception as e:
        logger.error("[S12] %s LOAD FAILED: %s", model_id, str(e)[:200])
        return {
            "model_id": model_id, "params": params,
            "corpus_size": len(corpus_texts),
            "encode_wall_time_sec": None, "peak_vram_mb_during_encode": None,
            "peak_vram_mb_after_load": None, "throughput_docs_per_sec": None,
            "load_time_sec": None, "batch_size": batch_size,
            "error": f"load failed: {str(e)[:200]}",
        }
    t_load = time.time() - t_load_start
    peak_vram_after_load = int(torch.cuda.max_memory_allocated() // (1024 * 1024))

    prefix = PASSAGE_PREFIX.get(model_id, "")
    prefixed = [prefix + t for t in corpus_texts] if prefix else corpus_texts

    # Reset AGAIN so we get encode-only peak VRAM
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        _model_dtype = next(model.parameters()).dtype
        _autocast_dtype = _model_dtype if _model_dtype in (torch.bfloat16, torch.float16) else torch.float16
        extra = ENCODE_EXTRA_KWARGS.get(model_id, {})
        with torch.amp.autocast("cuda", dtype=_autocast_dtype):
            _ = model.encode(
                prefixed, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
                **extra,
            )
    except Exception as e:
        logger.error("[S12] %s ENCODE FAILED: %s", model_id, str(e)[:200])
        del model
        torch.cuda.empty_cache()
        return {
            "model_id": model_id, "params": params,
            "corpus_size": len(corpus_texts),
            "encode_wall_time_sec": None, "peak_vram_mb_during_encode": None,
            "peak_vram_mb_after_load": peak_vram_after_load,
            "throughput_docs_per_sec": None,
            "load_time_sec": round(t_load, 2), "batch_size": batch_size,
            "error": f"encode failed: {str(e)[:200]}",
        }
    elapsed = time.time() - t0
    peak_mb = int(torch.cuda.max_memory_allocated() // (1024 * 1024))

    del model
    torch.cuda.empty_cache()

    out = {
        "model_id": model_id, "params": params,
        "corpus_size": len(corpus_texts),
        "encode_wall_time_sec": round(elapsed, 2),
        "peak_vram_mb_during_encode": peak_mb,
        "peak_vram_mb_after_load": peak_vram_after_load,
        "throughput_docs_per_sec": round(len(corpus_texts) / elapsed, 1),
        "load_time_sec": round(t_load, 2),
        "batch_size": batch_size,
    }
    logger.info("[S12] %s: t=%.1fs, vram=%dMB, throughput=%.1f docs/s",
                model_id, elapsed, peak_mb, out["throughput_docs_per_sec"])
    return out


def load_corpus_texts(lang: str = "ja", domain: str = "finance") -> list[str]:
    p = OUTPUT_DIR / lang / domain / "corpus.jsonl"
    out: list[str] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            out.append(doc.get("text", ""))
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="S12 — Index-build cost benchmark")
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--domain", default="finance")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip 4B / 7B / 8B models (use on <24GB VRAM)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to a subset of model IDs (substring match)")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus_texts(args.lang, args.domain)
    logger.info("Loaded %d corpus docs from %s/%s", len(corpus), args.lang, args.domain)

    models = MODELS_TO_TIME[:]
    if args.skip_large:
        models = [m for m in models if not any(t in m[1] for t in ("4B", "7B", "8B"))]
    if args.only:
        models = [m for m in models if any(o in m[0] for o in args.only)]

    logger.info("Will time %d models", len(models))

    results: list[dict] = []
    for model_id, params in models:
        try:
            r = time_one_model(model_id, params, corpus, batch_size=args.batch_size)
        except Exception as e:
            logger.exception("[S12] %s UNCAUGHT: %s", model_id, e)
            r = {"model_id": model_id, "params": params, "error": f"uncaught: {str(e)[:200]}",
                 "corpus_size": len(corpus)}
        results.append(r)
        # Persist after every model so a crash doesn't lose prior work
        OUT.write_text(json.dumps({
            "lang": args.lang, "domain": args.domain,
            "corpus_size": len(corpus), "batch_size": args.batch_size,
            "results": results,
        }, indent=2))

    # Final pretty print
    print("\n=== S12 RESULTS (sorted by encode time) ===")
    valid = [r for r in results if r.get("encode_wall_time_sec") is not None]
    valid.sort(key=lambda r: r["encode_wall_time_sec"])
    print(f"{'model':<50} {'params':>6} {'encode_s':>9} {'vram_MB':>8} {'docs/s':>7}")
    print("-" * 88)
    for r in valid:
        print(f"{r['model_id']:<50} {r['params']:>6} {r['encode_wall_time_sec']:>9.1f} "
              f"{r['peak_vram_mb_during_encode']:>8} {r['throughput_docs_per_sec']:>7.1f}")
    failed = [r for r in results if r.get("error")]
    if failed:
        print("\nFAILED:")
        for r in failed:
            print(f"  {r['model_id']:<50}  {r['error'][:80]}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
