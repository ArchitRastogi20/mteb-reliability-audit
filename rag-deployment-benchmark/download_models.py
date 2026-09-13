#!/usr/bin/env python3
"""
Download and cache all deployment benchmark models to disk.

Usage:
    python download_models.py
    python download_models.py --config config/models.yaml
    python download_models.py --model BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def load_config(config_path: Path) -> tuple[list[dict], Path]:
    data = yaml.safe_load(config_path.read_text())
    models = data["models"]
    cache_dir = Path(data.get("model_cache", "/workspace/model_cache"))
    return models, cache_dir


def is_cached(model_id: str, cache_dir: Path) -> bool:
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def download(model_id: str, cache_dir: Path) -> Path:
    for attempt in range(3):
        try:
            local = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
            )
            return Path(local)
        except Exception as exc:
            if attempt == 2:
                raise
            import time
            wait = 10 * (2 ** attempt)
            print(f"  Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download deployment benchmark models")
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "config" / "models.yaml",
        type=Path,
    )
    parser.add_argument("--model", metavar="HF_ID", help="Download a single model by ID")
    parser.add_argument("--skip-cached", action="store_true", default=True,
                        help="Skip models already in cache (default: true)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if already cached")
    args = parser.parse_args()

    models, cache_dir = load_config(args.config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        targets = [m for m in models if m["id"] == args.model]
        if not targets:
            print(f"Model '{args.model}' not found in config. Available models:")
            for m in models:
                print(f"  {m['id']}")
            sys.exit(1)
    else:
        targets = models

    print(f"Cache directory: {cache_dir.resolve()}")
    print(f"Models to process: {len(targets)}\n")

    ok = failed = skipped = 0
    for m in targets:
        model_id = m["id"]
        params = m.get("params", "?")
        print(f"[{model_id}]  ({params})")

        if not args.force and is_cached(model_id, cache_dir):
            print("  already cached — skip")
            skipped += 1
            continue

        try:
            path = download(model_id, cache_dir)
            print(f"  -> {path}")
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Downloaded: {ok}  Skipped: {skipped}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
