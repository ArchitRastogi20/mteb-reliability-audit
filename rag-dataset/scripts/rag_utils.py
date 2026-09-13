from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Generator

import psutil

DISK_TOTAL_BYTES = shutil.disk_usage("/workspace").total
DISK_BUDGET_BYTES = int(DISK_TOTAL_BYTES * 0.70)
RAM_LIMIT_PERCENT = 95.0


def batch_iter(items: list, batch_size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def sanitize_model_id(model_id: str) -> str:
    """ChromaDB-safe unique identifier. Format: <name20>_<md5_6> (<=27 chars)."""
    name = model_id.split("/")[-1]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name)[:20]
    h = hashlib.md5(model_id.encode()).hexdigest()[:6]
    return f"{safe_name}_{h}"


def corpus_collection_name(model_id: str, lang: str, domain: str) -> str:
    """ChromaDB corpus collection name (<=63 chars guaranteed)."""
    return f"c_{sanitize_model_id(model_id)}_{lang}_{domain}"


def query_collection_name(model_id: str, lang: str, domain: str) -> str:
    """ChromaDB query collection name (<=63 chars guaranteed)."""
    return f"q_{sanitize_model_id(model_id)}_{lang}_{domain}"


def disk_usage_bytes(path: str | Path = "/workspace") -> int:
    return shutil.disk_usage(str(path)).used


def ram_percent() -> float:
    return psutil.virtual_memory().percent


def vram_peak_mb() -> int:
    try:
        import torch
        return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    except Exception:
        return 0


def log_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
