from __future__ import annotations
import hashlib, json, re, shutil
from pathlib import Path
from typing import Generator

import psutil

DISK_TOTAL_BYTES = shutil.disk_usage("/workspace").total
DISK_BUDGET_BYTES = int(DISK_TOTAL_BYTES * 0.70)   # 70% for model cache
RAM_LIMIT_PERCENT = 95.0                             # pause writes above this


TASK_SHORT: dict[str, str] = {
    "JaqketRetrieval": "jaq",
    "MrTyDiJaRetrievalLite": "mrt",
    "JaGovFaqsRetrieval": "gov",
    "NLPJournalTitleAbsRetrieval": "ta",
    "NLPJournalAbsIntroRetrieval": "ai",
    "NLPJournalTitleIntroRetrieval": "ti",
}


def batch_iter(items: list, batch_size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def sanitize_model_id(model_id: str) -> str:
    """
    Unique, ChromaDB-safe identifier for a model.
    Format: <name_prefix20>_<md5_6>  (always <=27 chars, always unique)
    """
    name = model_id.split("/")[-1]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name)[:20]
    h = hashlib.md5(model_id.encode()).hexdigest()[:6]
    return f"{safe_name}_{h}"


def corpus_collection_name(model_id: str, task_name: str) -> str:
    """ChromaDB collection name for corpus embeddings. <=63 chars guaranteed."""
    short_task = TASK_SHORT[task_name]
    return f"c_{sanitize_model_id(model_id)}_{short_task}"


def query_collection_name(model_id: str, task_name: str) -> str:
    """ChromaDB collection name for query embeddings. <=63 chars guaranteed."""
    short_task = TASK_SHORT[task_name]
    return f"q_{sanitize_model_id(model_id)}_{short_task}"


def disk_usage_bytes(path: str | Path = "/workspace") -> int:
    return shutil.disk_usage(str(path)).used


def ram_percent() -> float:
    return psutil.virtual_memory().percent


def vram_peak_mb() -> int:
    """Peak GPU memory allocated since last reset, in MB."""
    try:
        import torch
        return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except Exception:
        return 0


def log_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
