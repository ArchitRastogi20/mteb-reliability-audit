from __future__ import annotations
import shutil
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Iterator

from huggingface_hub import snapshot_download

from scripts.rag_utils import disk_usage_bytes, DISK_BUDGET_BYTES

MODEL_CACHE_DIR = Path("/workspace/model_cache")
_SENTINEL = object()


class ModelPipeline:
    """
    Yields (model_id, local_path) pairs.
    Downloads model N+1 in background while model N is being evaluated.
    Deletes each model's files from disk after the caller finishes with it.
    """

    def __init__(
        self,
        model_ids: list[str],
        cache_dir: Path = MODEL_CACHE_DIR,
        disk_budget: int = DISK_BUDGET_BYTES,
        keep_models: bool = False,
    ) -> None:
        self.model_ids = model_ids
        self.cache_dir = Path(cache_dir)
        self.disk_budget = disk_budget
        self.keep_models = keep_models
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ready: Queue = Queue(maxsize=2)

    def __iter__(self) -> Iterator[tuple[str, Path]]:
        dl_thread = threading.Thread(target=self._download_loop, daemon=True)
        dl_thread.start()

        try:
            for _ in self.model_ids:
                item = self._ready.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                model_id, local_path = item
                yield model_id, local_path
                if not self.keep_models:
                    self._cleanup(local_path)
        finally:
            dl_thread.join()

    def _download_loop(self) -> None:
        for model_id in self.model_ids:
            try:
                local_path = self._download_with_quota(model_id)
                self._ready.put((model_id, local_path))
            except Exception as exc:
                print(f"[Downloader] FAILED {model_id}: {exc}")
                self._ready.put(exc)
        self._ready.put(_SENTINEL)

    def _download_with_quota(self, model_id: str) -> Path:
        while disk_usage_bytes(self.cache_dir.parent) > self.disk_budget:
            print(
                f"[Downloader] Disk quota exceeded "
                f"({disk_usage_bytes(self.cache_dir.parent)/1e9:.1f}GB). Waiting..."
            )
            time.sleep(30)

        print(f"[Downloader] Fetching {model_id}...")
        for attempt in range(3):
            try:
                local = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                )
                print(f"[Downloader] Ready: {model_id}")
                return Path(local)
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 10 * (2 ** attempt)
                print(f"[Downloader] Attempt {attempt+1} failed ({exc}). Retry in {wait}s...")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def _cleanup(self, local_path: Path) -> None:
        try:
            model_root = local_path.parent.parent
            if model_root.exists() and model_root.name.startswith("models--"):
                shutil.rmtree(model_root, ignore_errors=True)
                print(f"[Cleanup] Removed {model_root.name}")
            elif local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
                print(f"[Cleanup] Removed {local_path}")
        except Exception as e:
            print(f"[Cleanup] Warning: {e}")
