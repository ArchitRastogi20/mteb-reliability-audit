from __future__ import annotations
import threading
from queue import Queue
from typing import Any

import chromadb
import numpy as np

from scripts.rag_utils import batch_iter, ram_percent, RAM_LIMIT_PERCENT

_CHROMA_GET_BATCH = 500


class EmbeddingCache:
    """
    ChromaDB-backed embedding cache with async writer thread.

    cached_encode():
    - Checks ChromaDB for already-encoded ids
    - GPU-encodes only missing texts
    - Streams new embeddings to ChromaDB in a background thread
    - Returns embeddings in original input order
    Returns: (embeddings [n, dim], n_cache_hits, n_cache_misses)
    """

    def __init__(self, chroma_path: str) -> None:
        self.client = chromadb.PersistentClient(path=chroma_path)

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def cached_encode(
        self,
        model: Any,
        texts: list[str],
        ids: list[str],
        collection_name: str,
        batch_size: int,
        prefix: str = "",
        encode_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, int, int]:
        collection = self._get_or_create(collection_name)

        all_existing_ids: list[str] = []
        all_existing_embs: list[list[float]] = []
        for id_batch in batch_iter(ids, _CHROMA_GET_BATCH):
            result = collection.get(ids=id_batch, include=["embeddings"])
            all_existing_ids.extend(result["ids"])
            all_existing_embs.extend(result["embeddings"])

        cached_set = set(all_existing_ids)
        emb_map: dict[str, np.ndarray] = {
            eid: np.array(emb, dtype=np.float32)
            for eid, emb in zip(all_existing_ids, all_existing_embs)
        }

        missing_idx = [i for i, id_ in enumerate(ids) if id_ not in cached_set]
        missing_ids = [ids[i] for i in missing_idx]
        missing_texts = [texts[i] for i in missing_idx]

        n_hits = len(cached_set)
        n_misses = len(missing_idx)

        if missing_texts:
            new_embs = self._encode_and_cache(
                model, missing_texts, missing_ids, collection, batch_size, prefix,
                encode_kwargs or {},
            )
            for mid, emb in zip(missing_ids, new_embs):
                emb_map[mid] = emb

        result_arr = np.stack([emb_map[id_] for id_ in ids])
        return result_arr, n_hits, n_misses

    def _encode_and_cache(
        self,
        model: Any,
        texts: list[str],
        ids: list[str],
        collection: chromadb.Collection,
        batch_size: int,
        prefix: str,
        encode_kwargs: dict | None = None,
    ) -> np.ndarray:
        write_queue: Queue = Queue(maxsize=8)
        writer = threading.Thread(
            target=_chroma_writer, args=(collection, write_queue), daemon=True
        )
        writer.start()

        all_embs: list[np.ndarray] = []
        prefixed = [prefix + t for t in texts] if prefix else texts

        for batch_texts, batch_ids in zip(
            batch_iter(prefixed, batch_size), batch_iter(ids, batch_size)
        ):
            while ram_percent() > RAM_LIMIT_PERCENT:
                import time; time.sleep(2)

            embs: np.ndarray = model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                **(encode_kwargs or {}),
            )
            if not isinstance(embs, np.ndarray):
                embs = np.array(embs, dtype=np.float32)
            all_embs.append(embs)
            write_queue.put((batch_ids, embs.tolist()))

        write_queue.put(None)
        writer.join()
        return np.vstack(all_embs)


def _chroma_writer(collection: chromadb.Collection, queue: Queue) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        batch_ids, batch_embs = item
        try:
            collection.add(ids=batch_ids, embeddings=batch_embs)
        except Exception:
            try:
                collection.upsert(ids=batch_ids, embeddings=batch_embs)
            except Exception as e:
                print(f"[ChromaDB writer] upsert failed: {e}")
