import numpy as np
import pytest
import chromadb
from unittest.mock import MagicMock


def make_mock_model(dim=8, seed=42):
    model = MagicMock()
    call_count = {"n": 0}
    def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True):
        call_count["n"] += 1
        rng = np.random.default_rng(seed + call_count["n"])
        embs = rng.random((len(texts), dim), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms
    model.encode.side_effect = fake_encode
    model._call_count = call_count
    return model


def test_cold_cache_encodes_all(tmp_path):
    from scripts.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    texts = ["hello", "world", "foo"]
    ids = ["d1", "d2", "d3"]
    embs, hits, misses = cache.cached_encode(model, texts, ids, "c_test_jaq", batch_size=10)
    assert embs.shape == (3, 8)
    assert hits == 0
    assert misses == 3
    assert model._call_count["n"] == 1


def test_warm_cache_skips_gpu(tmp_path):
    from scripts.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    texts = ["hello", "world", "foo"]
    ids = ["d1", "d2", "d3"]
    embs1, _, _ = cache.cached_encode(model, texts, ids, "c_test_jaq", batch_size=10)
    calls_after_first = model._call_count["n"]
    embs2, hits, misses = cache.cached_encode(model, texts, ids, "c_test_jaq", batch_size=10)
    assert model._call_count["n"] == calls_after_first
    assert hits == 3
    assert misses == 0
    np.testing.assert_allclose(embs1, embs2, atol=1e-5)


def test_partial_cache(tmp_path):
    from scripts.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    cache.cached_encode(model, ["a", "b"], ["d1", "d2"], "c_test_jaq", batch_size=10)
    calls_before = model._call_count["n"]
    embs, hits, misses = cache.cached_encode(
        model, ["a", "b", "c", "d"], ["d1", "d2", "d3", "d4"], "c_test_jaq", batch_size=10
    )
    assert embs.shape == (4, 8)
    assert hits == 2
    assert misses == 2
    assert model._call_count["n"] == calls_before + 1


def test_output_ordering(tmp_path):
    from scripts.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    ids = ["z", "a", "m"]
    texts = ["zzz", "aaa", "mmm"]
    embs1, _, _ = cache.cached_encode(model, texts, ids, "c_test_jaq", batch_size=10)
    embs2, _, _ = cache.cached_encode(
        model, ["mmm", "zzz", "aaa"], ["m", "z", "a"], "c_test_jaq", batch_size=10
    )
    np.testing.assert_allclose(embs2[0], embs1[2], atol=1e-5)
    np.testing.assert_allclose(embs2[1], embs1[0], atol=1e-5)
