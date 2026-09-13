import numpy as np
from tests.scripts.conftest import make_mock_model


def test_cold_cache_encodes_all(tmp_path):
    from scripts.rag_embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    embs, hits, misses = cache.cached_encode(
        model, ["a", "b", "c"], ["d1", "d2", "d3"], "c_test_ja_finance", batch_size=10
    )
    assert embs.shape == (3, 8)
    assert hits == 0
    assert misses == 3
    assert model._call_count["n"] == 1


def test_warm_cache_skips_gpu(tmp_path):
    from scripts.rag_embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    embs1, _, _ = cache.cached_encode(
        model, ["a", "b", "c"], ["d1", "d2", "d3"], "c_test_ja_finance", batch_size=10
    )
    calls_after_first = model._call_count["n"]
    embs2, hits, misses = cache.cached_encode(
        model, ["a", "b", "c"], ["d1", "d2", "d3"], "c_test_ja_finance", batch_size=10
    )
    assert model._call_count["n"] == calls_after_first
    assert hits == 3
    assert misses == 0
    np.testing.assert_allclose(embs1, embs2, atol=1e-5)


def test_partial_cache(tmp_path):
    from scripts.rag_embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    cache.cached_encode(model, ["a", "b"], ["d1", "d2"], "c_test_ja_finance", batch_size=10)
    calls_before = model._call_count["n"]
    embs, hits, misses = cache.cached_encode(
        model, ["a", "b", "c", "d"], ["d1", "d2", "d3", "d4"],
        "c_test_ja_finance", batch_size=10,
    )
    assert embs.shape == (4, 8)
    assert hits == 2
    assert misses == 2
    assert model._call_count["n"] == calls_before + 1


def test_output_ordering_preserved(tmp_path):
    from scripts.rag_embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    embs1, _, _ = cache.cached_encode(
        model, ["zzz", "aaa", "mmm"], ["z", "a", "m"], "c_test_ja_law", batch_size=10
    )
    embs2, _, _ = cache.cached_encode(
        model, ["mmm", "zzz", "aaa"], ["m", "z", "a"], "c_test_ja_law", batch_size=10
    )
    np.testing.assert_allclose(embs2[0], embs1[2], atol=1e-5)
    np.testing.assert_allclose(embs2[1], embs1[0], atol=1e-5)


def test_prefix_applied_to_texts(tmp_path):
    from scripts.rag_embedding_cache import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path / "chroma"))
    model = make_mock_model()
    cache.cached_encode(
        model, ["hello"], ["d1"], "c_prefix_test", batch_size=10, prefix="passage: "
    )
    called_texts = model.encode.call_args[0][0]
    assert called_texts[0] == "passage: hello"
