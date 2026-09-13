import numpy as np
import pytest
from unittest.mock import MagicMock


def make_mock_model(dim: int = 8, seed: int = 42) -> MagicMock:
    """Mock SentenceTransformer returning deterministic normalized embeddings."""
    model = MagicMock()
    call_count = {"n": 0}

    def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True):
        call_count["n"] += 1
        rng = np.random.default_rng(seed + call_count["n"])
        embs = rng.random((len(texts), dim), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / norms).astype(np.float32)

    model.encode.side_effect = fake_encode
    model._call_count = call_count
    return model


@pytest.fixture
def tiny_model():
    return make_mock_model()
