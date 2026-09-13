import pytest
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def tiny_model():
    """Mock SentenceTransformer that returns deterministic 8-dim embeddings."""
    model = MagicMock()
    def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True):
        np.random.seed(42)
        return np.random.rand(len(texts), 8).astype(np.float32)
    model.encode.side_effect = fake_encode
    return model

@pytest.fixture
def tmp_chroma(tmp_path):
    """Ephemeral ChromaDB client for tests."""
    import chromadb
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))
