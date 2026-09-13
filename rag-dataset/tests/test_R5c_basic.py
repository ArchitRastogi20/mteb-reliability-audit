"""Offline smoke test for R5c. Does not require GPU or API keys."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports():
    mod = _load("eval_R5c", REPO_ROOT / "scripts" / "eval_R5c_open_weight_propagation.py")
    assert hasattr(mod, "MODEL_PAIR")
    assert hasattr(mod, "main_async")


def test_model_pair_constants():
    mod = _load("eval_R5c", REPO_ROOT / "scripts" / "eval_R5c_open_weight_propagation.py")
    assert mod.MODEL_PAIR == [
        ("hf", "BAAI/bge-m3", "ja", "finance"),
        ("hf", "Snowflake/snowflake-arctic-embed-l-v2.0", "ja", "finance"),
    ]
    assert mod.JUDGES == ["gpt-5.4-mini", "judge-llm-small"]
    assert mod.N_QUERIES == 50
    assert mod.TOP_K == 3


def test_hf_embed_helper_signature():
    mod = _load("eval_R5c", REPO_ROOT / "scripts" / "eval_R5c_open_weight_propagation.py")
    assert callable(mod.embed_hf_model)
    # Signature: embed_hf_model(model_id, texts, role="passage") -> np.ndarray of shape (N, D)
    import inspect
    sig = inspect.signature(mod.embed_hf_model)
    assert {"model_id", "texts", "role"}.issubset(sig.parameters.keys())


def test_retrieve_top_k_basic():
    mod = _load("eval_R5c", REPO_ROOT / "scripts" / "eval_R5c_open_weight_propagation.py")
    import numpy as np
    # 3 docs, 2 queries, dim=4
    corpus = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    queries = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    out = mod.retrieve_top_k(corpus, ids, queries, top_k=2)
    assert out[0][0] == "a"  # query 0 best match = doc a
    assert out[1][0] == "c"  # query 1 best match = doc c
