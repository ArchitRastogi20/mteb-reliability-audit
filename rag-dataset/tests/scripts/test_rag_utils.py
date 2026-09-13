from __future__ import annotations
import json
import shutil
from pathlib import Path


def test_sanitize_model_id_is_chromadb_safe():
    from scripts.rag_utils import sanitize_model_id
    san = sanitize_model_id("intfloat/multilingual-e5-small")
    assert len(san) <= 63
    assert san[0].isalpha()
    assert all(c.isalnum() or c in "_-" for c in san)


def test_sanitize_model_id_is_unique():
    from scripts.rag_utils import sanitize_model_id
    a = sanitize_model_id("org/model-a")
    b = sanitize_model_id("org/model-b")
    assert a != b


def test_sanitize_model_id_same_input_same_output():
    from scripts.rag_utils import sanitize_model_id
    assert sanitize_model_id("BAAI/bge-m3") == sanitize_model_id("BAAI/bge-m3")


def test_batch_iter_splits_correctly():
    from scripts.rag_utils import batch_iter
    items = list(range(10))
    batches = list(batch_iter(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batch_iter_exact_multiple():
    from scripts.rag_utils import batch_iter
    items = list(range(6))
    batches = list(batch_iter(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5]]


def test_log_json_creates_file(tmp_path):
    from scripts.rag_utils import log_json
    out = tmp_path / "sub" / "result.json"
    log_json(out, {"score": 0.85})
    assert out.exists()
    assert json.loads(out.read_text()) == {"score": 0.85}


def test_disk_budget_is_70_percent():
    from scripts.rag_utils import DISK_BUDGET_BYTES
    total = shutil.disk_usage("/workspace").total
    assert DISK_BUDGET_BYTES == int(total * 0.70)


def test_ram_limit_percent_is_95():
    from scripts.rag_utils import RAM_LIMIT_PERCENT
    assert RAM_LIMIT_PERCENT == 95.0


def test_vram_peak_mb_returns_int():
    from scripts.rag_utils import vram_peak_mb
    result = vram_peak_mb()
    assert isinstance(result, int)
    assert result >= 0


def test_corpus_collection_name_length():
    from scripts.rag_utils import corpus_collection_name
    name = corpus_collection_name("intfloat/multilingual-e5-small", "ja", "finance")
    assert 3 <= len(name) <= 63
    assert name.startswith("c_")


def test_query_collection_name_length():
    from scripts.rag_utils import query_collection_name
    name = query_collection_name("intfloat/multilingual-e5-small", "ja", "finance")
    assert 3 <= len(name) <= 63
    assert name.startswith("q_")
