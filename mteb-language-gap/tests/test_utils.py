import hashlib, re
import pytest
import numpy as np

def test_batch_iter_full_batches():
    from scripts.utils import batch_iter
    items = list(range(10))
    batches = list(batch_iter(items, 3))
    assert batches == [[0,1,2],[3,4,5],[6,7,8],[9]]

def test_batch_iter_exact_fit():
    from scripts.utils import batch_iter
    items = list(range(6))
    assert list(batch_iter(items, 3)) == [[0,1,2],[3,4,5]]

def test_batch_iter_empty():
    from scripts.utils import batch_iter
    assert list(batch_iter([], 10)) == []

def test_sanitize_model_id_uniqueness():
    from scripts.utils import sanitize_model_id
    ids = [
        "intfloat/multilingual-e5-small",
        "intfloat/e5-small-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-multilingual-gemma2",
    ]
    sanitized = [sanitize_model_id(i) for i in ids]
    assert len(set(sanitized)) == len(ids), "sanitized IDs must be unique"

def test_sanitize_model_id_chromadb_safe():
    from scripts.utils import sanitize_model_id
    for mid in [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-multilingual-gemma2",
    ]:
        s = sanitize_model_id(mid)
        assert len(s) <= 40, f"{mid} -> {s!r} is {len(s)} chars"
        assert re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', s), f"unsafe chars in {s!r}"

def test_collection_name_length():
    from scripts.utils import corpus_collection_name, query_collection_name
    long_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    long_task = "NLPJournalTitleIntroRetrieval"
    assert len(corpus_collection_name(long_model, long_task)) <= 63
    assert len(query_collection_name(long_model, long_task)) <= 63

def test_log_json(tmp_path):
    from scripts.utils import log_json
    p = tmp_path / "sub" / "out.json"
    log_json(p, {"key": "value", "num": 42})
    import json
    data = json.loads(p.read_text())
    assert data == {"key": "value", "num": 42}
