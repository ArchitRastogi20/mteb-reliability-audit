# tests/test_eval_belebele_openai.py
import pytest
from scripts.eval_openai_embed import load_belebele

def test_load_belebele_three_languages():
    """Belebele loader returns dict with 3 languages, each with corpus + queries + qrels."""
    data = load_belebele(["ita_Latn", "jpn_Jpan", "hin_Deva"])
    assert set(data.keys()) == {"ita_Latn", "jpn_Jpan", "hin_Deva"}
    for lang, d in data.items():
        assert "corpus" in d and "queries" in d and "qrels" in d
        # Belebele Retrieval per-language is ~488 unique passages × 900 queries
        assert len(d["corpus"]) >= 400, f"{lang} corpus too small"
        assert len(d["queries"]) >= 800, f"{lang} queries too small"
        assert len(d["qrels"]) == len(d["queries"])
