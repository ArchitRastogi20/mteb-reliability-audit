# tests/test_openai_embed_lib.py
import pytest
from scripts._openai_embed_lib import (
    truncate_to_tokens, estimate_cost_usd, EMBED_DIM,
)

def test_truncate_short_text_unchanged():
    text = "Quick brown fox."
    out = truncate_to_tokens(text, max_tokens=8192, model="text-embedding-3-large")
    assert out == text

def test_truncate_long_text_clipped():
    text = "word " * 20000  # ~20k tokens
    out = truncate_to_tokens(text, max_tokens=8192, model="text-embedding-3-large")
    # encode/decode round-trip should give us <= 8192 tokens
    import tiktoken
    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    assert len(enc.encode(out)) <= 8192

def test_estimate_cost_large():
    # 1M tokens on text-embedding-3-large = $0.13
    assert abs(estimate_cost_usd(1_000_000, "text-embedding-3-large") - 0.13) < 1e-6

def test_estimate_cost_small():
    assert abs(estimate_cost_usd(1_000_000, "text-embedding-3-small") - 0.02) < 1e-6

def test_embed_dim():
    assert EMBED_DIM["text-embedding-3-large"] == 3072
    assert EMBED_DIM["text-embedding-3-small"] == 1536
