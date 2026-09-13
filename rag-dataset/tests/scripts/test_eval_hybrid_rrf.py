"""Tests for RRF fusion math."""
import pytest
from scripts.eval_hybrid_rrf import rrf_fuse


def test_rrf_two_runs_top1_each():
    """If two retrievers each rank one doc as #1, RRF score = 2/(60+1) = 0.0328."""
    bm25 = {"q1": {"docA": 5.0, "docB": 4.0}}
    dense = {"q1": {"docA": 0.9, "docB": 0.8}}
    fused = rrf_fuse([bm25, dense], k=60)
    assert "q1" in fused
    # docA is #1 in both → score = 1/(60+1) + 1/(60+1) = 2/61
    assert fused["q1"]["docA"] == pytest.approx(2 / 61, rel=1e-6)
    # docB is #2 in both → 2/(60+2) = 2/62
    assert fused["q1"]["docB"] == pytest.approx(2 / 62, rel=1e-6)
    # docA should outrank docB
    assert fused["q1"]["docA"] > fused["q1"]["docB"]


def test_rrf_disagreement():
    """Doc only in one run still scored, but lower than docs in both."""
    bm25 = {"q1": {"docA": 5.0}}  # docA #1 in BM25 only
    dense = {"q1": {"docB": 0.9}}  # docB #1 in dense only
    fused = rrf_fuse([bm25, dense], k=60)
    # Both docs only appear in one run → both score 1/61
    assert fused["q1"]["docA"] == pytest.approx(1 / 61, rel=1e-6)
    assert fused["q1"]["docB"] == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_empty():
    """Empty queries are handled gracefully."""
    assert rrf_fuse([{}, {}], k=60) == {}


def test_rrf_k_parameter():
    """Different k values change the relative weights."""
    bm25 = {"q1": {"docA": 5.0, "docB": 4.0, "docC": 3.0}}
    dense = {"q1": {"docA": 0.9, "docB": 0.8, "docC": 0.7}}
    fused_k1 = rrf_fuse([bm25, dense], k=1)
    fused_k60 = rrf_fuse([bm25, dense], k=60)
    # k=1 amplifies top ranks more than k=60
    ratio_k1 = fused_k1["q1"]["docA"] / fused_k1["q1"]["docC"]
    ratio_k60 = fused_k60["q1"]["docA"] / fused_k60["q1"]["docC"]
    assert ratio_k1 > ratio_k60
