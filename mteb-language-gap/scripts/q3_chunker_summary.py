# mteb-language-gap/scripts/q3_chunker_summary.py
"""Q3: nemotron-8b IT-Wikipedia chunker sweep summary.

Reads results/ita_lb_maxseq{128,256,512,1024}/llama_embed*/WikipediaRetrievalMultilingual.json
Outputs verdict: model-failure / preprocessing-artifact / preprocessing-sensitive.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "q3_chunker_summary.json"

VARIANTS = [128, 256, 512, 1024]

def get_score(d, key="ndcg_at_10"):
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() == key: return v
            r = get_score(v, key)
            if r is not None: return r
    elif isinstance(d, list):
        for x in d:
            r = get_score(x, key)
            if r is not None: return r
    return None

def main():
    rows = []
    for L in VARIANTS:
        cands = list((REPO_ROOT / f"results/ita_lb_maxseq{L}").glob("llama_embed*/WikipediaRetrievalMultilingual.json"))
        if not cands:
            rows.append({"max_seq_length": L, "ndcg_at_10": None, "note": "missing"})
            continue
        ndcg = get_score(json.loads(cands[0].read_text()))
        rows.append({"max_seq_length": L, "ndcg_at_10": float(ndcg) * 100 if ndcg else None})

    valid = [r["ndcg_at_10"] for r in rows if r["ndcg_at_10"] is not None]
    if not valid:
        verdict = "INSUFFICIENT_DATA"
    elif all(v < 50 for v in valid):
        verdict = "MODEL_FAILURE_CONFIRMED"
    elif any(v > 60 for v in valid):
        verdict = "PREPROCESSING_ARTIFACT"
    else:
        verdict = "PREPROCESSING_SENSITIVE"

    summary = {
        "model": "nvidia/llama-embed-nemotron-8b",
        "task": "WikipediaRetrievalMultilingual (Italian)",
        "paper_baseline_ndcg10": 46.3,
        "variants": rows,
        "verdict": verdict,
        "decision_rules": {
            "MODEL_FAILURE_CONFIRMED": "all 4 variants <50 NDCG@10 → keep §3.4 Mechanism B as hypothesis-only, robust to preprocessing",
            "PREPROCESSING_ARTIFACT": "any variant >60 NDCG@10 → relabel §3.4: not a hidden failure, drop from headline",
            "PREPROCESSING_SENSITIVE": "intermediate → report as weaker hypothesis with this sensitivity result",
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
