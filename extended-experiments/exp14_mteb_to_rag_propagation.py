"""Per-config Spearman rho between MTEB-aggregate rank and
"RAG-coverage" rank, for all 6 RAG configs.

Signal choice per config:
  - If a model has judged keypoint-coverage data in
    analysis_output/stats/downstream_rag.json under
    'detailed.{provider}/{model}/{config}', use the average coverage rate
    as the RAG signal (this is the strict "downstream keypoint coverage").
  - Otherwise fall back to dense-retrieval overall.ndcg_at_10 from
    Indic-Japanese-RAG-Dataset/output/evaluation/{model_dir}/{lang}_{domain}.json
    -- the higher this is, the better the retriever served the answerer.
The output table records which signal each config used.

A rho near 1 means MTEB-agg predicts the chosen RAG signal well; rho near 0
or negative means the per-config inversion propagates downstream.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

from _common import (
    ROSTER_CSV, RAG_EVAL, RAG_CONFIGS, DOWNSTREAM_JSON, OUTPUTS,
)

REPO = Path(__file__).resolve().parent.parent
MTEB_AGG_CSV = REPO / "mteb-language-gap" / "config" / "mteb_agg_scores.csv"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_dir_to_model(model_dir: str, candidates: list[str]) -> str | None:
    base = re.sub(r"_[0-9a-f]{6}$", "", model_dir)
    nb = _norm(base)
    best = None; best_len = 0
    for mid in candidates:
        leaf = mid.split("/")[-1]
        nleaf = _norm(leaf)
        overlap = min(len(nb), len(nleaf))
        if overlap < 6:
            continue
        if nb.startswith(nleaf[:overlap]) or nleaf.startswith(nb[:overlap]):
            if overlap > best_len:
                best = mid; best_len = overlap
    return best


def _judged_coverage(downstream: dict, lang: str, domain: str) -> dict[str, float]:
    """Return {model_id_leaf: avg coverage rate} for one config from the
    downstream JSON's `detailed` map, if present."""
    out: dict[str, float] = {}
    cfg_suffix = f"/{lang}_{domain}"
    detailed = downstream.get("detailed", {})
    for key, entry in detailed.items():
        if not key.endswith(cfg_suffix):
            continue
        # key format: "{provider}/{model}/{config}" -- take model leaf
        parts = key.split("/")
        if len(parts) < 3:
            continue
        model_leaf = parts[1]
        per_q = entry.get("per_query") or []
        if not per_q:
            continue
        rates = []
        for q in per_q:
            covered = q.get("covered") or q.get("correct")
            total   = q.get("total")   or q.get("n_keypoints")
            if covered is None or not total:
                continue
            rates.append(covered / total)
        if rates:
            out[model_leaf] = sum(rates) / len(rates)
    return out


def _dense_ndcg(lang: str, domain: str) -> dict[str, float]:
    """Per-model overall NDCG@10 from existing eval JSONs (excluding bm25)."""
    out: dict[str, float] = {}
    if not RAG_EVAL.exists():
        return out
    for model_dir in RAG_EVAL.iterdir():
        if model_dir.name == "bm25":
            continue
        f = model_dir / f"{lang}_{domain}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        overall = d.get("overall") or {}
        n = overall.get("ndcg_at_10")
        if n is not None:
            out[model_dir.name] = float(n)
    return out


def main():
    roster = pd.read_csv(ROSTER_CSV)
    agg = pd.read_csv(MTEB_AGG_CSV).set_index("model_id")["mteb_agg"]
    downstream = json.loads(DOWNSTREAM_JSON.read_text())

    rows = []
    for lang, domain in RAG_CONFIGS:
        signal = "dense_ndcg_at_10"
        cfg = f"{lang}_{domain}"

        # First-choice signal: judged coverage (only present for a tiny subset)
        judged = _judged_coverage(downstream, lang, domain)

        # Fallback signal: dense-retrieval overall NDCG@10
        dense = _dense_ndcg(lang, domain)

        # Build joined frame (model_id -> mteb_agg, rag_signal)
        # Match eval dir names to mteb_agg.index via the substring heuristic.
        candidates = list(agg.index)
        rag_by_mid: dict[str, float] = {}
        for dir_name, val in dense.items():
            mid = _match_dir_to_model(dir_name, candidates)
            if mid is not None and mid not in rag_by_mid:
                rag_by_mid[mid] = val

        # If judged coverage covers more models for this config than dense,
        # prefer it (this is rare but possible for closed-API models).
        if len(judged) > len(rag_by_mid):
            rag_by_mid = {}
            for leaf, val in judged.items():
                # judged keys are like "text-embedding-3-large" -- match
                # against canonical agg.index by leaf
                mid = _match_dir_to_model(leaf, candidates)
                if mid is not None:
                    rag_by_mid[mid] = val
            signal = "judged_keypoint_coverage"

        common = [m for m in rag_by_mid if m in agg.index]
        n = len(common)
        if n < 4:
            rows.append({
                "config": cfg, "rho": None, "n_models": n,
                "propagates": None, "signal": signal,
            })
            continue
        mteb_scores = agg.loc[common].values
        rag_scores  = pd.Series(rag_by_mid).loc[common].values
        rho, _ = spearmanr(mteb_scores, rag_scores)
        rows.append({
            "config": cfg,
            "rho": round(float(rho), 3),
            "n_models": n,
            "propagates": "no" if rho is not None and rho >= 0.5 else "yes",
            "signal": signal,
        })

    df = pd.DataFrame(rows)
    path = OUTPUTS / "exp14_mteb_to_rag_propagation.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
