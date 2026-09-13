"""Per-config Spearman rho between log(param_count) and per-query-type
NDCG@10 for neural retrievers (BM25 excluded).

Source for per-(model, config, query_type) NDCG@10:
  Indic-Japanese-RAG-Dataset/output/evaluation/{model_dir}/{lang}_{domain}.json
  -> by_type.{factual, multi_hop, summarization}.ndcg_at_10

Source for params: union of
  more_exp_claude_10_may/analysis/roster_membership.csv  (curated tier)
  rag-deployment-benchmark/results/deployment_results.csv  (deployment-tractable)
The deployment-tractable set is the population the RAG eval was actually
run against, so it carries the param info for most eval directories.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import ROSTER_CSV, RAG_EVAL, RAG_CONFIGS, OUTPUTS

REPO = Path(__file__).resolve().parent.parent
DEPLOYMENT_CSV = REPO / "rag-deployment-benchmark" / "results" / "deployment_results.csv"

QUERY_TYPES = ["factual", "multi_hop", "summarization"]


def _params_to_millions(p) -> float:
    p = str(p).strip().lower()
    if p in ("unk", "nan", ""):
        return float("nan")
    if p.endswith("b"):
        return float(p[:-1]) * 1000
    if p.endswith("m"):
        return float(p[:-1])
    try:
        return float(p)
    except ValueError:
        return float("nan")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_dir_to_model(model_dir: str, params_tbl: pd.DataFrame) -> str | None:
    """Match an eval dir name like 'multilingual_e5_larg_571f3e' to a
    canonical model_id. Strategy: strip the trailing hash, normalise both
    sides, then look for the longest substring match into known model_ids."""
    base = re.sub(r"_[0-9a-f]{6}$", "", model_dir)
    nb = _norm(base)
    best = None; best_len = 0
    for mid in params_tbl["model_id"]:
        leaf = mid.split("/")[-1]
        nleaf = _norm(leaf)
        # require non-trivial overlap: min(len(nb), len(nleaf)) >= 6
        overlap = min(len(nb), len(nleaf))
        if overlap < 6:
            continue
        # check prefix containment in either direction
        if nb.startswith(nleaf[:overlap]) or nleaf.startswith(nb[:overlap]):
            if overlap > best_len:
                best = mid; best_len = overlap
    return best


def _load_eval(model_dir: Path, lang: str, domain: str) -> dict | None:
    f = model_dir / f"{lang}_{domain}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _read_model_id_from_dir(model_dir: Path) -> str | None:
    """Read the canonical model_id from the first JSON file found in the dir."""
    for f in sorted(model_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            mid = data.get("model_id")
            if mid:
                return str(mid)
        except Exception:
            continue
    return None


def _load_params_table() -> pd.DataFrame:
    """Union the curated roster and deployment-tractable param tables.
    Returns DataFrame with columns model_id, params_m."""
    parts = []
    if ROSTER_CSV.exists():
        a = pd.read_csv(ROSTER_CSV)
        a["params_m"] = a["params"].map(_params_to_millions)
        parts.append(a[["model_id", "params_m"]].dropna())
    if DEPLOYMENT_CSV.exists():
        # Skip the leading "# GPU: ..." comment lines
        b = pd.read_csv(DEPLOYMENT_CSV, comment="#")
        b["params_m"] = b["params"].map(_params_to_millions)
        parts.append(b[["model_id", "params_m"]].dropna())
    out = (pd.concat(parts, ignore_index=True)
             .drop_duplicates(subset="model_id", keep="first"))
    return out


def main():
    params_tbl = _load_params_table()
    params_lookup = dict(zip(params_tbl["model_id"], params_tbl["params_m"]))
    print(f"params table: {len(params_tbl)} unique models")

    # Build long-format frame: model_id, config, query_type, ndcg, params_m
    long_rows = []
    unmatched_dirs = []
    if not RAG_EVAL.exists():
        raise SystemExit(f"missing eval dir: {RAG_EVAL}")

    skip_dirs = {"bm25", "hybrid", "runs"}

    for model_dir in sorted(RAG_EVAL.iterdir()):
        if model_dir.name in skip_dirs:
            continue
        if not model_dir.is_dir():
            continue

        # Primary: read canonical model_id from inside the JSON file (always present)
        mid = _read_model_id_from_dir(model_dir)

        # Fallback: heuristic dir-name match
        if mid is None:
            mid = _match_dir_to_model(model_dir.name, params_tbl)

        if mid is None or mid not in params_lookup:
            unmatched_dirs.append(f"{model_dir.name} -> {mid}")
            continue

        params_m = params_lookup[mid]
        for lang, domain in RAG_CONFIGS:
            d = _load_eval(model_dir, lang, domain)
            if d is None:
                continue
            by_type = d.get("by_type") or {}
            for qt in QUERY_TYPES:
                cell = by_type.get(qt) or {}
                ndcg = cell.get("ndcg_at_10")
                if ndcg is None:
                    continue
                long_rows.append({
                    "model_id": mid, "model_dir": model_dir.name,
                    "config": f"{lang}_{domain}",
                    "query_type": qt,
                    "ndcg_at_10": float(ndcg),
                    "params_m": float(params_m),
                })

    if unmatched_dirs:
        print(f"unmatched dirs ({len(unmatched_dirs)}): {unmatched_dirs}")

    long_df = pd.DataFrame(long_rows)
    print(f"long-format rows: {len(long_df)}; "
          f"distinct models matched: {long_df['model_id'].nunique()}")

    rows = []
    for lang, domain in RAG_CONFIGS:
        cfg = f"{lang}_{domain}"
        row = {"config": cfg}
        for qt in QUERY_TYPES:
            sub = long_df[(long_df["config"] == cfg) & (long_df["query_type"] == qt)]
            if len(sub) < 4:
                row[qt] = float("nan"); row[f"{qt}_n"] = len(sub); continue
            x = np.log(sub["params_m"].values)
            y = sub["ndcg_at_10"].values
            rho, _ = spearmanr(x, y)
            row[qt] = round(float(rho), 3) if not math.isnan(rho) else float("nan")
            row[f"{qt}_n"] = len(sub)
        rows.append(row)

    out = pd.DataFrame(rows, columns=[
        "config",
        "factual", "factual_n",
        "multi_hop", "multi_hop_n",
        "summarization", "summarization_n",
    ])
    path = OUTPUTS / "exp13_params_vs_querytype.csv"
    out.to_csv(path, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
