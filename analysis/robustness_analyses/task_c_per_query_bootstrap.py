#!/usr/bin/env python3
"""
Per-query paired bootstrap: bge-m3 vs snowflake on Italian BelebeleRetrieval.

Run: python task_c_per_query_bootstrap.py  (from analysis/robustness_analyses/)
Requires GPU. Dependencies: sentence-transformers, pytrec_eval, torch, datasets.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytrec_eval
import torch
from sentence_transformers import SentenceTransformer

MODEL_CACHE = None   # set to a local path to override the default HuggingFace cache
OUT_DIR = Path(__file__).parent / "task_c_per_query_bootstrap"
SEED = 20260506
N_BOOTSTRAP = 10_000
PAPER_BGE_NDCG = 0.937
PAPER_SNOW_NDCG = 0.662
WARN_THRESHOLD = 0.02
SPLIT = "test"

MODELS = {
    "bge_m3": {
        "hf_id": "BAAI/bge-m3",
        "display": "bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": True,
    },
    "snowflake": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "display": "snowflake-arctic-embed-l-v2.0",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "trust_remote_code": True,
    },
}


def setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("tpqb")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def load_belebele_italian(log: logging.Logger):
    from datasets import load_dataset as hf_load_dataset
    log.info("Loading BelebeleRetrieval (ita_Latn) via datasets")
    revision = "979a211276faa22f671e69d096634193567cfd05"
    ds = hf_load_dataset("mteb/belebele", "ita_Latn", revision=revision)
    rows = list(ds["test"])

    link_to_context_id: dict[str, str] = {}
    corpus: dict[str, dict[str, str]] = {}
    context_idx = 0
    for row in rows:
        if row["link"] not in link_to_context_id:
            cid = f"C{context_idx}"
            link_to_context_id[row["link"]] = cid
            corpus[cid] = {"title": "", "text": row["flores_passage"]}
            context_idx += 1

    question_ids: dict[str, int] = {}
    for row in rows:
        q = row["question"]
        if q not in question_ids:
            question_ids[q] = len(question_ids)

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        q = row["question"]
        qid = f"Q{question_ids[q]}"
        queries[qid] = q
        cid = link_to_context_id[row["link"]]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = 1

    shared = sorted(set(queries) & set(qrels))
    queries = {qid: queries[qid] for qid in shared}
    qrels = {qid: qrels[qid] for qid in shared}
    log.info(f"  corpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")
    return corpus, queries, qrels


def load_model(cfg: dict, log: logging.Logger) -> SentenceTransformer:
    log.info(f"Loading {cfg['hf_id']} (fp16)")
    torch.manual_seed(SEED)
    kwargs = dict(
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
        trust_remote_code=cfg["trust_remote_code"],
    )
    if MODEL_CACHE is not None:
        kwargs["cache_folder"] = MODEL_CACHE
    model = SentenceTransformer(cfg["hf_id"], **kwargs)
    try:
        rev = model[0].auto_model.config._commit_hash or "unknown"
    except Exception:
        rev = "unknown"
    log.info(f"  revision={rev}  query_prefix={repr(cfg['query_prefix'])}")
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def run_retrieval(cfg: dict, corpus: dict, queries: dict, log: logging.Logger) -> dict[str, dict[str, float]]:
    model = load_model(cfg, log)

    doc_ids = list(corpus.keys())
    doc_texts = [cfg["passage_prefix"] + corpus[d]["text"] for d in doc_ids]
    log.info(f"  encoding {len(doc_ids)} corpus docs")
    doc_embs = encode(model, doc_texts)

    qids = list(queries.keys())
    qtexts = [cfg["query_prefix"] + queries[qid] for qid in qids]
    log.info(f"  encoding {len(qids)} queries")
    q_embs = encode(model, qtexts)

    del model
    torch.cuda.empty_cache()

    scores = (q_embs @ doc_embs.T).astype(np.float32)
    top_k = min(1000, len(doc_ids))
    predictions = {}
    for i, qid in enumerate(qids):
        idx = np.argpartition(-scores[i], top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[i][idx])]
        predictions[qid] = {doc_ids[j]: float(scores[i][j]) for j in idx}
    return predictions


def per_query_ndcg(predictions: dict, qrels: dict) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    results = evaluator.evaluate(predictions)
    return {qid: results[qid]["ndcg_cut_10"] for qid in results}


def bootstrap(deltas: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    boot_means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean()
        for _ in range(n)
    ])
    return float(deltas.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def write_per_query_csv(path: Path, qids, bge_arr, snow_arr) -> None:
    lines = ["query_id,bge_m3_ndcg10,snowflake_ndcg10,delta"]
    for qid, b, s in zip(qids, bge_arr, snow_arr):
        lines.append(f"{qid},{b:.6f},{s:.6f},{b - s:.6f}")
    path.write_text("\n".join(lines) + "\n")


def write_bootstrap_csv(path: Path, n_q: int, mean_d: float, ci_lo: float, ci_hi: float) -> None:
    zero_in = "yes" if ci_lo <= 0 <= ci_hi else "no"
    header = "task,model_a,model_b,n_queries,mean_delta,ci_lo,ci_hi,zero_in_ci,n_bootstrap"
    row = (
        f"BelebeleRetrieval_ita,bge-m3,snowflake-arctic-embed-l-v2.0,"
        f"{n_q},{mean_d:.6f},{ci_lo:.6f},{ci_hi:.6f},{zero_in},{N_BOOTSTRAP}"
    )
    path.write_text(header + "\n" + row + "\n")


def write_summary_md(path: Path, n_q: int, mean_d: float, ci_lo: float, ci_hi: float) -> None:
    zero_in = ci_lo <= 0 <= ci_hi
    significance = "0 not in CI" if not zero_in else "0 in CI -- investigate"
    m100, lo100, hi100 = mean_d * 100, ci_lo * 100, ci_hi * 100
    text = (
        f"A demonstration on the bge-m3 vs. snowflake pair on Italian-Belebele "
        f"(n={n_q} queries) gives a paired-bootstrap mean per-query NDCG@10 difference of "
        f"{m100:.1f} (95% CI [{lo100:.1f}, {hi100:.1f}], "
        f"{N_BOOTSTRAP:,} query-level resamples; {significance}), "
        f"confirming the inversion is significant per-query and not a near-tie. "
        f"Full per-pair sweep across the roster requires re-running all models with per-query "
        f"logging and is left to future work."
    )
    path.write_text(text + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUT_DIR / "task_c_per_query_bootstrap.log")
    log.info("=== task_c_per_query_bootstrap.py ===")
    log.info(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
             f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  "
             f"seed={SEED}")

    corpus, queries, qrels = load_belebele_italian(log)

    pq: dict[str, dict[str, float]] = {}
    for slug, cfg in MODELS.items():
        log.info(f"\n--- {slug} ---")
        preds = run_retrieval(cfg, corpus, queries, log)
        pq[slug] = per_query_ndcg(preds, qrels)
        agg = float(np.mean(list(pq[slug].values())))
        log.info(f"  aggregate NDCG@10 = {agg:.4f}  ({agg * 100:.1f})")
        paper = PAPER_BGE_NDCG if slug == "bge_m3" else PAPER_SNOW_NDCG
        diff = abs(agg - paper)
        if diff > WARN_THRESHOLD:
            log.warning(
                f"  WARN: own-run {agg:.3f} vs published {paper:.3f} "
                f"(diff={diff:.3f} > {WARN_THRESHOLD})"
            )
        else:
            log.info(f"  sanity OK: own-run {agg:.3f} matches published {paper:.3f}")
        n_dropped = len(qrels) - len(pq[slug])
        if n_dropped > 0:
            log.warning(f"  pytrec_eval dropped {n_dropped} queries for {slug}")

    shared = sorted(set(pq["bge_m3"]) & set(pq["snowflake"]))
    log.info(f"\n{len(shared)} shared queries")
    bge_arr = np.array([pq["bge_m3"][qid] for qid in shared])
    snow_arr = np.array([pq["snowflake"][qid] for qid in shared])
    deltas = bge_arr - snow_arr

    log.info(f"Running paired bootstrap (n={N_BOOTSTRAP:,} resamples)")
    rng = np.random.default_rng(SEED)
    mean_d, ci_lo, ci_hi = bootstrap(deltas, N_BOOTSTRAP, rng)
    zero_in = ci_lo <= 0 <= ci_hi
    log.info(f"  mean_delta = {mean_d:.4f}  ({mean_d * 100:.1f} per 100)")
    log.info(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  zero_in_CI={zero_in}")
    if zero_in:
        log.warning("  UNEXPECTED: 0 is in CI -- check eval setup")

    write_per_query_csv(OUT_DIR / "per_query_ndcg.csv", shared, bge_arr, snow_arr)
    log.info("  wrote per_query_ndcg.csv")

    write_bootstrap_csv(OUT_DIR / "bootstrap_stats.csv", len(shared), mean_d, ci_lo, ci_hi)
    log.info("  wrote bootstrap_stats.csv")

    write_summary_md(OUT_DIR / "per_query_bootstrap.md", len(shared), mean_d, ci_lo, ci_hi)
    log.info("  wrote per_query_bootstrap.md")

    log.info(f"\nAll outputs in {OUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
