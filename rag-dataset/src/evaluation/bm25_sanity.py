from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi


def _tokenize(text: str, lang: str = "ja") -> list[str]:
    """
    Language-aware tokenizer.
    Hindi uses spaces between words — word-level tokenization preserves meaning.
    Japanese/Chinese have no spaces — character-level is the only option.
    """
    if lang in ("hi", "it"):
        return text.split()
    return list(text)


def _compute_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute NDCG@k and Recall@k using pytrec_eval."""
    try:
        import pytrec_eval  # type: ignore[import]
    except ImportError:
        raise ImportError("Install pytrec-eval-terrier: pip install pytrec-eval-terrier")

    measures = {f"ndcg_cut_{k}" for k in k_values} | {f"recall_{k}" for k in k_values}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)

    aggregated: dict[str, float] = {}
    for measure in measures:
        scores = [results[qid].get(measure, 0.0) for qid in results]
        aggregated[measure] = sum(scores) / len(scores) if scores else 0.0
    return aggregated


def run_bm25_sanity(
    lang: str,
    domain: str,
    output_dir: Path,
    evaluation_dir: Path,
    k_values: list[int] | None = None,
) -> dict:
    """
    Load BEIR files from output_dir/lang/domain/, run BM25, compute metrics,
    write evaluation_dir/bm25_sanity_{lang}_{domain}.json, and return results.
    """
    if k_values is None:
        k_values = [10]

    dataset_dir = output_dir / lang / domain
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / "test.tsv"

    # Load corpus
    corpus: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus[rec["_id"]] = rec["text"]

    # Load queries
    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            queries[rec["_id"]] = rec["text"]

    # Load qrels — only answerable queries have entries
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                qid, _, doc_id, score = parts
                qrels.setdefault(qid, {})[doc_id] = int(score)

    # Filter queries to only those with qrels (answerable only)
    answerable_qids = [qid for qid in queries if qid in qrels]
    if not answerable_qids:
        return {"error": "No answerable queries found in qrels"}

    # Build BM25 index
    doc_ids = list(corpus.keys())
    tokenized_corpus = [_tokenize(corpus[did], lang) for did in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    # Retrieve top-100 for each answerable query
    run: dict[str, dict[str, float]] = {}
    top_k = max(k_values) * 10  # retrieve more than needed for recall
    for qid in answerable_qids:
        tokenized_query = _tokenize(queries[qid], lang)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        run[qid] = {doc_ids[i]: float(scores[i]) for i in top_indices}

    # Compute metrics
    metrics = _compute_metrics(run, {qid: qrels[qid] for qid in answerable_qids}, k_values)

    # Interpret NDCG@10
    ndcg10 = metrics.get("ndcg_cut_10", 0.0)
    if ndcg10 > 0.7:
        interpretation = "too_easy"
    elif ndcg10 < 0.1:
        interpretation = "broken"
    else:
        interpretation = "healthy"

    result = {
        "lang": lang,
        "domain": domain,
        "num_docs": len(corpus),
        "num_answerable_queries": len(answerable_qids),
        "metrics": metrics,
        "ndcg10_interpretation": interpretation,
    }

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    out_path = evaluation_dir / f"bm25_sanity_{lang}_{domain}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[BM25] {lang}/{domain}: NDCG@10={ndcg10:.4f} ({interpretation})")
    return result
