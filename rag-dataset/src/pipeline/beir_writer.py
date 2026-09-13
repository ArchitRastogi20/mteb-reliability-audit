from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Literal

from src.models.corpus_models import Document
from src.models.query_models import QARPair


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that LLMs sometimes inject into generated articles."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)   # **bold** → bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)         # *italic* → italic
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ## Header → Header
    text = re.sub(r'\n{3,}', '\n\n', text)              # collapse excess blank lines
    return text.strip()


def write_beir_dataset(
    lang: Literal["ja", "hi"],
    domain: Literal["finance", "law"],
    documents: list[Document],
    queries: list[QARPair],
    output_dir: Path,
) -> None:
    """
    Write BEIR-format files:
      - corpus.jsonl  : {"_id": doc_id, "title": "", "text": content}
      - queries.jsonl : {"_id": query_id, "text": question}
      - qrels/test.tsv: query_id TAB 0 TAB doc_id TAB 1
                        (unanswerable queries have NO entry in qrels)
    Only queries with valid=True are written.
    """
    target_dir = output_dir / lang / domain
    qrels_dir = target_dir / "qrels"
    target_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    # corpus.jsonl — strip markdown so BM25 tokenization is clean
    corpus_path = target_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            record = {"_id": doc.doc_id, "title": "", "text": _strip_markdown(doc.content)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # queries.jsonl — only valid queries (BEIR format + extended metadata)
    valid_queries = [q for q in queries if q.valid]
    queries_path = target_dir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as f:
        for q in valid_queries:
            record = {"_id": q.query_id, "text": q.question}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # full_queries.jsonl — complete dataset for release (all metadata)
    full_path = target_dir / "full_queries.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for q in valid_queries:
            record = {
                "_id": q.query_id,
                "query_type": q.query_type,
                "language": q.language,
                "domain": q.domain,
                "question": q.question,
                "ground_truth": {
                    "doc_id": q.doc_id,
                    "answer": q.answer,
                    "references": q.references,
                    "keypoints": q.keypoints,
                },
                "prediction": {
                    "content": q.generated_answer,
                    "answer_valid": q.answer_valid,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # qrels/test.tsv — answerable valid queries only
    qrels_path = qrels_dir / "test.tsv"
    with qrels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for q in valid_queries:
            if q.query_type != "unanswerable" and q.doc_id is not None:
                writer.writerow([q.query_id, "0", q.doc_id, "1"])
