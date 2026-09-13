from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from tqdm.asyncio import tqdm

from src.client.async_openai import MINI, NANO, AsyncOpenAIClient
from src.models.query_models import KeypointExtraction, QARPair, RagAnswer


def _rag_answer_messages(question: str, references: list[str]) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] {ref}" for i, ref in enumerate(references))
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer the question using ONLY the provided "
                "reference passages. Do not add information from outside the passages. "
                "Answer in the same language as the question and references."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference passages:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Provide a concise, accurate answer based only on the references above."
            ),
        },
    ]


def _keypoint_messages(question: str, answer: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are extracting key facts from an answer. "
                "Break the answer into a list of atomic, self-contained factual statements. "
                "Each keypoint should be a single verifiable fact. "
                "Respond in the same language as the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Answer: {answer}\n\n"
                "Extract the key factual points from this answer as a list."
            ),
        },
    ]


def _checkpoint_path(lang: str, domain: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"answers_{lang}_{domain}.jsonl"


def _load_checkpoint(path: Path) -> dict[str, QARPair]:
    if not path.exists():
        return {}
    completed: dict[str, QARPair] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pair = QARPair(**json.loads(line))
                completed[pair.query_id] = pair
    return completed


async def _append_checkpoint(path: Path, pair: QARPair) -> None:
    """Write checkpoint using asyncio.to_thread to avoid blocking the event loop."""
    data = pair.model_dump_json()
    await asyncio.to_thread(_write_line, path, data)


def _write_line(path: Path, data: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(data + "\n")


async def _process_one(
    pair: QARPair,
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> QARPair:
    """Generate RAG answer + extract keypoints for a single answerable query."""
    # Step 1: Generate RAG answer from GT references (mini — quality matters)
    rag_response: RagAnswer = await client.complete(
        messages=_rag_answer_messages(pair.question, pair.references),
        model=MINI,
        response_format=RagAnswer,
        semaphore=semaphore,
    )

    # Step 2: Extract keypoints from expected answer (nano — simple extraction)
    kp_response: KeypointExtraction = await client.complete(
        messages=_keypoint_messages(pair.question, pair.answer),
        model=NANO,
        response_format=KeypointExtraction,
        semaphore=semaphore,
    )

    updated = pair.model_copy(update={
        "generated_answer": rag_response.content,
        "keypoints": kp_response.keypoints,
    })
    await _append_checkpoint(checkpoint_path, updated)
    return updated


async def run_stage5(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    queries: list[QARPair],
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    semaphore_limit: int = 30,
) -> list[QARPair]:
    """
    For each valid answerable query:
      - Generate a RAG answer using GT references as context (mini)
      - Extract keypoints from the expected answer (nano)
    Unanswerable and invalid queries pass through unchanged.
    """
    ckpt_path = _checkpoint_path(lang, domain, checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoint(ckpt_path)
    semaphore = asyncio.Semaphore(semaphore_limit)

    results: list[QARPair] = []
    pending: list[QARPair] = []

    for pair in queries:
        if pair.query_type == "unanswerable" or not pair.valid:
            results.append(pair)
        elif pair.query_id in completed:
            results.append(completed[pair.query_id])
        elif not pair.references:
            # No references to generate from — pass through
            results.append(pair)
        else:
            pending.append(pair)

    if pending:
        tasks = [_process_one(p, client, semaphore, ckpt_path) for p in pending]
        new_results = await tqdm.gather(*tasks, desc=f"Stage5 {lang}/{domain}")
        results.extend(new_results)

    return results
