from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from tqdm.asyncio import tqdm

from src.client.async_openai import NANO, AsyncOpenAIClient
from src.models.corpus_models import Document
from src.models.query_models import QARPair, RefinedReferences
from src.prompts import PromptSet
from src.prompts.finance_hi import FinanceHiPrompts
from src.prompts.finance_it import FinanceItPrompts
from src.prompts.finance_ja import FinanceJaPrompts
from src.prompts.law_hi import LawHiPrompts
from src.prompts.law_it import LawItPrompts
from src.prompts.law_ja import LawJaPrompts

_PROMPT_MAP: dict[tuple[str, str], PromptSet] = {
    ("ja", "finance"): FinanceJaPrompts(),
    ("hi", "finance"): FinanceHiPrompts(),
    ("it", "finance"): FinanceItPrompts(),
    ("ja", "law"): LawJaPrompts(),
    ("hi", "law"): LawHiPrompts(),
    ("it", "law"): LawItPrompts(),
}


def _checkpoint_path(lang: str, domain: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"queries_{lang}_{domain}_refined.jsonl"


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


def _append_checkpoint(path: Path, pair: QARPair) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(pair.model_dump_json() + "\n")


def _build_doc_index(documents: list[Document]) -> dict[str, Document]:
    return {doc.doc_id: doc for doc in documents}


async def _refine_one(
    pair: QARPair,
    doc: Document,
    prompts: PromptSet,
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> QARPair:
    messages = [
        {"role": "system", "content": prompts.refine_system},
        {"role": "user", "content": prompts.refine_user(pair.question, pair.answer, doc.content)},
    ]
    refined: RefinedReferences = await client.complete(
        messages=messages,
        model=NANO,
        response_format=RefinedReferences,
        semaphore=semaphore,
    )
    updated = pair.model_copy(update={"references": refined.references})
    _append_checkpoint(checkpoint_path, updated)
    return updated


async def run_stage4(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    queries: list[QARPair],
    documents: list[Document],
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    semaphore_limit: int = 50,
) -> list[QARPair]:
    """
    Refine references for answerable queries. Unanswerable queries pass through.
    Scope: factual + multi_hop + summarization only.
    """
    prompts = _PROMPT_MAP[(lang, domain)]
    ckpt_path = _checkpoint_path(lang, domain, checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    doc_index = _build_doc_index(documents)

    completed = _load_checkpoint(ckpt_path)
    semaphore = asyncio.Semaphore(semaphore_limit)

    results: list[QARPair] = []
    pending_pairs: list[QARPair] = []

    for pair in queries:
        if pair.query_type == "unanswerable":
            # Pass through unchanged — no refinement needed
            results.append(pair)
        elif pair.query_id in completed:
            results.append(completed[pair.query_id])
        else:
            pending_pairs.append(pair)

    if pending_pairs:
        tasks = [
            _refine_one(p, doc_index[p.doc_id], prompts, client, semaphore, ckpt_path)
            for p in pending_pairs
            if p.doc_id in doc_index
        ]
        refined = await tqdm.gather(*tasks, desc=f"Stage4 {lang}/{domain}")
        results.extend(refined)

    return results
