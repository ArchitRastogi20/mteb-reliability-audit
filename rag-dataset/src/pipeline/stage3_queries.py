from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Union

from tqdm.asyncio import tqdm

from src.client.async_openai import MINI, NANO, AsyncOpenAIClient
from src.models.config_models import FinanceConfig, LawConfig
from src.models.corpus_models import Document
from src.models.query_models import GeneratedQuery, QARPair, UnanswerableQuery
from src.prompts import PromptSet
from src.prompts.finance_hi import FinanceHiPrompts
from src.prompts.finance_it import FinanceItPrompts
from src.prompts.finance_ja import FinanceJaPrompts
from src.prompts.law_hi import LawHiPrompts
from src.prompts.law_it import LawItPrompts
from src.prompts.law_ja import LawJaPrompts

Config = Union[FinanceConfig, LawConfig]

_PROMPT_MAP: dict[tuple[str, str], PromptSet] = {
    ("ja", "finance"): FinanceJaPrompts(),
    ("hi", "finance"): FinanceHiPrompts(),
    ("it", "finance"): FinanceItPrompts(),
    ("ja", "law"): LawJaPrompts(),
    ("hi", "law"): LawHiPrompts(),
    ("it", "law"): LawItPrompts(),
}


def _checkpoint_path(lang: str, domain: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"queries_{lang}_{domain}.jsonl"


def _load_checkpoint(path: Path) -> dict[str, QARPair]:
    """Return mapping of query_id -> QARPair for already-completed queries."""
    if not path.exists():
        return {}
    completed: dict[str, QARPair] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                pair = QARPair(**item)
                completed[pair.query_id] = pair
    return completed


def _append_checkpoint(path: Path, pair: QARPair) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(pair.model_dump_json() + "\n")


def _entity_names(configs: list[Config]) -> list[str]:
    """Extract entity names from configs for unanswerable query generation."""
    names = []
    for cfg in configs:
        if isinstance(cfg, FinanceConfig):
            names.append(cfg.company_name)
        elif isinstance(cfg, LawConfig):
            names.append(cfg.case_name)
    return names


async def _generate_answerable(
    query_id: str,
    doc: Document,
    query_type: Literal["factual", "multi_hop", "summarization"],
    prompts: PromptSet,
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> QARPair:
    if query_type == "factual":
        user_prompt = prompts.factual_query_user(doc.content)
        model = NANO
    elif query_type == "multi_hop":
        user_prompt = prompts.multihop_query_user(doc.content)
        model = MINI
    else:  # summarization
        user_prompt = prompts.summarization_query_user(doc.content)
        model = MINI

    messages = [
        {"role": "system", "content": prompts.query_system},
        {"role": "user", "content": user_prompt},
    ]
    generated: GeneratedQuery = await client.complete(
        messages=messages,
        model=model,
        response_format=GeneratedQuery,
        semaphore=semaphore,
    )
    pair = QARPair(
        query_id=query_id,
        doc_id=doc.doc_id,
        language=doc.language,
        domain=doc.domain,
        query_type=query_type,
        question=generated.question,
        answer=generated.answer,
        references=generated.references,
    )
    _append_checkpoint(checkpoint_path, pair)
    return pair


async def _generate_unanswerable(
    query_id: str,
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    prompts: PromptSet,
    existing_entities: list[str],
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> QARPair:
    messages = [
        {"role": "system", "content": prompts.query_system},
        {"role": "user", "content": prompts.unanswerable_query_user(existing_entities)},
    ]
    generated: UnanswerableQuery = await client.complete(
        messages=messages,
        model=NANO,
        response_format=UnanswerableQuery,
        semaphore=semaphore,
    )
    pair = QARPair(
        query_id=query_id,
        doc_id=None,
        language=lang,
        domain=domain,
        query_type="unanswerable",
        question=generated.question,
        answer="",
        references=[],
    )
    _append_checkpoint(checkpoint_path, pair)
    return pair


async def run_stage3(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    documents: list[Document],
    configs: list[Config],
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    semaphore_limit: int = 40,
) -> list[QARPair]:
    """
    Generate queries for all 4 types:
      - factual:       docs[0:175]  → 175 queries (nano)
      - multi_hop:     docs[0:175]  → 175 queries (mini)
      - summarization: docs[150:250] → 100 queries (mini)
      - unanswerable:  no doc       → 50 queries  (nano)
    Total: 500 per domain per language.
    """
    prompts = _PROMPT_MAP[(lang, domain)]
    ckpt_path = _checkpoint_path(lang, domain, checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoint(ckpt_path)
    semaphore = asyncio.Semaphore(semaphore_limit)
    entity_names = _entity_names(configs)

    # Build all (query_id, task_coroutine) pairs
    planned: list[tuple[str, QARPair | None]] = []

    def _add(qid: str, coro_fn):
        if qid in completed:
            planned.append((qid, completed[qid]))
        else:
            planned.append((qid, coro_fn()))

    # Factual: docs 0–174
    for i, doc in enumerate(documents[:175]):
        qid = f"{lang}_{domain}_factual_{i:04d}"
        doc_ref = doc
        _add(qid, lambda d=doc_ref, q=qid: _generate_answerable(q, d, "factual", prompts, client, semaphore, ckpt_path))

    # Multi-hop: docs 0–174
    for i, doc in enumerate(documents[:175]):
        qid = f"{lang}_{domain}_multihop_{i:04d}"
        doc_ref = doc
        _add(qid, lambda d=doc_ref, q=qid: _generate_answerable(q, d, "multi_hop", prompts, client, semaphore, ckpt_path))

    # Summarization: last 100 docs (docs 150–249)
    for i, doc in enumerate(documents[150:250]):
        qid = f"{lang}_{domain}_summ_{i:04d}"
        doc_ref = doc
        _add(qid, lambda d=doc_ref, q=qid: _generate_answerable(q, d, "summarization", prompts, client, semaphore, ckpt_path))

    # Unanswerable: 50 queries
    for i in range(50):
        qid = f"{lang}_{domain}_unans_{i:04d}"
        _add(qid, lambda q=qid: _generate_unanswerable(q, lang, domain, prompts, entity_names, client, semaphore, ckpt_path))

    # Execute pending tasks
    pending_indices = [idx for idx, (_, v) in enumerate(planned) if not isinstance(v, QARPair)]
    pending_coros = [planned[idx][1] for idx in pending_indices]

    if pending_coros:
        new_results = await tqdm.gather(*pending_coros, desc=f"Stage3 {lang}/{domain}")
        for idx, result in zip(pending_indices, new_results):
            planned[idx] = (planned[idx][0], result)

    return [pair for _, pair in planned]
