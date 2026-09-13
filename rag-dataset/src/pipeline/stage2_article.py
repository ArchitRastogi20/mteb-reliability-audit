from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Union

from tqdm.asyncio import tqdm

from src.client.async_openai import MINI, AsyncOpenAIClient
from src.models.config_models import FinanceConfig, LawConfig
from src.models.corpus_models import Document, GeneratedArticle
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
    return checkpoint_dir / f"articles_{lang}_{domain}.jsonl"


def _load_checkpoint(path: Path) -> dict[int, Document]:
    if not path.exists():
        return {}
    completed: dict[int, Document] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                idx = item.pop("_index")
                completed[idx] = Document(**item)
    return completed


def _append_checkpoint(path: Path, index: int, doc: Document) -> None:
    data = {"_index": index, **doc.model_dump()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def _generate_one(
    index: int,
    config: Config,
    lang: str,
    domain: str,
    prompts: PromptSet,
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> Document:
    messages = [
        {"role": "system", "content": prompts.article_system},
        {"role": "user", "content": prompts.article_user(config)},
    ]
    generated = await client.complete(
        messages=messages,
        model=MINI,
        response_format=GeneratedArticle,
        semaphore=semaphore,
    )
    doc = Document(
        doc_id=f"{lang}_{domain}_{index:04d}",
        language=lang,  # type: ignore[arg-type]
        domain=domain,  # type: ignore[arg-type]
        content=generated.content,
    )
    _append_checkpoint(checkpoint_path, index, doc)
    return doc


async def run_stage2(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    configs: list[Config],
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    semaphore_limit: int = 30,
) -> list[Document]:
    """Generate one article per config, resuming from checkpoint."""
    prompts = _PROMPT_MAP[(lang, domain)]
    ckpt_path = _checkpoint_path(lang, domain, checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoint(ckpt_path)
    results: list[Document | None] = [None] * len(configs)

    for idx, doc in completed.items():
        if idx < len(configs):
            results[idx] = doc

    missing = [i for i in range(len(configs)) if results[i] is None]
    if not missing:
        return [r for r in results if r is not None]

    semaphore = asyncio.Semaphore(semaphore_limit)
    tasks = [
        _generate_one(i, configs[i], lang, domain, prompts, client, semaphore, ckpt_path)
        for i in missing
    ]
    new_docs = await tqdm.gather(*tasks, desc=f"Stage2 {lang}/{domain}")
    for i, doc in zip(missing, new_docs):
        results[i] = doc

    return [r for r in results if r is not None]
