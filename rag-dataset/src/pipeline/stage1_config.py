from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Union

from tqdm.asyncio import tqdm

from src.client.async_openai import NANO, AsyncOpenAIClient
from src.models.config_models import FinanceConfig, LawConfig
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

_CONFIG_MODEL: dict[str, type[Config]] = {
    "finance": FinanceConfig,
    "law": LawConfig,
}


def _checkpoint_path(lang: str, domain: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"configs_{lang}_{domain}.jsonl"


def _load_checkpoint(path: Path) -> dict[int, dict]:
    """Return mapping of index -> raw dict for already-completed items."""
    if not path.exists():
        return {}
    completed: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                completed[item["_index"]] = item
    return completed


def _append_checkpoint(path: Path, index: int, config: Config) -> None:
    data = {"_index": index, **config.model_dump()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def _generate_one(
    index: int,
    prompts: PromptSet,
    config_model: type[Config],
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
) -> Config:
    messages = [
        {"role": "system", "content": prompts.config_system},
        {"role": "user", "content": prompts.config_user(index)},
    ]
    config = await client.complete(
        messages=messages,
        model=NANO,
        response_format=config_model,
        semaphore=semaphore,
    )
    _append_checkpoint(checkpoint_path, index, config)
    return config


async def run_stage1(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    count: int,
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    semaphore_limit: int = 50,
) -> list[Config]:
    """Generate `count` configs for the given lang+domain, resuming from checkpoint."""
    prompts = _PROMPT_MAP[(lang, domain)]
    config_model = _CONFIG_MODEL[domain]
    ckpt_path = _checkpoint_path(lang, domain, checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoint(ckpt_path)
    results: list[Config | None] = [None] * count

    # Restore already-completed items
    for idx, raw in completed.items():
        if idx < count:
            raw_copy = {k: v for k, v in raw.items() if k != "_index"}
            results[idx] = config_model(**raw_copy)

    # Generate missing items
    missing = [i for i in range(count) if results[i] is None]
    if not missing:
        return [r for r in results if r is not None]

    semaphore = asyncio.Semaphore(semaphore_limit)
    tasks = [
        _generate_one(i, prompts, config_model, client, semaphore, ckpt_path)
        for i in missing
    ]
    new_results = await tqdm.gather(*tasks, desc=f"Stage1 {lang}/{domain}")
    for i, result in zip(missing, new_results):
        results[i] = result

    return [r for r in results if r is not None]
