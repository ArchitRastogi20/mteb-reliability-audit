from __future__ import annotations

import asyncio
import time
from typing import TypeVar, Type

import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

NANO = "gpt-5.4-nano"
MINI = "gpt-5.4-mini"

_PRICING: dict[str, dict[str, float]] = {
    NANO: {"input": 0.20, "output": 1.25},
    MINI: {"input": 0.75, "output": 4.50},
}


class BudgetExceededError(Exception):
    pass


class CostTracker:
    def __init__(self, budget: float = 25.0) -> None:
        self.budget = budget
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._enc = tiktoken.get_encoding("o200k_base")

    def estimate_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = _PRICING.get(model, _PRICING[MINI])
        cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if self.total_cost > self.budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.4f} > ${self.budget:.2f}"
            )
        return cost

    def summary(self) -> str:
        return (
            f"Total cost: ${self.total_cost:.4f} / ${self.budget:.2f} | "
            f"Tokens in: {self.total_input_tokens:,} out: {self.total_output_tokens:,}"
        )


class AsyncOpenAIClient:
    def __init__(self, cost_tracker: CostTracker, api_key: str | None = None) -> None:
        import os
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CHATGPT_API_KEY")
            or "not-set"
        )
        self._client = AsyncOpenAI(api_key=resolved_key)
        self.cost_tracker = cost_tracker

    async def complete(
        self,
        messages: list[dict],
        model: str,
        response_format: Type[T],
        semaphore: asyncio.Semaphore,
        max_retries: int = 5,
    ) -> T:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await self._client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                    )
                usage = response.usage
                self.cost_tracker.record_usage(
                    model, usage.prompt_tokens, usage.completion_tokens
                )
                return response.choices[0].message.parsed
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 16.0)
        raise last_exc
