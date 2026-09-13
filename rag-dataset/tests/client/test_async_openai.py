from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.client.async_openai import (
    MINI,
    NANO,
    AsyncOpenAIClient,
    BudgetExceededError,
    CostTracker,
)


# --- CostTracker tests ---

def test_cost_tracker_records_nano_cost():
    tracker = CostTracker(budget=25.0)
    cost = tracker.record_usage(NANO, input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 0.20) < 1e-9
    assert abs(tracker.total_cost - 0.20) < 1e-9


def test_cost_tracker_records_mini_cost():
    tracker = CostTracker(budget=25.0)
    cost = tracker.record_usage(MINI, input_tokens=0, output_tokens=1_000_000)
    assert abs(cost - 4.50) < 1e-9


def test_cost_tracker_accumulates():
    tracker = CostTracker(budget=25.0)
    tracker.record_usage(NANO, 1_000_000, 0)
    tracker.record_usage(NANO, 1_000_000, 0)
    assert abs(tracker.total_cost - 0.40) < 1e-9


def test_budget_exceeded_raises():
    tracker = CostTracker(budget=0.001)
    with pytest.raises(BudgetExceededError):
        tracker.record_usage(MINI, input_tokens=1_000_000, output_tokens=0)


def test_estimate_tokens_non_zero():
    tracker = CostTracker()
    count = tracker.estimate_tokens("Hello world")
    assert count > 0


def test_summary_format():
    tracker = CostTracker(budget=25.0)
    tracker.record_usage(NANO, 100, 50)
    summary = tracker.summary()
    assert "$" in summary
    assert "25.00" in summary


# --- AsyncOpenAIClient tests ---

class _SimpleModel(BaseModel):
    value: str


def _make_mock_response(parsed_obj: BaseModel, input_tokens: int = 100, output_tokens: int = 50):
    msg = MagicMock()
    msg.parsed = parsed_obj
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_complete_returns_parsed_model():
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    expected = _SimpleModel(value="hello")
    mock_response = _make_mock_response(expected)

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        sem = asyncio.Semaphore(5)
        result = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            model=NANO,
            response_format=_SimpleModel,
            semaphore=sem,
        )

    assert result.value == "hello"
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50


@pytest.mark.asyncio
async def test_complete_retries_on_failure():
    tracker = CostTracker()
    client = AsyncOpenAIClient(tracker)
    expected = _SimpleModel(value="retry_ok")
    mock_response = _make_mock_response(expected)

    call_count = 0

    async def flaky_parse(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient error")
        return mock_response

    with patch.object(client._client.beta.chat.completions, "parse", side_effect=flaky_parse):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            sem = asyncio.Semaphore(5)
            result = await client.complete(
                messages=[{"role": "user", "content": "test"}],
                model=NANO,
                response_format=_SimpleModel,
                semaphore=sem,
            )

    assert result.value == "retry_ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_complete_budget_exceeded_propagates():
    tracker = CostTracker(budget=0.000001)
    client = AsyncOpenAIClient(tracker)
    mock_response = _make_mock_response(_SimpleModel(value="x"), input_tokens=1_000_000)

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        sem = asyncio.Semaphore(5)
        with pytest.raises(BudgetExceededError):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                model=MINI,
                response_format=_SimpleModel,
                semaphore=sem,
            )
