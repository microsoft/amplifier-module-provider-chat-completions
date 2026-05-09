"""Tests verifying that cost_usd is JSON-serializable at every emit boundary.

Covers:
  1. llm:response event payload is fully JSON-serializable for a known model
  2. cost_usd in the event payload is a str (not Decimal) for a known model
  3. cost_usd in the event payload is None for an unknown model
  4. cost_usd survives a json.dumps / json.loads round-trip unchanged
  5. Usage model stores cost_usd as Decimal internally (invariant docs)

These tests correspond to Bug 1 (emit boundary) fixed in __init__.py:
    usage_dict["cost_usd"] = str(_cost_usd) if _cost_usd is not None else None

The internal Decimal invariant (test 5) ensures we do NOT change
_convert_to_chat_response / model_copy — the fix is only at the emit boundary.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = FakeHooks()


def _make_provider(model: str = "gpt-4o", coordinator: object | None = None):
    """Build a ChatCompletionsProvider wired for non-streaming, zero-retry tests."""
    from amplifier_module_provider_chat_completions import ChatCompletionsProvider

    return ChatCompletionsProvider(
        config={
            "model": model,
            "use_streaming": "false",
            "max_retries": "0",
        },
        coordinator=coordinator,
    )


def _make_mock_completion(
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a minimal mock of an OpenAI ChatCompletion with an explicit model string.

    The *model* attribute on the mock drives compute_cost() inside _build_response().
    Without it (i.e. with a bare MagicMock), the model name is not in _RATES and
    cost_usd would always be None regardless of token counts.
    """
    message = MagicMock()
    message.content = "Test response"
    message.tool_calls = None
    message.reasoning_content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    usage.prompt_tokens_details = None  # no cache → cached_tokens = 0

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model  # KEY: real model string so compute_cost hits _RATES
    return response


# ---------------------------------------------------------------------------
# 1. llm:response event is fully JSON-serializable for a known model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_event_is_json_serializable_known_model():
    """llm:response payload must be fully JSON-serializable when a known model is used.

    gpt-4o is in _cost.py's _RATES table, so cost_usd will be a non-None Decimal
    internally. The fix serializes it to str before the event emit, making the
    entire payload safe for json.dumps().
    """
    from amplifier_core.message_models import ChatRequest, Message

    coordinator = FakeCoordinator()
    provider = _make_provider(model="gpt-4o", coordinator=coordinator)
    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_completion(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
    )

    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        model="gpt-4o",
    )
    await provider.complete(request)

    response_events = [
        e
        for e in coordinator.hooks.events
        if e[0] == "llm:response" and e[1].get("status") != "error"
    ]
    assert len(response_events) >= 1, "llm:response event must be emitted"
    _, payload = response_events[0]

    # Must not raise — this is the regression guard for the Decimal serialization bug.
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 2. cost_usd in the event payload is a str for a known model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_event_cost_usd_is_str_for_known_model():
    """cost_usd in the llm:response usage dict must be a str, not a raw Decimal.

    After the fix, the emit boundary converts Decimal → str so the value is
    JSON-safe. The str must also represent a positive monetary amount (i.e. the
    compute_cost() result was non-zero for the given token counts).
    """
    from amplifier_core.message_models import ChatRequest, Message

    coordinator = FakeCoordinator()
    provider = _make_provider(model="gpt-4o", coordinator=coordinator)
    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_completion(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
    )

    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        model="gpt-4o",
    )
    await provider.complete(request)

    response_events = [
        e
        for e in coordinator.hooks.events
        if e[0] == "llm:response" and e[1].get("status") != "error"
    ]
    _, payload = response_events[0]

    cost = payload["usage"].get("cost_usd")
    assert isinstance(cost, str), (
        f"cost_usd must be str at the emit boundary, got {type(cost).__name__!r}: {cost!r}"
    )
    assert Decimal(cost) > 0, f"cost_usd str must parse to a positive Decimal, got {cost!r}"


# ---------------------------------------------------------------------------
# 3. cost_usd is None for an unknown model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_event_cost_usd_is_none_for_unknown_model():
    """cost_usd must be None in the event payload when the model has no pricing entry.

    Unknown models return None from compute_cost() — DO NOT default to $0.00.
    The emit boundary must propagate None faithfully (not convert it to the
    string "None").
    """
    from amplifier_core.message_models import ChatRequest, Message

    coordinator = FakeCoordinator()
    provider = _make_provider(model="unknown-model-9999", coordinator=coordinator)
    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_completion(
        model="unknown-model-9999",
        prompt_tokens=100,
        completion_tokens=50,
    )

    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        model="unknown-model-9999",
    )
    await provider.complete(request)

    response_events = [
        e
        for e in coordinator.hooks.events
        if e[0] == "llm:response" and e[1].get("status") != "error"
    ]
    _, payload = response_events[0]

    cost = payload["usage"].get("cost_usd")
    assert cost is None, (
        f"cost_usd must be None for an unknown model, got {type(cost).__name__!r}: {cost!r}"
    )


# ---------------------------------------------------------------------------
# 4. cost_usd round-trips through json.dumps / json.loads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_event_cost_usd_round_trips_through_json():
    """cost_usd must survive a json.dumps → json.loads round-trip unchanged.

    After serialization and deserialization the value must still be the same
    string (downstream consumers that read from a JSON log must see the same
    numeric string they can pass to Decimal()).
    """
    from amplifier_core.message_models import ChatRequest, Message

    coordinator = FakeCoordinator()
    provider = _make_provider(model="gpt-4o", coordinator=coordinator)
    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_completion(
        model="gpt-4o",
        prompt_tokens=1_000,
        completion_tokens=200,
    )

    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        model="gpt-4o",
    )
    await provider.complete(request)

    response_events = [
        e
        for e in coordinator.hooks.events
        if e[0] == "llm:response" and e[1].get("status") != "error"
    ]
    _, payload = response_events[0]

    original_cost = payload["usage"].get("cost_usd")
    assert original_cost is not None, "Expected a non-None cost_usd for gpt-4o"

    # Round-trip
    round_tripped = json.loads(json.dumps(payload))
    rt_cost = round_tripped["usage"].get("cost_usd")

    assert rt_cost == original_cost, (
        f"cost_usd must survive json round-trip: before={original_cost!r}, after={rt_cost!r}"
    )
    # Still parses as a valid Decimal after the round-trip
    assert Decimal(rt_cost) > 0


# ---------------------------------------------------------------------------
# 5. Usage model stores cost_usd as Decimal internally (invariant)
# ---------------------------------------------------------------------------


def test_usage_model_stores_decimal_internally():
    """result.usage.cost_usd must be a Decimal, not a str, inside the ChatResponse.

    The serialization fix is applied only at the emit boundary (usage_dict for
    the llm:response event).  The internal ChatCompletionsChatResponse.usage
    object retains the raw Decimal from compute_cost() so that arithmetic
    (e.g. accumulation in _accumulate()) stays exact.
    """
    from amplifier_module_provider_chat_completions import ChatCompletionsProvider

    provider = ChatCompletionsProvider(
        config={
            "model": "gpt-4o",
            "use_streaming": "false",
            "max_retries": "0",
        }
    )

    mock_response = _make_mock_completion(
        model="gpt-4o",
        prompt_tokens=1_000,
        completion_tokens=200,
    )

    result = provider._build_response(mock_response)

    assert result.usage is not None, "Usage must be populated for a known model"
    assert result.usage.cost_usd is not None, (
        "cost_usd must be set for gpt-4o with non-zero tokens"
    )
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"Internal cost_usd must remain Decimal, got {type(result.usage.cost_usd).__name__!r}"
    )
    assert not isinstance(result.usage.cost_usd, float), (
        "cost_usd must be Decimal, not float — float arithmetic loses precision"
    )
