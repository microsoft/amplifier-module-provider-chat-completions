"""Tests for _accumulate hook and register_contributor in mount().

Verifies that mount() registers:
  - an `llm:response` hook (_accumulate) that sums cost_usd into a closure-captured dict
  - a lazy contributor callback on session.cost channel under name 'provider-chat-completions'

Also verifies cache_read_tokens extraction from prompt_tokens_details.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_chat_completions import mount


# ---------------------------------------------------------------------------
# Mock coordinator fixture
# ---------------------------------------------------------------------------


class _MockHooks:
    def __init__(self):
        self._handlers: dict = {}

    def register(self, event: str, handler) -> None:
        self._handlers[event] = handler

    async def emit(self, event: str, data: dict) -> None:
        if event in self._handlers:
            await self._handlers[event](event, data)


class _MockCoordinator:
    def __init__(self):
        self.hooks = _MockHooks()
        self.registered_hooks = self.hooks._handlers  # shared reference
        self.registered_contributors: dict = {}

    async def mount(self, *args, **kwargs) -> None:
        pass

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.registered_contributors[(channel, name)] = callback

    def get_capability(self, *args, **kwargs):
        return None


@pytest.fixture
def mock_coordinator():
    return _MockCoordinator()


# ---------------------------------------------------------------------------
# test_contributor_registered_at_mount
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_registered_at_mount(mock_coordinator, monkeypatch):
    """mount() must register a contributor on ('session.cost', 'provider-chat-completions')."""
    monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:8080/v1")
    await mount(mock_coordinator, config={})
    assert (
        "session.cost",
        "provider-chat-completions",
    ) in mock_coordinator.registered_contributors


# ---------------------------------------------------------------------------
# test_contributor_returns_none_before_any_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_returns_none_before_any_calls(mock_coordinator, monkeypatch):
    """Contributor callback returns None when no llm:response events have fired."""
    monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:8080/v1")
    await mount(mock_coordinator, config={})
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-chat-completions")
    ]
    assert callback() is None


# ---------------------------------------------------------------------------
# test_contributor_accumulates_after_llm_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_accumulates_after_llm_response(
    mock_coordinator, monkeypatch
):
    """_accumulate sums cost_usd over multiple events; callback returns Decimal total."""
    monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:8080/v1")
    await mount(mock_coordinator, config={})

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-chat-completions")
    ]

    await accumulate("llm:response", {"provider": "chat-completions", "usage": {"cost_usd": "0.05"}})
    await accumulate("llm:response", {"provider": "chat-completions", "usage": {"cost_usd": "0.03"}})

    result = callback()
    assert result is not None, "Callback should return a dict after cost events"
    assert "cost_usd" in result
    assert result["cost_usd"] == Decimal("0.08"), (
        f"Expected Decimal('0.08'), got {result['cost_usd']!r}"
    )
    assert isinstance(result["cost_usd"], Decimal), (
        f"cost_usd must be Decimal, got {type(result['cost_usd'])}"
    )


# ---------------------------------------------------------------------------
# test_contributor_ignores_none_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_ignores_none_cost(mock_coordinator, monkeypatch):
    """_accumulate ignores events where cost_usd is None; has_data stays False."""
    monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:8080/v1")
    await mount(mock_coordinator, config={})

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-chat-completions")
    ]

    await accumulate("llm:response", {"provider": "chat-completions", "usage": {"cost_usd": None}})

    assert callback() is None, (
        "Callback should still return None after a None-cost event"
    )


# ---------------------------------------------------------------------------
# test_cache_read_tokens_extracted_from_prompt_tokens_details
# (EXTRA step - unique to chat-completions)
# ---------------------------------------------------------------------------


def test_cache_read_tokens_extracted_from_prompt_tokens_details():
    """Usage.cache_read_tokens is populated when prompt_tokens_details.cached_tokens present."""
    from amplifier_module_provider_chat_completions import ChatCompletionsProvider

    provider = ChatCompletionsProvider(
        config={"model": "test-model", "use_streaming": "false", "max_retries": "0"}
    )

    # Build a mock response with prompt_tokens_details.cached_tokens
    prompt_tokens_details = MagicMock()
    prompt_tokens_details.cached_tokens = 500

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 1000
    mock_usage.completion_tokens = 200
    mock_usage.total_tokens = 1200
    mock_usage.prompt_tokens_details = prompt_tokens_details

    mock_message = MagicMock()
    mock_message.content = "Hello"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "test-model"

    result = provider._build_response(mock_response)

    assert result.usage is not None
    assert result.usage.cache_read_tokens == 500, (
        f"Expected cache_read_tokens=500, got {result.usage.cache_read_tokens!r}"
    )


# ---------------------------------------------------------------------------
# test_cache_read_tokens_none_when_absent
# ---------------------------------------------------------------------------


def test_cache_read_tokens_none_when_absent():
    """Usage.cache_read_tokens is None when prompt_tokens_details is absent."""
    from amplifier_module_provider_chat_completions import ChatCompletionsProvider

    provider = ChatCompletionsProvider(
        config={"model": "test-model", "use_streaming": "false", "max_retries": "0"}
    )

    # Build a mock response WITHOUT prompt_tokens_details
    mock_usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
    mock_usage.prompt_tokens = 1000
    mock_usage.completion_tokens = 200
    mock_usage.total_tokens = 1200

    mock_message = MagicMock()
    mock_message.content = "Hello"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "test-model"

    result = provider._build_response(mock_response)

    assert result.usage is not None
    assert result.usage.cache_read_tokens is None, (
        f"Expected cache_read_tokens=None, got {result.usage.cache_read_tokens!r}"
    )


# ---------------------------------------------------------------------------
# test_cost_usd_stamped_on_usage_for_known_model
# ---------------------------------------------------------------------------


def test_cost_usd_stamped_on_usage_for_known_model():
    """Usage.cost_usd is populated for known models via compute_cost."""
    from amplifier_module_provider_chat_completions import ChatCompletionsProvider

    provider = ChatCompletionsProvider(
        config={"model": "gpt-4o", "use_streaming": "false", "max_retries": "0"}
    )

    mock_usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
    mock_usage.prompt_tokens = 1000
    mock_usage.completion_tokens = 200
    mock_usage.total_tokens = 1200
    # No prompt_tokens_details (not in spec → hasattr returns False)

    mock_message = MagicMock()
    mock_message.content = "Hello"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4o"

    result = provider._build_response(mock_response)

    assert result.usage is not None
    assert result.usage.cost_usd is not None, (
        "cost_usd should be set for known model gpt-4o"
    )
    # 1000 prompt tokens at $2.50/1M + 200 completion at $10/1M
    expected = Decimal("1000") * Decimal("2.50") / Decimal("1000000") + Decimal(
        "200"
    ) * Decimal("10.00") / Decimal("1000000")
    assert result.usage.cost_usd == expected, (
        f"Expected {expected!r}, got {result.usage.cost_usd!r}"
    )
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"cost_usd must be Decimal, got {type(result.usage.cost_usd)}"
    )
