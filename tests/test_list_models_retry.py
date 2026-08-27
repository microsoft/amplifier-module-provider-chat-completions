"""Retry behavior tests for list_models().

Verifies that list_models() uses the same shared retry_with_backoff()/
RetryConfig machinery as complete(): transient failures (connection errors,
timeouts, 5xx) are retried with backoff via the shared _translate_error()
classification, and non-retryable failures (e.g. 401) skip retries.

Unlike the sibling fixes in provider-openai (PR #61), provider-anthropic
(PR #90), and provider-gemini (PR #39) -- which all *raise* to the caller
once retries are exhausted -- this module's list_models() has a
deliberate soft-failure contract: it never raises, it degrades to a
one-element list containing the configured model. That contract is
preserved here; retry is added *before* the degrade, not instead of it.
The key regression this guards is a transient blip no longer causing an
immediate (un-retried) fallback to the configured-model list.

See tests/test_provider.py::TestRetry for the equivalent tests on the
complete() path -- this file mirrors that call shape for list_models().
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from amplifier_core.llm_errors import AuthenticationError as KernelAuthenticationError
from amplifier_module_provider_chat_completions import ChatCompletionsProvider

# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_provider.py's FakeHooks/FakeCoordinator/
# _make_openai_error -- duplicated here to keep this file self-contained,
# matching the sibling fixes' test file convention.)
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_openai_error(cls, message="error", status_code=400):
    """Construct an OpenAI SDK error with the expected shape."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": message}}
    return cls(message, response=mock_response, body=None)


def _make_provider(coordinator=None, **config_overrides) -> ChatCompletionsProvider:
    config = {
        "model": "test-model",
        "max_retries": "3",
        "min_retry_delay": "0.01",
        "max_retry_delay": "0.02",
        **config_overrides,
    }
    provider = ChatCompletionsProvider(config=config, coordinator=coordinator)
    provider._client = AsyncMock()
    return provider


def _fake_models_response(model_ids: list[str]) -> SimpleNamespace:
    """Create a fake OpenAI-compatible models.list() response."""
    return SimpleNamespace(data=[SimpleNamespace(id=mid) for mid in model_ids])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListModelsRetry:
    async def test_list_models_succeeds_first_try(self):
        """No transient failure: exactly one API call, result unchanged."""
        provider = _make_provider(filtered="false")
        provider._client.models.list = AsyncMock(
            return_value=_fake_models_response(["test-model"])
        )

        with pytest.MonkeyPatch.context() as mp:
            mock_sleep = AsyncMock()
            mp.setattr(asyncio, "sleep", mock_sleep)
            models = await provider.list_models()

            assert provider.client.models.list.await_count == 1
            mock_sleep.assert_not_awaited()

        assert len(models) == 1
        assert models[0].id == "test-model"
        # Real server data, not the degraded fallback shape.
        assert models[0].capabilities == ["tools", "streaming"]

    async def test_list_models_recovers_from_transient_error(self):
        """THE KEY REGRESSION: a single transient APIConnectionError is
        retried, then the call succeeds -- the degraded configured-model
        fallback must NOT be returned when the retry succeeds.
        """
        provider = _make_provider(filtered="false", max_retries="2")
        conn_error = openai.APIConnectionError(request=MagicMock())
        provider._client.models.list = AsyncMock(
            side_effect=[conn_error, _fake_models_response(["test-model"])]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        assert provider.client.models.list.await_count == 2
        # Degraded fallback NOT returned -- real server data came back.
        assert len(models) == 1
        assert models[0].id == "test-model"
        assert models[0].capabilities == ["tools", "streaming"]

    async def test_list_models_recovers_from_transient_500(self):
        """A transient 5xx (InternalServerError) is also retryable and
        recovers.
        """
        provider = _make_provider(filtered="false", max_retries="2")
        server_error = _make_openai_error(
            openai.InternalServerError, "internal error", 500
        )
        provider._client.models.list = AsyncMock(
            side_effect=[server_error, _fake_models_response(["test-model"])]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        assert provider.client.models.list.await_count == 2
        assert len(models) == 1
        assert models[0].id == "test-model"

    async def test_list_models_exhaustion_returns_degraded_fallback(self, caplog):
        """Persistent transient failure exhausts retries, then degrades to
        the one-element configured-model list (not a raise) with an
        unmistakable WARNING naming the attempt count and the degraded
        return.
        """
        import logging

        provider = _make_provider(max_retries="2")
        conn_error = openai.APIConnectionError(request=MagicMock())
        provider._client.models.list = AsyncMock(side_effect=conn_error)

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING),
        ):
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        # 1 initial + 2 retries = 3 total attempts
        assert provider.client.models.list.await_count == 3
        # Soft-failure contract preserved: degrades, does not raise.
        assert len(models) == 1
        assert models[0].id == "test-model"
        # Degraded fallback shape: no capabilities set (unlike real data).
        assert models[0].capabilities == []

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("list_models" in r.getMessage() for r in warnings)
        assert any("3 attempt" in r.getMessage() for r in warnings)
        assert any("DEGRADED" in r.getMessage() for r in warnings)

    async def test_list_models_non_retryable_error_skips_retries(self, caplog):
        """A non-retryable error (401) is not retried -- one attempt only --
        but still degrades to the configured-model fallback rather than
        raising, per this module's soft-failure contract.
        """
        import logging

        provider = _make_provider(max_retries="3")
        auth_error = _make_openai_error(openai.AuthenticationError, "invalid key", 401)
        provider._client.models.list = AsyncMock(side_effect=auth_error)

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING),
        ):
            mock_sleep = AsyncMock()
            mp.setattr(asyncio, "sleep", mock_sleep)
            models = await provider.list_models()

            assert provider.client.models.list.await_count == 1
            mock_sleep.assert_not_awaited()

        assert len(models) == 1
        assert models[0].id == "test-model"

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("list_models" in r.getMessage() for r in warnings)
        assert any("1 attempt" in r.getMessage() for r in warnings)

    async def test_list_models_retry_emits_provider_retry_event(self):
        """provider:retry event is emitted on each retry attempt, matching
        the shape complete() emits.
        """
        coordinator = FakeCoordinator()
        provider = _make_provider(
            coordinator=coordinator, filtered="false", max_retries="2"
        )
        conn_error = openai.APIConnectionError(request=MagicMock())
        provider._client.models.list = AsyncMock(
            side_effect=[conn_error, _fake_models_response(["test-model"])]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            await provider.list_models()

        retry_events = [e for e in coordinator.hooks.events if e[0] == "provider:retry"]
        assert len(retry_events) == 1
        _, payload = retry_events[0]
        assert payload["provider"] == "chat-completions"
        assert payload["max_retries"] == 2
        assert payload["attempt"] == 1
        assert "delay" in payload
        assert "error_type" in payload
        assert "error_message" in payload

    async def test_list_models_non_retryable_error_type_is_authentication(self):
        """Sanity check: the translated error type used to decide
        retryability for a 401 is KernelAuthenticationError (non-retryable),
        matching complete()'s classification via _translate_error.
        """
        provider = _make_provider()
        auth_error = _make_openai_error(openai.AuthenticationError, "invalid key", 401)
        translated = provider._translate_error(auth_error)
        assert isinstance(translated, KernelAuthenticationError)
        assert translated.retryable is False
