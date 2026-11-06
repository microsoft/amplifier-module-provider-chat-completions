"""Strict discovery uses current endpoint evidence, never legacy fallbacks."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core.llm_errors import (
    AuthenticationError,
    LLMError,
    LLMTimeoutError,
    ProviderUnavailableError,
)

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


def provider_for(handler, **config):
    provider = ChatCompletionsProvider(
        config={
            "base_url": "https://endpoint.invalid/custom/v1",
            "api_key": "test-account-key",
            "model": "configured-not-in-api",
            "max_retries": 0,
            **config,
        }
    )
    # Exercise the public SDK transport, without reading credentials or networking.
    provider._client = openai.AsyncOpenAI(
        base_url=provider._base_url,
        api_key=provider._api_key,
        default_headers={"X-Discovery-Tenant": "test-tenant"},
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


def wire_models(*ids):
    return {"object": "list", "data": [{"id": mid, "object": "model"} for mid in ids]}


def model_ids(result):
    assert result["version"] == 1
    assert result["source"] == "provider-api"
    # Existing public ModelInfo serialization must remain JSON-compatible.
    json.dumps({**result, "models": [m.model_dump(mode="json") for m in result["models"]]})
    return [m.id for m in result["models"]]


@pytest.mark.asyncio
async def test_current_endpoint_ids_ignore_selection_and_refresh_every_call(monkeypatch):
    monkeypatch.delenv("CHAT_COMPLETIONS_BASE_URL", raising=False)
    requests = []
    responses = [
        wire_models("account-a", "account-b"),
        wire_models(),
        wire_models("account-c"),
    ]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=responses.pop(0))

    provider = provider_for(
        handler,
        default_model="another-configured-default",
        reasoning_effort="high",
        priority=7,
        filtered=True,
    )
    try:
        assert model_ids(await provider.discover_models()) == ["account-a", "account-b"]
        assert model_ids(await provider.discover_models()) == []
        assert model_ids(await provider.discover_models()) == ["account-c"]
        assert len(requests) == 3
        for request in requests:
            assert str(request.url) == "https://endpoint.invalid/custom/v1/models"
            assert request.headers["authorization"] == "Bearer test-account-key"
            assert request.headers["x-discovery-tenant"] == "test-tenant"
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type"), [(401, AuthenticationError), (503, ProviderUnavailableError)]
)
async def test_http_failure_raises_despite_legacy_configured_fallback(status, error_type):
    provider = provider_for(
        lambda request: httpx.Response(status, json={"error": {"message": "rejected"}})
    )
    try:
        with pytest.raises(error_type):
            await provider.discover_models()
        # The older independent client contract still deliberately falls back.
        assert [m.id for m in await provider.list_models()] == ["configured-not-in-api"]
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_raises_not_an_inventory(error_type):
    def handler(request):
        raise error_type("offline test", request=request)

    provider = provider_for(handler)
    try:
        expected = (
            LLMTimeoutError
            if error_type is httpx.ReadTimeout
            else ProviderUnavailableError
        )
        with pytest.raises(expected):
            await provider.discover_models()
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": None},
        {"data": [{}]},
        {"data": [{"id": ""}]},
        {"data": [{"id": 42}]},
        {"data": [{"id": "   "}]},
    ],
)
async def test_malformed_response_never_becomes_empty_or_configured_rows(payload):
    provider = provider_for(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMError):
            await provider.discover_models()
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_malformed_json_raises():
    provider = provider_for(lambda request: httpx.Response(200, text="not-json"))
    try:
        with pytest.raises(LLMError):
            await provider.discover_models()
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_legacy_filtered_first_model_contract_remains():
    provider = provider_for(
        lambda request: httpx.Response(200, json=wire_models("first", "second"))
    )
    try:
        assert [m.id for m in await provider.list_models()] == ["first"]
        assert model_ids(await provider.discover_models()) == ["first", "second"]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_pagination_includes_all_ids_and_propagates_later_page_failure():
    provider = ChatCompletionsProvider(config={"model": "not-real", "max_retries": 0})
    second = SimpleNamespace(
        data=[SimpleNamespace(id="second")], has_next_page=lambda: False
    )
    first = SimpleNamespace(
        data=[SimpleNamespace(id="first")],
        has_next_page=lambda: True,
        get_next_page=AsyncMock(return_value=second),
    )
    provider._client = SimpleNamespace(
        models=SimpleNamespace(list=AsyncMock(return_value=first))
    )
    assert model_ids(await provider.discover_models()) == ["first", "second"]
    first.get_next_page.side_effect = openai.APIConnectionError(
        request=httpx.Request("GET", "https://endpoint.invalid/models")
    )
    with pytest.raises(ProviderUnavailableError):
        await provider.discover_models()


@pytest.mark.asyncio
async def test_retry_uses_existing_classification_without_fallback(monkeypatch):
    provider = ChatCompletionsProvider(config={"model": "not-real", "max_retries": 1})
    call = AsyncMock(
        side_effect=[
            openai.APIConnectionError(
                request=httpx.Request("GET", "https://endpoint.invalid/models")
            ),
            SimpleNamespace(data=[SimpleNamespace(id="actual")]),
        ]
    )
    provider._client = SimpleNamespace(models=SimpleNamespace(list=call))
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    assert model_ids(await provider.discover_models()) == ["actual"]
    assert call.await_count == 2
