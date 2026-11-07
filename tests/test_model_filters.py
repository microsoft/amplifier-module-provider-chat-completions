"""Canonical filter conformance and unchanged explicit inference."""

import asyncio
import json

import httpx
import pytest
from amplifier_core.llm_errors import AuthenticationError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_chat_completions import ChatCompletionsProvider
from amplifier_module_provider_chat_completions._model_listing import ModelFilters
from tests.test_discover_models import model_ids, provider_for, wire_models

IDS = [
    "production-chat",
    "Vendor/Model:7b-q4",
    "Vendor/Model:70b-q8",
    "future-2099-01-01",
    "future",
]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, sorted(IDS)),
        ({"include": []}, []),
        ({"include": ["VENDOR/*"], "exclude": ["*:70B-*"]}, ["Vendor/Model:7b-q4"]),
        ({"include": ["future*"], "hide_dated": True}, ["future"]),
        ({"include": ["future-*"], "hide_dated": True}, ["future-2099-01-01"]),
    ],
)
def test_vectors(filters, expected):
    assert ModelFilters.parse(filters).select(IDS) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "*",
        {"include": "*"},
        {"include": [""]},
        {"exclude": [1]},
        {"hide_dated": "yes"},
        {"extra": True},
    ],
)
def test_invalid_config(invalid):
    with pytest.raises((TypeError, ValueError)):
        ChatCompletionsProvider({"model_filters": invalid})


@pytest.mark.asyncio
async def test_default_override_mutation_metadata_and_explicit_model(monkeypatch):
    monkeypatch.delenv("CHAT_COMPLETIONS_BASE_URL", raising=False)
    cfg = {"include": ["production-*"], "exclude": []}
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=wire_models(*IDS))
        body = json.loads(request.content)
        assert body["model"] == "hidden-exact:Q4"
        assert body["top_p"] == 0.25
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = provider_for(
        handler, model_filters=cfg, filtered=True, use_streaming=False, top_p=0.25
    )
    cfg["include"].clear()
    try:
        assert model_ids(await provider.list_models()) == ["production-chat"]
        assert model_ids(await provider.list_models(filters=None)) == [
            "production-chat"
        ]
        assert model_ids(await provider.list_models(filters={})) == sorted(IDS)
        assert model_ids(
            await provider.list_models(filters={"include": ["vendor/*"]})
        ) == sorted(IDS[1:3])
        assert await provider.list_models(filters={"include": []}) == []
        info = provider.get_model_list_info()
        assert info["source"] == "provider-api" and info["filters_keyword"] == "filters"
        assert info["override"] == "replace" and info["unfiltered"] == {}
        info["default_filters"]["include"].clear()
        info["filters_schema"]["properties"].clear()
        assert provider.get_model_list_info()["default_filters"]["include"] == [
            "production-*"
        ]
        assert (
            "include" in provider.get_model_list_info()["filters_schema"]["properties"]
        )
        before = len(requests)
        with pytest.raises(ValueError):
            await provider.list_models(filters={"include": "bad"})
        assert len(requests) == before
        assert not hasattr(provider, "discover_models")
        await provider.complete(
            ChatRequest(
                model="hidden-exact:Q4",
                temperature=0.25,
                messages=[Message(role="user", content="test")],
            )
        )
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_sanitized_errors_and_cancellation():
    provider = provider_for(
        lambda r: httpx.Response(401, json={"error": {"message": "secret-account-key"}})
    )
    try:
        with pytest.raises(AuthenticationError) as caught:
            await provider.list_models()
        assert "secret-account-key" not in str(caught.value)
        assert caught.value.__cause__ is None
    finally:
        await provider.close()

    async def cancelled(request):
        raise asyncio.CancelledError()

    provider = provider_for(cancelled)
    try:
        with pytest.raises(asyncio.CancelledError):
            await provider.list_models()
    finally:
        await provider.close()
