"""Default model waits survive virtual elapsed time; cancellation/errors remain live."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from amplifier_core.llm_errors import LLMTimeoutError, ProviderUnavailableError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_chat_completions import ChatCompletionsProvider


def setup_call(streaming, config=None, failure=None):
    provider = ChatCompletionsProvider(
        config={
            "use_streaming": streaming,
            "max_retries": 0,
            **(config or {}),
        }
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = []

    async def wait():
        entered.set()
        await release.wait()
        if failure:
            raise failure

    async def stream():
        try:
            await wait()
            if False:
                yield
        finally:
            closed.append(True)

    async def create(**kwargs):
        if streaming:
            return stream()
        await wait()
        return SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    client = MagicMock()
    provider._client = client
    call = client.chat.completions.create = AsyncMock(side_effect=create)

    request = ChatRequest(messages=[Message(role="user", content="hello")])
    return provider, request, entered, release, closed, call


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_default_wait_survives_an_hour_then_completes(monkeypatch, streaming):
    provider, request, entered, release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    # No real sleeping: move beyond every former model-work deadline while
    # the mock provider remains healthy but silent.
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 3600)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
    release.set()
    await task
    timeout = call.call_args.kwargs["timeout"]
    assert timeout.read is None
    assert timeout.connect == 5.0
    assert timeout.pool == 5.0
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_cancellation_propagates_without_retry(streaming):
    provider, request, entered, _release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_deadline_still_stops_model_work(monkeypatch, streaming):
    provider, request, entered, _release, closed, call = setup_call(
        streaming, {"timeout": 10}
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 11)
        with pytest.raises(LLMTimeoutError):
            await task
    assert call.call_args.kwargs["timeout"].read == 10
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_transport_failure_is_still_reported(streaming):
    provider, request, entered, release, closed, call = setup_call(
        streaming, failure=openai.APIConnectionError(request=MagicMock())
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    release.set()
    with pytest.raises(ProviderUnavailableError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_actual_sdk_request_disables_hidden_read_deadline(streaming):
    import json

    from openai import _base_client

    sdk_module = openai
    body = {
        "id": "chat_wait",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    chunk = {
        **body,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}
        ],
    }
    stream_body = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
    provider = ChatCompletionsProvider(
        config={"use_streaming": streaming, "max_retries": 0}
    )
    client_type = sdk_module.AsyncOpenAI

    transport = getattr(_base_client, "httpx2", None) or _base_client.httpx
    seen = []

    async def handle(request):
        seen.append(request)
        if streaming:
            return transport.Response(
                200, text=stream_body, headers={"content-type": "text/event-stream"}
            )
        return transport.Response(200, json=body)

    provider._client = client_type(
        api_key="fixture",
        timeout=0.001,
        http_client=transport.AsyncClient(transport=transport.MockTransport(handle)),
    )
    try:
        await provider.complete(
            ChatRequest(messages=[Message(role="user", content="hello")])
        )
        assert len(seen) == 1
        assert seen[0].extensions["timeout"] == {
            "read": None,
            "write": None,
            "connect": 5.0,
            "pool": 5.0,
        }
        assert "timeout" not in json.loads(seen[0].content)
    finally:
        await provider.close()
