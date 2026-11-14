"""Real SDK/HTTP serialization, no network, inference or account entitlement."""

import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import openai
import pytest
from openai import _base_client
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolSpec,
)

from amplifier_module_provider_chat_completions import ChatCompletionsProvider
from amplifier_module_provider_chat_completions._media import OFFICIAL_USER_IMAGE_MODELS

HTTP = getattr(_base_client, "httpx2", None) or _base_client.httpx
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAEElEQVR4nGP8zwAFTDAGAwATKQED8NgHhAAAAABJRU5ErkJggg=="
JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDi6KKK+ZP3E//Z"


def image(data=PNG, mime="image/png"):
    return ImageBlock(source={"type": "base64", "media_type": mime, "data": data})


def completion(model):
    return {
        "id": "fixture",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


class Cap:
    version = 1

    def __init__(self):
        self.request = None
        self.calls = []
        self.revoked = False

    def validate(self, value, *, purpose, request_options):
        assert value is self.request
        self.calls.append(copy.deepcopy(request_options))
        if self.revoked:
            raise ValueError("revoked fixture")
        return {"valid": True}


def provider(
    monkeypatch,
    handler,
    *,
    model="gpt-5.4",
    stream=False,
    hook=None,
    cap=None,
    base="https://api.openai.com/v1",
):
    monkeypatch.delenv("CHAT_COMPLETIONS_BASE_URL", raising=False)
    monkeypatch.setattr(
        openai,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: HTTP.AsyncClient(
            transport=HTTP.MockTransport(handler), **kwargs
        ),
    )
    events = []

    async def emit(name, payload):
        events.append((name, copy.deepcopy(payload)))
        if hook:
            hook(name, payload)

    p = ChatCompletionsProvider(
        {
            "base_url": base,
            "api_key": "fixture-not-credential",
            "model": model,
            "use_streaming": stream,
            "raw": True,
            "max_retries": 1,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
        },
        coordinator=SimpleNamespace(
            hooks=SimpleNamespace(emit=emit), get_capability=lambda key: cap
        ),
    )
    return p, events


@pytest.mark.asyncio
@pytest.mark.parametrize("model", list(OFFICIAL_USER_IMAGE_MODELS))
@pytest.mark.parametrize("stream", [False, True])
async def test_documented_user_images_actual_http_original_bytes_order(
    monkeypatch, model, stream
):
    seen = []

    def handle(req):
        body = json.loads(req.content)
        seen.append(body)
        assert str(req.url) == "https://api.openai.com/v1/chat/completions"
        assert body["model"] == model
        if stream:
            chunk = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
                ],
            }
            return HTTP.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return HTTP.Response(200, json=completion(model))

    cap = Cap()
    p, events = provider(monkeypatch, handle, model=model, stream=stream, cap=cap)
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content=[
                    TextBlock(text="historical-before"),
                    image(),
                    TextBlock(text="historical-after"),
                ],
            ),
            Message(role="assistant", content="previous"),
            Message(
                role="tool",
                tool_call_id="call1",
                content=[
                    TextBlock(text="tool-before"),
                    image(),
                    TextBlock(text="tool-after"),
                ],
            ),
            Message(
                role="user",
                content=[
                    TextBlock(text="before"),
                    image(),
                    TextBlock(text="between"),
                    image(JPEG, "image/jpeg"),
                    TextBlock(text="after"),
                ],
            ),
        ],
        model=model,
        metadata={"opaque": "preserve"},
    )
    cap.request = request
    before = request.model_dump()
    try:
        await p.complete(request, request_options={"model": model}, model=model)
        body = seen[0]
        assert [m["role"] for m in body["messages"]] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert body["messages"][0]["content"][0]["text"] == "historical-before"
        assert body["messages"][0]["content"][2]["text"] == "historical-after"
        parts = body["messages"][3]["content"]
        assert [b["type"] for b in parts] == [
            "text",
            "image_url",
            "text",
            "image_url",
            "text",
        ]
        assert [parts[i]["text"] for i in (0, 2, 4)] == ["before", "between", "after"]
        for index, data, mime in [(1, PNG, "image/png"), (3, JPEG, "image/jpeg")]:
            header, encoded = parts[index]["image_url"]["url"].split(",", 1)
            assert header == f"data:{mime};base64"
            assert (
                hashlib.sha256(base64.b64decode(encoded)).digest()
                == hashlib.sha256(base64.b64decode(data)).digest()
            )
        tool = body["messages"][2]
        assert (
            tool["tool_call_id"] == "call1" and "no pixels supplied]" in tool["content"]
        )
        assert PNG not in tool["content"] and tool["content"].index(
            "tool-before"
        ) < tool["content"].index("[Image unavailable:") < tool["content"].index(
            "tool-after"
        )
        assert cap.calls == [{"model": model}, {"model": model}]
        assert request.model_dump() == before
        assert PNG not in str(events) and JPEG not in str(events)
        assert p.client.max_retries == 0
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base",
    [
        "https://api.openai.com.attacker/v1",
        "http://api.openai.com/v1",
        "https://api.openai.com/v1/other",
        "https://proxy.invalid/v1",
    ],
)
async def test_similar_or_custom_routes_never_gain_vision(monkeypatch, base):
    p, _ = provider(
        monkeypatch,
        lambda req: HTTP.Response(200, json=completion("gpt-5.4")),
        base=base,
    )
    try:
        assert p.get_media_request_support(model="gpt-5.4")["positions"] == []
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_unknown_alias_metadata_and_client_injection_not_grants(monkeypatch):
    p, _ = provider(monkeypatch, lambda req: HTTP.Response(200, json={}))
    try:
        for model in ["gpt-5.4-custom", "gpt-5.4-mini", "deployment", "unknown"]:
            assert (
                p.get_media_request_support(
                    model=model, request_options={"supports_images": True}
                )["positions"]
                == []
            )
        assert p.get_media_request_support(model="gpt-5.4")["positions"] == ["user"]
        p.client.base_url = "https://proxy.invalid/v1/"
        assert p.get_media_request_support(model="gpt-5.4")["positions"] == []
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_sol_tools_refuse_not_silently_drop(monkeypatch):
    seen = []
    p, _ = provider(monkeypatch, lambda req: seen.append(req), model="gpt-6.1-sol")
    try:
        with pytest.raises(InvalidRequestError, match="Responses"):
            await p.complete(
                ChatRequest(
                    messages=[Message(role="user", content=[image()])],
                    tools=[ToolSpec(name="tool", parameters={})],
                )
            )
        assert not seen
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "endpoint",
        "api_key",
        "project",
        "retries",
        "resource",
        "redirect",
        "resource_post",
        "request_hook",
        "http_auth",
        "transport",
    ],
)
async def test_actual_instance_mutation_after_prepare_refuses_before_http(
    monkeypatch, mutation
):
    seen = []

    def hook(name, payload):
        if name == "llm:request":
            if mutation == "endpoint":
                p.client.base_url = "https://proxy.invalid/v1/"
            if mutation == "api_key":
                p.client.api_key = "changed"
            if mutation == "project":
                p.client.project = "changed"
            if mutation == "retries":
                p.client.max_retries = 1
            if mutation == "resource":
                p.client.chat.completions = SimpleNamespace(
                    create=lambda **kwargs: None
                )
            if mutation == "redirect":
                p.client._client.follow_redirects = True
            if mutation == "resource_post":
                original = p.client.chat.completions._post

                async def changed(*args, **kwargs):
                    kwargs["body"]["model"] = "unqualified-model"
                    return await original(*args, **kwargs)

                p.client.chat.completions._post = changed
            if mutation == "request_hook":

                async def changed(req):
                    req.url = HTTP.URL("https://proxy.invalid/v1/chat/completions")
                    req.headers["Authorization"] = "changed"

                p.client._client.event_hooks["request"].append(changed)
            if mutation == "http_auth":
                p.client._client.auth = HTTP.BasicAuth("other", "account")
            if mutation == "transport":
                p.client._client._transport = HTTP.MockTransport(
                    lambda req: seen.append(req)
                )

    p, _ = provider(monkeypatch, lambda req: seen.append(req), hook=hook)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(
                ChatRequest(messages=[Message(role="user", content=[image()])])
            )
        assert not seen
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_known_error_echo_pixels_not_diagnostic_or_exception(monkeypatch, status):
    calls = []

    def handle(req):
        calls.append(req)
        return HTTP.Response(
            status, json={"error": {"message": PNG, "type": "invalid_request_error"}}
        )

    p, events = provider(monkeypatch, handle)
    try:
        with pytest.raises(LLMError) as error:
            await p.complete(
                ChatRequest(messages=[Message(role="user", content=[image()])])
            )
        assert PNG not in str(error.value) and error.value.__suppress_context__
        assert PNG not in str(events)
        assert len(calls) == (2 if status in (429, 500) else 1)
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_early_sse_error_no_chunk_no_pixel_echo(monkeypatch):
    seen = []

    def handle(req):
        seen.append(req)
        return HTTP.Response(
            200,
            text=f"data: {json.dumps({'error': {'message': PNG, 'type': 'invalid_request_error'}})}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    p, events = provider(monkeypatch, handle, stream=True)
    try:
        with pytest.raises(LLMError) as error:
            await p.complete(
                ChatRequest(messages=[Message(role="user", content=[image()])])
            )
        assert PNG not in str(error.value) and PNG not in str(events)
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_retry_keeps_exact_serialized_pixels_and_revalidates(monkeypatch):
    bodies = []
    cap = Cap()

    def handle(req):
        bodies.append(json.loads(req.content))
        if len(bodies) == 1:
            return HTTP.Response(
                429, json={"error": {"message": PNG, "type": "rate_limit_error"}}
            )
        return HTTP.Response(200, json=completion("gpt-5.4"))

    p, events = provider(monkeypatch, handle, cap=cap)
    request = ChatRequest(
        model="gpt-5.4", messages=[Message(role="user", content=[image()])]
    )
    cap.request = request
    try:
        await p.complete(request)
        assert len(bodies) == 2 and bodies[0] == bodies[1]
        assert cap.calls == [{}, {}, {}]
        assert PNG not in str(events)
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_revocation_during_actual_retry_wait_stops_second_http(monkeypatch):
    seen = []
    cap = Cap()

    def handle(req):
        seen.append(req)
        cap.revoked = True
        return HTTP.Response(429, json={"error": {"message": "rate limited"}})

    p, _ = provider(monkeypatch, handle, cap=cap)
    request = ChatRequest(messages=[Message(role="user", content=[image()])])
    cap.request = request
    try:
        with pytest.raises(ValueError, match="revoked"):
            await p.complete(request)
        assert len(seen) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_early_sse_eof_after_text_never_replays_or_echoes_error(monkeypatch):
    seen = []

    class Broken(HTTP.AsyncByteStream):
        async def __aiter__(self):
            chunk = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-5.4",
                "choices": [
                    {"index": 0, "delta": {"content": "first"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            raise HTTP.ReadError(PNG)

    def handle(req):
        seen.append(req)
        return HTTP.Response(
            200, stream=Broken(), headers={"content-type": "text/event-stream"}
        )

    p, events = provider(monkeypatch, handle, stream=True)
    try:
        with pytest.raises(LLMError) as error:
            await p.complete(
                ChatRequest(messages=[Message(role="user", content=[image()])])
            )
        assert error.value.retryable is False and len(seen) == 1
        assert PNG not in str(error.value) and PNG not in str(events)
        assert any(name == "llm:stream_aborted" for name, _ in events)
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_official_redirect_refuses_without_following_image_payload(monkeypatch):
    seen = []

    def handle(req):
        seen.append(str(req.url))
        return HTTP.Response(
            307, headers={"location": "https://attacker.invalid/collect"}
        )

    p, events = provider(monkeypatch, handle)
    try:
        with pytest.raises(LLMError):
            await p.complete(
                ChatRequest(messages=[Message(role="user", content=[image()])])
            )
        assert seen == ["https://api.openai.com/v1/chat/completions"]
        assert PNG not in str(events)
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_only_effective_selected_model_qualifies_not_configured_name(monkeypatch):
    bodies = []

    def handle(req):
        body = json.loads(req.content)
        bodies.append(body)
        return HTTP.Response(200, json=completion(body["model"]))

    p, _ = provider(monkeypatch, handle, model="unknown")
    try:
        await p.complete(
            ChatRequest(
                model="unknown", messages=[Message(role="user", content=[image()])]
            ),
            request_options={"model": "gpt-5.4"},
        )
        assert bodies[0]["model"] == "gpt-5.4"
        assert bodies[0]["messages"][0]["content"][0]["type"] == "image_url"
        await p.complete(
            ChatRequest(
                model="gpt-5.4", messages=[Message(role="user", content=[image()])]
            ),
            model="unknown",
        )
        assert (
            bodies[1]["model"] == "unknown"
            and "no pixels supplied]" in bodies[1]["messages"][0]["content"]
        )
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_fresh_unqualified_client_rate_limit_retry_still_works(monkeypatch):
    seen = []

    def handle(req):
        seen.append(req)
        if len(seen) == 1:
            return HTTP.Response(429, json={"error": {"message": "limited"}})
        return HTTP.Response(200, json=completion("unknown"))

    p, _ = provider(monkeypatch, handle, model="unknown")
    try:
        assert p._client is None
        await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
        assert len(seen) == 2
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("optout", ["expert", "headers"])
async def test_sol_tools_restriction_independent_of_image_qualification(
    monkeypatch, optout
):
    seen = []
    p, _ = provider(monkeypatch, lambda req: seen.append(req), model="gpt-6.1-sol")
    if optout == "expert":
        p._extra_request_params = {"temperature": 0.1}
    else:
        p._default_headers = {"X-Example": "custom"}
    try:
        assert p.get_media_request_support(model="gpt-6.1-sol")["positions"] == []
        with pytest.raises(InvalidRequestError, match="Responses"):
            await p.complete(
                ChatRequest(
                    messages=[Message(role="user", content="hello")],
                    tools=[ToolSpec(name="tool", parameters={})],
                )
            )
        assert seen == []
    finally:
        await p.close()
