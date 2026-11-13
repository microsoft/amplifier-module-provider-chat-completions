"""Offline actual SDK-call payload checks for the explicitly no-vision route."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
)

from amplifier_module_provider_chat_completions import ChatCompletionsProvider
from amplifier_module_provider_chat_completions._media import (
    MEDIA_NAMESPACE,
    NOT_RETAINED,
    UNAVAILABLE,
    diagnostic_copy,
)


def image():
    return ImageBlock(
        source={"type": "base64", "media_type": "image/png", "data": "cGl4ZWxz"}
    )


def response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="unknown-model",
        usage=None,
        model_dump=lambda: {
            "content": [{"type": "image_url", "image_url": {"url": "private"}}],
            "status": "ok",
        },
    )


class Refusal(Exception):
    pass


class Capability:
    version = 1

    def __init__(self):
        self.calls = []
        self.revoked = False

    def validate(self, request, *, purpose, request_options):
        self.calls.append((request, purpose, copy.deepcopy(request_options)))
        if self.revoked:
            raise Refusal("expired")
        return {"valid": True}


def provider(*, cap=None, stream=False, retries=0, raw=True, hook=None):
    events = []

    async def emit(name, payload):
        events.append((name, copy.deepcopy(payload)))
        if hook:
            await hook(name, payload)

    coord = SimpleNamespace(
        hooks=SimpleNamespace(emit=emit),
        get_capability=lambda key: cap,
    )
    p = ChatCompletionsProvider(
        {
            "model": "unknown-model",
            "raw": raw,
            "use_streaming": stream,
            "max_retries": retries,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
        },
        coordinator=coord,
    )
    p._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response()))
        )
    )
    return p, events


@pytest.mark.parametrize("role", ["user", "tool", "assistant", "system", "developer"])
def test_every_image_position_is_explicit_text_only(role):
    p, _ = provider()
    msg = Message(
        role=role,
        tool_call_id="call-1" if role == "tool" else None,
        content=[TextBlock(text="before"), image(), TextBlock(text="after")],
        metadata={"owner": "preserved"},
    )
    before = msg.model_dump()
    wire = p._convert_messages_to_wire([msg])
    assert (
        wire[0]["content"].index("before")
        < wire[0]["content"].index(UNAVAILABLE)
        < wire[0]["content"].index("after")
    )
    assert "cGl4ZWxz" not in str(wire)
    if role == "tool":
        assert wire[0]["tool_call_id"] == "call-1"
        assert wire[0]["role"] == "tool"
    assert msg.model_dump() == before


def test_tool_rich_output_is_not_dictionary_string_pixels():
    p, _ = provider()
    msgs = [
        Message(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_call_id="one",
                    output=[TextBlock(text="first"), image(), TextBlock(text="last")],
                ),
                ToolResultBlock(
                    tool_call_id="two",
                    output={"status": "failed", "content": [image().model_dump()]},
                ),
            ],
        )
    ]
    wire = p._convert_messages_to_wire(msgs)
    assert [m["tool_call_id"] for m in wire] == ["one", "two"]
    assert wire[0]["content"] == f"first\n{UNAVAILABLE}\nlast"
    assert UNAVAILABLE in wire[1]["content"] and "failed" in wire[1]["content"]
    assert "cGl4ZWxz" not in str(wire)


def test_unknown_model_support_no_prefix_guesses_or_optional_fake_methods():
    p, _ = provider()
    for name in ["gpt-4o", "qwen-vl", "custom", "default"]:
        support = p.get_media_request_support(
            model=name, request_options={"model": "different"}
        )
        assert support["v"] == 1 and support["positions"] == []
    for method in [
        "request_budget",
        "count_tokens",
        "compact_context",
        "stream",
        "refit_request",
    ]:
        assert not hasattr(p, method)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_actual_sdk_payload_fallback_and_same_validation_options(stream):
    cap = Capability()
    p, events = provider(cap=cap, stream=stream)
    if stream:

        async def chunks():
            yield SimpleNamespace(choices=[], usage=None)

        p._client.chat.completions.create.return_value = chunks()
    request = ChatRequest(
        model="selected",
        tool_choice="none",
        messages=[
            Message(
                role="user", content=[TextBlock(text="A"), image(), TextBlock(text="B")]
            ),
            Message(role="tool", tool_call_id="c1", content=[image()]),
        ],
        metadata={"opaque": "continuation"},
    )
    snapshot = request.model_dump()
    await p.complete(request, selected_option={"value": 1})
    wire = p._client.chat.completions.create.call_args.kwargs
    assert wire["model"] == "selected" and wire["tool_choice"] == "none"
    assert wire["messages"][0]["content"] == f"A\n{UNAVAILABLE}\nB"
    assert wire["messages"][1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": UNAVAILABLE,
    }
    assert "cGl4ZWxz" not in str(wire)
    assert cap.calls[-1] == (request, "transport", {"selected_option": {"value": 1}})
    assert request.model_dump() == snapshot
    assert "cGl4ZWxz" not in str(events)


@pytest.mark.asyncio
async def test_retry_revocation_is_local_refusal_no_second_send():
    cap = Capability()
    p, _ = provider(cap=cap, retries=2)
    error = openai.RateLimitError(
        "rate limited", response=MagicMock(status_code=429, headers={}), body=None
    )

    async def first(**params):
        cap.revoked = True
        raise error

    p._client.chat.completions.create.side_effect = first
    request = ChatRequest(messages=[Message(role="user", content="placeholder")])
    with pytest.raises(Refusal):
        await p.complete(request)
    assert p._client.chat.completions.create.await_count == 1
    assert len(cap.calls) == 3  # preflight, first send, refused retry
    assert all(call[0] is request for call in cap.calls)


@pytest.mark.asyncio
async def test_retry_uses_same_frozen_wire_even_if_sdk_mutates_its_copy():
    cap = Capability()
    p, _ = provider(cap=cap, retries=1)
    seen = []

    async def create(**params):
        seen.append(copy.deepcopy(params["messages"]))
        if len(seen) == 1:
            params["messages"][0]["content"] = "SDK mutation"
            raise openai.RateLimitError(
                "limited", response=MagicMock(status_code=429, headers={}), body=None
            )
        return response()

    p._client.chat.completions.create.side_effect = create
    await p.complete(ChatRequest(messages=[Message(role="user", content=[image()])]))
    assert seen[0] == seen[1] == [{"role": "user", "content": UNAVAILABLE}]


@pytest.mark.asyncio
async def test_hook_mutation_cannot_change_wire_and_selection_change_refuses():
    async def hook(name, payload):
        if name == "llm:request":
            payload["raw"]["messages"][0]["content"] = "malicious hook"

    p, _ = provider(hook=hook)
    await p.complete(ChatRequest(messages=[Message(role="user", content="original")]))
    assert (
        p._client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        == "original"
    )

    async def mutate(name, payload):
        if name == "llm:request":
            p._model = "changed-account-model"

    p, _ = provider(hook=mutate)
    with pytest.raises(InvalidRequestError):
        await p.complete(
            ChatRequest(messages=[Message(role="user", content="original")])
        )
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_capability_refuses_ref_bearing_request():
    p, _ = provider()
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="image label",
                metadata={MEDIA_NAMESPACE: {"refs": []}},
            )
        ]
    )
    with pytest.raises(InvalidRequestError):
        await p.complete(request)
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"model": "uncounted"},
        {"messages": [{"role": "user", "content": "replacement"}]},
        {"tools": []},
        {"stream": True},
    ],
)
async def test_media_expert_control_overrides_refuse(override):
    p, _ = provider(cap=Capability())
    p._extra_request_params = override
    with pytest.raises(InvalidRequestError):
        await p.complete(
            ChatRequest(messages=[Message(role="user", content="placeholder")])
        )
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_nested_sdk_extra_body_cannot_override_prepared_controls(stream):
    p, _ = provider(cap=Capability(), stream=stream)
    p._extra_request_params = {"extra_body": {"model": "other", "messages": []}}
    with pytest.raises(InvalidRequestError):
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_schema_type_lists_and_mappings_are_not_images():
    p, _ = provider()
    schema = {
        "type": "object",
        "properties": {
            "nullable": {"type": ["string", "null"]},
            "opaque": {"type": {"custom": True}},
        },
    }
    await p.complete(
        ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[ToolSpec(name="tool", parameters=schema)],
        )
    )
    assert diagnostic_copy(schema) == schema
    output = p._convert_messages_to_wire(
        [
            Message(
                role="tool",
                content=[
                    ToolResultBlock(tool_call_id="one", output=schema),
                ],
            )
        ]
    )
    assert "nullable" in output[0]["content"]


def test_tuples_and_typed_images_never_become_dictionary_string_pixels():
    p, _ = provider()
    output = {"status": "failed", "content": (image(), "after")}
    wire = p._convert_messages_to_wire(
        [
            Message(
                role="tool",
                content=[
                    ToolResultBlock(tool_call_id="one", output=output),
                ],
            )
        ]
    )
    assert UNAVAILABLE in wire[0]["content"]
    assert "cGl4ZWxz" not in str(wire)
    safe = diagnostic_copy({"raw": (image(),)})
    assert "cGl4ZWxz" not in str(safe) and NOT_RETAINED in str(safe)


@pytest.mark.asyncio
async def test_each_validation_gets_fresh_options_copy():
    cap = Capability()
    original = cap.validate

    def mutate(request, *, purpose, request_options):
        result = original(request, purpose=purpose, request_options=request_options)
        request_options["nested"]["value"] = "validator changed"
        return result

    cap.validate = mutate
    p, _ = provider(cap=cap)
    kwargs = {"nested": {"value": "original"}}
    await p.complete(
        ChatRequest(messages=[Message(role="user", content="hi")]), **kwargs
    )
    assert kwargs == {"nested": {"value": "original"}}
    assert all(call[2] == kwargs for call in cap.calls)


@pytest.mark.asyncio
async def test_removed_capability_cannot_bypass_witness_after_hook():
    async def hook(name, payload):
        if name == "llm:request":
            p.coordinator.get_capability = lambda key: None

    p, _ = provider(cap=Capability(), hook=hook)
    with pytest.raises(InvalidRequestError):
        await p.complete(
            ChatRequest(messages=[Message(role="user", content="placeholder")])
        )
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "abort-hook", "postprocess"])
async def test_all_postchunk_failures_refuse_replay(failure):
    async def hook(name, payload):
        if failure == "abort-hook" and name == "llm:stream_aborted":
            raise ValueError("broken observer")

    p, _ = provider(stream=True, retries=2, hook=hook)
    if failure == "timeout":
        p._timeout = 0.01

    async def chunks():
        if failure == "postprocess":
            delta = SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call",
                        function=SimpleNamespace(name="tool", arguments="[]"),
                    ),
                ],
            )
        else:
            delta = SimpleNamespace(
                content="first", reasoning_content=None, tool_calls=None
            )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None
        )
        if failure == "timeout":
            await asyncio.sleep(10)
        elif failure == "abort-hook":
            raise openai.APIConnectionError(request=MagicMock())

    p._client.chat.completions.create.return_value = chunks()
    with pytest.raises(LLMError) as exc:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert exc.value.retryable is False
    assert p._client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_expert_message_override_cannot_smuggle_pixels_without_capability():
    p, _ = provider()
    p._extra_request_params = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,cGl4ZWxz"},
                    },
                ],
            }
        ]
    }
    with pytest.raises(InvalidRequestError):
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    p._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonstream_received_response_decode_failure_never_replays():
    p, _ = provider(retries=2)
    p._build_response = MagicMock(side_effect=ValueError("bad tool arguments"))
    with pytest.raises(InvalidRequestError):
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert p._client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_client_replacement_in_hook_refuses_before_dispatch():
    async def hook(name, payload):
        if name == "llm:request":
            p._client = SimpleNamespace()

    p, _ = provider(hook=hook)
    first = p._client
    with pytest.raises(InvalidRequestError):
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    first.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_failure_after_any_chunk_never_replays():
    p, _ = provider(stream=True, retries=2)

    async def chunks():
        yield SimpleNamespace(choices=[], usage=None)
        raise openai.APIConnectionError(request=MagicMock())

    p._client.chat.completions.create.return_value = chunks()
    with pytest.raises(LLMError) as exc:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert exc.value.retryable is False
    assert p._client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_diagnostics_omit_before_hook_without_capability_logging_operations():
    p, events = provider()
    payload = {
        "raw": {"messages": [{"role": "user", "content": [image().model_dump()]}]},
        "data_url": "data:image/png;base64,cGl4ZWxz",
        "status": "unknown",
        "reasoning": "opaque signature",
        "tool_call_id": "c1",
    }
    before = copy.deepcopy(payload)
    await p._emit_event("llm:request", payload)
    saved = events[0][1]
    assert payload == before
    assert "cGl4ZWxz" not in str(saved) and NOT_RETAINED in str(saved)
    assert saved["status"] == "unknown" and saved["reasoning"] == "opaque signature"
    assert "sha256" not in str(saved) and "blob_ref" not in str(saved)
    assert (
        diagnostic_copy('{"type":"image","data":"not parsed"}')
        == '{"type":"image","data":"not parsed"}'
    )
