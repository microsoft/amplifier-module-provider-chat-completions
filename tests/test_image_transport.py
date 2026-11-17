"""Offline public-entry -> actual OpenAI SDK wire regression fixtures."""

import copy
import json
import traceback
from types import SimpleNamespace

import httpx
import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    ReasoningBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)

from amplifier_module_provider_chat_completions import ChatCompletionsProvider

# A complete 1x1 PNG, not an opaque base64-looking placeholder.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
URL = "https://images.invalid/private-screenshot.png?token=do-not-retain"


def image(source=None):
    return ImageBlock(
        source=source or {"type": "base64", "media_type": "image/png", "data": PNG}
    )


def call(*ids):
    return Message(
        role="assistant",
        content=[ToolCallBlock(id=id, name="inspect", input={}) for id in ids],
    )


class WireFixture:
    def __init__(self, stream=False, status=200, mutate_hook=False, echo=False):
        self.requests = []
        self.events = []
        self.stream = stream
        self.status = status
        self.mutate_hook = mutate_hook
        self.echo = echo
        self.provider = ChatCompletionsProvider(
            {
                "model": "unknown-backend-alias",
                "use_streaming": stream,
                "raw": True,
                "top_p": 0.9,
                "seed": 7,
            },
            coordinator=SimpleNamespace(hooks=SimpleNamespace(emit=self.emit)),
        )
        self.provider._client = openai.AsyncOpenAI(
            api_key="fixture-not-a-key",
            base_url="https://backend.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.respond)),
        )

    async def emit(self, name, payload):
        # This is the FIRST hook subscriber, not a later recorder/redactor.
        self.events.append((name, copy.deepcopy(payload)))
        if self.mutate_hook and name == "llm:request":
            payload["raw"]["messages"].clear()

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        status = self.status.pop(0) if isinstance(self.status, list) else self.status
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"message": f"invalid image {URL} {PNG}"}},
            )
        if body.get("stream"):
            chunk = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
                ],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"echo {PNG} {URL}" if self.echo else "ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def complete(self, messages):
        request = ChatRequest(
            messages=messages, model="selected-alias", temperature=0.2
        )
        before = copy.deepcopy(request.model_dump())
        try:
            self.response = await self.provider.complete(request)
            assert request.model_dump() == before
            return self.requests[-1]["messages"]
        finally:
            await self.provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
async def test_ordered_user_images_and_history_reach_sdk(stream, as_dict):
    fixture = WireFixture(stream)
    blocks = [
        TextBlock(text="before"),
        image(),
        TextBlock(text="between"),
        image({"type": "url", "url": URL}),
        TextBlock(text="after"),
    ]
    if as_dict:
        blocks = [block.model_dump() for block in blocks]
    wire = await fixture.complete(
        [
            Message(role="user", content=blocks),
            Message(role="assistant", content="earlier answer"),
            Message(role="user", content="plain followup"),
        ]
    )
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[0]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        {"type": "text", "text": "between"},
        {"type": "image_url", "image_url": {"url": URL}},
        {"type": "text", "text": "after"},
    ]
    assert fixture.requests[0]["model"] == "selected-alias"
    # Preserve existing request parameter policy (temperature is not sent here).
    assert fixture.requests[0]["top_p"] == 0.9
    assert fixture.requests[0]["seed"] == 7
    assert "temperature" not in fixture.requests[0]
    assert fixture.requests[0]["stream"] is stream
    diagnostics = json.dumps(fixture.events)
    assert PNG not in diagnostics and URL not in diagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_direct_tool_image_is_unavailable_in_original_correlated_slot(stream):
    fixture = WireFixture(stream)
    wire = await fixture.complete(
        [
            Message(role="user", content="inspect"),
            call("one"),
            Message(
                role="tool",
                tool_call_id="one",
                name="inspect",
                content=[TextBlock(text="before"), image(), TextBlock(text="after")],
            ),
            Message(role="user", content="continue without the image"),
        ]
    )
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "user"]
    result = wire[2]
    assert result["tool_call_id"] == "one" and result["name"] == "inspect"
    text = result["content"]
    assert isinstance(text, str)
    assert text.index("before") < text.index("Image unavailable") < text.index("after")
    assert "no pixels" in text and "user" in text
    assert PNG not in json.dumps(wire)
    assert wire[-1]["content"] == "continue without the image"


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_nested_results_keep_call_order_and_error_status(nested):
    output = [TextBlock(text="before"), image(), {"type": "text", "text": "after"}]
    if nested:
        output = {"success": False, "error": "capture failed", "output": output}
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one", "two"),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(tool_call_id="one", output=output),
                    ToolResultBlock(tool_call_id="two", output="second result"),
                ],
            ),
        ]
    )
    assert [m["tool_call_id"] for m in wire[1:]] == ["one", "two"]
    text = wire[1]["content"]
    assert "before" in text and "after" in text and "Image unavailable" in text
    assert PNG not in text and "ImageBlock" not in text
    if nested:
        assert "'success': False" in text and "capture failed" in text
    assert wire[2]["content"] == "second result"


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "system", "developer"])
async def test_nonuser_images_are_explicit_and_do_not_poison_plain_followup(role):
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            Message(role=role, content=[TextBlock(text="context"), image()]),
            Message(role="user", content="a plain followup"),
        ]
    )
    assert len(wire) == 2  # Existing developer/system normalization is unchanged.
    assert "Image unavailable" in wire[0]["content"]
    assert PNG not in json.dumps(wire)
    assert wire[-1]["content"] == "a plain followup"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/png", "data": "malformed-secret"},
        {"type": "base64", "media_type": "text/plain", "data": PNG},
        {"type": "url", "url": "file:///private/screenshot.png"},
        {"type": "url"},
        {"type": "future", "secret": URL},
    ],
)
async def test_unsupported_source_is_payload_free_in_its_user_slot(source):
    fixture = WireFixture()
    wire = await fixture.complete([Message(role="user", content=[image(source)])])
    assert len(wire) == 1 and wire[0]["role"] == "user"
    assert "Image unavailable" in wire[0]["content"]
    assert URL not in json.dumps(wire) and PNG not in json.dumps(wire)
    assert "malformed-secret" not in json.dumps(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "orphan", "mismatch", "duplicate", "late"])
async def test_invalid_image_result_correlation_refuses_before_sdk(case):
    result = Message(role="tool", tool_call_id="one", content=[image()])
    messages = [call("one"), result]
    if case == "missing":
        result.tool_call_id = None
    elif case == "orphan":
        messages = [result]
    elif case == "mismatch":
        result.tool_call_id = "other"
    elif case == "duplicate":
        messages.append(result)
    elif case == "late":
        messages.insert(1, Message(role="user", content="interrupted"))
    fixture = WireFixture()
    with pytest.raises((ValueError, InvalidRequestError)) as exc:
        await fixture.complete(messages)
    assert fixture.requests == []
    assert PNG not in str(exc.value)


@pytest.mark.asyncio
async def test_diagnostic_copy_is_not_the_transport_object():
    fixture = WireFixture(mutate_hook=True)
    wire = await fixture.complete([Message(role="user", content=[image()])])
    assert wire[0]["content"][0]["image_url"]["url"].endswith(PNG)
    assert PNG not in json.dumps(fixture.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_backend_image_error_echo_does_not_escape_or_retry(stream):
    fixture = WireFixture(stream, status=400)
    with pytest.raises(InvalidRequestError) as exc:
        await fixture.complete([Message(role="user", content=[image()])])
    assert len(fixture.requests) == 1
    assert not exc.value.retryable
    rendered = "".join(traceback.format_exception(exc.value))
    assert PNG not in rendered and URL not in rendered
    assert PNG not in json.dumps(fixture.events)
    assert URL not in json.dumps(fixture.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [False, 0, "", None, {}, []])
async def test_falsey_nonimage_outputs_keep_legacy_serialization(output):
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[ToolResultBlock(tool_call_id="one", output=output)],
            ),
        ]
    )
    assert wire[-1]["content"] == str(output)


@pytest.mark.asyncio
async def test_business_type_image_mapping_and_status_are_not_image_blocks():
    business = {
        "type": "image",
        "id": "product-record",
        "success": False,
        "count": 0,
        "output": None,
    }
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(
                        tool_call_id="one",
                        output={"content": [business, image().model_dump()]},
                    )
                ],
            ),
        ]
    )
    assert str(business) in wire[-1]["content"]
    assert "Image unavailable" in wire[-1]["content"]
    assert PNG not in wire[-1]["content"]


def test_base64_only_interleaving_source_regression():
    """Isolate the accepted-main reorder bug from the separate URL KeyError."""
    provider = ChatCompletionsProvider()
    wire = provider._convert_messages_to_wire(
        [
            Message(
                role="user",
                content=[
                    TextBlock(text="one"),
                    image(),
                    TextBlock(text="two"),
                    image(),
                    TextBlock(text="three"),
                ],
            )
        ]
    )
    assert [part["type"] for part in wire[0]["content"]] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]


@pytest.mark.asyncio
async def test_url_image_in_tool_output_is_not_textualized():
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(
                        tool_call_id="one",
                        output={
                            "success": True,
                            "output": [
                                {"type": "text", "text": "capture metadata"},
                                image({"type": "url", "url": URL}).model_dump(),
                            ],
                        },
                    )
                ],
            ),
        ]
    )
    assert "Image unavailable" in wire[-1]["content"]
    assert "'success': True" in wire[-1]["content"]
    assert URL not in json.dumps(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_existing_rate_limit_retry_keeps_identical_pixels_and_safe_events(stream):
    fixture = WireFixture(stream, status=[429, 200])
    fixture.provider._max_retries = 1
    fixture.provider._min_retry_delay = 0
    fixture.provider._max_retry_delay = 0
    await fixture.complete([Message(role="user", content=[image()])])
    assert len(fixture.requests) == 2
    assert fixture.requests[0] == fixture.requests[1]
    sent_image = fixture.requests[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert sent_image.endswith(PNG)
    assert any(name == "provider:retry" for name, _ in fixture.events)
    assert PNG not in json.dumps(fixture.events)
    assert URL not in json.dumps(fixture.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_content", [False, True])
async def test_ordinary_nested_business_record_is_not_transport(with_content):
    business = {
        "record": {
            "type": "image",
            "source": {"type": "url", "url": "businessvalue"},
        },
        "status": False,
        "count": 0,
        "empty": None,
    }
    output = copy.deepcopy(business)
    if with_content:
        output["content"] = [
            {"type": "text", "text": "actual capture"},
            image().model_dump(),
        ]
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[ToolResultBlock(tool_call_id="one", output=output)],
            ),
        ]
    )
    result = wire[-1]["content"]
    if with_content:
        assert str(business["record"]) in result
        assert "Image unavailable" in result
        assert "'status': False" in result and "'count': 0" in result
        assert PNG not in result
    else:
        assert result == str(business)


@pytest.mark.asyncio
async def test_image_looking_ordinary_tool_text_is_not_parsed_or_erased():
    text = f"Ordinary opaque text, not a content block: {URL} {PNG}"
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[ToolResultBlock(tool_call_id="one", output=text)],
            ),
        ]
    )
    assert wire[-1]["content"] == text


@pytest.mark.asyncio
async def test_raw_response_omits_known_bare_image_data_and_signed_url_echoes():
    fixture = WireFixture(echo=True)
    await fixture.complete(
        [
            Message(
                role="user",
                content=[image(), image({"type": "url", "url": URL})],
            )
        ]
    )
    # Omission is in diagnostic copies, not the model response content path.
    assert fixture.response.text == f"echo {PNG} {URL}"
    raw = [payload["raw"] for name, payload in fixture.events if name == "llm:response"]
    assert len(raw) == 1
    assert PNG not in json.dumps(raw) and URL not in json.dumps(raw)


def test_diagnostics_do_not_guess_arbitrary_text_is_a_known_image():
    from amplifier_module_provider_chat_completions._images import diagnostic_copy

    ordinary = {"content": f"ordinary opaque text: {PNG} {URL}"}
    assert diagnostic_copy(ordinary) == ordinary


@pytest.mark.asyncio
async def test_system_reminder_is_coalesced_before_image_tool_correlation():
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(role="system", content="current reminder"),
            Message(
                role="tool",
                tool_call_id="one",
                name="read_file",
                content=[TextBlock(text="tool text"), image()],
            ),
        ]
    )
    assert [message["role"] for message in wire] == ["system", "assistant", "tool"]
    assert wire[-1]["tool_call_id"] == "one"
    assert wire[-1]["name"] == "read_file"
    assert "Image unavailable" in wire[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("direct", [False, True])
async def test_explicit_tool_envelope_error_status_is_preserved(is_error, direct):
    if direct:
        result = Message(
            role="tool",
            tool_call_id="one",
            is_error=is_error,
            content=[TextBlock(text="capture text"), image()],
        )
    else:
        result = Message(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_call_id="one",
                    output=[TextBlock(text="capture text"), image()],
                    is_error=is_error,
                )
            ],
        )
    fixture = WireFixture()
    wire = await fixture.complete([call("one"), result])
    text = wire[-1]["content"]
    assert f"is_error={is_error}" in text
    assert "capture text" in text and "Image unavailable" in text
    assert PNG not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "shape", ["direct", "direct_dict", "nested", "nested_dict", "no_image"]
)
async def test_tool_thinking_and_signature_never_become_output_text(stream, shape):
    fixture = WireFixture(stream)
    blocks = [
        TextBlock(text="before"),
        ThinkingBlock(
            thinking="private-reasoning-fixture", signature="private-signature"
        ),
    ]
    if shape != "no_image":
        blocks.append(image())
    blocks.append(TextBlock(text="after"))
    if shape.endswith("_dict"):
        blocks = [block.model_dump() for block in blocks]
    if shape.startswith("nested"):
        result = Message(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_call_id="one",
                    output={"status": False, "content": blocks},
                )
            ],
        )
    else:
        result = Message(
            role="tool", tool_call_id="one", name="read_file", content=blocks
        )
    wire = await fixture.complete(
        [Message(role="user", content=[image()]), call("one"), result]
    )
    assert [message["role"] for message in wire] == ["user", "assistant", "tool"]
    # Preserve genuine user pixels; native tool-image support stays unavailable.
    assert wire[0]["content"][0]["image_url"]["url"].endswith(PNG)
    assert wire[2]["tool_call_id"] == "one"
    text = wire[2]["content"]
    assert text.index("before") < text.index("after")
    if shape == "no_image":
        assert text == "before\nafter"
    else:
        assert (
            text.index("before") < text.index("Image unavailable") < text.index("after")
        )
        assert PNG not in text
    if shape.startswith("nested"):
        assert "'status': False" in text
    else:
        assert wire[2]["name"] == "read_file"
    transport_and_diagnostics = json.dumps([fixture.requests, fixture.events])
    for private in ("private-reasoning-fixture", "private-signature", "ThinkingBlock"):
        assert private not in transport_and_diagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("kind", ["redacted_thinking", "reasoning", "tool_call"])
async def test_known_nonoutput_core_blocks_are_not_tool_text(stream, as_dict, kind):
    private = {
        "redacted_thinking": RedactedThinkingBlock(data="private-core-data"),
        "reasoning": ReasoningBlock(
            content=["private-core-data"], summary=["private-core-summary"]
        ),
        "tool_call": ToolCallBlock(
            id="not-an-output", name="inspect", input={"value": "private-core-data"}
        ),
    }[kind]
    blocks = [TextBlock(text="before"), private, image(), TextBlock(text="after")]
    if as_dict:
        blocks = [block.model_dump() for block in blocks]
    fixture = WireFixture(stream)
    wire = await fixture.complete(
        [
            call("one"),
            Message(role="tool", tool_call_id="one", name="read_file", content=blocks),
        ]
    )
    text = wire[1]["content"]
    assert text.startswith("before\n[Image unavailable") and text.endswith("]\nafter")
    assert wire[1]["tool_call_id"] == "one" and wire[1]["name"] == "read_file"
    combined = json.dumps([fixture.requests, fixture.events])
    assert "private-core" not in combined and "not-an-output" not in combined


@pytest.mark.asyncio
async def test_private_block_filter_preserves_business_fields_and_falsey_status():
    business = {"type": "thinking", "thinking": "ordinary-record", "signature": "id"}
    fixture = WireFixture()
    wire = await fixture.complete(
        [
            call("one"),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(
                        tool_call_id="one",
                        is_error=False,
                        output={
                            "record": business,
                            "success": False,
                            "count": 0,
                            "empty": None,
                            "output": {
                                "content": [
                                    ThinkingBlock(thinking="private-to-omit"),
                                    TextBlock(text="metadata"),
                                    image(),
                                ]
                            },
                        },
                    )
                ],
            ),
        ]
    )
    text = wire[1]["content"]
    assert str(business) in text and "is_error=False" in text
    assert (
        "'success': False" in text and "'count': 0" in text and "'empty': None" in text
    )
    assert "metadata" in text and "Image unavailable" in text
    assert "private-to-omit" not in text and PNG not in text
