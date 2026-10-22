"""Unit tests for llm:stream_* event contract in _complete_streaming().

Contract reference: docs/provider-streaming-contract.md (in streaming-text repo).

Four events, all sharing one request_id per call:
  llm:stream_block_start  – once per block, before its deltas
  llm:stream_block_delta  – each non-empty content fragment (text OR thinking);
                            block_type ("text"|"thinking") carried on every delta
  llm:stream_block_end    – when a block completes
  llm:stream_aborted      – mid-stream exception AFTER partial emit

There is NO separate llm:stream_thinking_delta event.  Both text and reasoning
fragments use llm:stream_block_delta; consumers route on block_type.

Per-request override:
  request.metadata == {"stream": False}  ->  non-streaming path (no stream_* events)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message, TextBlock, ThinkingBlock

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def names(self) -> list[str]:
        return [e[0] for e in self.events]

    def payloads_for(self, name: str) -> list[dict]:
        return [p for n, p in self.events if n == name]

    def stream_events(self) -> list[tuple[str, dict]]:
        return [(n, p) for n, p in self.events if n.startswith("llm:stream_")]


class _FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = _FakeHooks()


def _make_provider(*, use_streaming: bool = True) -> ChatCompletionsProvider:
    """Return a ChatCompletionsProvider wired with a FakeCoordinator."""
    provider = ChatCompletionsProvider(
        config={
            "model": "test-model",
            "use_streaming": str(use_streaming).lower(),
            "max_retries": "0",
        },
        coordinator=_FakeCoordinator(),
    )
    return provider


def _make_mock_chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage=None,
):
    """Build a minimal MagicMock that looks like an OpenAI streaming chunk."""
    delta = MagicMock(spec=[])  # spec=[] prevents auto-creating unknown attrs
    delta.content = content
    delta.tool_calls = tool_calls
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _make_tc_delta(index: int, tc_id: str | None, name: str | None, args: str):
    """Build a partial tool-call delta."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = args

    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.function = fn
    return tc


def _set_fake_stream(provider: ChatCompletionsProvider, chunks: list) -> None:
    """Wire a fake async-generator stream onto provider._client."""
    mock_client = AsyncMock()
    provider._client = mock_client

    async def _gen():
        for c in chunks:
            yield c

    mock_client.chat.completions.create = AsyncMock(return_value=_gen())


def _simple_request(**kwargs) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        model="test-model",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Text-only stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_stream_emits_block_start_deltas_block_end():
    """text-only stream -> block_start(text) -> block_delta x N -> block_end."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content="Hello"),
        _make_mock_chunk(content=" world"),
        _make_mock_chunk(content="!"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream = hooks.stream_events()
    names = [n for n, _ in stream]

    assert "llm:stream_block_start" in names
    assert "llm:stream_block_delta" in names
    assert "llm:stream_block_end" in names
    assert "llm:stream_aborted" not in names

    # block_start comes before any block_delta; block_delta before block_end
    start_idx = names.index("llm:stream_block_start")
    delta_idx = names.index("llm:stream_block_delta")
    end_idx = names.index("llm:stream_block_end")
    assert start_idx < delta_idx < end_idx

    # Only one block (text)
    starts = hooks.payloads_for("llm:stream_block_start")
    ends = hooks.payloads_for("llm:stream_block_end")
    assert len(starts) == 1
    assert starts[0]["block_type"] == "text"
    assert len(ends) == 1
    assert ends[0]["block_type"] == "text"

    # Three deltas (one per non-empty content chunk), each carrying block_type:"text"
    deltas = hooks.payloads_for("llm:stream_block_delta")
    assert len(deltas) == 3
    assert all(d["block_type"] == "text" for d in deltas)


@pytest.mark.asyncio
async def test_text_delta_sequence_zero_based_per_block():
    """sequence numbers are 0-based and per-block."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content="a"),
        _make_mock_chunk(content="b"),
        _make_mock_chunk(content="c"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")
    seqs = [d["sequence"] for d in deltas]
    assert seqs == [0, 1, 2]


@pytest.mark.asyncio
async def test_single_request_id_across_all_events():
    """All stream events for one call share the same request_id."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content="x"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream_events = hooks.stream_events()
    assert stream_events, "Expected at least one stream event"

    request_ids = {p["request_id"] for _, p in stream_events}
    assert len(request_ids) == 1, f"Expected one request_id, got: {request_ids}"


@pytest.mark.asyncio
async def test_block_index_in_payloads():
    """block_start, block_delta, and block_end all carry the same block_index."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content="hi"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    start = hooks.payloads_for("llm:stream_block_start")[0]
    delta = hooks.payloads_for("llm:stream_block_delta")[0]
    end = hooks.payloads_for("llm:stream_block_end")[0]

    assert start["block_index"] == delta["block_index"] == end["block_index"]
    assert start["block_index"] == 0  # first and only block


@pytest.mark.asyncio
async def test_empty_content_fragment_not_emitted():
    """Empty-string content does not produce a block_delta event."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content=""),  # empty -- must be skipped
        _make_mock_chunk(content="ok"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")
    # Exactly one delta for "ok"; empty string must not produce one
    assert len(deltas) == 1
    assert deltas[0]["text"] == "ok"
    assert deltas[0]["block_type"] == "text"


# ---------------------------------------------------------------------------
# 2. Reasoning / thinking model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_model_emits_thinking_block_delta_before_text_block_delta():
    """Thinking and text both use llm:stream_block_delta; block_type routes them.

    Contract: ONE delta event for all content.  No llm:stream_thinking_delta.
    Thinking deltas (block_type="thinking") come before text deltas (block_type="text").
    """
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(reasoning_content="step1"),
        _make_mock_chunk(reasoning_content="step2"),
        _make_mock_chunk(content="answer"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream = hooks.stream_events()
    names = [n for n, _ in stream]

    # No separate thinking_delta event -- merged into block_delta
    assert "llm:stream_thinking_delta" not in names
    assert "llm:stream_block_delta" in names

    # All block_delta payloads carry block_type
    all_deltas = hooks.payloads_for("llm:stream_block_delta")
    assert all("block_type" in d for d in all_deltas)

    # Thinking deltas (block_type="thinking") come before text deltas (block_type="text")
    thinking_deltas = [(i, n) for i, (n, p) in enumerate(stream)
                       if n == "llm:stream_block_delta" and p.get("block_type") == "thinking"]
    text_deltas = [(i, n) for i, (n, p) in enumerate(stream)
                   if n == "llm:stream_block_delta" and p.get("block_type") == "text"]
    assert thinking_deltas, "Expected at least one thinking block_delta"
    assert text_deltas, "Expected at least one text block_delta"
    assert thinking_deltas[-1][0] < text_deltas[0][0], (
        "Last thinking delta must come before first text delta"
    )

    # Exactly two block_starts: thinking (idx 0) and text (idx 1)
    starts = hooks.payloads_for("llm:stream_block_start")
    assert len(starts) == 2
    assert starts[0]["block_type"] == "thinking"
    assert starts[0]["block_index"] == 0
    assert starts[1]["block_type"] == "text"
    assert starts[1]["block_index"] == 1

    # Two block_ends
    ends = hooks.payloads_for("llm:stream_block_end")
    assert len(ends) == 2
    end_types = {e["block_type"] for e in ends}
    assert end_types == {"thinking", "text"}


@pytest.mark.asyncio
async def test_thinking_block_delta_sequence_zero_based():
    """thinking block_delta sequence numbers start at 0 and are per-block."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(reasoning_content="a"),
        _make_mock_chunk(reasoning_content="b"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    # Filter block_delta events to thinking blocks only
    all_deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")
    deltas = [d for d in all_deltas if d.get("block_type") == "thinking"]
    seqs = [d["sequence"] for d in deltas]
    assert seqs == [0, 1]


@pytest.mark.asyncio
async def test_thinking_block_delta_carries_block_type_and_text():
    """llm:stream_block_delta for thinking carries block_type="thinking" and text."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(reasoning_content="some reasoning"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    all_deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")
    thinking_deltas = [d for d in all_deltas if d.get("block_type") == "thinking"]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]["text"] == "some reasoning"
    assert thinking_deltas[0]["block_type"] == "thinking"


@pytest.mark.asyncio
async def test_reasoning_content_does_not_become_persisted_thinking_block():
    """The persisted ChatResponse.content must never contain a ThinkingBlock.

    reasoning_content deltas still drive the ephemeral llm:stream_block_delta /
    llm:stream_block_end UI events above (render-only, never replayed to a
    provider) -- but the persisted content returned from complete() must be
    free of ThinkingBlock, since chat-completions has no signed extended
    thinking and a fabricated block would carry signature=None. Replaying
    that history to provider-anthropic 400s the whole request (see
    microsoft-amplifier/amplifier-support#206).
    """
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(reasoning_content="step1"),
        _make_mock_chunk(content="answer"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    resp = await provider.complete(_simple_request())

    assert not any(isinstance(b, ThinkingBlock) for b in resp.content)
    text_blocks = [b for b in resp.content if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "answer"

    # The ephemeral UI stream is untouched: thinking deltas still fired.
    hooks = provider.coordinator.hooks
    thinking_deltas = [
        d
        for d in hooks.payloads_for("llm:stream_block_delta")
        if d.get("block_type") == "thinking"
    ]
    assert thinking_deltas, "Expected ephemeral thinking block_delta to still fire"


# ---------------------------------------------------------------------------
# 3. Tool use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_use_emits_block_start_and_block_end_no_arg_deltas():
    """tool_use block: block_start with name + block_end; no per-argument deltas."""
    provider = _make_provider()
    tc1 = _make_tc_delta(0, "call_1", "grep", '{"pat":')
    tc2 = _make_tc_delta(0, None, None, '"test"}')

    chunks = [
        _make_mock_chunk(tool_calls=[tc1]),
        _make_mock_chunk(tool_calls=[tc2]),
        _make_mock_chunk(finish_reason="tool_calls"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream_names = hooks.names()

    # block_start must appear
    assert "llm:stream_block_start" in stream_names
    # block_end must appear
    assert "llm:stream_block_end" in stream_names
    # No block_delta for tool use (tool-use blocks have no per-fragment deltas)
    assert "llm:stream_block_delta" not in stream_names

    start = hooks.payloads_for("llm:stream_block_start")[0]
    assert start["block_type"] == "tool_use"
    assert start["name"] == "grep"

    end = hooks.payloads_for("llm:stream_block_end")[0]
    assert end["block_type"] == "tool_use"
    assert end["block_index"] == start["block_index"]


@pytest.mark.asyncio
async def test_tool_use_block_gets_index_zero_when_first():
    """tool_use only response: block_index is 0."""
    provider = _make_provider()
    tc1 = _make_tc_delta(0, "call_1", "fn", '{"a":1}')

    chunks = [
        _make_mock_chunk(tool_calls=[tc1]),
        _make_mock_chunk(finish_reason="tool_calls"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    start = provider.coordinator.hooks.payloads_for("llm:stream_block_start")[0]
    assert start["block_index"] == 0


# ---------------------------------------------------------------------------
# 4. Per-request stream override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_stream_false_uses_nonstreaming_path():
    """metadata={"stream": False} routes to non-streaming even if use_streaming=True."""
    provider = _make_provider(use_streaming=True)

    # Non-streaming mock returns a plain completion object
    mock_client = AsyncMock()
    provider._client = mock_client

    ns_response = MagicMock()
    ns_response.choices = [MagicMock()]
    ns_response.choices[0].message.content = "hello"
    ns_response.choices[0].message.tool_calls = None
    ns_response.choices[0].message.reasoning_content = None
    ns_response.choices[0].finish_reason = "stop"
    ns_response.usage = MagicMock()
    ns_response.usage.prompt_tokens = 5
    ns_response.usage.completion_tokens = 3
    ns_response.usage.total_tokens = 8
    ns_response.usage.prompt_tokens_details = None
    mock_client.chat.completions.create.return_value = ns_response

    request = _simple_request(metadata={"stream": False})
    await provider.complete(request)

    hooks = provider.coordinator.hooks
    stream_names = [n for n, _ in hooks.stream_events()]
    assert stream_names == [], (
        f"Expected no stream events for metadata={{stream: False}}, got {stream_names}"
    )


@pytest.mark.asyncio
async def test_metadata_stream_false_identity_check_not_truthiness():
    """metadata={"stream": 0} (falsy but not False) must NOT override streaming."""
    provider = _make_provider(use_streaming=True)

    chunks = [
        _make_mock_chunk(content="hi"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    # stream=0 is falsy but not `is False`, so streaming should still be used
    request = _simple_request(metadata={"stream": 0})
    await provider.complete(request)

    hooks = provider.coordinator.hooks
    stream_events = hooks.stream_events()
    assert len(stream_events) > 0, (
        "metadata={'stream': 0} should NOT suppress streaming (identity check, not truthiness)"
    )


@pytest.mark.asyncio
async def test_config_use_streaming_false_no_stream_events():
    """use_streaming=False at config level -> no stream events (existing behavior)."""
    provider = _make_provider(use_streaming=False)

    mock_client = AsyncMock()
    provider._client = mock_client

    ns_response = MagicMock()
    ns_response.choices = [MagicMock()]
    ns_response.choices[0].message.content = "hello"
    ns_response.choices[0].message.tool_calls = None
    ns_response.choices[0].message.reasoning_content = None
    ns_response.choices[0].finish_reason = "stop"
    ns_response.usage = MagicMock()
    ns_response.usage.prompt_tokens = 5
    ns_response.usage.completion_tokens = 3
    ns_response.usage.total_tokens = 8
    ns_response.usage.prompt_tokens_details = None
    mock_client.chat.completions.create.return_value = ns_response

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream_names = [n for n, _ in hooks.stream_events()]
    assert stream_names == [], f"Expected no stream events, got {stream_names}"


# ---------------------------------------------------------------------------
# 5. Mid-stream error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_stream_error_after_partial_emits_aborted():
    """If an exception occurs after at least one delta, emit llm:stream_aborted."""
    provider = _make_provider()

    async def _bad_gen():
        yield _make_mock_chunk(content="partial")
        raise RuntimeError("connection lost")

    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create = AsyncMock(return_value=_bad_gen())

    with pytest.raises(Exception):
        await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    aborted = hooks.payloads_for("llm:stream_aborted")
    assert len(aborted) == 1, "Expected exactly one llm:stream_aborted event"

    payload = aborted[0]
    assert "request_id" in payload
    assert "error" in payload
    assert payload["error"]["type"] == "RuntimeError"
    assert "connection lost" in payload["error"]["msg"]


@pytest.mark.asyncio
async def test_mid_stream_error_before_any_emit_no_aborted():
    """If the error fires before any delta, llm:stream_aborted must NOT be emitted."""
    provider = _make_provider()

    async def _bad_gen():
        raise RuntimeError("immediate failure")
        yield  # make it an async generator

    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create = AsyncMock(return_value=_bad_gen())

    with pytest.raises(Exception):
        await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    aborted = hooks.payloads_for("llm:stream_aborted")
    assert aborted == [], (
        "llm:stream_aborted must not be emitted when no deltas were sent"
    )


@pytest.mark.asyncio
async def test_mid_stream_error_aborted_carries_request_id():
    """llm:stream_aborted shares the same request_id as the preceding stream events."""
    provider = _make_provider()

    async def _bad_gen():
        yield _make_mock_chunk(content="hello")
        raise ValueError("oops")

    mock_client = AsyncMock()
    provider._client = mock_client
    mock_client.chat.completions.create = AsyncMock(return_value=_bad_gen())

    with pytest.raises(Exception):
        await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    stream_events = hooks.stream_events()
    request_ids = {p["request_id"] for _, p in stream_events}
    assert len(request_ids) == 1, (
        "All stream events (incl. aborted) must share one request_id"
    )


# ---------------------------------------------------------------------------
# 6. No coordinator -- must not crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_coordinator_streaming_does_not_crash():
    """Provider without a coordinator must complete without raising."""
    provider = ChatCompletionsProvider(
        config={
            "model": "test-model",
            "max_retries": "0",
        }
        # coordinator intentionally absent (defaults to None)
    )

    chunks = [
        _make_mock_chunk(content="ok"),
        _make_mock_chunk(finish_reason="stop"),
    ]

    mock_client = AsyncMock()
    provider._client = mock_client

    async def _gen():
        for c in chunks:
            yield c

    mock_client.chat.completions.create = AsyncMock(return_value=_gen())

    result = await provider.complete(_simple_request())
    assert result is not None


# ---------------------------------------------------------------------------
# 7. Block-end emitted for every started block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_end_emitted_for_every_started_block_thinking_and_text():
    """Two blocks started -> two block_ends emitted."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(reasoning_content="think"),
        _make_mock_chunk(content="reply"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    ends = provider.coordinator.hooks.payloads_for("llm:stream_block_end")
    assert len(ends) == 2
    end_types = {e["block_type"] for e in ends}
    assert end_types == {"thinking", "text"}


@pytest.mark.asyncio
async def test_block_end_after_loop_not_inside_loop():
    """block_end must come after all deltas -- not interleaved inside the loop."""
    provider = _make_provider()
    chunks = [
        _make_mock_chunk(content="a"),
        _make_mock_chunk(content="b"),
        _make_mock_chunk(finish_reason="stop"),
    ]
    _set_fake_stream(provider, chunks)

    await provider.complete(_simple_request())

    hooks = provider.coordinator.hooks
    names = [n for n, _ in hooks.stream_events()]
    last_delta_idx = max(
        i for i, n in enumerate(names) if n == "llm:stream_block_delta"
    )
    end_idx = names.index("llm:stream_block_end")
    assert end_idx > last_delta_idx, "block_end must be after the last block_delta"
