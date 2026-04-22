"""Integration tests for ChatCompletionsProvider against a live llama-server.

These tests require a running llama-server or Ollama-compatible instance.
They are skipped automatically if the server is not reachable.

Environment variables:
    INTEGRATION_BASE_URL: Override the server base URL
                          (default: http://localhost:8080/v1 on host,
                           or http://host.docker.internal:8080/v1 from container)
    INTEGRATION_MODEL:    Override the model name
                          (default: gemma26-long)

Run integration tests explicitly:
    pytest tests/test_integration.py -v

These tests are NOT included in the default pytest run (see conftest.py).
"""

import os
import urllib.request
import urllib.error

import pytest

from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_chat_completions import ChatCompletionsProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INTEGRATION_BASE_URL: str = os.environ.get(
    "INTEGRATION_BASE_URL", "http://localhost:8080/v1"
)
INTEGRATION_MODEL: str = os.environ.get("INTEGRATION_MODEL", "gemma26-long")


def _server_reachable() -> bool:
    """Return True if the configured llama-server health endpoint responds OK."""
    # Strip trailing /v1 or /v1/ to get the base server URL
    base = INTEGRATION_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health_url = f"{base}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


_SERVER_UP = _server_reachable()

pytestmark = pytest.mark.skipif(
    not _SERVER_UP,
    reason=f"llama-server not reachable at {INTEGRATION_BASE_URL} — skipping integration tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(*, use_streaming: bool = False) -> ChatCompletionsProvider:
    """Create a ChatCompletionsProvider pointed at the integration server."""
    return ChatCompletionsProvider(
        config={
            "base_url": INTEGRATION_BASE_URL,
            "model": INTEGRATION_MODEL,
            "timeout": 120.0,
            "max_retries": 1,
            "use_streaming": use_streaming,
        }
    )


# ---------------------------------------------------------------------------
# Test: simple text prompt
# ---------------------------------------------------------------------------


class TestSimplePrompt:
    """Integration test: simple text prompt returns content and usage."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Provider returns ChatResponse with text content and non-zero usage."""
        provider = _make_provider()
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="Say hello in exactly 3 words.")],
                )
            ]
        )

        try:
            response = await provider.complete(request)

            # Response must exist and have content
            assert response is not None, "Response should not be None"
            assert response.content is not None, "Response content should not be None"
            assert len(response.content) > 0, (
                "Response should have at least one content block"
            )

            # At least one TextBlock with non-empty text
            text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
            assert len(text_blocks) > 0, (
                f"Response should have at least one TextBlock, got: "
                f"{[type(b).__name__ for b in response.content]}"
            )
            assert text_blocks[0].text.strip(), "TextBlock text should be non-empty"

            # Usage must be present and non-zero
            assert response.usage is not None, "Response should include usage data"
            assert response.usage.input_tokens > 0, (
                f"Input tokens should be > 0, got {response.usage.input_tokens}"
            )
            assert response.usage.output_tokens > 0, (
                f"Output tokens should be > 0, got {response.usage.output_tokens}"
            )

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_simple_text_response_streaming(self):
        """Provider returns ChatResponse via streaming with text content and usage."""
        provider = _make_provider(use_streaming=True)
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="Say hello in exactly 3 words.")],
                )
            ]
        )

        try:
            response = await provider.complete(request)

            assert response is not None, "Response should not be None"
            assert response.content is not None, "Response content should not be None"
            assert len(response.content) > 0, (
                "Response should have at least one content block"
            )

            text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
            assert len(text_blocks) > 0, (
                "Streaming response should have at least one TextBlock"
            )
            assert text_blocks[0].text.strip(), "TextBlock text should be non-empty"

            # Note: llama.cpp streaming may omit usage — accept None gracefully
            if response.usage is not None:
                assert response.usage.output_tokens >= 0, (
                    "Output tokens should be non-negative"
                )

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_provider_closes_cleanly(self):
        """Provider.close() completes without error."""
        provider = _make_provider()
        # Trigger client initialisation
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="Hi.")],
                )
            ]
        )
        await provider.complete(request)
        # close() should not raise
        await provider.close()


# ---------------------------------------------------------------------------
# Test: tool calling
# ---------------------------------------------------------------------------


class TestToolCalling:
    """Integration test: tool calling returns tool_calls or text content."""

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """Provider returns response with tool_calls or text when tools are provided."""
        provider = _make_provider()
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="What is 2+2? Use the calculator tool.")],
                )
            ],
            tools=[
                ToolSpec(
                    name="calculator",
                    description="Evaluate math",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                )
            ],
        )

        try:
            response = await provider.complete(request)

            # Response must exist and have content
            assert response is not None, "Response should not be None"
            assert response.content is not None, "Response content should not be None"
            assert len(response.content) > 0, (
                "Response should have at least one content block"
            )

            # Response must have tool_calls OR text — either is acceptable
            tool_call_blocks = [
                b for b in response.content if isinstance(b, ToolCallBlock)
            ]
            text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

            has_tool_calls = len(tool_call_blocks) > 0
            has_text = len(text_blocks) > 0

            assert has_tool_calls or has_text, (
                f"Response should have tool_calls or text content, "
                f"got: {[type(b).__name__ for b in response.content]}"
            )

            if has_tool_calls:
                # Verify tool call has expected structure
                tc = tool_call_blocks[0]
                assert tc.id, "ToolCallBlock should have an id"
                assert tc.name, "ToolCallBlock should have a name"

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_tool_call_response_streaming(self):
        """Provider returns tool_calls or text via streaming when tools are provided."""
        provider = _make_provider(use_streaming=True)
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="What is 2+2? Use the calculator tool.")],
                )
            ],
            tools=[
                ToolSpec(
                    name="calculator",
                    description="Evaluate math",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                )
            ],
        )

        try:
            response = await provider.complete(request)

            assert response is not None, "Response should not be None"
            assert response.content is not None, "Response content should not be None"
            assert len(response.content) > 0, (
                "Response should have at least one content block"
            )

            tool_call_blocks = [
                b for b in response.content if isinstance(b, ToolCallBlock)
            ]
            text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

            assert len(tool_call_blocks) > 0 or len(text_blocks) > 0, (
                "Streaming response should have tool_calls or text content"
            )

        finally:
            await provider.close()
