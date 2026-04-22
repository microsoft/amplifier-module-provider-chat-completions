"""Amplifier provider module for OpenAI-compatible chat completions.

This module mounts a ChatCompletionsProvider into the Amplifier coordinator,
making it available as the 'chat-completions' provider.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

import openai

from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.utils import redact_secrets
from amplifier_core.llm_errors import AccessDeniedError as KernelAccessDeniedError
from amplifier_core.llm_errors import AuthenticationError as KernelAuthenticationError
from amplifier_core.llm_errors import ContentFilterError as KernelContentFilterError
from amplifier_core.llm_errors import ContextLengthError as KernelContextLengthError
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.llm_errors import LLMError as KernelLLMError
from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError
from amplifier_core.llm_errors import NotFoundError as KernelNotFoundError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError
from amplifier_core.models import ConfigField, ModelInfo, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)
from amplifier_core.content_models import TextContent, ThinkingContent, ToolCallContent

__all__ = ["mount", "ChatCompletionsProvider", "ChatCompletionsChatResponse"]
__amplifier_module_type__ = "provider"

logger = logging.getLogger(__name__)


class ChatCompletionsChatResponse(ChatResponse):
    """Extended ChatResponse with event-block support for streaming UI."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


class ChatCompletionsProvider:
    """Provider for OpenAI-compatible chat completions API."""

    @staticmethod
    def _config_bool(value: Any) -> bool:
        """Parse config booleans from YAML or CLI string values."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _config_int(value: Any, default: int) -> int:
        """Parse an int config value with a safe fallback."""
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid integer config value %r; using default %s",
                value,
                default,
            )
            return default

    @staticmethod
    def _config_float(value: Any, default: float) -> float:
        """Parse a float config value with a safe fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid float config value %r; using default %s",
                value,
                default,
            )
            return default

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: Any | None = None,
    ) -> None:
        """Initialise the provider with config and coordinator.

        Args:
            config: Provider configuration object.
            coordinator: Amplifier coordinator instance.
        """
        self.config = config or {}
        self.coordinator = coordinator

        # Name: honor per-instance config so multiple chat-completions providers
        # can be mounted with distinct routing identities. Falls back to the
        # generic "chat-completions" name when not specified.
        self.name: str = str(self.config.get("name", "chat-completions"))

        # base_url: env var takes precedence over config. Intentionally has NO
        # hardcoded default — an absent base_url must remain empty so that the
        # mount() silent-skip guard below fires, and so the ValueError in the
        # `client` property can signal a real misconfiguration at call time.
        # A default of "http://localhost:8080/v1" would mask missing config
        # behind a broken connection to a probably-non-existent local server.
        _configured_base_url = self.config.get("base_url")
        self._base_url: str = os.environ.get(
            "CHAT_COMPLETIONS_BASE_URL",
            str(_configured_base_url) if _configured_base_url else "",
        )

        # api_key: env var takes precedence over config, then "not-needed".
        # Empty string is rejected by the OpenAI client library, so we use
        # "not-needed" as a safe placeholder for local/keyless deployments.
        self._api_key: str = (
            os.environ.get("CHAT_COMPLETIONS_API_KEY")
            or self.config.get("api_key")
            or "not-needed"
        )

        # Prefer `default_model` (written by amplifier-app-cli's wizard to match
        # anthropic/openai conventions). Fall back to `model` for settings.yaml
        # files that still use the legacy key. Finally fall back to "default".
        self._model: str = str(
            self.config.get("default_model") or self.config.get("model") or "default"
        )
        self._client: openai.AsyncOpenAI | None = None
        self._timeout: float = self._config_float(
            self.config.get("timeout", 300.0), 300.0
        )
        self._temperature: float = self._config_float(
            self.config.get("temperature", 0.7), 0.7
        )
        self._max_tokens: int = self._config_int(
            self.config.get("max_tokens", 4096), 4096
        )
        self._max_retries: int = self._config_int(self.config.get("max_retries", 3), 3)
        self._min_retry_delay: float = self._config_float(
            self.config.get("min_retry_delay", 1.0), 1.0
        )
        self._max_retry_delay: float = self._config_float(
            self.config.get("max_retry_delay", 30.0), 30.0
        )
        self._repaired_tool_ids: set[str] = set()
        self._use_streaming: bool = self._config_bool(
            self.config.get("use_streaming", True)
        )

        # Task 5: Optional generation params
        _top_p_val = self.config.get("top_p")
        self._top_p: float | None = (
            self._config_float(_top_p_val, 0.0) if _top_p_val is not None else None
        )
        self._stop: list[str] | None = self.config.get("stop")
        _seed_val = self.config.get("seed")
        self._seed: int | None = (
            self._config_int(_seed_val, 0) if _seed_val is not None else None
        )
        self._parallel_tool_calls: bool = self._config_bool(
            self.config.get("parallel_tool_calls", True)
        )

        # Task 6: priority
        self._priority: int = self._config_int(self.config.get("priority", 100), 100)

        # Task 7: filtered
        self._filtered: bool = self._config_bool(self.config.get("filtered", True))

        # Task 8: raw
        self._raw: bool = self._config_bool(self.config.get("raw", False))

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Lazily initialize the OpenAI client on first access."""
        if self._client is None:
            if not self._base_url:
                raise ValueError(
                    "base_url must be configured for API calls. "
                    "Set base_url in config or CHAT_COMPLETIONS_BASE_URL env var."
                )
            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=0,  # Intentional: disables the openai SDK's built-in retry
                # layer. The provider manages retries itself via
                # retry_with_backoff() from amplifier-core.
            )
        return self._client

    def _translate_error(self, exc: Exception) -> KernelLLMError:
        """Translate an OpenAI SDK exception to a kernel error type.

        Maps OpenAI SDK exceptions to the shared kernel error vocabulary so that
        downstream code can catch rate limits, auth failures, etc. without
        provider-specific knowledge.

        Args:
            exc: The original exception from the OpenAI SDK or asyncio.

        Returns:
            A KernelLLMError subclass with provider, model, and retryable set.
            The __cause__ attribute is set to the original exception.
        """
        provider = self.name
        model = self._model
        err: KernelLLMError

        # Check specific OpenAI error types before the broader APIStatusError,
        # since many specific errors (RateLimitError, AuthenticationError, etc.)
        # are subclasses of APIStatusError.
        if isinstance(exc, openai.APITimeoutError):
            err = KernelLLMTimeoutError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.APIConnectionError):
            err = KernelProviderUnavailableError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.RateLimitError):
            err = KernelRateLimitError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.BadRequestError):
            msg = str(exc).lower()
            if "context length" in msg or "too many tokens" in msg:
                err = KernelContextLengthError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
            elif "content filter" in msg or "safety" in msg or "blocked" in msg:
                err = KernelContentFilterError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
            else:
                err = KernelInvalidRequestError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
        elif isinstance(exc, openai.AuthenticationError):
            err = KernelAuthenticationError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.PermissionDeniedError):
            err = KernelAccessDeniedError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.NotFoundError):
            err = KernelNotFoundError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.APIStatusError):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and 500 <= status_code < 600:
                err = KernelProviderUnavailableError(
                    str(exc),
                    provider=provider,
                    model=model,
                    status_code=status_code,
                    retryable=True,
                )
            else:
                err = KernelLLMError(
                    str(exc),
                    provider=provider,
                    model=model,
                )
        elif isinstance(exc, asyncio.TimeoutError):
            err = KernelLLMTimeoutError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        else:
            err = KernelLLMError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )

        err.__cause__ = exc
        return err

    async def _emit_event(self, name: str, payload: dict[str, Any]) -> None:
        """Emit an observability event via the coordinator's hooks, if available.

        Safely checks that self.coordinator is not None and has a hooks
        attribute before delegating to coordinator.hooks.emit.  This allows
        the provider to be used without a coordinator (e.g. in unit tests)
        without crashing.

        Args:
            name: The event name (e.g. 'llm:request', 'llm:response').
            payload: The event payload dict.
        """
        if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(name, payload)

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str]]:
        """Find tool calls that have no corresponding tool result.

        Returns list of (msg_idx, call_id, tool_name) tuples for each
        orphaned tool call, skipping already-repaired IDs.
        """
        # Collect all tool_call_ids that already have results
        existing_results: set[str] = set()
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id:
                existing_results.add(msg.tool_call_id)
            if msg.role == "tool" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        existing_results.add(block.tool_call_id)

        orphans: list[tuple[int, str, str]] = []
        for idx, msg in enumerate(messages):
            if msg.role != "assistant" or not isinstance(msg.content, list):
                continue
            for block in msg.content:
                if isinstance(block, ToolCallBlock):
                    call_id = block.id
                    if (
                        call_id
                        and call_id not in existing_results
                        and call_id not in self._repaired_tool_ids
                    ):
                        orphans.append((idx, call_id, block.name or "unknown"))
        return orphans

    def _create_synthetic_result_message(self, call_id: str, tool_name: str) -> Message:
        """Create a synthetic tool result for an orphaned tool call."""
        return Message(
            role="tool",
            tool_call_id=call_id,
            content=(
                f"[Tool '{tool_name}' result not available"
                " \u2014 call was orphaned in conversation history]"
            ),
        )

    def _apply_jit_repair(
        self,
        messages: list[Message],
        orphans: list[tuple[int, str, str]],
    ) -> list[Message]:
        """Insert synthetic tool results and bridging messages.

        Processes orphan groups in REVERSE index order so that insertions
        don't shift indices of earlier groups. For each group of orphans
        at a given assistant message index:
          1. Insert synthetic tool results immediately after the assistant message
          2. If the message following the synthetics is a user message,
             insert a bridging assistant message to maintain valid turn ordering
        """
        result = list(messages)

        # Group orphans by assistant message index
        groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for msg_idx, call_id, tool_name in orphans:
            groups[msg_idx].append((call_id, tool_name))

        # Process in reverse index order so insertions don't shift earlier indices
        for msg_idx in sorted(groups.keys(), reverse=True):
            insert_pos = msg_idx + 1
            synthetics: list[Message] = []
            for call_id, tool_name in groups[msg_idx]:
                synthetics.append(
                    self._create_synthetic_result_message(call_id, tool_name)
                )
                self._repaired_tool_ids.add(call_id)

            # Check if next message after insertion point is a user message
            needs_bridge = (
                insert_pos < len(result) and result[insert_pos].role == "user"
            )

            # Insert synthetics
            for i, synthetic in enumerate(synthetics):
                result.insert(insert_pos + i, synthetic)

            # Insert bridging assistant message if needed
            if needs_bridge:
                bridge_pos = insert_pos + len(synthetics)
                result.insert(
                    bridge_pos,
                    Message(
                        role="assistant",
                        content="I'll continue with the next step.",
                    ),
                )

        return result

    async def _repair_tool_sequence(self, messages: list[Message]) -> list[Message]:
        """Repair orphaned tool calls with synthetic results and bridging messages.

        Detects assistant messages with ToolCallBlock entries that have no
        corresponding tool result. For each orphan, inserts a synthetic result
        and optionally a bridging assistant message before the next user message.

        Tracks repaired IDs to prevent duplicate repairs across calls.
        Clears the tracking set when it exceeds 1000 entries to bound memory.

        Args:
            messages: Internal message list to inspect and (possibly) repair.

        Returns:
            The (possibly modified) message list with synthetic results injected.
        """
        # Bound memory: clear tracking set if it has grown too large
        if len(self._repaired_tool_ids) > 1000:
            self._repaired_tool_ids.clear()

        orphans = self._find_missing_tool_results(messages)
        if not orphans:
            return messages

        result = self._apply_jit_repair(messages, orphans)

        # Emit repair event
        repaired_ids = [orphan[1] for orphan in orphans]
        if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "provider:tool_sequence_repaired",
                {
                    "provider": self.name,
                    "model": self._model,
                    "repaired_count": len(repaired_ids),
                    "repaired_tool_ids": repaired_ids,
                },
            )

        return result

    def _convert_messages_to_wire(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert internal Message list to OpenAI-compatible wire format.

        Maps roles, joins text blocks, drops thinking blocks, converts tool
        call blocks to the ``tool_calls`` array, and converts tool result blocks
        to ``role: 'tool'`` messages.  Images are converted to ``image_url``
        content items.

        Developer messages are wrapped in <context_file> XML and mapped to
        user role (matching the Anthropic provider convention).

        Args:
            messages: Internal message list from the ChatRequest.

        Returns:
            List of dicts suitable for ``client.chat.completions.create``.
        """
        wire: list[dict[str, Any]] = []

        # Prepend developer messages before user messages in wire format.
        ordered = [m for m in messages if m.role == "developer"] + [
            m for m in messages if m.role != "developer"
        ]

        for message in ordered:
            # Task 9: developer role -> user role with <context_file> XML wrapping
            if message.role == "developer":
                text = message.content if isinstance(message.content, str) else ""
                if isinstance(message.content, list):
                    text_parts_dev: list[str] = []
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts_dev.append(block.text)
                    text = "\n".join(text_parts_dev)
                wire.append(
                    {
                        "role": "user",
                        "content": f"<context_file>\n{text}\n</context_file>",
                    }
                )
                continue

            role = message.role
            content = message.content

            if isinstance(content, str):
                wire.append({"role": role, "content": content})
                continue

            # List of content blocks — iterate and classify.
            text_parts: list[str] = []
            tool_calls_wire: list[dict[str, Any]] = []
            image_parts: list[dict[str, Any]] = []
            tool_result_block: ToolResultBlock | None = None

            for block in content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    pass  # Silently drop thinking blocks
                elif isinstance(block, ToolCallBlock):
                    tool_calls_wire.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    tool_result_block = block
                elif isinstance(block, ImageBlock):
                    src = block.source
                    url = f"data:{src['media_type']};base64,{src['data']}"
                    image_parts.append({"type": "image_url", "image_url": {"url": url}})

            if tool_result_block is not None:
                # ToolResultBlock overrides everything else in this message.
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result_block.tool_call_id,
                        "content": str(tool_result_block.output),
                    }
                )
                continue

            msg: dict[str, Any] = {"role": role}

            if image_parts:
                # Build a multimodal content array.
                content_array: list[dict[str, Any]] = []
                if text_parts:
                    content_array.append(
                        {"type": "text", "text": "\n".join(text_parts)}
                    )
                content_array.extend(image_parts)
                msg["content"] = content_array
            elif text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None

            if tool_calls_wire:
                msg["tool_calls"] = tool_calls_wire

            wire.append(msg)

        return wire

    def _convert_tools_to_wire(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert internal ToolSpec list to OpenAI function-calling wire format.

        Args:
            tools: List of ToolSpec from the ChatRequest.

        Returns:
            List of ``{type: 'function', function: {name, description, parameters}}``
            dicts.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _build_response(self, response: Any) -> ChatCompletionsChatResponse:
        """Build a ChatCompletionsChatResponse from an OpenAI ChatCompletion object.

        Args:
            response: The ChatCompletion returned by the OpenAI SDK.

        Returns:
            A ChatCompletionsChatResponse with content blocks, tool calls, usage,
            content_blocks (for streaming UI), and text.
        """
        choice = response.choices[0]
        message = choice.message

        content: list[Any] = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_parts: list[str] = []

        # Preserve any extended reasoning content as a ThinkingBlock.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            content.append(ThinkingBlock(thinking=reasoning_content))
            event_blocks.append(ThinkingContent(text=reasoning_content))

        # Text content → TextBlock.
        if message.content:
            content.append(TextBlock(text=message.content))
            event_blocks.append(TextContent(text=message.content))
            text_parts.append(message.content)

        # Tool calls → ToolCallBlock in content + ToolCall in tool_calls.
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                arguments = json.loads(tc.function.arguments)
                content.append(
                    ToolCallBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=arguments,
                    )
                )
                event_blocks.append(
                    ToolCallContent(
                        id=tc.id, name=tc.function.name, arguments=arguments
                    )
                )
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # Usage mapping: OpenAI prompt/completion → kernel input/output.
        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatCompletionsChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            content_blocks=event_blocks if event_blocks else None,
            text="".join(text_parts) if text_parts else None,
        )

    async def _complete_non_streaming(
        self,
        wire_messages: list[dict[str, Any]],
        wire_tools: list[dict[str, Any]] | None,
        request: "ChatRequest",
    ) -> "tuple[ChatCompletionsChatResponse, Any]":
        """Execute a non-streaming chat completion and return a ChatCompletionsChatResponse.

        Args:
            wire_messages: Messages already converted to OpenAI wire format.
            wire_tools: Tools already converted to OpenAI wire format, or None.
            request: The original ChatRequest (used to resolve model).

        Returns:
            A tuple of (ChatCompletionsChatResponse, raw_api_response).
        """
        model = request.model or self._model
        params: dict[str, Any] = dict(
            model=model,
            messages=wire_messages,
            tools=wire_tools,
            stream=False,
        )
        # Task 5: Add optional generation params when configured
        if self._top_p is not None:
            params["top_p"] = self._top_p
        if self._stop is not None:
            params["stop"] = self._stop
        if self._seed is not None:
            params["seed"] = self._seed
        if wire_tools and self._parallel_tool_calls is not None:
            params["parallel_tool_calls"] = self._parallel_tool_calls

        response = await self.client.chat.completions.create(**params)
        return self._build_response(response), response

    async def _complete_streaming(
        self,
        wire_messages: list[dict[str, Any]],
        wire_tools: list[dict[str, Any]] | None,
        request: "ChatRequest",
    ) -> "tuple[ChatCompletionsChatResponse, None]":
        """Execute a streaming chat completion and accumulate chunks into a ChatCompletionsChatResponse.

        Calls the API with ``stream=True``, iterates the async chunk stream, and
        accumulates:
        - ``delta.content`` into a text buffer
        - ``delta.tool_calls`` by index (first chunk per index carries ``id`` and
          ``function.name``; subsequent chunks append to ``function.arguments``)
        - ``reasoning_content`` into a thinking buffer (via ``getattr`` for safety)
        - ``finish_reason`` from the last non-empty choice
        - ``usage`` from the final chunk if present

        Args:
            wire_messages: Messages already converted to OpenAI wire format.
            wire_tools: Tools already converted to OpenAI wire format, or None.
            request: The original ChatRequest (used to resolve model).

        Returns:
            A tuple of (ChatCompletionsChatResponse, None). Streaming has no
            single raw response object to dump, so the second element is always None.
        """
        model = request.model or self._model

        text_buffer: str = ""
        thinking_buffer: str = ""
        # Maps chunk index -> accumulated tool call data
        tool_call_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None

        params: dict[str, Any] = dict(
            model=model,
            messages=wire_messages,
            tools=wire_tools,
            stream=True,
        )
        # Task 5: Add optional generation params when configured
        if self._top_p is not None:
            params["top_p"] = self._top_p
        if self._stop is not None:
            params["stop"] = self._stop
        if self._seed is not None:
            params["seed"] = self._seed
        if wire_tools and self._parallel_tool_calls is not None:
            params["parallel_tool_calls"] = self._parallel_tool_calls

        stream = await self.client.chat.completions.create(**params)

        async for chunk in stream:
            # Capture usage if present on any chunk (typically the final one).
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Track finish_reason from the last chunk that has one.
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Accumulate text content.
            if delta.content:
                text_buffer += delta.content

            # Accumulate reasoning/thinking content (provider-specific extension).
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thinking_buffer += reasoning

            # Accumulate tool call deltas by index.
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        # First chunk for this index: capture id and name.
                        tool_call_accum[idx] = {
                            "id": tc_delta.id,
                            "name": tc_delta.function.name,
                            "arguments": "",
                        }
                    # All chunks: append to arguments.
                    if tc_delta.function.arguments:
                        tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

        # Build content blocks from accumulated buffers.
        content: list[Any] = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []

        if thinking_buffer:
            content.append(ThinkingBlock(thinking=thinking_buffer))
            event_blocks.append(ThinkingContent(text=thinking_buffer))

        if text_buffer:
            content.append(TextBlock(text=text_buffer))
            event_blocks.append(TextContent(text=text_buffer))

        # Build tool calls from accumulated data (ordered by index).
        tool_calls: list[ToolCall] | None = None
        if tool_call_accum:
            tool_calls = []
            for idx in sorted(tool_call_accum.keys()):
                tc_data = tool_call_accum[idx]
                arguments = json.loads(tc_data["arguments"])
                content.append(
                    ToolCallBlock(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        input=arguments,
                    )
                )
                event_blocks.append(
                    ToolCallContent(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    )
                )
                tool_calls.append(
                    ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    )
                )

        # Map usage if captured from the stream.
        usage_obj: Usage | None = None
        if usage is not None:
            usage_obj = Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )

        chat_response = ChatCompletionsChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage_obj,
            finish_reason=finish_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=text_buffer if text_buffer else None,
        )
        return chat_response, None

    async def list_models(self) -> list[ModelInfo]:
        """Return a list of models available on the server.

        When ``filtered=True`` (the default), returns only the configured
        model if found in the server's model list.  When ``filtered=False``,
        returns all models from the server.

        Returns:
            A list of :class:`~amplifier_core.models.ModelInfo` objects.  On
            failure, returns a one-element list containing the configured model.
        """
        try:
            response = await self.client.models.list()
            all_models = [
                ModelInfo(
                    id=model.id,
                    display_name=model.id,
                    context_window=0,
                    max_output_tokens=self._max_tokens,
                    capabilities=["tools", "streaming"],
                )
                for model in response.data
            ]
            # Task 7: filtered mode — return only configured model when filtered=True
            if self._filtered and self._model and self._model != "default":
                configured = [m for m in all_models if m.id == self._model]
                return configured if configured else all_models[:1]
            return all_models
        except Exception as exc:
            logger.warning("Failed to list models from server: %s", exc)
            return [
                ModelInfo(
                    id=self._model,
                    display_name=self._model,
                    context_window=0,
                    max_output_tokens=self._max_tokens,
                )
            ]

    async def complete(
        self, request: ChatRequest, **kwargs: Any
    ) -> ChatCompletionsChatResponse:
        """Send a chat request and return a ChatCompletionsChatResponse.

        Lazily initialises the AsyncOpenAI client via the `client` property on
        the first call.  Wraps each individual attempt in ``asyncio.timeout`` and
        retries retryable errors with exponential backoff via ``retry_with_backoff``.

        Args:
            request: The unified ChatRequest to execute.

        Returns:
            A ChatCompletionsChatResponse with content, tool calls, and usage.

        Raises:
            KernelLLMError subclass on any provider failure.
        """
        messages = await self._repair_tool_sequence(request.messages)
        wire_messages = self._convert_messages_to_wire(messages)
        wire_tools = (
            self._convert_tools_to_wire(request.tools) if request.tools else None
        )

        retry_config = RetryConfig(
            max_retries=self._max_retries,
            initial_delay=self._min_retry_delay,
            max_delay=self._max_retry_delay,
        )

        async def _on_retry(attempt: int, delay: float, error: Any) -> None:
            if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "delay": delay,
                        "max_retries": self._max_retries,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        async def _single_attempt() -> ChatCompletionsChatResponse:
            start_time = time.monotonic()

            # Task 8: Build llm:request event payload, include raw params when raw=True
            request_payload: dict[str, Any] = {
                "provider": self.name,
                "model": self._model,
            }
            if self._raw:
                raw_params: dict[str, Any] = {
                    "model": request.model or self._model,
                    "messages": wire_messages,
                    "stream": self._use_streaming,
                }
                if wire_tools:
                    raw_params["tools"] = wire_tools
                request_payload["raw"] = redact_secrets(raw_params)

            await self._emit_event("llm:request", request_payload)

            try:
                async with asyncio.timeout(self._timeout):
                    if self._use_streaming:
                        (
                            chat_response,
                            raw_api_response,
                        ) = await self._complete_streaming(
                            wire_messages, wire_tools, request
                        )
                    else:
                        (
                            chat_response,
                            raw_api_response,
                        ) = await self._complete_non_streaming(
                            wire_messages, wire_tools, request
                        )
            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                kernel_error = self._translate_error(exc)
                await self._emit_event(
                    "llm:response",
                    {
                        "provider": self.name,
                        "status": "error",
                        "duration_ms": duration_ms,
                        "error_type": type(exc).__name__,
                    },
                )
                raise kernel_error from exc

            duration_ms = (time.monotonic() - start_time) * 1000
            usage_dict: dict[str, Any] = {}
            if chat_response.usage:
                usage_dict = {
                    "input_tokens": chat_response.usage.input_tokens,
                    "output_tokens": chat_response.usage.output_tokens,
                    "total_tokens": chat_response.usage.total_tokens,
                }

            # Task 8: Build llm:response event payload, include raw response when raw=True
            response_payload: dict[str, Any] = {
                "provider": self.name,
                "usage": usage_dict,
                "duration_ms": duration_ms,
                "stop_reason": chat_response.finish_reason,
            }
            if self._raw and raw_api_response is not None:
                try:
                    response_payload["raw"] = redact_secrets(
                        raw_api_response.model_dump()
                    )
                except Exception:
                    pass  # Don't crash on model_dump failure

            await self._emit_event("llm:response", response_payload)
            return chat_response

        return await retry_with_backoff(
            _single_attempt, config=retry_config, on_retry=_on_retry
        )

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Extract tool calls from a ChatResponse.

        Args:
            response: The ChatResponse returned by complete().

        Returns:
            The list of ToolCall objects, or an empty list if none present.
        """
        return response.tool_calls or []

    def get_info(self) -> ProviderInfo:
        """Return metadata describing this provider's capabilities and configuration.

        Returns:
            A ProviderInfo instance with the provider's id, display name,
            credential environment variables, capabilities, defaults, and
            all config field declarations.
        """
        return ProviderInfo(
            id=self.name,
            display_name=f"Chat Completions ({self.name})"
            if self.name != "chat-completions"
            else "Chat Completions",
            credential_env_vars=["CHAT_COMPLETIONS_API_KEY"],
            capabilities=["tools", "streaming", "json_mode"],
            defaults={
                "model": self._model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 300.0,
            },
            # Wizard exposes only the fields a real user has to think about
            # the first time they add this provider: api_key, base_url, priority.
            #
            # `model` is deliberately NOT declared here. app-cli's wizard has a
            # dedicated "Default Model" phase (provider_config_utils.configure_provider
            # phase 2) that calls our list_models() and presents an interactive
            # picker of whatever the server's /v1/models returns, falling back
            # to a free-text prompt when the server is unreachable. Declaring
            # `model` here would produce a duplicate text prompt *before* the
            # picker. Matches the pattern in amplifier-module-provider-anthropic.
            #
            # All other fields (max_tokens, temperature, timeout, max_retries,
            # min_retry_delay, max_retry_delay, use_streaming,
            # parallel_tool_calls, filtered, raw, top_p, stop, seed) keep their
            # in-code defaults and remain overridable via settings.yaml for
            # power users.
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your API key (leave blank for local/keyless servers)",
                    env_var="CHAT_COMPLETIONS_API_KEY",
                    required=False,
                ),
                ConfigField(
                    id="base_url",
                    display_name="Base URL",
                    field_type="text",
                    prompt="Enter the API base URL",
                    env_var="CHAT_COMPLETIONS_BASE_URL",
                    required=False,
                    default="http://localhost:8080/v1",
                ),
                ConfigField(
                    id="priority",
                    display_name="Priority",
                    field_type="text",
                    prompt="Provider selection priority (lower = preferred, 0 = promoted by spawn_utils)",
                    required=False,
                    default="100",
                ),
            ],
        )

    async def close(self) -> None:
        """Release any resources held by this provider.

        Uses asyncio.shield so the client cleanup completes even if the
        enclosing task is cancelled.  All exceptions are suppressed.
        """
        if self._client is not None:
            try:
                await asyncio.shield(self._client.close())
            except (asyncio.CancelledError, Exception):
                pass


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> Any:
    """Mount the chat-completions provider into the coordinator.

    Returns None (silent non-mount) when base_url is not configured in
    either config or the CHAT_COMPLETIONS_BASE_URL environment variable.

    Args:
        coordinator: Amplifier coordinator instance (first arg per amplifier-core convention).
        config: Provider configuration dict, or None for defaults.

    Returns:
        An async cleanup function that closes the provider, or None if skipped.
    """
    config = config or {}

    # Resolve base_url: config takes precedence, then env var
    base_url = config.get("base_url") or os.environ.get("CHAT_COMPLETIONS_BASE_URL", "")
    if not base_url:
        logger.info("chat-completions provider: no base_url configured, skipping mount")
        return None

    provider = ChatCompletionsProvider(config=config, coordinator=coordinator)
    if provider.name != "chat-completions":
        logger.warning(
            "chat-completions provider mounted with custom name %r; this overrides the "
            "coordinator's default-name remap and any configured 'instance_id' will not "
            "take effect for this provider. See mod docs for details.",
            provider.name,
        )
    await coordinator.mount("providers", provider, name=provider.name)
    logger.info(
        "chat-completions provider mounted (name=%s, base_url=%s)",
        provider.name,
        base_url,
    )

    async def cleanup() -> None:
        await provider.close()

    return cleanup
