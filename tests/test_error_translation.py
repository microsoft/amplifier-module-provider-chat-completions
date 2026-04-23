"""Tests for error-detection paths in _translate_error, _extract_structured_error,
and _format_context_error_message.

Covers the widened context-overflow detection logic (llama.cpp 'context size',
LM Studio 'context window', generic 'exceed_context_size_error', and structured
body checks) plus the diagnostic message formatting that surfaces n_prompt_tokens
and n_ctx for actionable guidance.
"""

from unittest.mock import MagicMock

import openai

from amplifier_core.llm_errors import ContextLengthError as KernelContextLengthError

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> ChatCompletionsProvider:
    """Return a minimal provider instance suitable for error-translation tests."""
    return ChatCompletionsProvider(
        config={
            "model": "test-model",
            "use_streaming": "false",
            "max_retries": "0",
        }
    )


def _make_bad_request(message: str, body=None) -> openai.BadRequestError:
    """Construct a BadRequestError with the given message string and optional body.

    Mirrors the _make_openai_error helper in test_provider.py.  body is passed
    directly to the SDK constructor so exc.body is set correctly (the SDK
    preserves it without further modification for BadRequestError).
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": message}}
    return openai.BadRequestError(message, response=mock_response, body=body)


# ---------------------------------------------------------------------------
# TestContextErrorMessageDetection — widened keyword matching
# ---------------------------------------------------------------------------


class TestContextErrorMessageDetection:
    """Verify all context-overflow message variants map to KernelContextLengthError."""

    def test_context_size_in_message(self):
        """'context size' in message → KernelContextLengthError (llama.cpp variant)."""
        provider = _make_provider()
        exc = _make_bad_request("Request exceeds context size")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError), (
            f"Expected KernelContextLengthError, got {type(err).__name__}"
        )
        assert err.retryable is False

    def test_context_window_in_message(self):
        """'context window' in message → KernelContextLengthError (LM Studio / misc variant)."""
        provider = _make_provider()
        exc = _make_bad_request("Input exceeds the model's context window")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError), (
            f"Expected KernelContextLengthError, got {type(err).__name__}"
        )
        assert err.retryable is False

    def test_exceed_context_size_error_in_message(self):
        """'exceed_context_size_error' in message → KernelContextLengthError."""
        provider = _make_provider()
        exc = _make_bad_request("exceed_context_size_error: prompt too long")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError), (
            f"Expected KernelContextLengthError, got {type(err).__name__}"
        )
        assert err.retryable is False

    def test_context_size_case_insensitive(self):
        """Keyword matching is case-insensitive ('Context Size' still matches)."""
        provider = _make_provider()
        exc = _make_bad_request("Context Size limit reached")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)

    def test_context_window_uppercase(self):
        """'CONTEXT WINDOW' (uppercase) still matches via lowercased msg check."""
        provider = _make_provider()
        exc = _make_bad_request("CONTEXT WINDOW exceeded")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)


# ---------------------------------------------------------------------------
# TestContextErrorStructuredBodyDetection — body-based detection
# ---------------------------------------------------------------------------


class TestContextErrorStructuredBodyDetection:
    """Verify structured-body checks catch context errors that plain messages miss."""

    def test_body_error_type_exceed_context_size_error(self):
        """body={'error': {'type': 'exceed_context_size_error'}} → KernelContextLengthError.

        The message does NOT contain any keyword — detection relies solely on the
        structured body's 'type' field.
        """
        provider = _make_provider()
        body = {
            "error": {
                "type": "exceed_context_size_error",
                "message": "prompt too long for the model",
            }
        }
        # Use a neutral message that doesn't match any keyword
        exc = _make_bad_request("Bad request", body=body)
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError), (
            f"Expected KernelContextLengthError via body type check, got {type(err).__name__}"
        )
        assert err.retryable is False

    def test_body_error_type_in_inner_dict(self):
        """Verify _extract_structured_error returns the inner 'error' dict (not full body)
        when body['error'] is a dict, and body['error']['type'] is checked correctly.
        """
        provider = _make_provider()
        body = {
            "error": {
                "type": "exceed_context_size_error",
                "code": 42,
                "n_prompt_tokens": 8192,
                "n_ctx": 4096,
            }
        }
        exc = _make_bad_request("Request failed", body=body)
        # _extract_structured_error should return the inner error dict
        structured = provider._extract_structured_error(exc)
        assert structured is not None
        assert structured == body["error"]
        assert structured.get("type") == "exceed_context_size_error"


# ---------------------------------------------------------------------------
# TestExtractStructuredError — unit tests for the helper itself
# ---------------------------------------------------------------------------


class TestExtractStructuredError:
    """Unit tests for _extract_structured_error logic."""

    def test_returns_inner_error_dict_when_present(self):
        """Returns inner 'error' dict when body is a dict with dict-valued 'error'."""
        exc = _make_bad_request("err", body={"error": {"type": "bad", "code": 1}})
        result = ChatCompletionsProvider._extract_structured_error(exc)
        assert result == {"type": "bad", "code": 1}

    def test_returns_full_body_when_no_error_key(self):
        """Returns the full body dict when 'error' key is absent."""
        exc = _make_bad_request("err", body={"n_prompt_tokens": 100, "n_ctx": 50})
        result = ChatCompletionsProvider._extract_structured_error(exc)
        assert result == {"n_prompt_tokens": 100, "n_ctx": 50}

    def test_returns_full_body_when_error_is_string(self):
        """Returns the full body dict when body['error'] is a string (not a dict).

        The docstring specifies: 'Returns the inner error dict if it's a dict,
        otherwise the full body dict (when body is itself a dict), otherwise None.'
        A string-valued 'error' key falls through to the full-body return path.
        """
        exc = _make_bad_request("err", body={"error": "some plain string error"})
        result = ChatCompletionsProvider._extract_structured_error(exc)
        # Falls through: 'error' is not a dict, so return the full body dict
        assert result == {"error": "some plain string error"}

    def test_returns_none_when_body_is_none(self):
        """Returns None when exc.body is None."""
        exc = _make_bad_request("err", body=None)
        result = ChatCompletionsProvider._extract_structured_error(exc)
        assert result is None

    def test_returns_none_when_body_is_not_dict(self):
        """Returns None when exc.body is a non-dict value (e.g. string)."""
        exc = _make_bad_request("err", body="plain string body")
        result = ChatCompletionsProvider._extract_structured_error(exc)
        assert result is None


# ---------------------------------------------------------------------------
# TestFormatContextErrorMessage — diagnostic field surfacing
# ---------------------------------------------------------------------------


class TestFormatContextErrorMessage:
    """Unit tests for _format_context_error_message."""

    def test_includes_n_prompt_tokens_and_n_ctx_values(self):
        """When structured has n_prompt_tokens and n_ctx, both appear in the message."""
        exc = _make_bad_request(
            "context length exceeded",
            body={"n_prompt_tokens": 1234, "n_ctx": 512},
        )
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)

        assert "1234" in msg, f"n_prompt_tokens (1234) not found in: {msg!r}"
        assert "512" in msg, f"n_ctx (512) not found in: {msg!r}"

    def test_n_prompt_and_ctx_values_verbatim_in_message(self):
        """Diagnostic values appear verbatim, not mangled or rounded."""
        exc = _make_bad_request(
            "context length exceeded",
            body={"n_prompt_tokens": 99999, "n_ctx": 8192},
        )
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)

        assert "99999" in msg
        assert "8192" in msg

    def test_returns_str_exc_when_n_prompt_absent(self):
        """Falls back to str(exc) when n_prompt_tokens is missing from structured."""
        exc = _make_bad_request("context length exceeded", body={"n_ctx": 4096})
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)

        # Without both fields, returns the plain exc string
        assert msg == str(exc)

    def test_returns_str_exc_when_structured_is_none(self):
        """Falls back to str(exc) when structured is None (no body)."""
        exc = _make_bad_request("context length exceeded", body=None)
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        assert structured is None
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)
        assert msg == str(exc)

    def test_error_is_string_body_fallback_returns_str_exc(self):
        """When body.error is a string, _extract_structured_error returns the full body dict.

        _format_context_error_message receives structured={'error': 'plain string'}.
        Since n_prompt_tokens and n_ctx are absent, it returns str(exc).

        This validates the docstring contract: 'present but not a dict' falls through
        to the full-body return (not None), and the formatter correctly handles a
        body dict that lacks diagnostic fields.
        """
        body = {"error": "plain string error message"}
        exc = _make_bad_request("Context size exceeded", body=body)

        # Confirm extraction returns full body (not None, not inner dict)
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        assert structured == body  # full body dict returned

        # Confirm formatter falls through to str(exc) since no n_prompt/n_ctx
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)
        assert msg == str(exc)

    def test_translate_error_uses_str_exc_when_body_error_is_string(self):
        """Integration: string body.error → KernelContextLengthError with str(exc) message.

        The message attribute on the kernel error must equal str(exc) because no
        diagnostic fields were present to override it.
        """
        provider = _make_provider()
        body = {"error": "plain text error"}
        exc = _make_bad_request("Context size exceeded", body=body)

        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)
        # The error message should be the str(exc) fallback
        assert str(exc) in str(err) or err.args[0] == str(exc)

    def test_message_mentions_server_flag_examples(self):
        """Formatted message includes flag examples for the main server implementations."""
        exc = _make_bad_request(
            "context length exceeded",
            body={"n_prompt_tokens": 5000, "n_ctx": 2048},
        )
        structured = ChatCompletionsProvider._extract_structured_error(exc)
        msg = ChatCompletionsProvider._format_context_error_message(exc, structured)

        # Should mention at least llama.cpp and vLLM flags (server-agnostic message)
        assert "ctx-size" in msg or "--ctx-size" in msg, (
            f"llama.cpp flag missing: {msg!r}"
        )
        assert "max-model-len" in msg or "--max-model-len" in msg, (
            f"vLLM flag missing: {msg!r}"
        )


# ---------------------------------------------------------------------------
# TestTranslateErrorIntegration — end-to-end through _translate_error
# ---------------------------------------------------------------------------


class TestTranslateErrorIntegration:
    """Integration tests that run full error through _translate_error."""

    def test_body_with_n_prompt_and_n_ctx_produces_rich_message(self):
        """Body with diagnostic fields produces a KernelContextLengthError with rich message."""
        provider = _make_provider()
        body = {"n_prompt_tokens": 7777, "n_ctx": 4096}
        exc = _make_bad_request("context length exceeded", body=body)

        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)
        err_msg = err.args[0]
        assert "7777" in err_msg, f"n_prompt_tokens not in error message: {err_msg!r}"
        assert "4096" in err_msg, f"n_ctx not in error message: {err_msg!r}"

    def test_context_size_maps_to_context_length_error(self):
        """Full path: 'context size' message → KernelContextLengthError."""
        provider = _make_provider()
        exc = _make_bad_request("Prompt exceeds context size limit")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)
        assert err.provider == "chat-completions"
        assert err.model == "test-model"
        assert err.retryable is False
        assert err.__cause__ is exc

    def test_context_window_maps_to_context_length_error(self):
        """Full path: 'context window' message → KernelContextLengthError."""
        provider = _make_provider()
        exc = _make_bad_request("Your message exceeds the context window")
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)
        assert err.provider == "chat-completions"
        assert err.model == "test-model"

    def test_structured_body_type_maps_to_context_length_error(self):
        """Full path: body type=exceed_context_size_error → KernelContextLengthError."""
        provider = _make_provider()
        body = {"error": {"type": "exceed_context_size_error"}}
        exc = _make_bad_request("Unknown error", body=body)
        err = provider._translate_error(exc)
        assert isinstance(err, KernelContextLengthError)
        assert err.provider == "chat-completions"
        assert err.retryable is False
        assert err.__cause__ is exc
