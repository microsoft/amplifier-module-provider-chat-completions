"""Tests for context_window config field and budget-path sinks.

Verifies that:
1. Absent config → ModelInfo.context_window == 0, get_info().defaults has no
   'context_window' key, and a WARNING is logged exactly once.
2. context_window: "32768" (string) → ModelInfo reports 32768,
   get_info().defaults["context_window"] == 32768, no warning.
3. context_window: 32768 (int, settings.yaml authors) → same as above.
4. Invalid string ("abc") → falls back to 0 with a warning log about the
   parse failure (tested with pytest caplog).
5. Env var CHAT_COMPLETIONS_CONTEXT_WINDOW=16384 → honored when config absent.
6. Both set → config field wins over env var.
7. Both context_window and max_tokens populate get_info().defaults correctly.
8. Warning emitted exactly once across multiple list_models() calls.
9. INFO log emitted exactly once when context_window > 0.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> ChatCompletionsProvider:
    """Create a ChatCompletionsProvider with sensible test defaults."""
    config = {"model": "test-model", **config_overrides}
    return ChatCompletionsProvider(config=config)


def _fake_models_response(model_ids: list[str]):
    """Create a fake OpenAI models.list() response."""
    data = [SimpleNamespace(id=mid) for mid in model_ids]
    return SimpleNamespace(data=data)


def _make_list_models_provider(**config_overrides) -> ChatCompletionsProvider:
    """Create a provider wired to return specific model IDs from list_models()."""
    config = {"model": "test-model", "filtered": False, **config_overrides}
    provider = ChatCompletionsProvider(config=config)
    provider._client = MagicMock()
    provider._client.models.list = AsyncMock(
        return_value=_fake_models_response(["test-model"])
    )
    return provider


# ---------------------------------------------------------------------------
# Case 1: Absent config → context_window == 0, no key in defaults, warn once
# ---------------------------------------------------------------------------


class TestAbsentConfig:
    """No context_window in config → unknown, no budgeting, warning logged."""

    def test_context_window_attribute_zero(self):
        """Provider without context_window config → _context_window == 0."""
        provider = _make_provider()
        assert provider._context_window == 0

    def test_get_info_defaults_no_context_window_key(self):
        """get_info().defaults must NOT contain 'context_window' when value is 0."""
        provider = _make_provider()
        info = provider.get_info()
        assert "context_window" not in info.defaults

    def test_list_models_context_window_zero(self):
        """list_models() returns ModelInfo with context_window=0 when not configured."""
        from amplifier_core.models import ModelInfo

        provider = _make_list_models_provider()
        models = asyncio.run(provider.list_models())
        assert len(models) >= 1
        assert all(isinstance(m, ModelInfo) for m in models)
        assert all(m.context_window == 0 for m in models)

    def test_warning_logged_when_context_window_zero(self, caplog):
        """A WARNING is logged when context_window is 0 (unknown)."""
        provider = _make_list_models_provider()

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "context_window" in r.getMessage()
        ]
        assert len(warning_records) >= 1, (
            "Expected a WARNING log mentioning 'context_window' when value is 0"
        )
        assert "bypass" in warning_records[0].getMessage().lower() or (
            "unknown" in warning_records[0].getMessage().lower()
        )


# ---------------------------------------------------------------------------
# Case 2: context_window: "32768" (string) → honored
# ---------------------------------------------------------------------------


class TestStringContextWindow:
    """context_window passed as a string → parsed and propagated correctly."""

    def test_string_parsed_to_int(self):
        """String '32768' in config → _context_window == 32768."""
        provider = _make_provider(context_window="32768")
        assert provider._context_window == 32768

    def test_model_info_context_window_set(self):
        """list_models() reports the configured context_window on each ModelInfo."""
        provider = _make_list_models_provider(context_window="32768")
        models = asyncio.run(provider.list_models())
        assert len(models) >= 1
        assert all(m.context_window == 32768 for m in models)

    def test_get_info_defaults_context_window(self):
        """get_info().defaults['context_window'] == 32768 when configured."""
        provider = _make_provider(context_window="32768")
        info = provider.get_info()
        assert "context_window" in info.defaults
        assert info.defaults["context_window"] == 32768

    def test_no_warning_when_context_window_set(self, caplog):
        """No 'unknown' WARNING is logged when context_window > 0."""
        provider = _make_list_models_provider(context_window="32768")

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())

        bypass_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "bypass" in r.getMessage().lower()
        ]
        assert len(bypass_warnings) == 0, (
            "No 'bypass' WARNING should be emitted when context_window > 0"
        )


# ---------------------------------------------------------------------------
# Case 3: context_window: 32768 (int) → same result
# ---------------------------------------------------------------------------


class TestIntContextWindow:
    """context_window passed as an int (settings.yaml authors) → same as string."""

    def test_int_stored_correctly(self):
        """Integer 32768 in config → _context_window == 32768."""
        provider = _make_provider(context_window=32768)
        assert provider._context_window == 32768

    def test_model_info_context_window_set(self):
        """list_models() reports 32768 when config uses int literal."""
        provider = _make_list_models_provider(context_window=32768)
        models = asyncio.run(provider.list_models())
        assert all(m.context_window == 32768 for m in models)

    def test_get_info_defaults_context_window(self):
        """get_info().defaults['context_window'] == 32768 for int config."""
        provider = _make_provider(context_window=32768)
        info = provider.get_info()
        assert info.defaults["context_window"] == 32768


# ---------------------------------------------------------------------------
# Case 4: Invalid string ("abc") → fallback to 0 + parse-failure warning
# ---------------------------------------------------------------------------


class TestInvalidContextWindow:
    """Non-numeric context_window → falls back to 0 with a warning log."""

    def test_invalid_string_falls_back_to_zero(self):
        """_config_int('abc', 0) falls back to 0."""
        provider = _make_provider(context_window="abc")
        assert provider._context_window == 0

    def test_parse_failure_warning_logged(self, caplog):
        """A WARNING is logged when an invalid context_window value is parsed."""
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            _make_provider(context_window="abc")

        # The _config_int helper logs "[PROVIDER] Invalid integer config value..."
        parse_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "abc" in r.getMessage()
        ]
        assert len(parse_warnings) >= 1, (
            "Expected a WARNING log mentioning the invalid value 'abc'"
        )


# ---------------------------------------------------------------------------
# Case 5: Env var CHAT_COMPLETIONS_CONTEXT_WINDOW honored when config absent
# ---------------------------------------------------------------------------


class TestEnvVarContextWindow:
    """CHAT_COMPLETIONS_CONTEXT_WINDOW env var is used when config field absent."""

    def test_env_var_sets_context_window(self, monkeypatch):
        """Env var CHAT_COMPLETIONS_CONTEXT_WINDOW=16384 → _context_window == 16384."""
        monkeypatch.setenv("CHAT_COMPLETIONS_CONTEXT_WINDOW", "16384")
        provider = _make_provider()  # No context_window in config
        assert provider._context_window == 16384

    def test_env_var_used_in_model_info(self, monkeypatch):
        """list_models() uses env-var value in ModelInfo when config absent."""
        monkeypatch.setenv("CHAT_COMPLETIONS_CONTEXT_WINDOW", "16384")
        provider = _make_list_models_provider()
        models = asyncio.run(provider.list_models())
        assert all(m.context_window == 16384 for m in models)

    def test_env_var_populates_get_info_defaults(self, monkeypatch):
        """get_info().defaults['context_window'] == 16384 when set via env var."""
        monkeypatch.setenv("CHAT_COMPLETIONS_CONTEXT_WINDOW", "16384")
        provider = _make_provider()
        info = provider.get_info()
        assert info.defaults.get("context_window") == 16384


# ---------------------------------------------------------------------------
# Case 6: Both set → config field wins over env var
# ---------------------------------------------------------------------------


class TestConfigWinsOverEnvVar:
    """When both config and env var are set, config field takes precedence."""

    def test_config_beats_env_var(self, monkeypatch):
        """config['context_window']='32768' wins over env var '16384'."""
        monkeypatch.setenv("CHAT_COMPLETIONS_CONTEXT_WINDOW", "16384")
        provider = _make_provider(context_window="32768")
        assert provider._context_window == 32768

    def test_config_beats_env_var_in_model_info(self, monkeypatch):
        """list_models() uses config value, not env var, when both set."""
        monkeypatch.setenv("CHAT_COMPLETIONS_CONTEXT_WINDOW", "16384")
        provider = _make_list_models_provider(context_window="32768")
        models = asyncio.run(provider.list_models())
        assert all(m.context_window == 32768 for m in models)


# ---------------------------------------------------------------------------
# Case 7: Both context_window and max_tokens populate get_info().defaults
# ---------------------------------------------------------------------------


class TestDefaultsPopulation:
    """get_info().defaults contains both context_window and max_output_tokens."""

    def test_defaults_contain_max_output_tokens(self):
        """max_output_tokens is in defaults when max_tokens is configured."""
        provider = _make_provider(max_tokens="8192")
        info = provider.get_info()
        assert "max_output_tokens" in info.defaults
        assert info.defaults["max_output_tokens"] == 8192

    def test_defaults_contain_both_when_context_window_set(self):
        """Both context_window and max_output_tokens in defaults when configured."""
        provider = _make_provider(context_window="32768", max_tokens="4096")
        info = provider.get_info()
        assert info.defaults.get("context_window") == 32768
        assert info.defaults.get("max_output_tokens") == 4096

    def test_max_output_tokens_default_4096(self):
        """max_output_tokens falls back to 4096 when max_tokens not configured.

        _max_tokens is None by default (conditional-send pattern: don't cap the
        wire call when the user hasn't asked).  But the defaults dict must still
        carry a value so context-manager budget math doesn't fall back to its own
        200k sentinel.  We use 4096 as a conservative reservation — matches the
        old always-4096 wire behaviour for budget purposes.
        """
        provider = _make_provider()
        assert provider._max_tokens is None  # conditional-send: absent from API call
        info = provider.get_info()
        # max_output_tokens is ALWAYS populated (see get_info() comment block)
        assert "max_output_tokens" in info.defaults
        assert info.defaults["max_output_tokens"] == 4096

    def test_model_info_max_output_tokens_set(self):
        """ModelInfo.max_output_tokens matches the configured max_tokens value."""
        provider = _make_list_models_provider(max_tokens="8192")
        models = asyncio.run(provider.list_models())
        assert len(models) >= 1
        assert all(m.max_output_tokens == 8192 for m in models)

    def test_model_info_max_output_tokens_fallback_when_absent(self):
        """ModelInfo.max_output_tokens falls back to 4096 when max_tokens not configured.

        Verifies the int|None conditional-send pattern: _max_tokens is None when
        not configured, but ModelInfo still receives 4096 as a conservative
        budget-math reservation value (matches the old always-4096 wire default).
        """
        provider = _make_list_models_provider()  # no max_tokens configured
        assert provider._max_tokens is None  # conditional-send: absent from API call
        models = asyncio.run(provider.list_models())
        assert len(models) >= 1
        assert all(m.max_output_tokens == 4096 for m in models)


# ---------------------------------------------------------------------------
# Case 8: Warning emitted exactly once across multiple list_models() calls
# ---------------------------------------------------------------------------


class TestWarnOnce:
    """One-shot warning: emitted exactly once per provider instance."""

    def test_warning_emitted_once_for_zero_context_window(self, caplog):
        """WARNING about unknown context_window is logged exactly once."""
        provider = _make_list_models_provider()  # context_window=0

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())
            asyncio.run(provider.list_models())
            asyncio.run(provider.list_models())

        bypass_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "bypass" in r.getMessage().lower()
        ]
        assert len(bypass_warnings) == 1, (
            f"Expected exactly 1 bypass warning, got {len(bypass_warnings)}"
        )

    def test_sentinel_set_after_first_call(self):
        """_context_window_warn_emitted is False initially, True after first call."""
        provider = _make_list_models_provider()
        assert provider._context_window_warn_emitted is False
        asyncio.run(provider.list_models())
        assert provider._context_window_warn_emitted is True

    def test_sentinel_prevents_duplicate_warnings(self, caplog):
        """Multiple list_models() calls do not produce multiple warnings."""
        provider = _make_list_models_provider()  # context_window=0

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            for _ in range(5):
                asyncio.run(provider.list_models())

        bypass_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "bypass" in r.getMessage().lower()
        ]
        assert len(bypass_warnings) == 1


# ---------------------------------------------------------------------------
# Case 9: INFO log emitted exactly once when context_window > 0
# ---------------------------------------------------------------------------


class TestInfoLogOnce:
    """One-shot INFO log when context_window > 0: emitted exactly once."""

    def test_info_logged_when_context_window_nonzero(self, caplog):
        """An INFO log is emitted the first time list_models() runs with context_window > 0."""
        provider = _make_list_models_provider(context_window="32768")

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())

        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "context_window" in r.getMessage()
        ]
        assert len(info_records) >= 1, (
            "Expected an INFO log mentioning 'context_window' when value > 0"
        )
        msg = info_records[0].getMessage()
        assert "32768" in msg
        assert "max_output_tokens" in msg or "truncation" in msg.lower()

    def test_info_logged_exactly_once(self, caplog):
        """INFO log about context_window is emitted exactly once per instance."""
        provider = _make_list_models_provider(context_window="32768")

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())
            asyncio.run(provider.list_models())
            asyncio.run(provider.list_models())

        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "context_window" in r.getMessage()
        ]
        assert len(info_records) == 1, (
            f"Expected exactly 1 INFO log about context_window, got {len(info_records)}"
        )

    def test_no_info_log_when_context_window_zero(self, caplog):
        """No 'context_window=...' INFO log when context_window is 0."""
        provider = _make_list_models_provider()  # context_window=0

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_chat_completions",
        ):
            asyncio.run(provider.list_models())

        # The INFO log only appears for context_window > 0
        ctx_info = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "context_window=" in r.getMessage()
        ]
        assert len(ctx_info) == 0


# ---------------------------------------------------------------------------
# ConfigField presence for context_window
# ---------------------------------------------------------------------------


class TestContextWindowConfigField:
    """get_info() must include the context_window ConfigField."""

    def test_config_field_present(self):
        """context_window ConfigField appears in get_info().config_fields."""
        provider = _make_provider()
        info = provider.get_info()
        field_ids = [f.id for f in info.config_fields]
        assert "context_window" in field_ids

    def test_config_field_is_text_type(self):
        """context_window ConfigField has field_type='text'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "context_window")
        assert field.field_type == "text"

    def test_config_field_default_is_zero_string(self):
        """context_window ConfigField has default='0'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "context_window")
        assert field.default == "0"

    def test_config_field_has_env_var(self):
        """context_window ConfigField has env_var='CHAT_COMPLETIONS_CONTEXT_WINDOW'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "context_window")
        assert field.env_var == "CHAT_COMPLETIONS_CONTEXT_WINDOW"

    def test_config_field_not_required(self):
        """context_window ConfigField is not required."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "context_window")
        assert field.required is False

    def test_config_field_display_name(self):
        """context_window ConfigField display_name is 'Context Window Override'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "context_window")
        assert field.display_name == "Context Window Override"


# ---------------------------------------------------------------------------
# Case 10: Negative context_window is clamped to 0 (COE finding 1)
# ---------------------------------------------------------------------------


class TestNegativeContextWindow:
    """Negative context_window values are clamped to 0 with a warning."""

    def test_negative_context_window_clamped_to_zero(self, caplog):
        """context_window='-1' → _context_window == 0; warning is logged."""
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            provider = _make_provider(context_window="-1")

        assert provider._context_window == 0, (
            "Negative context_window must be clamped to 0"
        )
        negative_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "negative" in r.getMessage().lower()
        ]
        assert len(negative_warnings) >= 1, (
            "Expected a WARNING log mentioning 'negative' when context_window < 0"
        )

    def test_negative_integer_context_window_clamped(self, caplog):
        """context_window=-100 (int) → _context_window == 0; warning is logged."""
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_chat_completions",
        ):
            provider = _make_provider(context_window=-100)

        assert provider._context_window == 0
        negative_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "negative" in r.getMessage().lower()
        ]
        assert len(negative_warnings) >= 1

    def test_negative_clamped_to_zero_not_in_defaults(self, caplog):
        """Clamped negative value → context_window absent from get_info().defaults."""
        with caplog.at_level(logging.WARNING):
            provider = _make_provider(context_window="-1")

        info = provider.get_info()
        assert "context_window" not in info.defaults, (
            "A clamped (0) context_window must not appear in get_info().defaults"
        )


# ---------------------------------------------------------------------------
# Case 11: Explicit '0' string behaves identically to absent (COE finding 1)
# ---------------------------------------------------------------------------


class TestExplicitZeroContextWindow:
    """Explicit context_window='0' is indistinguishable from absent config."""

    def test_explicit_zero_string_behaves_like_absent(self):
        """config context_window='0' → _context_window == 0; no 'context_window' key
        in get_info().defaults."""
        provider = _make_provider(context_window="0")

        assert provider._context_window == 0
        info = provider.get_info()
        assert "context_window" not in info.defaults, (
            "Explicit '0' must not inject 'context_window' into defaults"
        )

    def test_explicit_zero_vs_absent_produce_identical_defaults(self):
        """get_info().defaults is identical whether context_window is '0' or absent."""
        provider_explicit_zero = _make_provider(context_window="0")
        provider_absent = _make_provider()  # no context_window key at all

        defaults_explicit = provider_explicit_zero.get_info().defaults
        defaults_absent = provider_absent.get_info().defaults

        assert defaults_explicit == defaults_absent, (
            f"Defaults differ:\n  explicit '0': {defaults_explicit}\n"
            f"  absent:       {defaults_absent}"
        )
