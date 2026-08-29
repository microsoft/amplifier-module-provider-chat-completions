"""Config hygiene tests: unknown-key sweep and extra_request_params merge
order.

This provider already had clean bool/numeric coercion (`_config_bool`/
`_config_int`/`_config_float` already handle string forms). This file adds
the missing pieces from the "family hygiene wave" survey:
  - a mount-time unknown-config-key sweep (didn't exist before).
  - `extra_request_params`, merged last into both the streaming and
    non-streaming request builds.
"""

from __future__ import annotations

import logging

import amplifier_module_provider_chat_completions as _provider_module

ChatCompletionsProvider = _provider_module.ChatCompletionsProvider
_warn_unknown_config_keys = _provider_module._warn_unknown_config_keys  # type: ignore[attr-defined]


def _provider(**config_overrides):
    config = {"base_url": "http://localhost:11434/v1", **config_overrides}
    return ChatCompletionsProvider(config=config)


class TestUnknownConfigKeySweep:
    def test_known_keys_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"base_url": "x", "priority": 1})
        assert caplog.text == ""

    def test_extra_request_params_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"extra_request_params": {}})
        assert caplog.text == ""

    def test_default_headers_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"default_headers": {}})
        assert caplog.text == ""

    def test_priority_never_flagged(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"priority": 50})
        assert caplog.text == ""

    def test_unknown_key_warns_with_suggestion(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"tiemout": 5})
        assert "tiemout" in caplog.text
        assert "timeout" in caplog.text

    def test_sweep_runs_at_construction(self, caplog):
        """The sweep is wired into __init__, not just callable standalone."""
        with caplog.at_level(logging.WARNING):
            _provider(bogus_key_xyz=1)
        assert "bogus_key_xyz" in caplog.text


class TestExtraRequestParams:
    def test_stored_on_provider(self):
        provider = _provider(extra_request_params={"presence_penalty": 0.5})
        assert provider._extra_request_params == {"presence_penalty": 0.5}

    def test_non_dict_ignored_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider = _provider(extra_request_params="nope")
        assert provider._extra_request_params == {}
        assert "extra_request_params" in caplog.text

    def test_default_is_empty_dict(self):
        provider = _provider()
        assert provider._extra_request_params == {}

    def test_merged_last_into_non_streaming_params(self, monkeypatch):
        """extra_request_params overrides computed defaults and adds new keys
        in the non-streaming request build."""
        provider = _provider(
            top_p=0.9,
            extra_request_params={"top_p": 0.1, "presence_penalty": 0.5},
        )

        captured: dict = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("stop before actually calling the API")

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        provider._client = _FakeClient()  # type: ignore[assignment]

        import asyncio

        from amplifier_core.message_models import ChatRequest, Message

        request = ChatRequest(messages=[Message(role="user", content="hi")])

        async def _run():
            try:
                await provider._complete_non_streaming([], None, request)
            except RuntimeError:
                pass

        asyncio.run(_run())

        assert captured["top_p"] == 0.1
        assert captured["presence_penalty"] == 0.5

    def test_merged_last_into_streaming_params(self):
        provider = _provider(
            use_streaming=True,
            extra_request_params={"logit_bias": {"123": -100}},
        )

        captured: dict = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("stop before actually calling the API")

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        provider._client = _FakeClient()  # type: ignore[assignment]

        import asyncio

        from amplifier_core.message_models import ChatRequest, Message

        request = ChatRequest(messages=[Message(role="user", content="hi")])

        async def _run():
            try:
                await provider._complete_streaming([], None, request)
            except RuntimeError:
                pass

        asyncio.run(_run())

        assert captured["logit_bias"] == {"123": -100}
