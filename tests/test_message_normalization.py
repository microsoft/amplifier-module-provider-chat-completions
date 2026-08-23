"""Tests for two OpenAI-compatibility normalizations:

1. System-message coalescing -- every ``system`` message is merged into a
   single LEADING system message. This mirrors provider-anthropic and
   provider-openai (which extract all system messages regardless of position
   and join them with "\\n\\n" before routing to a top-level param), and it
   fixes the opaque HTTP 500 that strict chat templates (e.g. Qwen3 on vLLM)
   return when a system message appears after user/assistant turns.

2. ``default_headers`` config -- forwarded to the OpenAI client constructor
   (``default_headers=``), matching provider-anthropic's convention. Lets a
   caller override the User-Agent to get past a WAF that blocks non-browser
   clients (e.g. a Runpod pod proxy behind Cloudflare, "error code: 1010").
"""

from amplifier_core.message_models import Message

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


def _provider(**config):
    config.setdefault("model", "test-model")
    return ChatCompletionsProvider(config=config)


class TestSystemMessageCoalescing:
    def test_trailing_system_message_is_hoisted_and_merged(self):
        """A system message after user turns must be merged into ONE leading system.

        This is the exact shape Amplifier emits (a trailing system reminder)
        that made strict Qwen3 chat templates return HTTP 500.
        """
        msgs = [
            Message(role="system", content="LEADING"),
            Message(role="user", content="hi"),
            Message(role="user", content="more"),
            Message(role="system", content="TRAILING REMINDER"),
        ]
        wire = _provider()._convert_messages_to_wire(msgs)

        system_positions = [i for i, m in enumerate(wire) if m["role"] == "system"]
        assert system_positions == [0], (
            "exactly one system message, and it must be first"
        )
        assert wire[0]["content"] == "LEADING\n\nTRAILING REMINDER"
        # Non-system messages keep their original order and content.
        assert [(m["role"], m["content"]) for m in wire[1:]] == [
            ("user", "hi"),
            ("user", "more"),
        ]

    def test_single_leading_system_is_unchanged(self):
        wire = _provider()._convert_messages_to_wire(
            [
                Message(role="system", content="system prompt"),
                Message(role="user", content="hi"),
            ]
        )
        assert wire[0] == {"role": "system", "content": "system prompt"}

    def test_coalesce_helper_merges_with_double_newline(self):
        wire = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "B"},
        ]
        out = ChatCompletionsProvider._coalesce_system_messages(wire)
        assert out[0] == {"role": "system", "content": "A\n\nB"}
        assert out[1:] == [{"role": "user", "content": "u1"}]

    def test_coalesce_helper_noop_without_system(self):
        wire = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert ChatCompletionsProvider._coalesce_system_messages(wire) == wire


class TestDefaultHeaders:
    def test_default_headers_forwarded_to_client(self):
        ua = "Mozilla/5.0 (compatible; AmplifierTest/1.0)"
        provider = _provider(
            base_url="http://localhost:9/v1",
            api_key="x",
            default_headers={"User-Agent": ua},
        )
        assert provider._default_headers == {"User-Agent": ua}

        client = provider.client  # constructs AsyncOpenAI with default_headers=
        merged = {k.lower(): v for k, v in dict(client.default_headers).items()}
        assert merged.get("user-agent") == ua

    def test_default_headers_absent_is_none(self):
        provider = _provider(base_url="http://localhost:9/v1", api_key="x")
        assert provider._default_headers is None
        # Client still constructs fine with default_headers=None.
        assert provider.client is not None

    def test_non_dict_default_headers_ignored(self):
        provider = _provider(
            base_url="http://localhost:9/v1", api_key="x", default_headers="nope"
        )
        assert provider._default_headers is None
