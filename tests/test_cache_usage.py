"""Reported cache misses and unreported usage have different meanings."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_chat_completions import ChatCompletionsProvider


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('details,expected', [(None, None), (SimpleNamespace(), None),
    (SimpleNamespace(cached_tokens=None), None), (SimpleNamespace(cached_tokens=0), 0),
    (SimpleNamespace(cached_tokens=300), 300)])
async def test_reported_cache_usage_survives_both_response_paths(streaming, details, expected):
    provider = ChatCompletionsProvider(config={'use_streaming': streaming, 'max_retries': 0})
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=10, total_tokens=1010,
                            prompt_tokens_details=details)
    response = SimpleNamespace(model='test-model', usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content='done', tool_calls=None), finish_reason='stop')])
    async def chunks():
        yield SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason='stop',
            delta=SimpleNamespace(content='done', tool_calls=None, reasoning_content=None))])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=chunks() if streaming else response)
    provider._client = client
    result = await provider.complete(ChatRequest(messages=[Message(role='user', content='hello')]))
    assert result.usage.input_tokens == 1000
    assert result.usage.total_tokens == 1010
    assert result.usage.cache_read_tokens == expected
