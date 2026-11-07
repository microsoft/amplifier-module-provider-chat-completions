"""Behavioral tests for chat-completions provider.

Inherits authoritative tests from amplifier-core.
"""

from amplifier_core.validation.behavioral import ProviderBehaviorTests
import httpx
import openai
import pytest


class TestChatCompletionsProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for chat-completions.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Qualify the inherited contract against an available offline endpoint.

        The inherited fixture has no server; strict listing must not fabricate
        success for that unreachable address. Exercise the SDK, not a fallback.
        """
        provider_module._client = openai.AsyncOpenAI(
            api_key="test",
            base_url="https://endpoint.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, json={"data": [{"id": "actual"}]}
                    )
                )
            ),
        )
        try:
            await super().test_list_models_returns_list(provider_module)
        finally:
            await provider_module.close()
