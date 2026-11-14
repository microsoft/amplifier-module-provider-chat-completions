"""The declared SDK floor must construct a genuine client, including for text."""

import tomllib
from pathlib import Path

import openai
import pytest
from packaging.requirements import Requirement

from amplifier_module_provider_chat_completions import ChatCompletionsProvider


def test_declared_and_locked_sdk_floor_match_qualified_baseline():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    requirement = next(
        Requirement(value)
        for value in project["project"]["dependencies"]
        if Requirement(value).name == "openai"
    )
    assert str(requirement.specifier) == ">=2.32.0"
    assert "1.0.0" not in requirement.specifier
    assert "2.32.0" in requirement.specifier
    metadata = next(
        package["metadata"]["requires-dist"]
        for package in lock["package"]
        if package["name"] == "amplifier-module-provider-chat-completions"
    )
    assert {"name": "openai", "specifier": ">=2.32.0"} in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["unknown-text-model", "gpt-5.4"])
async def test_real_unpatched_official_client_constructs_for_text_and_images(
    monkeypatch, model
):
    # No patch of DefaultAsyncHttpxClient or AsyncOpenAI: this is the constructor
    # boundary the previous SDK1.0.0 declaration failed before any request.
    monkeypatch.delenv("CHAT_COMPLETIONS_BASE_URL", raising=False)
    provider = ChatCompletionsProvider(
        {
            "base_url": "https://api.openai.com/v1",
            "api_key": "offline-fixture-not-a-credential",
            "model": model,
        }
    )
    try:
        client = provider.client
        assert type(client) is openai.AsyncOpenAI
        assert isinstance(client._client, openai.DefaultAsyncHttpxClient)
        assert client._client.follow_redirects is False
        assert client.max_retries == 0
        support = provider.get_media_request_support(model=model)
        assert support["positions"] == (["user"] if model == "gpt-5.4" else [])
    finally:
        await provider.close()
