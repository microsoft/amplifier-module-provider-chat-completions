"""Pytest configuration and shared fixtures for chat-completions provider tests.

The amplifier-core pytest plugin provides fixtures automatically:
- provider_module: Mounted module instance for behavioral tests
"""

import os

import pytest


@pytest.fixture(autouse=True)
def set_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CHAT_COMPLETIONS_BASE_URL is set so mount() proceeds during tests.

    The provider's mount() does a silent non-mount when base_url is not
    configured (production behaviour). For unit/behavioral tests we supply a
    dummy URL so the provider is always mounted; no real server is required.
    """
    if not os.environ.get("CHAT_COMPLETIONS_BASE_URL"):
        monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:11434/v1")
