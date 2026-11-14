"""Pytest configuration and shared fixtures for chat-completions provider tests.

The amplifier-core pytest plugin provides fixtures automatically:
- provider_module: Mounted module instance for behavioral tests
"""

import os
import socket

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live-provider",
        action="store_true",
        default=False,
        help="Explicitly allow the live-server integration tests (not offline qualification)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_provider: requires explicit live network authority"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live-provider"):
        for item in items:
            if item.get_closest_marker("live_provider"):
                item.add_marker(
                    pytest.mark.skip(
                        reason="Live provider tests require --live-provider"
                    )
                )


@pytest.fixture(autouse=True)
def set_base_url_env(monkeypatch: pytest.MonkeyPatch, request) -> None:
    """Ensure CHAT_COMPLETIONS_BASE_URL is set so mount() proceeds during tests.

    The provider's mount() does a silent non-mount when base_url is not
    configured (production behaviour). For unit/behavioral tests we supply a
    dummy URL so the provider is always mounted; no real server is required.
    """
    if not request.node.get_closest_marker("live_provider"):
        monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", "http://localhost:11434/v1")


@pytest.fixture(autouse=True)
def offline_transport_fence(monkeypatch: pytest.MonkeyPatch, request) -> None:
    """Synthetic SDK fixtures must never escape to a real account or socket."""
    if request.node.get_closest_marker("live_provider") and request.config.getoption(
        "--live-provider"
    ):
        return
    for key in list(os.environ):
        if key.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
            monkeypatch.delenv(key, raising=False)

    def blocked(*args, **kwargs):
        raise AssertionError("Network access forbidden in offline provider tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
