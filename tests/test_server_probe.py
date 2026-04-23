"""Tests for server-type probing (PR 3).

Covers all 16 test cases from the PR 3 design doc:

 1. llama.cpp /props success
 2. llama.cpp /props disabled (404) → fall through to next probe
 3. LM Studio /api/v0/models/{id} success
 4. LM Studio model_id with '/' → URL-encoded properly
 5. TGI /info success
 6. Ollama /api/tags detection → advisory log, no context value consumed
 7. All probes fail → unknown-sentinel
 8. Vendor extension (max_model_len) wins over probe — no extra HTTP
 9. Config wins over vendor extension (and probe)
10. Concurrent probe dedup → exactly one HTTP call per endpoint
11. Probe cached across sequential calls → zero HTTP on second call
12. Negative caching → after all-fail, subsequent calls don't re-probe
13. Integration: probe result flows to get_info().defaults AND ModelInfo.context_window
14. Drift detection: configured + probe differ → warning log, config wins
15. auto_probe_context: false → no probes fire
16. auto_probe_timeout_seconds: slow probe cancelled at timeout, falls through

Plus a structural guardrail:
    The training-context trap string must not appear as "n_ctx_train" in
    _server_probe.py (design guardrail against reading training context as
    a runtime budget).

HTTP mocking uses ``respx`` (httpx-native; pytest-friendly).
"""

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx  # type: ignore[import-untyped]

from amplifier_module_provider_chat_completions import ChatCompletionsProvider
from amplifier_module_provider_chat_completions._server_probe import (
    _server_root,
    detect_ollama,
    probe_lm_studio_v0,
    probe_llama_cpp_props,
    probe_server,
    probe_tgi_info,
)

# ---------------------------------------------------------------------------
# Constants — must match the base URL the conftest.py autouse fixture injects
# via CHAT_COMPLETIONS_BASE_URL so that provider integration tests align with
# what the provider actually uses internally.
# ---------------------------------------------------------------------------

# conftest.py sets CHAT_COMPLETIONS_BASE_URL="http://localhost:11434/v1" unless
# already set.  Provider integration tests rely on the provider picking up this
# URL, so we mirror it here and override it with our own value via the autouse
# fixture below.
BASE_URL = "http://localhost:8080/v1"
SERVER_ROOT = "http://localhost:8080"
MODEL_ID = "test-model"


@pytest.fixture(autouse=True)
def _override_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CHAT_COMPLETIONS_BASE_URL to BASE_URL for all probe tests.

    conftest.py's autouse fixture sets the env var to localhost:11434.  This
    local fixture runs *after* that one (same scope, later discovery) and
    overrides it so providers created by _make_provider() use the same base
    URL as our respx mock routes (localhost:8080).
    """
    monkeypatch.setenv("CHAT_COMPLETIONS_BASE_URL", BASE_URL)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> ChatCompletionsProvider:
    """Create a ChatCompletionsProvider with test-friendly defaults."""
    config: dict = {
        "base_url": BASE_URL,
        "model": MODEL_ID,
        "filtered": False,
        **config_overrides,
    }
    return ChatCompletionsProvider(config=config)


def _mock_openai_client(
    provider: ChatCompletionsProvider, *model_ids: str, **model_attrs
) -> MagicMock:
    """Inject a mock openai client returning the given model IDs."""
    provider._client = MagicMock()
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id=mid, **model_attrs) for mid in model_ids]
        )
    )
    return provider._client  # type: ignore[return-value]


# ===========================================================================
# Case 1: llama.cpp /props success
# ===========================================================================


class TestLlamaCppPropsSuccess:
    """Case 1: GET /props returns a valid llama.cpp n_ctx."""

    @pytest.mark.asyncio
    async def test_probe_result_fields(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(
                return_value=httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 32768}},
                )
            )
            async with httpx.AsyncClient() as client:
                result = await probe_llama_cpp_props(BASE_URL, MODEL_ID, client)

        assert result is not None
        assert result.context_window == 32768
        assert result.server_type == "llama.cpp"
        assert result.confidence == "high"
        assert "/props" in result.source_endpoint

    @pytest.mark.asyncio
    async def test_probe_server_short_circuits_on_llama_cpp(self) -> None:
        """probe_server returns immediately on llama.cpp success; LM Studio/TGI skipped."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(
                return_value=httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 32768}},
                )
            )
            # Register but don't call — verifies short-circuit.
            lm_route = mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            tgi_route = mock.get(f"{SERVER_ROOT}/info").mock(
                return_value=httpx.Response(404)
            )
            async with httpx.AsyncClient() as client:
                result = await probe_server(BASE_URL, MODEL_ID, client, timeout=2.0)

        assert result.context_window == 32768
        assert result.server_type == "llama.cpp"
        assert result.confidence == "high"
        assert not lm_route.called
        assert not tgi_route.called


# ===========================================================================
# Case 2: llama.cpp /props disabled (404) → fall through to LM Studio
# ===========================================================================


class TestLlamaCppPropsDisabled:
    """Case 2: --no-endpoint-props returns 404; probe falls through."""

    @pytest.mark.asyncio
    async def test_probe_fn_returns_none_on_404(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            async with httpx.AsyncClient() as client:
                result = await probe_llama_cpp_props(BASE_URL, MODEL_ID, client)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_server_falls_through_to_lm_studio(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(200, json={"loaded_context_length": 16384})
            )
            async with httpx.AsyncClient() as client:
                result = await probe_server(BASE_URL, MODEL_ID, client, timeout=2.0)

        assert result.context_window == 16384
        assert result.server_type == "lm-studio"
        assert result.confidence == "high"


# ===========================================================================
# Case 3: LM Studio /api/v0/models/{id} success
# ===========================================================================


class TestLmStudioV0Success:
    """Case 3: LM Studio v0 API returns loaded_context_length."""

    @pytest.mark.asyncio
    async def test_probe_fn_returns_loaded_context_length(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(200, json={"loaded_context_length": 8192})
            )
            async with httpx.AsyncClient() as client:
                result = await probe_lm_studio_v0(BASE_URL, MODEL_ID, client)

        assert result is not None
        assert result.context_window == 8192
        assert result.server_type == "lm-studio"
        assert result.confidence == "high"


# ===========================================================================
# Case 4: LM Studio model_id with '/' → URL-encoded properly
# ===========================================================================


class TestLmStudioUrlEncoding:
    """Case 4: model_id containing '/' must be percent-encoded in the URL."""

    @pytest.mark.asyncio
    async def test_slash_in_model_id_is_percent_encoded(self) -> None:
        model_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        encoded_id = "bartowski%2FLlama-3.2-3B-Instruct-GGUF"

        with respx.mock() as mock:
            route = mock.get(f"{SERVER_ROOT}/api/v0/models/{encoded_id}").mock(
                return_value=httpx.Response(200, json={"loaded_context_length": 4096})
            )
            async with httpx.AsyncClient() as client:
                result = await probe_lm_studio_v0(BASE_URL, model_id, client)

        assert result is not None
        assert result.context_window == 4096
        # The encoded URL was actually requested.
        assert route.called


# ===========================================================================
# Case 5: TGI /info success
# ===========================================================================


class TestTgiInfoSuccess:
    """Case 5: TGI /info returns max_input_length."""

    @pytest.mark.asyncio
    async def test_probe_fn_returns_max_input_length(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/info").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "max_input_length": 4096,
                        "max_total_tokens": 5120,
                        "model_id": "mistral-7b",
                    },
                )
            )
            async with httpx.AsyncClient() as client:
                result = await probe_tgi_info(BASE_URL, MODEL_ID, client)

        assert result is not None
        assert result.context_window == 4096
        assert result.server_type == "tgi"
        assert result.confidence == "high"

    @pytest.mark.asyncio
    async def test_probe_server_falls_through_to_tgi(self) -> None:
        """All earlier probes fail; TGI probe succeeds."""
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            mock.get(f"{SERVER_ROOT}/info").mock(
                return_value=httpx.Response(200, json={"max_input_length": 2048})
            )
            async with httpx.AsyncClient() as client:
                result = await probe_server(BASE_URL, MODEL_ID, client, timeout=2.0)

        assert result.context_window == 2048
        assert result.server_type == "tgi"


# ===========================================================================
# Case 6: Ollama /api/tags detection → advisory log, no context consumed
# ===========================================================================


class TestOllamaDetection:
    """Case 6: detect_ollama signals Ollama; advisory log fires, context stays 0."""

    @pytest.mark.asyncio
    async def test_detect_ollama_true_when_api_tags_has_models_list(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(
                return_value=httpx.Response(
                    200,
                    json={"models": [{"name": "llama3:latest"}]},
                )
            )
            async with httpx.AsyncClient() as client:
                result = await detect_ollama(BASE_URL, client)

        assert result is True

    @pytest.mark.asyncio
    async def test_detect_ollama_false_when_api_tags_absent(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            async with httpx.AsyncClient() as client:
                result = await detect_ollama(BASE_URL, client)

        assert result is False

    @pytest.mark.asyncio
    async def test_advisory_log_emitted_and_context_window_zero(self, caplog) -> None:
        """Ollama detected → advisory log; no context value read from Ollama."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(
                return_value=httpx.Response(
                    200,
                    json={"models": [{"name": "llama3:latest"}]},
                )
            )
            # All probe endpoints fail → context stays 0.
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            mock.get(f"{SERVER_ROOT}/info").mock(return_value=httpx.Response(404))

            provider = _make_provider()
            _mock_openai_client(provider, MODEL_ID)

            with caplog.at_level(logging.INFO):
                models = await provider.list_models()

        # Advisory log mentions Ollama and the correct alternative provider.
        log_text = " ".join(r.message for r in caplog.records)
        assert "Ollama" in log_text or "ollama" in log_text
        assert "provider-ollama" in log_text

        # No context_window was read from Ollama.
        assert models[0].context_window == 0


# ===========================================================================
# Case 7: All probes fail → unknown-sentinel
# ===========================================================================


class TestAllProbesFail:
    """Case 7: probe_server returns unknown-sentinel when all probes fail."""

    @pytest.mark.asyncio
    async def test_unknown_sentinel_returned(self) -> None:
        with respx.mock() as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            mock.get(f"{SERVER_ROOT}/info").mock(return_value=httpx.Response(404))
            async with httpx.AsyncClient() as client:
                result = await probe_server(BASE_URL, MODEL_ID, client, timeout=2.0)

        assert result.context_window == 0
        assert result.server_type == "unknown"
        assert result.confidence == "low"


# ===========================================================================
# Case 8: Vendor extension (max_model_len) wins over probe — no extra HTTP
# ===========================================================================


class TestVendorExtWinsOverProbe:
    """Case 8: max_model_len on /v1/models is consumed; no probe endpoint called."""

    @pytest.mark.asyncio
    async def test_max_model_len_used_probe_never_called(self) -> None:
        """When vendor extension provides context, probe endpoints are never invoked."""
        props_called = False
        lm_called = False
        tgi_called = False

        async def mark_called_props(req):
            nonlocal props_called
            props_called = True
            return httpx.Response(404)

        async def mark_called_lm(req):
            nonlocal lm_called
            lm_called = True
            return httpx.Response(404)

        async def mark_called_tgi(req):
            nonlocal tgi_called
            tgi_called = True
            return httpx.Response(404)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=mark_called_props)
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                side_effect=mark_called_lm
            )
            mock.get(f"{SERVER_ROOT}/info").mock(side_effect=mark_called_tgi)

            provider = _make_provider()
            # Model object from /v1/models carries max_model_len — free read.
            provider._client = MagicMock()
            provider._client.models.list = AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id=MODEL_ID, max_model_len=131072)]
                )
            )

            models = await provider.list_models()

        assert models[0].context_window == 131072
        # Verify that probe endpoints were NEVER contacted.
        assert not props_called, "/props was called but vendor-ext should have won"
        assert not lm_called, "/api/v0/models was called but vendor-ext should have won"
        assert not tgi_called, "/info was called but vendor-ext should have won"

    @pytest.mark.asyncio
    async def test_loaded_context_length_also_consumed(self) -> None:
        """loaded_context_length (LM Studio extension on /v1/models) is also free-read."""
        props_called = False

        async def mark_called(req):
            nonlocal props_called
            props_called = True
            return httpx.Response(404)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=mark_called)
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                side_effect=mark_called
            )
            mock.get(f"{SERVER_ROOT}/info").mock(side_effect=mark_called)

            provider = _make_provider()
            provider._client = MagicMock()
            provider._client.models.list = AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id=MODEL_ID, loaded_context_length=65536)]
                )
            )

            models = await provider.list_models()

        assert models[0].context_window == 65536
        assert not props_called, (
            "Probe fired when vendor-ext (loaded_context_length) should have won"
        )


# ===========================================================================
# Case 9: Config wins over vendor extension and probe
# ===========================================================================


class TestConfigWinsOverVendorExt:
    """Case 9: Explicit context_window config overrides vendor extensions and probe."""

    @pytest.mark.asyncio
    async def test_config_wins_over_max_model_len(self) -> None:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            # Probe runs for drift detection (config is set); return 404 → no drift.
            mock.get(f"{SERVER_ROOT}/props").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            mock.get(f"{SERVER_ROOT}/info").mock(return_value=httpx.Response(404))

            provider = _make_provider(context_window=4096)
            provider._client = MagicMock()
            provider._client.models.list = AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id=MODEL_ID, max_model_len=131072)]
                )
            )

            models = await provider.list_models()
            info = provider.get_info()

        # Config wins in ModelInfo and in get_info().defaults.
        assert models[0].context_window == 4096
        assert info.defaults.get("context_window") == 4096


# ===========================================================================
# Case 10: Concurrent probe dedup → exactly one HTTP call per endpoint
# ===========================================================================


class TestConcurrentProbeDedup:
    """Case 10: Two concurrent list_models() produce exactly one probe HTTP call."""

    @pytest.mark.asyncio
    async def test_one_probe_despite_two_concurrent_callers(self) -> None:
        probe_call_count = 0

        async def counted_props(request):
            nonlocal probe_call_count
            probe_call_count += 1
            # Yield so the second list_models() task can start before we return.
            await asyncio.sleep(0.02)
            return httpx.Response(
                200,
                json={"default_generation_settings": {"n_ctx": 32768}},
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=counted_props)

            provider = _make_provider()
            _mock_openai_client(provider, MODEL_ID)

            # Two concurrent callers — wizard model-picker + early routing check.
            results = await asyncio.gather(
                provider.list_models(),
                provider.list_models(),
            )

        # Both awaiters joined the same task; HTTP probe fired exactly once.
        assert probe_call_count == 1
        assert results[0][0].context_window == 32768
        assert results[1][0].context_window == 32768


# ===========================================================================
# Case 11: Probe cached across sequential calls → zero HTTP on second call
# ===========================================================================


class TestProbeCachedAcrossCalls:
    """Case 11: Second list_models() call reads from probe cache, no new HTTP."""

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self) -> None:
        probe_call_count = 0

        async def counted_props(request):
            nonlocal probe_call_count
            probe_call_count += 1
            return httpx.Response(
                200,
                json={"default_generation_settings": {"n_ctx": 16384}},
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=counted_props)

            provider = _make_provider()
            _mock_openai_client(provider, MODEL_ID)

            models1 = await provider.list_models()
            models2 = await provider.list_models()

        assert probe_call_count == 1  # Fired exactly once.
        assert models1[0].context_window == 16384
        assert models2[0].context_window == 16384


# ===========================================================================
# Case 12: Negative caching → after all-fail, subsequent calls don't re-probe
# ===========================================================================


class TestNegativeCaching:
    """Case 12: Failed probes cached as unknown; no re-probing on subsequent calls."""

    @pytest.mark.asyncio
    async def test_all_fail_cached_no_reprobing(self) -> None:
        probe_call_count = 0

        async def counted_404(request):
            nonlocal probe_call_count
            probe_call_count += 1
            return httpx.Response(404)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=counted_404)
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(404)
            )
            mock.get(f"{SERVER_ROOT}/info").mock(return_value=httpx.Response(404))

            provider = _make_provider()
            _mock_openai_client(provider, MODEL_ID)

            # First call: tries all probes, all fail → unknown result cached.
            models1 = await provider.list_models()
            # Second call: reads from negative cache, no new HTTP calls.
            models2 = await provider.list_models()

        # /props called exactly once (only for the first list_models() call).
        assert probe_call_count == 1
        assert models1[0].context_window == 0
        assert models2[0].context_window == 0


# ===========================================================================
# Case 13: Integration — probe flows to get_info().defaults AND ModelInfo
# ===========================================================================


class TestProbeFlowsToGetInfo:
    """Case 13: Probe result appears in get_info().defaults['context_window']."""

    @pytest.mark.asyncio
    async def test_probe_populates_model_info_and_defaults(self) -> None:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(
                return_value=httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 32768}},
                )
            )

            provider = _make_provider()
            _mock_openai_client(provider, MODEL_ID)

            # list_models() triggers the probe and caches the result.
            models = await provider.list_models()
            # get_info() is sync — reads from _probed_context_window.
            info = provider.get_info()

        # ModelInfo carries the probed context_window.
        assert models[0].context_window == 32768

        # get_info().defaults["context_window"] reflects the probed value.
        assert info.defaults.get("context_window") == 32768


# ===========================================================================
# Case 14: Drift detection → configured + probed differ → warning, config wins
# ===========================================================================


class TestDriftDetection:
    """Case 14: config context_window differs from probed value → drift warning."""

    @pytest.mark.asyncio
    async def test_drift_warning_emitted_and_config_wins(self, caplog) -> None:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(
                return_value=httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 32768}},
                )
            )

            provider = _make_provider(context_window=16384)
            _mock_openai_client(provider, MODEL_ID)

            with caplog.at_level(logging.WARNING):
                models = await provider.list_models()

        # Config wins in ModelInfo.
        assert models[0].context_window == 16384

        # Drift warning was logged exactly once with both values.
        drift_records = [
            r for r in caplog.records if "configured context_window" in r.message
        ]
        assert len(drift_records) == 1
        assert "16384" in drift_records[0].message
        assert "32768" in drift_records[0].message
        assert drift_records[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_drift_warning_only_once_per_session(self, caplog) -> None:
        """Drift warning fires at most once per provider instance."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/api/tags").mock(return_value=httpx.Response(404))
            mock.get(f"{SERVER_ROOT}/props").mock(
                return_value=httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 32768}},
                )
            )

            provider = _make_provider(context_window=16384)
            _mock_openai_client(provider, MODEL_ID)

            with caplog.at_level(logging.WARNING):
                await provider.list_models()
                await provider.list_models()

        drift_count = sum(
            1 for r in caplog.records if "configured context_window" in r.message
        )
        assert drift_count == 1


# ===========================================================================
# Case 15: auto_probe_context: false → no probes fire
# ===========================================================================


class TestAutoProbeDisabled:
    """Case 15: auto_probe_context=False suppresses all probing and Ollama check."""

    @pytest.mark.asyncio
    async def test_no_http_calls_when_probe_disabled(self) -> None:
        with respx.mock(assert_all_called=False) as mock:
            provider = _make_provider(auto_probe_context=False)
            _mock_openai_client(provider, MODEL_ID)

            models = await provider.list_models()

        # Zero HTTP calls to any probe or Ollama-detection endpoint.
        assert len(mock.calls) == 0
        assert models[0].context_window == 0

    @pytest.mark.asyncio
    async def test_no_context_window_in_get_info_when_disabled_no_config(self) -> None:
        """With probe disabled and no config, get_info() has no context_window key."""
        provider = _make_provider(auto_probe_context=False)
        info = provider.get_info()
        assert "context_window" not in info.defaults

    @pytest.mark.asyncio
    async def test_config_still_respected_when_probe_disabled(self) -> None:
        """Explicit config context_window works even with auto_probe_context=False."""
        with respx.mock(assert_all_called=False) as mock:
            provider = _make_provider(auto_probe_context=False, context_window=8192)
            _mock_openai_client(provider, MODEL_ID)

            models = await provider.list_models()
            info = provider.get_info()

        assert len(mock.calls) == 0  # No probe HTTP calls.
        assert models[0].context_window == 8192
        assert info.defaults.get("context_window") == 8192


# ===========================================================================
# Case 16: Slow probe cancelled at timeout; falls through to next probe
# ===========================================================================


class TestProbeTimeout:
    """Case 16: Per-probe timeout cancels a stalled probe; next probe is tried."""

    @pytest.mark.asyncio
    async def test_slow_llama_cpp_falls_through_to_lm_studio(self) -> None:
        async def slow_props(request):
            # Sleep much longer than the configured per-probe timeout.
            await asyncio.sleep(1.0)
            # If somehow reached (test failure), return something obviously wrong.
            return httpx.Response(
                200,
                json={"default_generation_settings": {"n_ctx": 99999}},
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SERVER_ROOT}/props").mock(side_effect=slow_props)
            mock.get(f"{SERVER_ROOT}/api/v0/models/{MODEL_ID}").mock(
                return_value=httpx.Response(200, json={"loaded_context_length": 8192})
            )

            async with httpx.AsyncClient() as client:
                result = await probe_server(
                    BASE_URL,
                    MODEL_ID,
                    client,
                    timeout=0.05,  # 50ms — far shorter than the 1s sleep.
                )

        # /props timed out; fell through to LM Studio.
        assert result.context_window == 8192
        assert result.server_type == "lm-studio"
        assert result.confidence == "high"


# ===========================================================================
# Structural guardrail: "n_ctx_train" must not appear in _server_probe.py
# ===========================================================================


class TestNCtxTrainNeverUsed:
    """Design guardrail: the literal string n_ctx_train must not appear in probe module.

    n_ctx_train is the model's *training* context size — different from and usually
    larger than the server's runtime ``num_ctx`` limit. Reading it and using it as
    a kernel budget silently over-budgets and produces overflows that are harder to
    diagnose than reporting 0. This grep test ensures the guardrail stays in place.
    """

    def test_n_ctx_train_absent_from_server_probe_module(self) -> None:
        probe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "amplifier_module_provider_chat_completions",
            "_server_probe.py",
        )
        with open(probe_path) as f:
            content = f.read()

        # The field name "n_ctx_train" must not appear anywhere in the module —
        # not in code, not in comments, not in docstrings.
        assert "n_ctx_train" not in content, (
            "_server_probe.py must not reference 'n_ctx_train'. "
            "Use the runtime n_ctx from /props, not the model's training context. "
            "See PR 3 design doc §'n_ctx_train trap' for background."
        )


# ===========================================================================
# Utility: _server_root helper
# ===========================================================================


class TestServerRoot:
    """Unit tests for the _server_root() helper."""

    def test_strips_v1_suffix(self) -> None:
        assert _server_root("http://localhost:8080/v1") == "http://localhost:8080"

    def test_strips_trailing_slash_then_v1(self) -> None:
        assert _server_root("http://localhost:8080/v1/") == "http://localhost:8080"

    def test_no_change_when_no_v1_suffix(self) -> None:
        assert _server_root("http://localhost:8080") == "http://localhost:8080"

    def test_preserves_path_prefix_before_v1(self) -> None:
        assert (
            _server_root("http://localhost:8080/some/v1")
            == "http://localhost:8080/some"
        )
