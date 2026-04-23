"""Server-type probing for OpenAI-compatible endpoints without dedicated providers.

Probes llama.cpp, LM Studio, and TGI for runtime context-window information.
Advisory detection is provided for Ollama (no context reading — users should
use provider-ollama which handles num_ctx correctly).

Architecture / why this file exists
-------------------------------------
``chat-completions`` is the generic OpenAI-compatible provider — the fallback
for servers that don't have their own Amplifier module. The scope is therefore:

* Probe what we can, for server types **we** are the de-facto home for.
* Don't duplicate work done by dedicated providers. Users on servers with a
  dedicated provider (``provider-ollama``, ``provider-vllm``) should use that
  provider; chat-completions won't intrude on their territory.
* Read vendor extensions that happen to appear on the **standard** ``/v1/models``
  endpoint — free information regardless of server type (handled in __init__.py).

Probe matrix
-------------
+---------------+-----------------------+------------------------------------------+
| Server        | Dedicated provider?   | Action here                              |
+===============+=======================+==========================================+
| llama.cpp     | No                    | GET /props → n_ctx (high confidence)     |
+---------------+-----------------------+------------------------------------------+
| LM Studio     | No                    | GET /api/v0/models/{id}                  |
|               |                       | → loaded_context_length (high)           |
+---------------+-----------------------+------------------------------------------+
| TGI           | No                    | GET /info → max_input_length (high)      |
+---------------+-----------------------+------------------------------------------+
| Ollama        | Yes (provider-ollama) | Advisory log only — no context reading.  |
|               |                       | ``/api/show`` reads training context,    |
|               |                       | not runtime ``num_ctx``. provider-ollama |
|               |                       | handles this correctly.                  |
+---------------+-----------------------+------------------------------------------+
| vLLM          | Yes (provider-vllm)   | ``max_model_len`` is on the standard     |
|               |                       | ``/v1/models`` response — read for free  |
|               |                       | in __init__.py; no extra probe needed.   |
+---------------+-----------------------+------------------------------------------+

The training-context trap
--------------------------
Model metadata endpoints (such as Ollama's ``/api/show`` ``model_info`` block)
expose the **training** context length — the maximum the model was trained on,
e.g. 131,072 tokens for Llama-3.2. This is NOT the server's runtime limit. A
single-GPU host may load that model with a context window of only 8 192 tokens
(``num_ctx=8192``). Using training context as a kernel budget silently
over-budgets by 16× and produces harder-to-diagnose overflows than reporting 0.

This module exclusively reads **runtime** limits from server-specific endpoints
(``/props``, ``/api/v0/models/{id}``, ``/info``) and never reads training-time
metadata. The ``detect_ollama()`` advisory is intentionally context-free for
this reason.
"""

import asyncio
import logging
import urllib.parse
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a server-type probe.

    Attributes:
        context_window: Effective per-request context in tokens. 0 means unknown.
        server_type: Detected server type.
            One of: "llama.cpp", "lm-studio", "tgi", "vllm", "ollama", "unknown".
        source_endpoint: The URL that provided the value (for logging).
        confidence: Quality of the measurement.
            "high"   — runtime, authoritative (e.g. per-slot n_ctx from /props).
            "medium" — informative but potentially inaccurate (reserved for future use).
            "low"    — unknown / all probes failed.
        notes: Optional human-readable annotation.
    """

    context_window: int
    server_type: str
    source_endpoint: str
    confidence: str
    notes: str = ""


def _server_root(base_url: str) -> str:
    """Strip the trailing ``/v1`` suffix to get the server root URL.

    Provider ``base_url`` values typically end with ``/v1`` (e.g.
    ``http://localhost:8080/v1``). Probes for ``/props``, ``/info``,
    ``/api/tags``, and ``/api/v0/*`` need the root without that suffix.

    Args:
        base_url: The provider's base URL.

    Returns:
        The server root URL without the trailing ``/v1`` suffix.

    Examples:
        >>> _server_root("http://localhost:8080/v1")
        'http://localhost:8080'
        >>> _server_root("http://localhost:8080/v1/")
        'http://localhost:8080'
        >>> _server_root("http://localhost:8080")
        'http://localhost:8080'
    """
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


async def probe_llama_cpp_props(
    base_url: str,
    model_id: str,
    http_client: httpx.AsyncClient,
) -> ProbeResult | None:
    """Probe a llama.cpp server for its runtime per-slot context size.

    llama.cpp — GET /props
    Field: default_generation_settings.n_ctx (per-slot runtime; authoritative)
    Last verified: llama.cpp b5180, 2026-04
    Disable flag: --no-endpoint-props (server returns 404 when set)

    The ``n_ctx`` field in ``default_generation_settings`` reflects the
    **per-slot** context size (total ``--ctx-size`` divided by ``--parallel``).
    This is the correct value to use as a kernel token budget — it represents
    the actual limit for a single request, not the aggregate pool size.

    Args:
        base_url: Provider base URL (e.g. ``http://localhost:8080/v1``).
        model_id: Model identifier (unused; present for signature uniformity).
        http_client: httpx.AsyncClient for making requests.

    Returns:
        ProbeResult with ``context_window`` and ``server_type="llama.cpp"``,
        or None if the endpoint is absent or returns unusable data.
    """
    root = _server_root(base_url)
    url = f"{root}/props"
    try:
        response = await http_client.get(url)
    except httpx.HTTPError:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        n_ctx = int(data["default_generation_settings"]["n_ctx"])
    except (ValueError, KeyError, TypeError):
        return None

    if n_ctx <= 0:
        return None

    return ProbeResult(
        context_window=n_ctx,
        server_type="llama.cpp",
        source_endpoint=url,
        confidence="high",
        notes="per-slot runtime n_ctx from /props",
    )


async def probe_lm_studio_v0(
    base_url: str,
    model_id: str,
    http_client: httpx.AsyncClient,
) -> ProbeResult | None:
    """Probe an LM Studio server for the loaded model's runtime context length.

    LM Studio — GET /api/v0/models/{model_id}
    Field: loaded_context_length (runtime; authoritative when the model is loaded)
    Last verified: LM Studio 0.3.x, 2026-04
    Note: model_id may contain '/' (e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF");
          the ID is URL-encoded before embedding in the path.

    Args:
        base_url: Provider base URL (e.g. ``http://localhost:1234/v1``).
        model_id: Model identifier. May contain path separators; URL-encoded.
        http_client: httpx.AsyncClient for making requests.

    Returns:
        ProbeResult with ``context_window`` and ``server_type="lm-studio"``,
        or None if the endpoint is absent or returns unusable data.
    """
    root = _server_root(base_url)
    encoded_id = urllib.parse.quote(model_id, safe="")
    url = f"{root}/api/v0/models/{encoded_id}"
    try:
        response = await http_client.get(url)
    except httpx.HTTPError:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        ctx = int(data["loaded_context_length"])
    except (ValueError, KeyError, TypeError):
        return None

    if ctx <= 0:
        return None

    return ProbeResult(
        context_window=ctx,
        server_type="lm-studio",
        source_endpoint=url,
        confidence="high",
        notes="loaded_context_length from /api/v0/models/{model_id}",
    )


async def probe_tgi_info(
    base_url: str,
    model_id: str,
    http_client: httpx.AsyncClient,
) -> ProbeResult | None:
    """Probe a TGI (HuggingFace text-generation-inference) server.

    TGI — GET /info
    Field: max_input_length (authoritative server configuration)
    Last verified: TGI 2.4.x, 2026-04

    Args:
        base_url: Provider base URL (e.g. ``http://localhost:8080/v1``).
        model_id: Model identifier (unused; present for signature uniformity).
        http_client: httpx.AsyncClient for making requests.

    Returns:
        ProbeResult with ``context_window`` and ``server_type="tgi"``,
        or None if the endpoint is absent or returns unusable data.
    """
    root = _server_root(base_url)
    url = f"{root}/info"
    try:
        response = await http_client.get(url)
    except httpx.HTTPError:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        max_input = int(data["max_input_length"])
    except (ValueError, KeyError, TypeError):
        return None

    if max_input <= 0:
        return None

    return ProbeResult(
        context_window=max_input,
        server_type="tgi",
        source_endpoint=url,
        confidence="high",
        notes="max_input_length from /info",
    )


async def detect_ollama(
    base_url: str,
    http_client: httpx.AsyncClient,
) -> bool:
    """Detect whether the server is an Ollama instance.

    Advisory detection only — no context reading. If True, callers should
    log a hint that ``provider-ollama`` handles ``num_ctx`` correctly and
    should be used instead.

    Uses ``GET /api/tags`` as a fingerprint: Ollama responds with a JSON
    object that has a top-level ``"models"`` key (a list). Other servers
    typically return 404 or a different shape.

    Why we don't read context from Ollama
    ----------------------------------------
    Ollama's ``POST /api/show`` returns ``model_info["<arch>.context_length"]``,
    which is the model's **training** context size (e.g. 131072 for Llama-3.2).
    This is NOT the server's runtime ``num_ctx`` limit. When a user runs Ollama
    with ``--parallel 4``, each slot gets ``num_ctx / 4`` tokens. Using training
    context as a budget silently over-budgets and produces overflows that are
    harder to diagnose than reporting 0.

    ``provider-ollama`` handles this correctly via its ``num_ctx`` parameter.
    Users pointing chat-completions at Ollama should switch providers.

    Args:
        base_url: Provider base URL.
        http_client: httpx.AsyncClient for making requests.

    Returns:
        True if the server appears to be an Ollama instance.
    """
    root = _server_root(base_url)
    url = f"{root}/api/tags"
    try:
        response = await http_client.get(url)
    except httpx.HTTPError:
        return False

    if response.status_code != 200:
        return False

    try:
        data = response.json()
        return isinstance(data.get("models"), list)
    except Exception:
        return False


async def probe_server(
    base_url: str,
    model_id: str,
    http_client: httpx.AsyncClient,
    timeout: float = 2.0,
) -> ProbeResult:
    """Probe the server for runtime context-window information.

    Tries each probe function in order (llama.cpp → LM Studio → TGI).
    Returns the first high-confidence result. Falls back to an
    unknown-sentinel ``ProbeResult`` if all probes fail or time out.

    Probe order: authoritative-first, short-circuit on first success.
    Per-probe timeout: applied individually so a stalled probe doesn't
    block the others. Worst-case wall time: ``3 × timeout`` seconds.

    Architecture notes
    -------------------
    * Does NOT probe Ollama — ``/api/show`` returns training context, not
      runtime ``num_ctx``. Use ``detect_ollama()`` for the advisory log.
    * Does NOT probe vLLM non-standard endpoints — ``max_model_len`` is
      already on the standard ``/v1/models`` response (free read in
      ``list_models()``); no extra probe needed.

    Args:
        base_url: Provider base URL.
        model_id: Model identifier passed to probes that need it.
        http_client: httpx.AsyncClient for making requests.
        timeout: Per-probe timeout in seconds (default 2.0).

    Returns:
        ProbeResult from the first successful high-confidence probe, or
        ``ProbeResult(context_window=0, server_type="unknown", confidence="low")``
        if all probes fail or time out.
    """
    probe_fns = (probe_llama_cpp_props, probe_lm_studio_v0, probe_tgi_info)

    for probe_fn in probe_fns:
        try:
            result = await asyncio.wait_for(
                probe_fn(base_url, model_id, http_client),
                timeout=timeout,
            )
            if result is not None and result.confidence == "high":
                return result
        except (httpx.HTTPError, ValueError, KeyError, asyncio.TimeoutError):
            continue

    return ProbeResult(
        context_window=0,
        server_type="unknown",
        source_endpoint="",
        confidence="low",
        notes="all probes failed or timed out",
    )
