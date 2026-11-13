# amplifier-module-provider-chat-completions

Amplifier provider module for any server implementing the OpenAI Chat Completions
wire format (`/v1/chat/completions`).

Unlike `amplifier-module-provider-openai` (which targets the OpenAI Responses API on
`api.openai.com`) and `amplifier-module-provider-azure-openai` (which extends it for
Azure-hosted deployments), this provider is deliberately endpoint-agnostic. Point it
at any Chat Completions-compatible server with a `base_url` and you are done.

Tested against:

- **llama-server** (llama.cpp)
- **vLLM** (Chat Completions mode)
- **SGLang**
- **LocalAI**
- **LM Studio**
- **text-generation-inference**
- Any other server speaking the OpenAI Chat Completions wire format

## Capabilities

- Tool calling (including parallel tool calls, configurable)
- Streaming and non-streaming modes
- JSON mode
- Automatic retries with exponential backoff (amplifier-core managed, SDK retries disabled)
- JIT tool-sequence repair for local models that emit non-compliant conversation history
- Per-instance naming so multiple chat-completions providers can be mounted in the same session with distinct routing identities

Tool results retain their `tool_call_id`. A message with several result blocks
produces one wire tool message per result, in block order. An uncorrelated result
or a mixture of result and other content blocks is rejected before dispatch,
rather than silently discarding evidence.

### Media compatibility and diagnostic privacy

This generic route does **not** qualify image support from an endpoint speaking
Chat Completions or a familiar model name. `get_media_request_support` reports
version 1 with no supported positions or MIME types for every selected model.
Hosts must prepare unavailable text before counting, without hydrating pixels.
User, historical, tool, assistant, system and developer image blocks become
`[Image unavailable: selected endpoint/model image support is unqualified; no pixels supplied]`.
Tool images stay in the matching call-bound tool result; they are never relocated
into a fabricated user message. Original text order, call IDs and canonical
metadata remain unchanged. This is honest text-only compatibility, not vision.

The host owns private request preparation and immutable count/send association.
When its `experimental.media.v1` capability is present, each transport attempt
validates that exact private request with the original call options. Ref-bearing
calls without a compatible capability fail locally. SDK retries remain disabled;
provider retries revalidate after backoff. Changes to the request or effective
provider selection refuse transport; media-bearing expert overrides of model,
messages, tools or stream are not accepted. Interrupted streams are not replayed
after any received chunk. This module has no native count/budget, public stream,
refit or compaction method and does not claim those independent paths.

Raw diagnostics remain opt-in and secret-redacted. Before any hook sees an event,
the provider makes a separate copy and replaces known Core/Chat Completions image
transports with an explicit not-retained marker. It neither ingests/publishes media
for logging nor invents a resolvable digest/ref. Unknown opaque text and continuation
fields are not JSON-decoded. Canonical history and actual SDK payloads are not
diagnostic copies.

Offline tests: `uv sync --frozen && uv run pytest -q`. Default tests clear API
credentials and block sockets. Live-server tests skip without
`--live-provider`; only use that flag under separate live-inference authority.

## Configuration

| Key | Type | Default | Env Var |
|---|---|---|---|
| `base_url` | text | *(required)* | `CHAT_COMPLETIONS_BASE_URL` |
| `api_key` | secret | `not-needed` | `CHAT_COMPLETIONS_API_KEY` |
| `default_model` / `model` | text | `default` | — |
| `max_tokens` | int | `4096` | — |
| `temperature` | float | `0.7` | — |
| `timeout` | float or null | `null` | Optional request deadline, seconds |
| `max_retries` | int | `3` | — |
| `min_retry_delay` | float | `1.0` | — |
| `max_retry_delay` | float | `30.0` | — |
| `use_streaming` | bool | `true` | — |
| `top_p` | float | `None` | — |
| `stop` | list[str] | `None` | — |
| `seed` | int | `None` | — |
| `parallel_tool_calls` | bool | `true` | — |
| `priority` | int | `100` | — |
| `filtered` | bool | `false` | Deprecated explicit configured-model filter; `model_filters` takes precedence |
| `model_filters` | mapping | `{}` (all endpoint IDs) | — |
| `raw` | bool | `false` | — |
| `default_headers` | dict | `None` | Extra HTTP headers forwarded to the OpenAI client constructor on every request (e.g. a browser-like `User-Agent` to get past a WAF fronting a self-hosted endpoint) |
| `extra_request_params` | dict | `{}` | Merged last into the `chat.completions.create()` kwargs -- an escape hatch for any param not listed above (e.g. `presence_penalty`, `frequency_penalty`, `logit_bias`, `response_format`) |

An explicit `config.api_key` takes precedence over `CHAT_COMPLETIONS_API_KEY`. If it is missing or empty, the environment variable is used; if both are absent, the provider sends `not-needed` for local endpoints.

Boolean and numeric keys accept native types or the string forms a config
wizard writes (`"true"`/`"false"`, `"3"`); invalid numeric strings warn and
fall back to the default rather than crashing at mount. Unrecognized
config keys produce a mount-time warning (with a did-you-mean suggestion).

## Model listing

`await provider.list_models()` returns all current endpoint IDs by default, including
custom aliases and tagged/quantized models. `filters=None` uses a validated immutable
snapshot of `config.model_filters`; any explicit mapping replaces that policy:
`filters={}` is unfiltered, `{"include": []}` includes none, and `exclude` wins.
`include`/`exclude` use case-insensitive, OS-independent glob matching without
changing returned IDs. Optional `hide_dated` suppresses a YYYY-MM-DD snapshot only
when its undated alias is selected. No vendor prefix is assumed.

An explicit legacy `filtered=true` still selects the configured ID, or the first
endpoint row if absent. Its `legacy_filtered_model` policy appears in
`get_model_list_info().default_filters`; canonical config and call overrides replace
it entirely. This legacy behavior is not the new default.

The bound client's endpoint, credentials and headers are preserved. Every call
fetches all pages afresh. A successful empty list stays empty; failures raise typed,
sanitized errors rather than inventing a configured-model fallback. Hosts should
offer manual exact entry when listing is unavailable. ModelInfo zero limits and
empty capabilities mean unknown, not measured endpoint capabilities.
`get_model_list_info()` reports the optional v1 filter schema and `provider-api`
source without I/O. The unshipped candidate `discover_models` method is removed.
Listing never restricts an explicitly selected inference model or changes its parameters.

## Silent-skip behavior

If `base_url` is not configured in either the module config or
`CHAT_COMPLETIONS_BASE_URL`, `mount()` returns `None` and logs an info-level
message rather than raising. This allows the provider to be pre-installed and
always available without forcing every user to configure it.

## Installation

This module is pre-installed by `amplifier-app-cli` alongside the standard
providers. To use it, install Amplifier and configure the provider
interactively:

```bash
uv tool install git+https://github.com/microsoft/amplifier@main
amplifier provider add
# Select "OpenAI-Compatible (self-hosted)" and supply base_url + model name.
```

### Installing standalone

```bash
amplifier module add provider-chat-completions \
  --source git+https://github.com/microsoft/amplifier-module-provider-chat-completions@main
```

### Installing for development

```bash
git clone https://github.com/microsoft/amplifier-module-provider-chat-completions.git
cd amplifier-module-provider-chat-completions
uv sync
```

## Related

- [amplifier-bundle-chat-completions](https://github.com/microsoft/amplifier-bundle-chat-completions) — thin bundle that composes this provider with an opinionated config context. Useful when composing into a larger bundle.
- [amplifier-module-provider-openai](https://github.com/microsoft/amplifier-module-provider-openai) — targets the OpenAI Responses API on `api.openai.com`.
- [amplifier-module-provider-azure-openai](https://github.com/microsoft/amplifier-module-provider-azure-openai) — extends provider-openai for Azure.

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
### Waiting for model work

Completion and streaming requests have no elapsed-time or read-idle deadline by default. They wait for completion, explicit cancellation, or a provider/transport error. Set `timeout` (seconds) to opt into a request deadline; `null` leaves model work unbounded. Connection and pool acquisition remain bounded to 5 seconds, and existing `close_timeout` cleanup limits are unchanged.

### Cache usage reporting

Both response modes preserve `prompt_tokens_details.cached_tokens` when reported,
including zero. Missing cache usage remains unknown. Cost calculation continues
to use the existing rates; the provider does not infer cache hits or enable
vendor-specific cache controls on arbitrary compatible endpoints.
