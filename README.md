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

## Configuration

| Key | Type | Default | Env Var |
|---|---|---|---|
| `base_url` | text | *(required)* | `CHAT_COMPLETIONS_BASE_URL` |
| `api_key` | secret | `not-needed` | `CHAT_COMPLETIONS_API_KEY` |
| `default_model` / `model` | text | `default` | — |
| `max_tokens` | int | `4096` | — |
| `temperature` | float | `0.7` | — |
| `timeout` | float | `300.0` | — |
| `max_retries` | int | `3` | — |
| `min_retry_delay` | float | `1.0` | — |
| `max_retry_delay` | float | `30.0` | — |
| `use_streaming` | bool | `true` | — |
| `top_p` | float | `None` | — |
| `stop` | list[str] | `None` | — |
| `seed` | int | `None` | — |
| `parallel_tool_calls` | bool | `true` | — |
| `priority` | int | `100` | — |
| `filtered` | bool | `true` | — |
| `raw` | bool | `false` | — |

## Context Window Configuration

By default the provider reports `context_window = 0` for every model, which tells
the kernel "unknown" — context-managers skip token budgeting entirely and the full
conversation is sent verbatim. Against a server with a finite runtime context
(llama.cpp, LM Studio, vLLM, …) this will eventually cause overflow errors.

Set `context_window` to the **per-slot runtime context** of your server:

| Config key | Env var | Default | Description |
|---|---|---|---|
| `context_window` | `CHAT_COMPLETIONS_CONTEXT_WINDOW` | `0` | Per-request token budget, or `0` to disable budgeting |

> **The llama.cpp trap:** the correct value is `--ctx-size / --parallel` (tokens
> per slot), *not* the model's training context window (e.g. 128 K). Example: if
> you run `llama-server --ctx-size 16384 --parallel 4`, set `context_window: 4096`.

```yaml
# settings.yaml
providers:
  - module: provider-chat-completions
    context_window: 4096    # --ctx-size 16384 / --parallel 4
    max_tokens: 1024
```

Once set, the kernel's context-manager modules (`context-simple`,
`context-persistent`, `context-managed`) will use this value to compute a token
budget and truncate or compact the conversation before it reaches the server.

> **Note:** Kernel-side budgeting only engages when a context-manager module is
> mounted alongside this provider. If you use `loop-basic` without a
> context-manager, the value is recorded but has no effect on truncation.

For the architecture details and the full tracing of how the budget value flows
through the kernel, see
[`docs/designs/001-pr2-context-window-config.md`](../docs/designs/001-pr2-context-window-config.md)
(if the `docs/` directory is present in your checkout; otherwise refer to the
inline explanation above).

## Server Auto-Detection

When `context_window` is not set in config, the provider probes the server
automatically on the first `list_models()` call to discover its runtime context
limit.  Probing is sequential, first-hit-wins, and cached for the session.

### Probe matrix

| Server | Action | Endpoint | Field |
|---|---|---|---|
| **llama.cpp** | Probe | `GET /props` | `default_generation_settings.n_ctx` |
| **LM Studio** (new API) | Free read | `GET /v1/models` | `loaded_context_length` extension |
| **LM Studio** (v0 API) | Probe | `GET /api/v0/models/{id}` | `loaded_context_length` |
| **TGI** (HuggingFace) | Probe | `GET /info` | `max_input_length` |
| **vLLM** | Free read | `GET /v1/models` | `max_model_len` extension |
| **Ollama** | Advisory only | `GET /api/tags` | *(no context reading — see below)* |
| **LocalAI**, unknown | Fall through | — | Config or 0 |

**Free reads** use the standard OpenAI `/v1/models` response that `list_models()`
already fetches — zero extra HTTP calls.  **Probes** issue one additional
request to a server-specific endpoint; they are only fired if the free read
produced nothing.

### Resolution order

```
config["context_window"]            # user override — always wins
  ↓ (if 0 or absent)
/v1/models vendor extensions        # max_model_len, loaded_context_length — free
  ↓ (if absent)
server probe                        # llama.cpp /props, LM Studio /api/v0, TGI /info
  ↓ (if all fail)
0                                   # honest "unknown"
```

### The Ollama advisory

If `auto_probe_context` detects an Ollama server (via `GET /api/tags`), it logs
an advisory and **does not** read the context window from Ollama:

```
chat-completions: this base_url appears to be an Ollama server. Consider using
provider-ollama instead — it handles num_ctx correctly and exposes additional
Ollama features.
```

**Why not probe Ollama?** Ollama's model-info endpoints expose the model's
*training* context size (e.g. 131,072 tokens for Llama-3.2), not the server's
runtime `num_ctx` limit. When Ollama runs with `--parallel 4`, each slot gets
`num_ctx / 4` tokens — but the model info still reports 131,072. Using training
context as a kernel budget silently over-budgets by 16× and produces overflows
that are harder to diagnose than reporting 0.

`provider-ollama` handles this correctly via the `num_ctx` parameter. Users
pointing `chat-completions` at Ollama should use `provider-ollama` instead.

### Observability

One INFO log line per resolved model on first `list_models()`:

```
chat-completions: context_window=32768 for gemma-4-26B-... (source=probe:llama.cpp)
chat-completions: context_window=16384 for gpt-oss-mini   (source=vendor-ext:max_model_len)
chat-completions: context_window=8192  for mistral-small  (source=config)
chat-completions: context_window=0     for unknown-model  (source=unknown; kernel budgeting disabled)
```

One WARNING when configured value disagrees with probed value (drift detection):

```
chat-completions: configured context_window=16384 but /props reports 32768. Using
configured value. If the server was reconfigured, update provider config or set
context_window=0 to trust server auto-detection.
```

### Escape hatches

Two power-user fields are available in `settings.yaml` (not shown by the setup
wizard) for environments where probing is undesirable:

| Key | Default | Description |
|---|---|---|
| `auto_probe_context` | `true` | Set `false` to disable all probing (air-gapped setups, strict firewall rules) |
| `auto_probe_timeout_seconds` | `2.0` | Per-probe timeout in seconds |

```yaml
providers:
  - module: provider-chat-completions
    base_url: http://localhost:8080/v1
    auto_probe_context: false     # Disable all probing
    context_window: 32768         # Set manually instead
```

### First-turn edge case

Probing is triggered by `list_models()`, which the wizard calls during setup.
If you bypass the wizard and fire a request immediately (no prior `list_models()`
call), `get_info()` returns `context_window=0` for the first turn. The PR 1
error message handles resulting overflows; subsequent turns benefit from the
probed value once it is cached.

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