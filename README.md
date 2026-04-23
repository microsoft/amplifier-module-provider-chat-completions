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