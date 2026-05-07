"""OpenAI Chat Completions pricing rates and cost computation.

Verification date: 2026-05-06
Source: https://openai.com/api/pricing

This provider speaks the OpenAI Chat Completions wire format.  Any server
that implements that format (llama.cpp, vLLM, SGLang, LM Studio, etc.) can be
targeted.  Cost computation is only possible for models in the table below;
all other models (local, custom) return None.

Unknown models return None — DO NOT default to $0.00.

Usage
-----
    from amplifier_module_provider_chat_completions._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "gpt-4o",
        prompt_tokens=1_000,
        completion_tokens=200,
        cached_tokens=100,
    )
    # Returns Decimal or None if the model is not recognised.

Notes
-----
- cached_tokens subtraction happens INSIDE compute_cost to prevent call-site
  double-charging.  Callers pass the raw API fields directly.
- No cache write cost for OpenAI Chat Completions (unlike Anthropic).
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

# _RATES maps model-id → {
#   "input_per_m":      Decimal,  # fresh input tokens, per 1M
#   "output_per_m":     Decimal,  # output/completion tokens, per 1M
#   "cache_read_per_m": Decimal,  # cached input tokens, per 1M (0.00 = no discount)
# }
#
# Rates are in USD.
# Unknown models → return None (DO NOT default to $0.00).
_RATES: dict[str, dict[str, Decimal]] = {
    # ------------------------------------------------------------------
    # GPT-4o  ($2.50 / $10.00, cache_read $1.25)
    # ------------------------------------------------------------------
    "gpt-4o": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("1.25"),
    },
    "gpt-4o-2024-11-20": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("1.25"),
    },
    "gpt-4o-2024-08-06": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("1.25"),
    },
    # ------------------------------------------------------------------
    # GPT-4o mini  ($0.15 / $0.60, cache_read $0.075)
    # ------------------------------------------------------------------
    "gpt-4o-mini": {
        "input_per_m": Decimal("0.15"),
        "output_per_m": Decimal("0.60"),
        "cache_read_per_m": Decimal("0.075"),
    },
    "gpt-4o-mini-2024-07-18": {
        "input_per_m": Decimal("0.15"),
        "output_per_m": Decimal("0.60"),
        "cache_read_per_m": Decimal("0.075"),
    },
    # ------------------------------------------------------------------
    # o1  ($15.00 / $60.00, cache_read $7.50)
    # ------------------------------------------------------------------
    "o1": {
        "input_per_m": Decimal("15.00"),
        "output_per_m": Decimal("60.00"),
        "cache_read_per_m": Decimal("7.50"),
    },
    "o1-2024-12-17": {
        "input_per_m": Decimal("15.00"),
        "output_per_m": Decimal("60.00"),
        "cache_read_per_m": Decimal("7.50"),
    },
    # ------------------------------------------------------------------
    # o1-mini  ($3.00 / $12.00, cache_read $1.50)
    # ------------------------------------------------------------------
    "o1-mini": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("12.00"),
        "cache_read_per_m": Decimal("1.50"),
    },
    "o1-mini-2024-09-12": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("12.00"),
        "cache_read_per_m": Decimal("1.50"),
    },
    # ------------------------------------------------------------------
    # o3-mini  ($1.10 / $4.40, cache_read $0.55)
    # ------------------------------------------------------------------
    "o3-mini": {
        "input_per_m": Decimal("1.10"),
        "output_per_m": Decimal("4.40"),
        "cache_read_per_m": Decimal("0.55"),
    },
    "o3-mini-2025-01-31": {
        "input_per_m": Decimal("1.10"),
        "output_per_m": Decimal("4.40"),
        "cache_read_per_m": Decimal("0.55"),
    },
    # ------------------------------------------------------------------
    # GPT 5.4  ($2.50 / $15.00, cache_read $0.25)
    # ------------------------------------------------------------------
    "gpt-5.4": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # GPT 5.5  ($5.00 / $30.00, cache_read $0.50)
    # ------------------------------------------------------------------
    "gpt-5.5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("30.00"),
        "cache_read_per_m": Decimal("0.50"),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> Decimal | None:
    """Compute the cost of an OpenAI Chat Completions API call in USD.

    Args:
        model: The model ID (e.g. 'gpt-4o').
        prompt_tokens: Total prompt tokens (TOTAL, includes cached).
            This is response.usage.prompt_tokens.
        completion_tokens: Completion tokens used.
        cached_tokens: Number of prompt tokens served from cache.
            This is response.usage.prompt_tokens_details.cached_tokens.

    Returns:
        Decimal cost in USD, or None if the model is not in the pricing table.

    Note:
        cached_tokens subtraction happens inside this function to prevent
        call-site double-charging.  Callers pass the raw API fields directly.
    """
    rates = _RATES.get(model)
    if rates is None:
        return None
    # Subtract cached from total INSIDE the function to prevent call-site double-charging.
    # Clamp to 0: if caller passes only cached_tokens without matching prompt_tokens,
    # fresh_input should not go negative.
    fresh_input = max(0, prompt_tokens - cached_tokens)
    cost = Decimal(fresh_input) * rates["input_per_m"] / _PER_M
    cost += Decimal(completion_tokens) * rates["output_per_m"] / _PER_M
    if cached_tokens:
        cost += Decimal(cached_tokens) * rates["cache_read_per_m"] / _PER_M
    return cost
