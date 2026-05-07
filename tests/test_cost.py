"""Tests for _cost.py: compute_cost() for provider-chat-completions.

Covers:
  (a) Known model: correct Decimal cost for prompt tokens
  (b) Output tokens cost
  (c) REQUIRED: Cached request does NOT double-charge
  (d) Unknown model returns None
  (e) None != Decimal('0')
  (f) Result type is always Decimal, never float
  (g) Cache-only (fresh_input=0)
  (h) zero cached_tokens → no change
"""

from decimal import Decimal

from amplifier_module_provider_chat_completions._cost import compute_cost


# ---------------------------------------------------------------------------
# (a) Known model: gpt-4o prompt cost
# ---------------------------------------------------------------------------
def test_known_model_input_cost():
    """gpt-4o: 1M prompt tokens (no cache) → $2.50."""
    result = compute_cost("gpt-4o", prompt_tokens=1_000_000)
    assert result == Decimal("2.50"), f"Expected Decimal('2.50'), got {result!r}"


# ---------------------------------------------------------------------------
# (b) Output tokens cost
# ---------------------------------------------------------------------------
def test_known_model_output_cost():
    """gpt-4o: 1M completion tokens → $10.00."""
    result = compute_cost("gpt-4o", completion_tokens=1_000_000)
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (c) REQUIRED: Cached request does NOT double-charge
# ---------------------------------------------------------------------------
def test_cached_request_does_not_double_charge():
    """gpt-4o: 1M prompt_tokens, 1M cached_tokens → $1.25 (cache_read only).

    fresh_input = 1M - 1M = 0
    cost = 0 × $2.50/M + 0 × $10.00/M + 1M × $1.25/M = $1.25
    """
    result = compute_cost("gpt-4o", prompt_tokens=1_000_000, cached_tokens=1_000_000)
    assert result == Decimal("1.25"), (
        f"Expected Decimal('1.25') (cache_read only, no double-charge), got {result!r}"
    )


# ---------------------------------------------------------------------------
# (d) Unknown model returns None
# ---------------------------------------------------------------------------
def test_unknown_model_returns_none():
    """An unrecognised model must return None (not 0, not raise)."""
    result = compute_cost("gpt-unknown-9999", prompt_tokens=1_000_000)
    assert result is None


# ---------------------------------------------------------------------------
# (e) None != Decimal('0')
# ---------------------------------------------------------------------------
def test_unknown_distinct_from_zero():
    """None returned for unknown model must not equal Decimal('0')."""
    result = compute_cost("gpt-unknown-9999", prompt_tokens=0)
    assert result is None
    assert result != Decimal("0")


# ---------------------------------------------------------------------------
# (f) Result type is Decimal, not float
# ---------------------------------------------------------------------------
def test_result_type_is_decimal():
    """compute_cost must return a Decimal, not a float."""
    result = compute_cost("gpt-4o", prompt_tokens=1_000)
    assert isinstance(result, Decimal)
    assert not isinstance(result, float)


# ---------------------------------------------------------------------------
# (g) Cache-only: prompt_tokens == cached_tokens → fresh_input = 0
# ---------------------------------------------------------------------------
def test_cache_only_no_fresh_input():
    """When prompt_tokens == cached_tokens, fresh cost is 0, only cache_read cost."""
    result = compute_cost("gpt-4o-mini", prompt_tokens=500_000, cached_tokens=500_000)
    expected = Decimal("500000") * Decimal("0.075") / Decimal("1000000")
    assert result == expected, f"Expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# (h) Zero cached_tokens → same as no cache arg
# ---------------------------------------------------------------------------
def test_zero_cached_tokens_no_discount():
    """cached_tokens=0 → same cost as no cache argument."""
    result_no_cache = compute_cost("gpt-4o", prompt_tokens=1_000_000)
    result_zero_cache = compute_cost("gpt-4o", prompt_tokens=1_000_000, cached_tokens=0)
    assert result_no_cache == result_zero_cache
