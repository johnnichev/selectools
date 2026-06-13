"""
Tests for OpenAI cache-aware cost calculation (v1.1 follow-up to #106).

OpenAI reports prompt-cache hits in ``usage.prompt_tokens_details.cached_tokens``;
the count is INCLUDED in ``usage.prompt_tokens`` and bills at the model's
published cached-input rate (registry ``cached_prompt_cost``). Covers the
shared ``_parse_response`` path (used by complete and acomplete) plus full
sync/async wiring with mocked SDK clients — never calls real APIs.

Rates verified 2026-06-12 against https://developers.openai.com/api/docs/pricing
(gpt-5.5: $5.00 input / $0.50 cached input / $30.00 output).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from selectools.types import Message, Role

MODEL = "gpt-5.5"
INPUT_RATE = 5.00
CACHED_RATE = 0.50
OUTPUT_RATE = 30.00


def _get_provider() -> Any:
    from selectools.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.default_model = MODEL
    provider.api_key = "test"
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    return provider


def _make_response(
    prompt_tokens: int = 1000,
    completion_tokens: int = 10,
    cached_tokens: Any = None,
    with_details: bool = True,
) -> MagicMock:
    message = MagicMock()
    message.content = "hello"
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    if with_details:
        details = MagicMock()
        details.cached_tokens = cached_tokens
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _messages() -> list:
    return [Message(role=Role.USER, content="hi")]


def _expected_cost(prompt: int, completion: int, cached: int) -> float:
    return (
        ((prompt - cached) / 1_000_000) * INPUT_RATE
        + (cached / 1_000_000) * CACHED_RATE
        + (completion / 1_000_000) * OUTPUT_RATE
    )


class TestParseResponseCachedTokens:
    def test_cached_tokens_discount_cost(self) -> None:
        provider = _get_provider()
        response = _make_response(prompt_tokens=1000, completion_tokens=10, cached_tokens=600)
        _, stats = provider._parse_response(response, MODEL)
        assert stats.cost_usd == pytest.approx(_expected_cost(1000, 10, 600))
        assert stats.cache_read_input_tokens == 600
        assert stats.prompt_tokens == 1000

    def test_zero_cached_tokens_matches_legacy_cost(self) -> None:
        from selectools.pricing import calculate_cost

        provider = _get_provider()
        response = _make_response(prompt_tokens=1000, completion_tokens=10, cached_tokens=0)
        _, stats = provider._parse_response(response, MODEL)
        assert stats.cost_usd == pytest.approx(calculate_cost(MODEL, 1000, 10))
        assert stats.cache_read_input_tokens == 0

    def test_missing_details_reports_none_and_full_price(self) -> None:
        from selectools.pricing import calculate_cost

        provider = _get_provider()
        response = _make_response(prompt_tokens=1000, completion_tokens=10, with_details=False)
        _, stats = provider._parse_response(response, MODEL)
        assert stats.cache_read_input_tokens is None
        assert stats.cost_usd == pytest.approx(calculate_cost(MODEL, 1000, 10))

    def test_non_int_cached_tokens_treated_as_unreported(self) -> None:
        """A bare MagicMock (or any non-int) must not poison the math."""
        provider = _get_provider()
        response = _make_response(
            prompt_tokens=1000, completion_tokens=10, cached_tokens=MagicMock()
        )
        _, stats = provider._parse_response(response, MODEL)
        assert stats.cache_read_input_tokens is None
        assert stats.cost_usd == pytest.approx(_expected_cost(1000, 10, 0))

    def test_model_without_cached_rate_bills_full_input_price(self) -> None:
        """gpt-4o has no published cached rate today: fallback = full price."""
        from selectools.pricing import calculate_cost

        provider = _get_provider()
        response = _make_response(prompt_tokens=1000, completion_tokens=10, cached_tokens=600)
        _, stats = provider._parse_response(response, "gpt-4o")
        assert stats.cost_usd == pytest.approx(calculate_cost("gpt-4o", 1000, 10))
        assert stats.cache_read_input_tokens == 600


class TestProviderWiring:
    def test_complete_passes_cached_tokens_to_cost(self) -> None:
        provider = _get_provider()
        provider._client.chat.completions.create.return_value = _make_response(
            prompt_tokens=2000, completion_tokens=50, cached_tokens=1500
        )
        _, stats = provider.complete(model=MODEL, system_prompt="sys", messages=_messages())
        assert stats.cost_usd == pytest.approx(_expected_cost(2000, 50, 1500))
        assert stats.cache_read_input_tokens == 1500

    def test_acomplete_passes_cached_tokens_to_cost(self) -> None:
        provider = _get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            return_value=_make_response(
                prompt_tokens=2000, completion_tokens=50, cached_tokens=1500
            )
        )
        _, stats = asyncio.run(
            provider.acomplete(model=MODEL, system_prompt="sys", messages=_messages())
        )
        assert stats.cost_usd == pytest.approx(_expected_cost(2000, 50, 1500))
        assert stats.cache_read_input_tokens == 1500
