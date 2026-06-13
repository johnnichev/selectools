"""
Tests for Gemini cache-aware cost calculation (v1.1 follow-up to #106).

Gemini reports context-caching hits in
``usage_metadata.cached_content_token_count``; the count is INCLUDED in
``prompt_token_count`` and bills at the model's published context-caching
rate (registry ``cached_prompt_cost``). Cache STORAGE ($/1M-token-hour) is
time-based and intentionally NOT part of cost_usd.

All tests use mocked genai responses — no API key required.

Rates verified 2026-06-12 against https://ai.google.dev/gemini-api/docs/pricing
(gemini-2.5-flash: $0.30 input / $0.03 context caching / $2.50 output).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from selectools.types import Message, Role

pytest.importorskip("google.genai")

from selectools.providers.gemini_provider import GeminiProvider  # noqa: E402

MODEL = "gemini-2.5-flash"
INPUT_RATE = 0.30
CACHED_RATE = 0.03
OUTPUT_RATE = 2.50


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.default_model = MODEL
    provider._client = MagicMock()
    return provider


def _make_response(
    prompt_tokens: int = 1000,
    completion_tokens: int = 10,
    cached_tokens: Any = None,
    with_usage: bool = True,
) -> MagicMock:
    response = MagicMock()
    response.text = "hello"
    response.candidates = []
    if with_usage:
        usage = MagicMock()
        usage.prompt_token_count = prompt_tokens
        usage.candidates_token_count = completion_tokens
        usage.cached_content_token_count = cached_tokens
        response.usage_metadata = usage
    else:
        response.usage_metadata = None
    return response


def _messages() -> list:
    return [Message(role=Role.USER, content="hi")]


def _expected_cost(prompt: int, completion: int, cached: int) -> float:
    return (
        ((prompt - cached) / 1_000_000) * INPUT_RATE
        + (cached / 1_000_000) * CACHED_RATE
        + (completion / 1_000_000) * OUTPUT_RATE
    )


class TestCompleteCachedTokens:
    def test_cached_tokens_discount_cost(self) -> None:
        provider = _make_provider()
        provider._client.models.generate_content.return_value = _make_response(
            prompt_tokens=1000, completion_tokens=10, cached_tokens=600
        )
        _, stats = provider.complete(model=MODEL, system_prompt="sys", messages=_messages())
        assert stats.cost_usd == pytest.approx(_expected_cost(1000, 10, 600))
        assert stats.cache_read_input_tokens == 600
        assert stats.prompt_tokens == 1000

    def test_zero_cached_tokens_matches_legacy_cost(self) -> None:
        from selectools.pricing import calculate_cost

        provider = _make_provider()
        provider._client.models.generate_content.return_value = _make_response(
            prompt_tokens=1000, completion_tokens=10, cached_tokens=0
        )
        _, stats = provider.complete(model=MODEL, system_prompt="sys", messages=_messages())
        assert stats.cost_usd == pytest.approx(calculate_cost(MODEL, 1000, 10))
        assert stats.cache_read_input_tokens == 0

    def test_unreported_cache_count_is_none_and_full_price(self) -> None:
        """cached_content_token_count=None (no caching used) -> None, not 0."""
        from selectools.pricing import calculate_cost

        provider = _make_provider()
        provider._client.models.generate_content.return_value = _make_response(
            prompt_tokens=1000, completion_tokens=10, cached_tokens=None
        )
        _, stats = provider.complete(model=MODEL, system_prompt="sys", messages=_messages())
        assert stats.cache_read_input_tokens is None
        assert stats.cost_usd == pytest.approx(calculate_cost(MODEL, 1000, 10))

    def test_missing_usage_metadata_still_zero_cost_tokens(self) -> None:
        provider = _make_provider()
        provider._client.models.generate_content.return_value = _make_response(with_usage=False)
        _, stats = provider.complete(model=MODEL, system_prompt="sys", messages=_messages())
        assert stats.prompt_tokens == 0
        assert stats.cache_read_input_tokens is None
        assert stats.cost_usd == 0.0


class TestAcompleteCachedTokens:
    def test_acomplete_cached_tokens_discount_cost(self) -> None:
        provider = _make_provider()
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=_make_response(
                prompt_tokens=2000, completion_tokens=50, cached_tokens=1500
            )
        )
        _, stats = asyncio.run(
            provider.acomplete(model=MODEL, system_prompt="sys", messages=_messages())
        )
        assert stats.cost_usd == pytest.approx(_expected_cost(2000, 50, 1500))
        assert stats.cache_read_input_tokens == 1500
