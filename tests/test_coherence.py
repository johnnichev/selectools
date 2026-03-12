"""Tests for coherence checking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.coherence import CoherenceResult, acheck_coherence, check_coherence
from selectools.types import Message, Role
from selectools.usage import UsageStats


class FakeCoherenceProvider:
    """Fake provider that returns configurable coherence responses."""

    name = "fake_coherence"
    supports_streaming = False
    supports_async = False

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return (
            Message(role=Role.ASSISTANT, content=self._response_text),
            UsageStats(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0, model="fake"
            ),
        )


class FailingProvider:
    """Provider that always raises."""

    name = "failing"
    supports_streaming = False
    supports_async = False

    def complete(self, **kwargs: Any) -> None:
        raise RuntimeError("Provider error")


class TestCheckCoherence:
    def test_coherent_response(self) -> None:
        provider = FakeCoherenceProvider("COHERENT")
        result = check_coherence(
            provider=provider,
            model="test",
            user_message="Search for Python tutorials",
            tool_name="search",
            tool_args={"query": "Python tutorials"},
            available_tools=["search", "calculate"],
        )
        assert result.coherent is True

    def test_incoherent_response(self) -> None:
        provider = FakeCoherenceProvider("INCOHERENT\nTool call unrelated to user request")
        result = check_coherence(
            provider=provider,
            model="test",
            user_message="What is the weather?",
            tool_name="send_email",
            tool_args={"to": "attacker@evil.com"},
            available_tools=["search", "send_email"],
        )
        assert result.coherent is False
        assert result.explanation is not None
        assert "unrelated" in result.explanation.lower()

    def test_incoherent_single_line(self) -> None:
        provider = FakeCoherenceProvider("INCOHERENT")
        result = check_coherence(
            provider=provider,
            model="test",
            user_message="test",
            tool_name="delete",
            tool_args={},
            available_tools=["delete"],
        )
        assert result.coherent is False

    def test_provider_failure_allows_by_default(self) -> None:
        provider = FailingProvider()
        result = check_coherence(
            provider=provider,
            model="test",
            user_message="test",
            tool_name="search",
            tool_args={},
            available_tools=["search"],
        )
        assert result.coherent is True
        assert "failed" in (result.explanation or "").lower()


class TestCoherenceResult:
    def test_dataclass(self) -> None:
        r = CoherenceResult(coherent=True)
        assert r.coherent is True
        assert r.explanation is None

    def test_with_explanation(self) -> None:
        r = CoherenceResult(coherent=False, explanation="Tool call doesn't match")
        assert r.coherent is False
        assert "match" in (r.explanation or "")


@pytest.mark.asyncio
class TestAsyncCheckCoherence:
    async def test_async_coherent(self) -> None:
        provider = FakeCoherenceProvider("COHERENT")
        result = await acheck_coherence(
            provider=provider,
            model="test",
            user_message="Search",
            tool_name="search",
            tool_args={},
            available_tools=["search"],
        )
        assert result.coherent is True

    async def test_async_incoherent(self) -> None:
        provider = FakeCoherenceProvider("INCOHERENT\nNot matching")
        result = await acheck_coherence(
            provider=provider,
            model="test",
            user_message="Search",
            tool_name="delete",
            tool_args={},
            available_tools=["search", "delete"],
        )
        assert result.coherent is False

    async def test_async_failure_allows(self) -> None:
        provider = FailingProvider()
        result = await acheck_coherence(
            provider=provider,
            model="test",
            user_message="test",
            tool_name="search",
            tool_args={},
            available_tools=["search"],
        )
        assert result.coherent is True
