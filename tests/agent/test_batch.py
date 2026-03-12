"""
Unit tests for Agent.batch() and Agent.abatch(): thread-safe history isolation,
progress callbacks, per-item error handling.

Previously only covered by E2E tests that were always skipped in CI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.providers.base import Provider, ProviderError
from selectools.tools import tool
from selectools.types import AgentResult, Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")


@tool()
def echo(text: str) -> str:
    """Echo text back."""
    return text


class _TrackingProvider(Provider):
    """Records messages passed to each complete() call for isolation checks."""

    name = "tracking"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self.default_model = "tracking"
        self.call_messages: List[List[Message]] = []

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        msgs = kwargs.get("messages", [])
        self.call_messages.append(list(msgs))
        return Message(role=Role.ASSISTANT, content="response"), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        msgs = kwargs.get("messages", [])
        self.call_messages.append(list(msgs))
        return Message(role=Role.ASSISTANT, content="response"), _DUMMY_USAGE


class _FailOnSecondProvider(Provider):
    """Succeeds on first call, fails on second."""

    name = "fail-second"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self.default_model = "test"
        self._count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._count += 1
        if self._count == 2:
            raise ProviderError("Simulated failure")
        return Message(role=Role.ASSISTANT, content=f"ok-{self._count}"), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._count += 1
        if self._count == 2:
            raise ProviderError("Simulated failure")
        return Message(role=Role.ASSISTANT, content=f"ok-{self._count}"), _DUMMY_USAGE


class TestBatchSync:
    def test_basic_batch(self) -> None:
        provider = _TrackingProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        inputs = [
            [Message(role=Role.USER, content="hello")],
            [Message(role=Role.USER, content="world")],
        ]
        results = agent.batch(inputs)

        assert len(results) == 2
        assert all(isinstance(r, AgentResult) for r in results)

    def test_history_isolation(self) -> None:
        """Each batch item must have independent history."""
        provider = _TrackingProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        inputs = [
            [Message(role=Role.USER, content="item-1")],
            [Message(role=Role.USER, content="item-2")],
            [Message(role=Role.USER, content="item-3")],
        ]
        agent.batch(inputs)

        for call_msgs in provider.call_messages:
            user_msgs = [m for m in call_msgs if m.role == Role.USER]
            assert len(user_msgs) == 1, (
                f"Expected 1 user message per call, got {len(user_msgs)} "
                f"- history leaked between batch items"
            )

    def test_progress_callback(self) -> None:
        provider = _TrackingProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        progress_events: List[Tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_events.append((done, total))

        inputs = [[Message(role=Role.USER, content=f"item-{i}")] for i in range(3)]
        agent.batch(inputs, on_progress=on_progress)

        assert len(progress_events) == 3
        totals = [t for _, t in progress_events]
        assert all(t == 3 for t in totals)
        dones = [d for d, _ in progress_events]
        assert sorted(dones) == [1, 2, 3]


class TestBatchAsync:
    @pytest.mark.asyncio
    async def test_basic_abatch(self) -> None:
        provider = _TrackingProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        inputs = [
            [Message(role=Role.USER, content="a")],
            [Message(role=Role.USER, content="b")],
        ]
        results = await agent.abatch(inputs)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_async_history_isolation(self) -> None:
        provider = _TrackingProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        inputs = [[Message(role=Role.USER, content=f"item-{i}")] for i in range(5)]
        await agent.abatch(inputs)

        for call_msgs in provider.call_messages:
            user_msgs = [m for m in call_msgs if m.role == Role.USER]
            assert len(user_msgs) == 1, "History leaked in async batch"


class TestBatchErrorHandling:
    def test_partial_failure_returns_all_results(self) -> None:
        provider = _FailOnSecondProvider()
        agent = Agent(
            tools=[echo],
            provider=provider,
            config=AgentConfig(max_iterations=1, max_retries=0),
        )
        inputs = [
            [Message(role=Role.USER, content="item-1")],
            [Message(role=Role.USER, content="item-2")],
            [Message(role=Role.USER, content="item-3")],
        ]
        results = agent.batch(inputs)
        assert len(results) == 3

        contents = [r.content for r in results]
        error_results = [c for c in contents if "error" in c.lower() or "Error" in c]
        success_results = [c for c in contents if "ok" in c.lower()]
        assert len(success_results) >= 1
