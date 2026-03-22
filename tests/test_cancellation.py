"""Tests for agent cancellation (R2)."""

import threading

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.cancellation import CancellationToken
from selectools.exceptions import CancellationError
from selectools.observer import AgentObserver
from selectools.tools.base import Tool
from selectools.trace import StepType

_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")


class TestCancellationToken:
    """Unit tests for the CancellationToken class."""

    def test_initial_state(self):
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled

    def test_reset(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled
        token.reset()
        assert not token.is_cancelled

    def test_raise_if_cancelled(self):
        token = CancellationToken()
        token.raise_if_cancelled()  # should not raise
        token.cancel()
        with pytest.raises(CancellationError):
            token.raise_if_cancelled()

    def test_thread_safety(self):
        """Token can be cancelled from another thread."""
        token = CancellationToken()
        result = []

        def cancel_after_delay():
            token.cancel()
            result.append("cancelled")

        t = threading.Thread(target=cancel_after_delay)
        t.start()
        t.join(timeout=5)
        assert token.is_cancelled
        assert result == ["cancelled"]


class TestAgentCancellation:
    """R2: Agent stops when cancellation token fires."""

    def test_cancel_before_first_iteration(self, fake_provider):
        """Pre-cancelled token stops agent immediately."""
        provider = fake_provider(responses=["should not run"])
        token = CancellationToken()
        token.cancel()
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6, cancellation_token=token),
        )
        result = agent.run("test")
        assert "cancelled" in result.content.lower()
        assert result.iterations <= 1

    def test_cancel_trace_step(self, fake_provider):
        """Trace contains CANCELLED step."""
        provider = fake_provider(responses=["should not run"])
        token = CancellationToken()
        token.cancel()
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6, cancellation_token=token),
        )
        result = agent.run("test")
        cancelled_steps = [s for s in result.trace.steps if s.type == StepType.CANCELLED]
        assert len(cancelled_steps) == 1

    def test_cancel_observer_event(self, fake_provider):
        """Observer receives on_cancelled event."""
        events = []

        class CancelObserver(AgentObserver):
            def on_cancelled(self, run_id, iteration, reason):
                events.append({"iteration": iteration, "reason": reason})

        provider = fake_provider(responses=["should not run"])
        token = CancellationToken()
        token.cancel()
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                max_iterations=6, cancellation_token=token, observers=[CancelObserver()]
            ),
        )
        agent.run("test")
        assert len(events) == 1
        assert "cancelled" in events[0]["reason"].lower()

    def test_no_token_runs_normally(self, fake_provider):
        """cancellation_token=None runs agent normally."""
        provider = fake_provider(responses=["answer"])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6),
        )
        result = agent.run("test")
        assert result.iterations == 1
        assert "cancelled" not in result.content.lower()

    @pytest.mark.asyncio
    async def test_cancel_arun(self, fake_provider):
        """Cancellation works with async arun()."""
        provider = fake_provider(responses=["should not run"])
        token = CancellationToken()
        token.cancel()
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6, cancellation_token=token),
        )
        result = await agent.arun("test")
        assert "cancelled" in result.content.lower()
