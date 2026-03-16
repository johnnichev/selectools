"""Tests for AsyncAgentObserver (FR-002).

Verifies that async observer hooks fire correctly in arun() and astream(),
with both blocking and non-blocking modes, and that errors are swallowed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from selectools.agent import Agent, AgentConfig
from selectools.observer import AgentObserver, AsyncAgentObserver
from selectools.tools import Tool, ToolParameter
from selectools.types import AgentResult, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TrackingAsyncObserver(AsyncAgentObserver):
    """Async observer that records calls for assertions."""

    def __init__(self, *, blocking: bool = False) -> None:
        self.blocking = blocking  # type: ignore[assignment]
        self.events: List[tuple] = []

    async def a_on_run_start(
        self, run_id: str, messages: List[Message], system_prompt: str
    ) -> None:
        self.events.append(("a_on_run_start", run_id))

    async def a_on_run_end(self, run_id: str, result: AgentResult) -> None:
        self.events.append(("a_on_run_end", run_id))

    async def a_on_llm_start(
        self, run_id: str, messages: List[Message], model: str, system_prompt: str
    ) -> None:
        self.events.append(("a_on_llm_start", run_id))

    async def a_on_llm_end(self, run_id: str, response: str, usage: Any) -> None:
        self.events.append(("a_on_llm_end", run_id))

    async def a_on_usage(self, run_id: str, usage: Any) -> None:
        self.events.append(("a_on_usage", run_id))

    async def a_on_iteration_start(
        self, run_id: str, iteration: int, messages: List[Message]
    ) -> None:
        self.events.append(("a_on_iteration_start", run_id, iteration))

    async def a_on_iteration_end(self, run_id: str, iteration: int, response: str) -> None:
        self.events.append(("a_on_iteration_end", run_id, iteration))

    async def a_on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self.events.append(("a_on_tool_start", run_id, tool_name, tool_args))

    async def a_on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        self.events.append(("a_on_tool_end", run_id, tool_name, result))

    async def a_on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self.events.append(("a_on_error", run_id, str(error)))


class TrackingSyncObserver(AgentObserver):
    """Sync observer that records calls alongside async observers."""

    def __init__(self) -> None:
        self.events: List[str] = []

    def on_run_start(self, run_id: str, messages: Any, system_prompt: str) -> None:
        self.events.append("on_run_start")

    def on_run_end(self, run_id: str, result: Any) -> None:
        self.events.append("on_run_end")

    def on_iteration_start(self, run_id: str, iteration: int, messages: Any) -> None:
        self.events.append("on_iteration_start")

    def on_iteration_end(self, run_id: str, iteration: int, response: str) -> None:
        self.events.append("on_iteration_end")

    def on_llm_start(self, run_id: str, messages: Any, model: str, system_prompt: str) -> None:
        self.events.append("on_llm_start")

    def on_llm_end(self, run_id: str, response: str, usage: Any) -> None:
        self.events.append("on_llm_end")


class ErrorAsyncObserver(AsyncAgentObserver):
    """Async observer that raises on every call."""

    blocking: bool = True

    async def a_on_run_start(
        self, run_id: str, messages: List[Message], system_prompt: str
    ) -> None:
        raise RuntimeError("observer boom")

    async def a_on_iteration_start(
        self, run_id: str, iteration: int, messages: List[Message]
    ) -> None:
        raise RuntimeError("observer boom")

    async def a_on_llm_start(
        self, run_id: str, messages: Any, model: str, system_prompt: str
    ) -> None:
        raise RuntimeError("observer boom")

    async def a_on_llm_end(self, run_id: str, response: str, usage: Any) -> None:
        raise RuntimeError("observer boom")

    async def a_on_run_end(self, run_id: str, result: Any) -> None:
        raise RuntimeError("observer boom")


def _noop_tool() -> Tool:
    return Tool(
        name="noop",
        description="does nothing",
        parameters=[],
        function=lambda **kw: "ok",
    )


def _make_agent(
    observers: List[Any],
    responses: Optional[list] = None,
    tools: Optional[List[Tool]] = None,
) -> Agent:
    """Create a simple agent with given observers and fake provider."""
    from tests.conftest import SharedFakeProvider

    provider = SharedFakeProvider(responses=responses or ["Hello!"])
    return Agent(
        tools=tools or [_noop_tool()],
        provider=provider,
        config=AgentConfig(
            model="fake-model",
            observers=observers,
        ),
    )


def _make_tool(tool_name: str = "greet", result: str = "hi") -> Tool:
    def _fn(name: str = "default") -> str:
        return result

    return Tool(
        name=tool_name,
        description="greet someone by name",
        parameters=[
            ToolParameter(name="name", param_type=str, description="name to greet", required=False),
        ],
        function=_fn,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocking_async_observer_awaited_in_arun() -> None:
    """blocking=True observer should have its events awaited inline during arun()."""
    obs = TrackingAsyncObserver(blocking=True)
    agent = _make_agent(observers=[obs])

    await agent.arun("hi")

    event_names = [e[0] for e in obs.events]
    assert "a_on_run_start" in event_names
    assert "a_on_run_end" in event_names
    assert "a_on_iteration_start" in event_names
    assert "a_on_iteration_end" in event_names
    assert "a_on_llm_start" in event_names
    assert "a_on_llm_end" in event_names


@pytest.mark.asyncio
async def test_nonblocking_async_observer_fires_in_arun() -> None:
    """blocking=False observer should have its events dispatched via ensure_future."""
    obs = TrackingAsyncObserver(blocking=False)
    agent = _make_agent(observers=[obs])

    await agent.arun("hi")
    # Give non-blocking tasks a chance to complete
    await asyncio.sleep(0.05)

    event_names = [e[0] for e in obs.events]
    assert "a_on_run_start" in event_names
    assert "a_on_run_end" in event_names
    assert "a_on_llm_start" in event_names
    assert "a_on_llm_end" in event_names


@pytest.mark.asyncio
async def test_sync_observers_work_alongside_async() -> None:
    """Sync and async observers should both receive their respective events."""
    sync_obs = TrackingSyncObserver()
    async_obs = TrackingAsyncObserver(blocking=True)
    agent = _make_agent(observers=[sync_obs, async_obs])

    await agent.arun("hi")

    # Sync events fired
    assert "on_run_start" in sync_obs.events
    assert "on_run_end" in sync_obs.events
    assert "on_iteration_start" in sync_obs.events

    # Async events also fired
    async_event_names = [e[0] for e in async_obs.events]
    assert "a_on_run_start" in async_event_names
    assert "a_on_run_end" in async_event_names
    assert "a_on_iteration_start" in async_event_names


@pytest.mark.asyncio
async def test_multiple_async_observers_mixed_blocking() -> None:
    """Multiple async observers with different blocking settings all fire."""
    blocking_obs = TrackingAsyncObserver(blocking=True)
    nonblocking_obs = TrackingAsyncObserver(blocking=False)
    agent = _make_agent(observers=[blocking_obs, nonblocking_obs])

    await agent.arun("hi")
    await asyncio.sleep(0.05)

    for obs in [blocking_obs, nonblocking_obs]:
        event_names = [e[0] for e in obs.events]
        assert "a_on_run_start" in event_names
        assert "a_on_run_end" in event_names


@pytest.mark.asyncio
async def test_tool_start_end_called_with_correct_args() -> None:
    """a_on_tool_start and a_on_tool_end should receive correct tool info."""
    from selectools.types import ToolCall

    tool = _make_tool(tool_name="greet", result="hello there")

    tool_call_msg = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name="greet", parameters={"name": "world"})],
    )
    final_msg = "Done"

    obs = TrackingAsyncObserver(blocking=True)
    agent = _make_agent(
        observers=[obs],
        responses=[tool_call_msg, final_msg],
        tools=[tool],
    )

    await agent.arun("say hello")

    tool_starts = [e for e in obs.events if e[0] == "a_on_tool_start"]
    tool_ends = [e for e in obs.events if e[0] == "a_on_tool_end"]

    assert len(tool_starts) >= 1
    assert tool_starts[0][2] == "greet"
    assert tool_starts[0][3] == {"name": "world"}

    assert len(tool_ends) >= 1
    assert tool_ends[0][2] == "greet"
    assert tool_ends[0][3] == "hello there"


@pytest.mark.asyncio
async def test_error_in_async_observer_does_not_crash_agent() -> None:
    """An async observer that raises should not prevent the agent from completing."""
    error_obs = ErrorAsyncObserver()
    tracking_obs = TrackingAsyncObserver(blocking=True)
    agent = _make_agent(observers=[error_obs, tracking_obs])

    result = await agent.arun("hi")

    assert result.content == "Hello!"
    # The tracking observer should still have received events despite the
    # error observer crashing on every call.
    event_names = [e[0] for e in tracking_obs.events]
    assert "a_on_run_start" in event_names
    assert "a_on_run_end" in event_names


@pytest.mark.asyncio
async def test_default_async_methods_are_noop() -> None:
    """All a_on_* methods on a bare AsyncAgentObserver should be callable without error."""
    obs = AsyncAgentObserver()

    # Call every async method with placeholder args
    await obs.a_on_run_start("rid", [], "prompt")
    await obs.a_on_run_end("rid", MagicMock())
    await obs.a_on_llm_start("rid", [], "model", "prompt")
    await obs.a_on_llm_end("rid", "resp", None)
    await obs.a_on_cache_hit("rid", "model", "resp")
    await obs.a_on_usage("rid", MagicMock())
    await obs.a_on_tool_start("rid", "cid", "tool", {})
    await obs.a_on_tool_end("rid", "cid", "tool", "result", 1.0)
    await obs.a_on_tool_error("rid", "cid", "tool", Exception(), {}, 1.0)
    await obs.a_on_tool_chunk("rid", "cid", "tool", "chunk")
    await obs.a_on_policy_decision("rid", "tool", "allow", "ok", {})
    await obs.a_on_structured_validate("rid", True, 1)
    await obs.a_on_iteration_start("rid", 1, [])
    await obs.a_on_iteration_end("rid", 1, "resp")
    await obs.a_on_batch_start("bid", 1)
    await obs.a_on_batch_end("bid", 1, 0, 100.0)
    await obs.a_on_provider_fallback("rid", "p1", "p2", Exception())
    await obs.a_on_llm_retry("rid", 1, 3, Exception(), 1.0)
    await obs.a_on_memory_trim("rid", 5, 10, "enforce_limits")
    await obs.a_on_session_load("rid", "sid", 5)
    await obs.a_on_session_save("rid", "sid", 5)
    await obs.a_on_memory_summarize("rid", "summary")
    await obs.a_on_entity_extraction("rid", 3)
    await obs.a_on_kg_extraction("rid", 2)
    await obs.a_on_error("rid", Exception("test"), {})

    # If we got here without raising, all methods are no-ops


@pytest.mark.asyncio
async def test_async_observer_in_astream() -> None:
    """Async observers should fire in astream() just like in arun()."""
    obs = TrackingAsyncObserver(blocking=True)
    agent = _make_agent(observers=[obs])

    chunks = []
    async for item in agent.astream("hi"):
        if isinstance(item, AgentResult):
            break
        chunks.append(item)

    event_names = [e[0] for e in obs.events]
    assert "a_on_run_start" in event_names
    assert "a_on_run_end" in event_names
    assert "a_on_iteration_start" in event_names
    assert "a_on_iteration_end" in event_names


@pytest.mark.asyncio
async def test_async_observer_is_subclass_of_agent_observer() -> None:
    """AsyncAgentObserver should be a subclass of AgentObserver."""
    assert issubclass(AsyncAgentObserver, AgentObserver)
    obs = AsyncAgentObserver()
    assert isinstance(obs, AgentObserver)


@pytest.mark.asyncio
async def test_blocking_default_is_false() -> None:
    """The default blocking value should be False."""
    obs = AsyncAgentObserver()
    assert obs.blocking is False


def test_async_observer_exported_from_package() -> None:
    """AsyncAgentObserver should be importable from the top-level package."""
    import selectools

    assert hasattr(selectools, "AsyncAgentObserver")
    assert selectools.AsyncAgentObserver is AsyncAgentObserver
