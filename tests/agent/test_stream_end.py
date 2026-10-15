"""StreamEnd: providers hand usage and provider_state back at the end of a native stream.

Before this contract, ``on_llm_end`` received ``None`` usage on every native
stream and the agent's totals skipped streamed turns entirely.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Iterator, List, Optional, Tuple, Union

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.observer import AgentObserver
from selectools.providers.base import Provider
from selectools.tools import tool
from selectools.trace import StepType
from selectools.types import Message, Role, StreamChunk, StreamEnd, ToolCall
from selectools.usage import UsageStats

_USAGE = UsageStats(prompt_tokens=11, completion_tokens=7, cost_usd=0.5, model="m", provider="p")
_STATE = {"items": [{"type": "reasoning", "encrypted_content": "abc"}]}


@tool()
def _dummy() -> str:
    """No-op."""
    return "ok"


class _EndProvider(Provider):
    """Streams text then a StreamEnd; ``end=None`` reproduces the legacy stream."""

    name = "end"
    supports_streaming = True
    supports_async = True

    def __init__(self, end: Optional[StreamEnd]):
        self._end = end

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise AssertionError("streaming agents must not call complete()")

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise AssertionError("streaming agents must not call acomplete()")

    def stream(self, **kwargs: Any) -> Iterator[Union[str, ToolCall, StreamEnd]]:
        yield "Hel"
        yield "lo"
        if self._end is not None:
            yield self._end

    async def astream(self, **kwargs: Any) -> AsyncGenerator[Union[str, ToolCall, StreamEnd], None]:
        yield "Hel"
        yield "lo"
        if self._end is not None:
            yield self._end


class _Recorder(AgentObserver):
    def __init__(self) -> None:
        self.llm_end_usage: List[Optional[UsageStats]] = []
        self.usage_events: List[UsageStats] = []

    def on_llm_end(self, run_id: str, content: Any, usage: Optional[UsageStats]) -> None:
        self.llm_end_usage.append(usage)

    def on_usage(self, run_id: str, usage: UsageStats) -> None:
        self.usage_events.append(usage)


def _agent(provider: Provider, recorder: _Recorder) -> Agent:
    return Agent(
        [_dummy],
        provider=provider,
        config=AgentConfig(model="m", stream=True, observers=[recorder]),
    )


def _llm_steps(result: Any) -> list:
    assert result.trace is not None
    return [s for s in result.trace.steps if s.type == StepType.LLM_CALL]


def test_stream_end_is_a_plain_container() -> None:
    end = StreamEnd()
    assert end.usage is None
    assert end.provider_state is None


def test_run_counts_streamed_usage_and_reports_it_to_observers() -> None:
    recorder = _Recorder()
    agent = _agent(_EndProvider(StreamEnd(usage=_USAGE)), recorder)

    result = agent.run("hi")

    assert result.message.content == "Hello"
    assert result.usage is not None and result.usage.total_tokens == 18
    assert recorder.llm_end_usage == [_USAGE]
    assert recorder.usage_events == [_USAGE]
    assert _llm_steps(result)[0].prompt_tokens == 11


@pytest.mark.asyncio
async def test_arun_counts_streamed_usage_and_reports_it_to_observers() -> None:
    recorder = _Recorder()
    agent = _agent(_EndProvider(StreamEnd(usage=_USAGE)), recorder)

    result = await agent.arun("hi")

    assert result.message.content == "Hello"
    assert result.usage is not None and result.usage.total_tokens == 18
    assert recorder.llm_end_usage == [_USAGE]
    assert recorder.usage_events == [_USAGE]
    assert _llm_steps(result)[0].prompt_tokens == 11


@pytest.mark.asyncio
async def test_astream_counts_streamed_usage_and_never_leaks_the_end_marker() -> None:
    recorder = _Recorder()
    agent = _agent(_EndProvider(StreamEnd(usage=_USAGE)), recorder)

    items: List[Any] = []
    async for item in agent.astream("hi"):
        items.append(item)

    assert not any(isinstance(i, StreamEnd) for i in items)
    assert "".join(i.content for i in items if isinstance(i, StreamChunk)) == "Hello"
    assert recorder.llm_end_usage == [_USAGE]
    assert recorder.usage_events == [_USAGE]
    assert agent.usage.total_tokens == 18


def test_run_attaches_provider_state_to_the_assistant_message() -> None:
    agent = _agent(_EndProvider(StreamEnd(provider_state=_STATE)), _Recorder())

    result = agent.run("hi")

    assert result.message.provider_state == _STATE
    assistant = [m for m in agent._history if m.role == Role.ASSISTANT]
    assert assistant[-1].provider_state == _STATE


@pytest.mark.asyncio
async def test_astream_attaches_provider_state_to_the_assistant_message() -> None:
    agent = _agent(_EndProvider(StreamEnd(provider_state=_STATE)), _Recorder())

    async for _ in agent.astream("hi"):
        pass

    assistant = [m for m in agent._history if m.role == Role.ASSISTANT]
    assert assistant[-1].provider_state == _STATE


def test_legacy_stream_without_end_marker_still_reports_none_usage() -> None:
    recorder = _Recorder()
    agent = _agent(_EndProvider(None), recorder)

    result = agent.run("hi")

    assert result.message.content == "Hello"
    assert recorder.llm_end_usage == [None]
    assert recorder.usage_events == []
    assert result.message.provider_state is None
