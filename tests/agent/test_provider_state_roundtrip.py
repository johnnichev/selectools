"""provider_state survives the tool loop on every execution path.

A stateless Responses-style API returns reasoning items between the assistant
message and its function calls, and needs them replayed verbatim on the next
request. The provider puts them in ``provider_state``; this proves the agent
hands the same payload back on the follow-up call after running the tool.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.providers.base import Provider
from selectools.tools import tool
from selectools.types import Message, Role, StreamEnd, ToolCall
from selectools.usage import UsageStats

_USAGE = UsageStats(1, 1, 2, 0.0, "mock", "mock")
_STATE = {
    "items": [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
    ]
}


@tool()
def lookup() -> str:
    """Return a fixed value."""
    return "42"


class _ResponsesLikeProvider(Provider):
    """Turn 1: tool call with reasoning state. Turn 2: plain answer."""

    name = "responses-like"
    supports_streaming = True
    supports_async = True

    def __init__(self) -> None:
        self.seen_histories: List[List[Message]] = []

    def _turn(self, messages: List[Message]) -> Message:
        self.seen_histories.append(list(messages))
        if len(self.seen_histories) == 1:
            return Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(tool_name="lookup", parameters={}, id="c1")],
                provider_state=_STATE,
            )
        return Message(role=Role.ASSISTANT, content="The answer is 42")

    def complete(self, *, messages: List[Message], **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self._turn(messages), _USAGE

    async def acomplete(
        self, *, messages: List[Message], **kwargs: Any
    ) -> Tuple[Message, UsageStats]:
        return self._turn(messages), _USAGE

    def stream(self, *, messages: List[Message], **kwargs: Any):
        msg = self._turn(messages)
        if msg.content:
            yield msg.content
        for call in msg.tool_calls or []:
            yield call
        yield StreamEnd(usage=_USAGE, provider_state=msg.provider_state)

    async def astream(
        self, *, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Union[str, ToolCall, StreamEnd], None]:
        for item in self.stream(messages=messages, **kwargs):
            yield item


def _assert_state_replayed(provider: _ResponsesLikeProvider) -> None:
    assert len(provider.seen_histories) == 2
    second = provider.seen_histories[1]
    assistant = [m for m in second if m.role == Role.ASSISTANT]
    assert assistant[-1].provider_state == _STATE
    assert assistant[-1].tool_calls is not None
    assert assistant[-1].tool_calls[0].id == "c1"
    tool_msgs = [m for m in second if m.role == Role.TOOL]
    assert tool_msgs and tool_msgs[-1].tool_call_id == "c1"


def _agent(provider: Provider, stream: bool) -> Agent:
    return Agent([lookup], provider=provider, config=AgentConfig(model="m", stream=stream))


def test_run_replays_provider_state_after_tool_execution() -> None:
    provider = _ResponsesLikeProvider()

    result = _agent(provider, stream=False).run("what is the answer?")

    assert result.message.content == "The answer is 42"
    _assert_state_replayed(provider)


def test_run_with_streaming_replays_provider_state_after_tool_execution() -> None:
    provider = _ResponsesLikeProvider()

    result = _agent(provider, stream=True).run("what is the answer?")

    assert result.message.content == "The answer is 42"
    _assert_state_replayed(provider)


@pytest.mark.asyncio
async def test_arun_replays_provider_state_after_tool_execution() -> None:
    provider = _ResponsesLikeProvider()

    result = await _agent(provider, stream=False).arun("what is the answer?")

    assert result.message.content == "The answer is 42"
    _assert_state_replayed(provider)


@pytest.mark.asyncio
async def test_astream_replays_provider_state_after_tool_execution() -> None:
    provider = _ResponsesLikeProvider()

    final: Dict[str, Any] = {}
    async for item in _agent(provider, stream=True).astream("what is the answer?"):
        if hasattr(item, "message"):
            final["result"] = item

    assert final["result"].message.content == "The answer is 42"
    _assert_state_replayed(provider)


def test_provider_state_survives_a_history_round_trip_through_dicts() -> None:
    provider = _ResponsesLikeProvider()
    agent = _agent(provider, stream=False)
    agent.run("what is the answer?")

    restored = [Message.from_dict(m.to_dict()) for m in agent._history]

    assistant = [m for m in restored if m.role == Role.ASSISTANT and m.tool_calls]
    assert assistant[0].provider_state == _STATE
