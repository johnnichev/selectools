"""BUG-01: Streaming run()/arun() silently drops ToolCall objects.

Source: Agno #6757 pattern -- competitor bug where tool function names
become empty strings in streaming responses.

Selectools variant: _streaming_call and _astreaming_call filter chunks
with `isinstance(chunk, str)` which drops ToolCall objects entirely.
Tools are never executed when AgentConfig(stream=True).
"""

from __future__ import annotations

from typing import Any, AsyncIterable, Iterable, List, Union

import pytest

from selectools import Agent, AgentConfig, Tool, ToolParameter
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role, ToolCall


class StreamingToolProvider(LocalProvider):
    """Sync provider that yields a ToolCall during streaming."""

    name = "streaming_tool_stub"
    supports_streaming = True
    supports_async = False

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: List[Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> Iterable[Union[str, ToolCall]]:
        self.call_count += 1
        if self.call_count == 1:
            yield "I will call a tool. "
            yield ToolCall(tool_name="echo", parameters={"text": "hello"})
        else:
            yield "Done. Got: hello"


class AsyncStreamingToolProvider(LocalProvider):
    """Async provider that yields a ToolCall during streaming via astream()."""

    name = "async_streaming_tool_stub"
    supports_streaming = True
    supports_async = True

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: List[Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        self.call_count += 1
        if self.call_count == 1:
            yield "I will call a tool. "
            yield ToolCall(tool_name="echo", parameters={"text": "hello"})
        else:
            yield "Done. Got: hello"


def _echo_fn(text: str) -> str:
    return text


def _make_echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo text",
        parameters=[
            ToolParameter(
                name="text",
                param_type=str,
                description="Text to echo",
                required=True,
            )
        ],
        function=_echo_fn,
    )


def test_streaming_preserves_tool_calls() -> None:
    """When stream=True, ToolCall objects from the provider must be executed."""
    provider = StreamingToolProvider()
    agent = Agent(
        tools=[_make_echo_tool()],
        provider=provider,
        config=AgentConfig(stream=True, max_iterations=3),
    )
    result = agent.run([Message(role=Role.USER, content="echo hello")])
    assert "Done" in result.content, f"Expected tool to execute; got: {result.content!r}"
    assert provider.call_count >= 2, "Agent should have looped after tool execution"


@pytest.mark.asyncio
async def test_astreaming_preserves_tool_calls() -> None:
    """When stream=True, ToolCall objects from provider.astream() must be executed."""
    provider = AsyncStreamingToolProvider()
    agent = Agent(
        tools=[_make_echo_tool()],
        provider=provider,
        config=AgentConfig(stream=True, max_iterations=3),
    )
    result = await agent.arun([Message(role=Role.USER, content="echo hello")])
    assert "Done" in result.content, f"Expected tool to execute; got: {result.content!r}"
    assert provider.call_count >= 2, "Agent should have looped after tool execution"
