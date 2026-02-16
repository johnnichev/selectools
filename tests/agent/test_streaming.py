from typing import AsyncGenerator, List, Tuple, Union

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.providers.base import Provider
from selectools.tools import tool
from selectools.types import AgentResult, Message, Role, StreamChunk, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")


@tool()
def mock_tool():
    """A mock tool."""
    return "ok"


class MockStreamingProvider(Provider):
    """Mock provider with astream support."""

    name = "mock-stream"
    supports_streaming = True
    supports_async = True

    def __init__(self, chunks: List[Union[str, ToolCall]]):
        self.chunks = chunks
        self.default_model = "mock-stream"

    def complete(self, **kwargs) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content="Done"), _DUMMY_USAGE

    def stream(self, **kwargs):
        for chunk in self.chunks:
            if isinstance(chunk, str):
                yield chunk

    async def acomplete(self, **kwargs) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content="Done"), _DUMMY_USAGE

    async def astream(self, **kwargs) -> AsyncGenerator[Union[str, ToolCall], None]:
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_agent_astream_yields_chunks():
    chunks = ["Hello", " ", "world", "!"]
    provider = MockStreamingProvider(chunks)
    agent = Agent(tools=[mock_tool], provider=provider, config=AgentConfig(max_iterations=1))

    streamed_content = []
    final_result = None

    async for item in agent.astream([Message(role=Role.USER, content="Hi")]):
        if isinstance(item, StreamChunk):
            streamed_content.append(item.content)
        elif isinstance(item, AgentResult):
            final_result = item

    assert "".join(streamed_content) == "Hello world!"
    assert final_result is not None
    assert final_result.content == "Hello world!"


@pytest.mark.asyncio
async def test_agent_astream_fallback_without_astream_method():
    """Verify fallback if provider doesn't implement astream."""

    class MockNoStreamProvider(Provider):
        name = "mock-no-stream"
        supports_streaming = False
        supports_async = True

        def __init__(self):
            self.default_model = "mock-no-stream"

        def complete(self, **kwargs) -> Tuple[Message, UsageStats]:
            return Message(role=Role.ASSISTANT, content="Fallback response"), _DUMMY_USAGE

        def stream(self, **kwargs):
            yield "Fallback response"

        async def acomplete(self, **kwargs) -> Tuple[Message, UsageStats]:
            return Message(role=Role.ASSISTANT, content="Fallback response"), _DUMMY_USAGE

    provider = MockNoStreamProvider()
    agent = Agent(tools=[mock_tool], provider=provider, config=AgentConfig(max_iterations=1))

    streamed_content = []
    final_result = None

    async for item in agent.astream([Message(role=Role.USER, content="Hi")]):
        if isinstance(item, StreamChunk):
            streamed_content.append(item.content)
        elif isinstance(item, AgentResult):
            final_result = item

    assert streamed_content == ["Fallback response"]
    assert final_result.content == "Fallback response"


@pytest.mark.asyncio
async def test_agent_astream_yields_tool_calls():
    """Verify that astream yields ToolCall objects and handles them."""
    chunks = [
        "Thinking...",
        ToolCall(tool_name="get_weather", parameters={"location": "San Francisco"}, id="call_123"),
        "The weather is nice.",
    ]

    provider = MockStreamingProvider(chunks)

    @tool()
    def get_weather(location: str):
        """Get weather"""
        return f"Weather in {location} is 72F"

    agent = Agent(tools=[get_weather], provider=provider, config=AgentConfig(max_iterations=2))

    items = []
    async for item in agent.astream([Message(role=Role.USER, content="Weather?")]):
        items.append(item)

    # Check that we got the text and the tool call
    texts = [i.content for i in items if isinstance(i, StreamChunk) and i.content]
    tool_calls = [i.tool_calls for i in items if isinstance(i, StreamChunk) and i.tool_calls]
    results = [i for i in items if isinstance(i, AgentResult)]

    assert "Thinking..." in texts
    assert any(tc[0].tool_name == "get_weather" for tc in tool_calls)
    assert len(results) > 0
