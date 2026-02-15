from typing import Any, AsyncGenerator, List, Tuple

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.providers.base import Provider, ProviderError
from selectools.tools import Tool
from selectools.types import AgentResult, Message, Role, StreamChunk
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "ok"


_DUMMY_TOOL = Tool(
    name="noop",
    description="Does nothing.",
    parameters=[],
    function=_noop,
)


_DUMMY_USAGE = UsageStats(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    cost_usd=0.0,
    model="mock",
    provider="mock",
)


class MockStreamingProvider(Provider):
    """Mock provider with astream support."""

    name = "mock-stream"
    supports_streaming = True
    supports_async = True

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.default_model = "mock-stream"

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content="".join(self.chunks)), _DUMMY_USAGE

    def stream(self, **kwargs: Any):  # type: ignore[override]
        yield from self.chunks

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content="".join(self.chunks)), _DUMMY_USAGE

    async def astream(self, **kwargs: Any) -> AsyncGenerator[str, None]:
        for chunk in self.chunks:
            yield chunk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_astream_yields_chunks():
    chunks = ["Hello", " ", "world", "!"]
    provider = MockStreamingProvider(chunks)
    agent = Agent(
        tools=[_DUMMY_TOOL],
        provider=provider,
        config=AgentConfig(max_iterations=1),
    )

    streamed_content: List[str] = []
    final_result = None

    async for item in agent.astream([Message(role=Role.USER, content="Hi")]):
        if isinstance(item, StreamChunk):
            streamed_content.append(item.content)
        elif isinstance(item, AgentResult):
            final_result = item

    assert streamed_content == chunks
    assert final_result is not None
    assert final_result.content == "Hello world!"


@pytest.mark.asyncio
async def test_agent_astream_fallback_without_astream_method():
    """Verify fallback if provider doesn't implement astream."""

    class MockNoStreamProvider:
        """Provider without astream — only acomplete."""

        name = "mock-no-stream"
        supports_streaming = False
        supports_async = True

        def __init__(self) -> None:
            self.default_model = "mock-no-stream"

        def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            return Message(role=Role.ASSISTANT, content="Fallback response"), _DUMMY_USAGE

        def stream(self, **kwargs: Any):  # type: ignore[override]
            yield "Fallback response"

        async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            return Message(role=Role.ASSISTANT, content="Fallback response"), _DUMMY_USAGE

    provider = MockNoStreamProvider()
    # Remove astream so hasattr check returns False
    assert not hasattr(provider, "astream")

    agent = Agent(
        tools=[_DUMMY_TOOL],
        provider=provider,
        config=AgentConfig(max_iterations=1),
    )

    streamed_content: List[str] = []
    final_result = None

    async for item in agent.astream([Message(role=Role.USER, content="Hi")]):
        if isinstance(item, StreamChunk):
            streamed_content.append(item.content)
        elif isinstance(item, AgentResult):
            final_result = item

    assert streamed_content == ["Fallback response"]
    assert final_result is not None
    assert final_result.content == "Fallback response"
