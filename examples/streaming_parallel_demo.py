#!/usr/bin/env python3
"""
Streaming and Parallel Tool Execution Demo.

Demonstrates E2E streaming (Agent.astream), parallel tool execution,
StreamChunk vs AgentResult, async tools with arun(), and native function calling.

Requirements:
    pip install selectools

Run:
    python examples/streaming_parallel_demo.py
"""

import asyncio
import time
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import AgentResult, StreamChunk, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Fake providers for offline demo
# ---------------------------------------------------------------------------


class MultiToolProvider:
    """Returns multiple tool calls in one response (for parallel execution)."""

    name = "multi-tool"
    supports_streaming = False
    supports_async = True

    def __init__(
        self,
        tool_calls: List[ToolCall],
        final_text: str = "All tasks completed.",
    ) -> None:
        self._tool_calls = tool_calls
        self._final_text = final_text
        self._call_count = 0

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        if self._call_count == 1:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=self._tool_calls,
                ),
                UsageStats(0, 0, 0, 0.0, "mock", "mock"),
            )
        return (
            Message(role=Role.ASSISTANT, content=self._final_text),
            UsageStats(0, 0, 0, 0.0, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class MockStreamingProvider:
    """Provider with astream support for token-by-token output."""

    name = "streaming-mock"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        chunks_iter1: List[Union[str, ToolCall]],
        chunks_iter2: Optional[List[str]] = None,
    ) -> None:
        self.chunks_iter1 = chunks_iter1
        self.chunks_iter2 = chunks_iter2 or ["All done!"]
        self._call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        content = "".join(c for c in self.chunks_iter1 + self.chunks_iter2 if isinstance(c, str))
        return (
            Message(role=Role.ASSISTANT, content=content),
            UsageStats(0, 0, 0, 0.0, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)

    async def astream(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[Union[str, ToolCall], None]:
        self._call_count += 1
        if self._call_count == 1:
            for chunk in self.chunks_iter1:
                yield chunk
        else:
            for chunk in self.chunks_iter2:
                yield chunk


# ---------------------------------------------------------------------------
# Tools (with different durations to show parallel vs sequential)
# ---------------------------------------------------------------------------

SLEEP_SEC = 0.2


@tool(description="Fast task, completes in 0.2s")
def fast_task(x: int) -> str:
    """Fast task."""
    time.sleep(SLEEP_SEC)
    return f"fast={x}"


@tool(description="Medium task, completes in 0.4s")
def medium_task(x: int) -> str:
    """Medium task."""
    time.sleep(SLEEP_SEC * 2)
    return f"medium={x}"


@tool(description="Slow task, completes in 0.6s")
def slow_task(x: int) -> str:
    """Slow task."""
    time.sleep(SLEEP_SEC * 3)
    return f"slow={x}"


@tool(description="Async task with arun()")
async def async_task(label: str) -> str:
    """Async tool demonstrating arun()."""
    await asyncio.sleep(0.15)
    return f"async_result={label}"


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------


def demo_sequential_vs_parallel() -> None:
    """Compare sequential vs parallel tool execution timing."""
    print("\n📌 Step 2 & 3: Sequential vs Parallel tool execution")

    tool_calls = [
        ToolCall(tool_name="fast_task", parameters={"x": 1}, id="c1"),
        ToolCall(tool_name="medium_task", parameters={"x": 2}, id="c2"),
        ToolCall(tool_name="slow_task", parameters={"x": 3}, id="c3"),
    ]

    # Sequential
    config_seq = AgentConfig(
        parallel_tool_execution=False,
        max_iterations=2,
    )
    provider_seq = MultiToolProvider(tool_calls)
    agent_seq = Agent(
        tools=[fast_task, medium_task, slow_task],
        provider=provider_seq,
        config=config_seq,
    )

    start = time.time()
    result_seq = agent_seq.run([Message(role=Role.USER, content="Run all tasks")])
    elapsed_seq = time.time() - start

    print(f"\n   Sequential execution: {elapsed_seq:.2f}s")
    print(f"   (3 tools × ~0.2–0.6s each ≈ 1.2s+)\n")

    # Parallel
    config_par = AgentConfig(
        parallel_tool_execution=True,
        max_iterations=2,
    )
    provider_par = MultiToolProvider(tool_calls)
    agent_par = Agent(
        tools=[fast_task, medium_task, slow_task],
        provider=provider_par,
        config=config_par,
    )

    start = time.time()
    result_par = agent_par.run([Message(role=Role.USER, content="Run all tasks")])
    elapsed_par = time.time() - start

    print(f"   Parallel execution: {elapsed_par:.2f}s")
    print(f"   (all 3 run concurrently ≈ slowest tool ~0.6s)")
    print(f"\n   ✅ Speedup: {elapsed_seq / max(elapsed_par, 0.01):.1f}x faster with parallel\n")


async def demo_astream() -> None:
    """Demonstrate Agent.astream() for token-by-token streaming."""
    print("\n📌 Step 4 & 5: Agent.astream() - StreamChunk vs AgentResult")

    chunks_iter1 = [
        "Thinking",
        " ",
        "about",
        " it",
        "...",
        ToolCall(
            tool_name="fast_task",
            parameters={"x": 42},
            id="call_1",
        ),
        " Done!",
    ]
    chunks_iter2 = ["Here", " is", " the", " result", "."]

    provider = MockStreamingProvider(chunks_iter1, chunks_iter2)
    config = AgentConfig(max_iterations=2)
    agent = Agent(tools=[fast_task], provider=provider, config=config)

    stream_chunks: List[str] = []
    final_result: Optional[AgentResult] = None

    print("\n   Streaming output: ", end="", flush=True)
    async for item in agent.astream([Message(role=Role.USER, content="Run fast task with 42")]):
        if isinstance(item, StreamChunk):
            if item.content:
                print(item.content, end="", flush=True)
                stream_chunks.append(item.content)
            if item.tool_calls:
                for tc in item.tool_calls:
                    print(f" [ToolCall:{tc.tool_name}] ", end="", flush=True)
        elif isinstance(item, AgentResult):
            final_result = item

    print()
    print(f"\n   StreamChunks received: {len(stream_chunks)} text chunks")
    if final_result is not None:
        print(
            f"   Final AgentResult: iterations={final_result.iterations}, "
            f"tool_calls={len(final_result.tool_calls)}"
        )
    print("   ✅ astream yields StreamChunk (deltas) then AgentResult (final)\n")


async def demo_async_tools() -> None:
    """Demonstrate async tools with arun()."""
    print("\n📌 Step 6: Async tools with arun()")

    provider = MultiToolProvider(
        [ToolCall(tool_name="async_task", parameters={"label": "demo"}, id="c1")],
        "Async task completed.",
    )
    config = AgentConfig(max_iterations=2)
    agent = Agent(tools=[async_task], provider=provider, config=config)

    result = await agent.arun([Message(role=Role.USER, content="Run async task with label demo")])
    print(f"   result.content: {result.content}")
    print(f"   result.tool_calls: {[tc.tool_name for tc in result.tool_calls]}")
    print("   ✅ Async tools work with agent.arun()\n")


def demo_native_function_calling() -> None:
    """Brief note on native function calling."""
    print("\n📌 Step 7: Native function calling (brief)")
    print(
        """
   When the provider supports native function calling, it returns Message
   objects with tool_calls=[ToolCall(...)] instead of parsed text.
   The agent handles both: native ToolCall objects and regex-parsed
   TOOL_CALL: {...} format from text responses.
   ✅ Our MultiToolProvider uses native ToolCall objects
"""
    )


def main() -> None:
    """Run the streaming and parallel demo."""
    print("\n" + "#" * 70)
    print("# Streaming and Parallel Tool Execution Demo")
    print("#" * 70)

    # --- Step 1: Define 3 tools with different durations ---
    print("\n📌 Step 1: Define 3 tools (fast=0.2s, medium=0.4s, slow=0.6s)")
    print("   - fast_task(x): sleeps 0.2s")
    print("   - medium_task(x): sleeps 0.4s")
    print("   - slow_task(x): sleeps 0.6s")
    print("   ✅ Tools defined\n")

    demo_sequential_vs_parallel()

    # Run async demos
    asyncio.run(demo_astream())
    asyncio.run(demo_async_tools())
    demo_native_function_calling()

    print("#" * 70)
    print("# Demo complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
