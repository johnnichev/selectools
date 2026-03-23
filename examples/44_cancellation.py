#!/usr/bin/env python3
"""
Agent Cancellation — cooperative stopping from any thread.

Demonstrates:
- CancellationToken for cooperative cancellation
- Cancel from a timeout task
- Partial results preserved after cancellation

Prerequisites:
    pip install selectools
    export OPENAI_API_KEY=your-key
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, CancellationToken, Message, Role
from selectools.tools import tool
from selectools.types import ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Slow provider that simulates long-running work
# ---------------------------------------------------------------------------


class SlowProvider:
    """Provider that keeps calling tools, simulating a long-running agent."""

    name = "mock"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
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
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        if self._call_count <= 20 and tools:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content=f"Processing batch {self._call_count}...",
                    tool_calls=[
                        ToolCall(
                            tool_name="slow_process",
                            parameters={"batch_id": self._call_count},
                            id=f"call_{self._call_count}",
                        )
                    ],
                ),
                UsageStats(100, 50, 150, 0.001, "mock", "gpt-4o"),
            )
        return (
            Message(role=Role.ASSISTANT, content="All batches processed."),
            UsageStats(100, 50, 150, 0.001, "mock", "gpt-4o"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(description="Process a data batch (slow)")
def slow_process(batch_id: int) -> str:
    return f"Batch {batch_id} complete — 5000 rows processed."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 70)
    print("  Cancellation Token Demo")
    print("=" * 70)

    # --- Demo 1: Cancel from a timeout task ---
    print("\n--- Demo 1: Cancel after 3 iterations via timeout task ---\n")

    token = CancellationToken()
    iteration_count = 0

    async def cancel_after_delay(cancel_token: CancellationToken, delay: float) -> None:
        """Watchdog that cancels the agent after a delay."""
        nonlocal iteration_count
        while iteration_count < 3:
            await asyncio.sleep(0.01)
        print(f"  [watchdog] Cancelling agent after {iteration_count} iterations")
        cancel_token.cancel()

    class IterationCounter:
        """Observer that counts iterations."""

        def __init__(self) -> None:
            self.count = 0

    counter = IterationCounter()

    agent = Agent(
        tools=[slow_process],
        provider=SlowProvider(),
        config=AgentConfig(
            max_iterations=20,
            cancellation_token=token,
        ),
    )

    # Monkey-patch to count iterations
    original_run = agent.run

    def counting_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal iteration_count
        result = original_run(*args, **kwargs)
        return result

    # Run agent with a cancellation watchdog
    watchdog = asyncio.create_task(cancel_after_delay(token, 0.05))

    # Use sync run in a thread (agent checks token between iterations)
    loop = asyncio.get_event_loop()

    def sync_run() -> Any:
        nonlocal iteration_count
        provider = SlowProvider()

        class CountingProvider:
            name = provider.name
            supports_streaming = provider.supports_streaming
            supports_async = provider.supports_async

            def complete(self, **kwargs: Any) -> Any:
                nonlocal iteration_count
                result = provider.complete(**kwargs)
                iteration_count += 1
                return result

            async def acomplete(self, **kwargs: Any) -> Any:
                nonlocal iteration_count
                result = provider.complete(**kwargs)
                iteration_count += 1
                return result

        counting_agent = Agent(
            tools=[slow_process],
            provider=CountingProvider(),
            config=AgentConfig(
                max_iterations=20,
                cancellation_token=token,
            ),
        )
        return counting_agent.run([Message(role=Role.USER, content="Process all 20 batches.")])

    result = await loop.run_in_executor(None, sync_run)
    await watchdog

    print(f"  Content:    {result.content[:70]}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Cancelled:  {token.is_cancelled}")

    # --- Demo 2: Reuse token after reset ---
    print("\n--- Demo 2: Reset token for reuse ---\n")

    token.reset()
    print(f"  Token reset: is_cancelled = {token.is_cancelled}")

    agent2 = Agent(
        tools=[slow_process],
        provider=SlowProvider(),
        config=AgentConfig(
            max_iterations=2,
            cancellation_token=token,
        ),
    )

    result2 = agent2.run([Message(role=Role.USER, content="Process a small batch.")])
    print(f"  Content:    {result2.content[:70]}")
    print(f"  Completed normally (no cancellation)")

    print("\n" + "=" * 70)
    print("  CancellationToken lets you stop agents cooperatively from any thread.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
