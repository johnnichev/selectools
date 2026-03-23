#!/usr/bin/env python3
"""
Token Budget Per Run — stop agents before they burn money.

Demonstrates:
- max_total_tokens: hard limit on cumulative tokens
- max_cost_usd: hard limit on cumulative cost
- Budget exceeded result includes partial content

Prerequisites:
    pip install selectools
    export OPENAI_API_KEY=your-key
"""

from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider that always calls a tool (burns tokens every iteration)
# ---------------------------------------------------------------------------


class TokenBurnerProvider:
    """Provider that keeps calling a tool to demonstrate budget limits."""

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

        if self._call_count <= 10 and tools:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content=f"Running step {self._call_count}...",
                    tool_calls=[
                        ToolCall(
                            tool_name="expensive_step",
                            parameters={"step": self._call_count},
                            id=f"call_{self._call_count}",
                        )
                    ],
                ),
                UsageStats(500, 200, 700, 0.005, "mock", "gpt-4o"),
            )

        return (
            Message(role=Role.ASSISTANT, content="All steps complete."),
            UsageStats(300, 100, 400, 0.003, "mock", "gpt-4o"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(description="Run an expensive computation step")
def expensive_step(step: int) -> str:
    return f"Step {step} completed — processed 10,000 records."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Token Budget Demo")
    print("=" * 70)

    # --- Demo 1: max_total_tokens ---
    print("\n--- Demo 1: max_total_tokens = 2000 ---\n")

    agent_tokens = Agent(
        tools=[expensive_step],
        provider=TokenBurnerProvider(),
        config=AgentConfig(
            max_iterations=10,
            max_total_tokens=2000,
        ),
    )

    result = agent_tokens.run([Message(role=Role.USER, content="Run all 10 steps.")])

    print(f"  Content:    {result.content[:70]}...")
    print(f"  Iterations: {result.iterations}")
    if result.usage:
        print(f"  Tokens:     {result.usage.total_tokens}")
        print(f"  Cost:       ${result.usage.total_cost_usd:.4f}")
    print(f"  Stopped by budget, not max_iterations!")

    # --- Demo 2: max_cost_usd ---
    print("\n--- Demo 2: max_cost_usd = 0.01 ---\n")

    agent_cost = Agent(
        tools=[expensive_step],
        provider=TokenBurnerProvider(),
        config=AgentConfig(
            max_iterations=10,
            max_cost_usd=0.01,
        ),
    )

    result = agent_cost.run([Message(role=Role.USER, content="Run all 10 steps.")])

    print(f"  Content:    {result.content[:70]}...")
    print(f"  Iterations: {result.iterations}")
    if result.usage:
        print(f"  Tokens:     {result.usage.total_tokens}")
        print(f"  Cost:       ${result.usage.total_cost_usd:.4f}")
    print(f"  Stopped before overspending!")

    # --- Demo 3: both limits ---
    print("\n--- Demo 3: Combined (tokens=5000, cost=$0.02) ---\n")

    agent_both = Agent(
        tools=[expensive_step],
        provider=TokenBurnerProvider(),
        config=AgentConfig(
            max_iterations=10,
            max_total_tokens=5000,
            max_cost_usd=0.02,
        ),
    )

    result = agent_both.run([Message(role=Role.USER, content="Run all 10 steps.")])

    print(f"  Content:    {result.content[:70]}...")
    print(f"  Iterations: {result.iterations}")
    if result.usage:
        print(f"  Tokens:     {result.usage.total_tokens}")
        print(f"  Cost:       ${result.usage.total_cost_usd:.4f}")

    print("\n" + "=" * 70)
    print("  Budget limits keep your agent spend predictable.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
