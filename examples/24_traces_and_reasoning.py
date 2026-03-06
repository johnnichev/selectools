#!/usr/bin/env python3
"""
Execution Traces & Reasoning Visibility — See exactly what the agent did and why.

Demonstrates:
  1. result.trace with TraceStep timeline
  2. Filtering trace by step type
  3. result.reasoning and reasoning_history
  4. trace.timeline() human-readable output
  5. trace.to_dict() / to_json() for export

No API key needed — uses a mock provider.

Prerequisites: pip install selectools
Run: python examples/24_traces_and_reasoning.py
"""

import json
import tempfile
from typing import Any, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.trace import AgentTrace, TraceStep
from selectools.types import AgentResult, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider that simulates tool calling with reasoning text
# ---------------------------------------------------------------------------


class ReasoningProvider:
    """Provider that returns reasoning text alongside tool calls."""

    name = "reasoning-mock"
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

        if self._call_count == 1:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="The customer is asking about their bill, so I need to look up their account first.",
                    tool_calls=[
                        ToolCall(
                            tool_name="lookup_account",
                            parameters={"customer_id": "cust-123"},
                            id="call_1",
                        )
                    ],
                ),
                UsageStats(200, 50, 250, 0.002, "mock", "gpt-4o-mini"),
            )

        if self._call_count == 2:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="Now that I have the account details, I can see the billing issue. Let me check the invoice.",
                    tool_calls=[
                        ToolCall(
                            tool_name="get_invoice",
                            parameters={"account_id": "acc-456", "month": "january"},
                            id="call_2",
                        )
                    ],
                ),
                UsageStats(300, 80, 380, 0.003, "mock", "gpt-4o-mini"),
            )

        return (
            Message(
                role=Role.ASSISTANT,
                content="Based on the account lookup and invoice review, I can see you were charged $49.99 on January 15th for your monthly subscription.",
            ),
            UsageStats(400, 100, 500, 0.004, "mock", "gpt-4o-mini"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Look up customer account details")
def lookup_account(customer_id: str) -> str:
    return f'{{"account_id": "acc-456", "name": "Alice", "plan": "premium", "status": "active"}}'


@tool(description="Get invoice details for a billing period")
def get_invoice(account_id: str, month: str) -> str:
    return (
        f'{{"invoice_id": "inv-789", "amount": "$49.99", "date": "2026-01-15", "status": "paid"}}'
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  Execution Traces & Reasoning Visibility Demo")
    print("=" * 70)

    # --- Step 1: Run agent and inspect trace ---
    print("\n--- Step 1: Run agent and inspect the execution trace ---\n")

    provider = ReasoningProvider()
    agent = Agent(
        tools=[lookup_account, get_invoice],
        provider=provider,
        config=AgentConfig(max_iterations=5),
    )

    result = agent.run([Message(role=Role.USER, content="Why was I charged $49.99?")])

    assert result.trace is not None
    print(f"  Trace has {len(result.trace)} steps\n")

    # --- Step 2: Print the timeline ---
    print("--- Step 2: Human-readable timeline ---\n")
    print(result.trace.timeline())
    print()

    # --- Step 3: Inspect individual steps ---
    print("--- Step 3: Inspect individual trace steps ---\n")

    for i, step in enumerate(result.trace):
        print(f"  Step {i + 1}:")
        print(f"    type         = {step.type}")
        print(f"    duration_ms  = {step.duration_ms:.1f}")
        if step.tool_name:
            print(f"    tool_name    = {step.tool_name}")
        if step.tool_args:
            print(f"    tool_args    = {step.tool_args}")
        if step.model:
            print(f"    model        = {step.model}")
        if step.prompt_tokens:
            print(
                f"    tokens       = {step.prompt_tokens} prompt + {step.completion_tokens} completion"
            )
        if step.reasoning:
            print(f"    reasoning    = {step.reasoning[:80]}...")
        if step.summary:
            print(f"    summary      = {step.summary[:80]}")
        print()

    # --- Step 4: Filter by step type ---
    print("--- Step 4: Filter trace by step type ---\n")

    llm_steps = result.trace.filter(type="llm_call")
    tool_steps = result.trace.filter(type="tool_execution")
    selection_steps = result.trace.filter(type="tool_selection")

    print(f"  LLM calls:       {len(llm_steps)}")
    print(f"  Tool selections: {len(selection_steps)}")
    print(f"  Tool executions: {len(tool_steps)}")

    total_llm_ms = result.trace.llm_duration_ms
    total_tool_ms = result.trace.tool_duration_ms
    print(f"\n  LLM time:  {total_llm_ms:.1f}ms")
    print(f"  Tool time: {total_tool_ms:.1f}ms")
    print(f"  Total:     {result.trace.total_duration_ms:.1f}ms\n")

    # --- Step 5: Reasoning visibility ---
    print("--- Step 5: Reasoning visibility ---\n")

    print(f"  result.reasoning = {result.reasoning}")
    print(f"  result.reasoning_history ({len(result.reasoning_history)} entries):")
    for i, r in enumerate(result.reasoning_history):
        if r:
            print(f"    [{i}] {r[:80]}...")
    print()

    # --- Step 6: Export trace ---
    print("--- Step 6: Export trace to dict / JSON ---\n")

    trace_dict = result.trace.to_dict()
    print(f"  trace.to_dict() keys: {list(trace_dict.keys())}")
    print(f"  step_count: {trace_dict['step_count']}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result.trace.to_json(f.name)
        print(f"  trace.to_json() wrote to: {f.name}")

        with open(f.name) as rf:
            exported = json.load(rf)
            print(f"  Exported {len(exported['steps'])} steps")

    print()

    # --- Assertions ---
    assert len(result.trace) >= 5
    assert len(llm_steps) == 3
    assert len(tool_steps) == 2
    assert result.reasoning_history is not None
    assert any(r for r in result.reasoning_history if r)

    print("=" * 70)
    print("  All trace & reasoning tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
