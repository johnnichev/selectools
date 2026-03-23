#!/usr/bin/env python3
"""
Model Switching — use different models per iteration.

Demonstrates:
- model_selector callback on AgentConfig
- Cheap model for tool selection, expensive for synthesis
- on_model_switch observer event

Prerequisites:
    pip install selectools
    export OPENAI_API_KEY=your-key
"""

from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role, SimpleStepObserver
from selectools.tools import tool
from selectools.types import ToolCall
from selectools.usage import AgentUsage, UsageStats

# ---------------------------------------------------------------------------
# Mock provider that tracks which model was requested
# ---------------------------------------------------------------------------


class ModelTrackingProvider:
    """Provider that records which model is used each call."""

    name = "mock"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self._call_count = 0
        self.model_log: List[str] = []

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
        self.model_log.append(model)

        if self._call_count == 1 and tools:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="Looking up the data.",
                    tool_calls=[
                        ToolCall(
                            tool_name="lookup_data",
                            parameters={"query": "quarterly revenue"},
                            id="call_1",
                        )
                    ],
                ),
                UsageStats(100, 30, 130, 0.001, "mock", model),
            )

        if self._call_count == 2 and tools:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="Running analysis.",
                    tool_calls=[
                        ToolCall(
                            tool_name="analyze",
                            parameters={"data": "revenue_q1_q4"},
                            id="call_2",
                        )
                    ],
                ),
                UsageStats(100, 30, 130, 0.001, "mock", model),
            )

        return (
            Message(
                role=Role.ASSISTANT,
                content="Revenue grew 15% YoY. Q3 was strongest at $4.2M.",
            ),
            UsageStats(200, 100, 300, 0.005, "mock", model),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Look up data from the database")
def lookup_data(query: str) -> str:
    return '{"q1": "$3.2M", "q2": "$3.5M", "q3": "$4.2M", "q4": "$3.8M"}'


@tool(description="Analyze data and produce insights")
def analyze(data: str) -> str:
    return "YoY growth: 15%. Q3 peak at $4.2M. Q2 strongest sequential growth."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Model Switching Demo")
    print("=" * 70)

    # --- Demo 1: Cheap model for tools, expensive for synthesis ---
    print("\n--- Demo 1: model_selector — cheap for tools, expensive for synthesis ---\n")

    CHEAP_MODEL = "gpt-4o-mini"
    EXPENSIVE_MODEL = "gpt-4o"

    def model_selector(iteration: int, messages: List[Any], usage: AgentUsage) -> str:
        """Use cheap model for tool-calling iterations, expensive for final."""
        if iteration <= 2:
            return CHEAP_MODEL
        return EXPENSIVE_MODEL

    provider = ModelTrackingProvider()

    # Track model switches via observer
    switches: List[Dict[str, str]] = []

    def on_event(event: str, run_id: str, **data: Any) -> None:
        if event == "model_switch":
            switches.append({"old": data["old_model"], "new": data["new_model"]})
            print(f"    [switch] {data['old_model']} -> {data['new_model']}")

    agent = Agent(
        tools=[lookup_data, analyze],
        provider=provider,
        config=AgentConfig(
            model=CHEAP_MODEL,
            max_iterations=5,
            model_selector=model_selector,
            observers=[SimpleStepObserver(on_event)],
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Analyze our quarterly revenue trends.")])

    print(f"\n  Answer:      {result.content[:60]}")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Models used: {provider.model_log}")
    print(f"  Switches:    {len(switches)}")

    # --- Demo 2: Cost-aware switching ---
    print("\n--- Demo 2: Switch to cheap model when cost exceeds threshold ---\n")

    COST_THRESHOLD = 0.003

    def cost_aware_selector(iteration: int, messages: List[Any], usage: AgentUsage) -> str:
        """Switch to cheap model if cost is getting high."""
        if usage.total_cost_usd > COST_THRESHOLD:
            return CHEAP_MODEL
        return EXPENSIVE_MODEL

    provider2 = ModelTrackingProvider()

    agent2 = Agent(
        tools=[lookup_data, analyze],
        provider=provider2,
        config=AgentConfig(
            model=EXPENSIVE_MODEL,
            max_iterations=5,
            model_selector=cost_aware_selector,
        ),
    )

    result2 = agent2.run([Message(role=Role.USER, content="Analyze revenue and create a summary.")])

    print(f"  Answer:      {result2.content[:60]}")
    print(f"  Models used: {provider2.model_log}")
    if result2.usage:
        print(f"  Total cost:  ${result2.usage.total_cost_usd:.4f}")

    print("\n" + "=" * 70)
    print("  Key takeaways:")
    print("    - model_selector(iteration, messages, usage) -> model_name")
    print("    - Use cheap models for tool selection, expensive for synthesis")
    print("    - Switch dynamically based on cost, iteration count, or context")
    print("    - on_model_switch observer event tracks every switch")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
