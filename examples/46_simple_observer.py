#!/usr/bin/env python3
"""
SimpleStepObserver — single callback for all agent events.

Demonstrates:
- SimpleStepObserver routes 31 events to one function
- Real-time visibility into agent execution
- Simpler alternative to subclassing AgentObserver

Prerequisites:
    pip install selectools
    export OPENAI_API_KEY=your-key
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role, SimpleStepObserver
from selectools.tools import tool
from selectools.types import ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class DemoProvider:
    """Provider that makes a tool call then responds."""

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
        if self._call_count == 1 and tools:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="Let me check the weather.",
                    tool_calls=[
                        ToolCall(
                            tool_name="get_weather",
                            parameters={"city": "London"},
                            id="call_1",
                        )
                    ],
                ),
                UsageStats(120, 30, 150, 0.001, "mock", "gpt-4o-mini"),
            )
        return (
            Message(role=Role.ASSISTANT, content="It's 55F and cloudy in London."),
            UsageStats(80, 20, 100, 0.0005, "mock", "gpt-4o-mini"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(description="Get the weather for a city")
def get_weather(city: str) -> str:
    return json.dumps({"city": city, "temp": "55F", "condition": "cloudy"})


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  SimpleStepObserver Demo")
    print("=" * 70)

    # --- Demo 1: SSE-style event stream ---
    print("\n--- Demo 1: All events via single callback ---\n")

    events: List[Dict[str, Any]] = []

    def on_event(event: str, run_id: str, **data: Any) -> None:
        events.append({"event": event, "run_id": run_id[:8], **data})
        # Print as SSE-style lines
        compact = {k: v for k, v in data.items() if k not in ("result", "system_prompt")}
        print(f"  event: {event:20s}  run={run_id[:8]}  {compact}")

    observer = SimpleStepObserver(on_event)

    agent = Agent(
        tools=[get_weather],
        provider=DemoProvider(),
        config=AgentConfig(
            max_iterations=3,
            observers=[observer],
        ),
    )

    result = agent.run([Message(role=Role.USER, content="What's the weather in London?")])

    print(f"\n  Final answer: {result.content}")
    print(f"  Events captured: {len(events)}")

    # --- Demo 2: Filter events by type ---
    print("\n--- Demo 2: Filter specific event types ---\n")

    tool_events = [e for e in events if e["event"].startswith("tool_")]
    llm_events = [e for e in events if e["event"].startswith("llm_")]

    print(f"  Tool events: {len(tool_events)}")
    for e in tool_events:
        print(f"    {e['event']:20s}  {e.get('tool_name', '')}")

    print(f"  LLM events:  {len(llm_events)}")
    for e in llm_events:
        print(f"    {e['event']:20s}  {e.get('model', '')}")

    # --- Demo 3: Combine with other observers ---
    print("\n--- Demo 3: Combine SimpleStepObserver with LoggingObserver ---\n")

    from selectools.observer import AgentObserver

    class CountingObserver(AgentObserver):
        """Minimal observer that counts LLM calls."""

        def __init__(self) -> None:
            self.llm_calls = 0

        def on_llm_end(self, run_id: str, response: Any, usage: Optional[UsageStats]) -> None:
            self.llm_calls += 1

    counter = CountingObserver()
    step_events: List[str] = []
    step_observer = SimpleStepObserver(lambda event, run_id, **d: step_events.append(event))

    agent2 = Agent(
        tools=[get_weather],
        provider=DemoProvider(),
        config=AgentConfig(
            max_iterations=3,
            observers=[counter, step_observer],
        ),
    )

    agent2.run([Message(role=Role.USER, content="Weather in Paris?")])

    print(f"  CountingObserver: {counter.llm_calls} LLM calls")
    print(f"  SimpleStepObserver: {len(step_events)} total events")
    print(f"  Events: {step_events}")

    print("\n" + "=" * 70)
    print("  Key takeaways:")
    print("    - SimpleStepObserver(callback) — one function gets all events")
    print("    - Event names match AgentObserver methods without 'on_' prefix")
    print("    - Combine with other observers via AgentConfig(observers=[...])")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
