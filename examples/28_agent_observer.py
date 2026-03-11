#!/usr/bin/env python3
"""
AgentObserver Protocol — structured lifecycle observability for production.

Demonstrates:
  1. Custom AgentObserver subclass with run_id/call_id correlation
  2. Built-in LoggingObserver for structured JSON logs
  3. Multiple observers on the same agent
  4. result.usage for aggregated stats
  5. result.trace.to_otel_spans() for OpenTelemetry export
  6. Observer with FallbackProvider (on_provider_fallback events)

No API key needed — uses mock providers.

Prerequisites: pip install selectools
Run: python examples/28_agent_observer.py
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, FallbackProvider, Message, Role
from selectools.observer import AgentObserver, LoggingObserver
from selectools.tools import tool
from selectools.types import AgentResult, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Provider that simulates tool calling for demo purposes."""

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
                    content="Let me look up that order for you.",
                    tool_calls=[
                        ToolCall(
                            tool_name="track_order",
                            parameters={"order_id": "ORD-42"},
                            id="call_1",
                        )
                    ],
                ),
                UsageStats(150, 40, 190, 0.001, "mock", "gpt-4o-mini"),
            )

        return (
            Message(
                role=Role.ASSISTANT,
                content="Your order ORD-42 shipped yesterday and arrives tomorrow.",
            ),
            UsageStats(200, 60, 260, 0.002, "mock", "gpt-4o-mini"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Track an order by ID")
def track_order(order_id: str) -> str:
    return f'{{"order_id": "{order_id}", "status": "shipped", "eta": "tomorrow"}}'


@tool(description="Cancel an order")
def cancel_order(order_id: str, reason: str) -> str:
    return f"Order {order_id} cancelled: {reason}"


# ---------------------------------------------------------------------------
# Demo 1: Custom Observer
# ---------------------------------------------------------------------------


class MetricsObserver(AgentObserver):
    """Collects structured metrics from agent execution."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self.events.append({"event": "run_start", "run_id": run_id})
        print(f"    [{run_id[:8]}] Run started")

    def on_llm_start(
        self, run_id: str, messages: List[Message], model: str, system_prompt: str
    ) -> None:
        self.events.append({"event": "llm_start", "run_id": run_id, "model": model})
        print(f"    [{run_id[:8]}] LLM call to {model}")

    def on_llm_end(self, run_id: str, response: Message, usage: Optional[UsageStats]) -> None:
        tokens = usage.total_tokens if usage else 0
        self.events.append({"event": "llm_end", "run_id": run_id, "tokens": tokens})
        print(f"    [{run_id[:8]}] LLM responded ({tokens} tokens)")

    def on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self.events.append(
            {
                "event": "tool_start",
                "run_id": run_id,
                "call_id": call_id,
                "tool": tool_name,
            }
        )
        print(f"    [{run_id[:8]}] Tool start: {tool_name}({tool_args})")

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        self.events.append(
            {
                "event": "tool_end",
                "run_id": run_id,
                "call_id": call_id,
                "tool": tool_name,
                "duration_ms": duration_ms,
            }
        )
        print(f"    [{run_id[:8]}] Tool end:   {tool_name} ({duration_ms:.0f}ms)")

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        self.events.append({"event": "run_end", "run_id": run_id})
        print(f"    [{run_id[:8]}] Run complete — {len(result.content)} chars")

    def on_error(self, run_id: str, error: Exception) -> None:
        self.events.append({"event": "error", "run_id": run_id, "error": str(error)})
        print(f"    [{run_id[:8]}] ERROR: {error}")


def demo_custom_observer() -> None:
    """Show a custom AgentObserver collecting metrics."""
    print("\n" + "=" * 70)
    print("  Demo 1: Custom AgentObserver")
    print("=" * 70 + "\n")

    observer = MetricsObserver()

    agent = Agent(
        tools=[track_order, cancel_order],
        provider=MockProvider(),
        config=AgentConfig(
            max_iterations=5,
            observers=[observer],
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Where is my order ORD-42?")])

    print(f"\n  Events captured: {len(observer.events)}")
    for ev in observer.events:
        print(f"    {ev['event']:12s}  run={ev['run_id'][:8]}")

    if result.usage:
        print(f"\n  Aggregated usage (result.usage):")
        print(f"    Total tokens: {result.usage.total_tokens}")
        print(f"    Total cost:   ${result.usage.total_cost_usd:.6f}")


# ---------------------------------------------------------------------------
# Demo 2: LoggingObserver
# ---------------------------------------------------------------------------


def demo_logging_observer() -> None:
    """Show the built-in LoggingObserver with Python logging."""
    print("\n" + "=" * 70)
    print("  Demo 2: Built-in LoggingObserver")
    print("=" * 70 + "\n")

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("    %(message)s"))
    logger = logging.getLogger("selectools.observer")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    agent = Agent(
        tools=[track_order],
        provider=MockProvider(),
        config=AgentConfig(
            max_iterations=5,
            observers=[LoggingObserver()],
        ),
    )

    print("  Running agent with LoggingObserver (structured JSON logs):\n")
    result = agent.run([Message(role=Role.USER, content="Track order ORD-99")])

    print(f"\n  Response: {result.content[:60]}...")

    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Demo 3: Multiple Observers
# ---------------------------------------------------------------------------


def demo_multiple_observers() -> None:
    """Show multiple observers on the same agent."""
    print("\n" + "=" * 70)
    print("  Demo 3: Multiple Observers")
    print("=" * 70 + "\n")

    metrics = MetricsObserver()
    logging_obs = LoggingObserver()

    agent = Agent(
        tools=[track_order],
        provider=MockProvider(),
        config=AgentConfig(
            max_iterations=5,
            observers=[metrics, logging_obs],
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Where is ORD-77?")])

    print(f"\n  MetricsObserver captured {len(metrics.events)} events")
    print(f"  LoggingObserver emitted structured logs to 'selectools.observer' logger")


# ---------------------------------------------------------------------------
# Demo 4: OTel Span Export
# ---------------------------------------------------------------------------


def demo_otel_export() -> None:
    """Show trace export to OpenTelemetry-compatible spans."""
    print("\n" + "=" * 70)
    print("  Demo 4: OpenTelemetry Span Export")
    print("=" * 70 + "\n")

    agent = Agent(
        tools=[track_order],
        provider=MockProvider(),
        config=AgentConfig(max_iterations=5),
    )

    result = agent.run([Message(role=Role.USER, content="Track ORD-55")])

    if result.trace:
        spans = result.trace.to_otel_spans()
        print(f"  Exported {len(spans)} OTel spans:\n")
        for span in spans:
            print(f"    name={span.get('name', 'N/A'):30s}  type={span.get('type', 'N/A')}")
            if "duration_ms" in span:
                print(f"      duration_ms={span['duration_ms']:.1f}")
            if span.get("attributes"):
                for k, v in list(span["attributes"].items())[:3]:
                    print(f"      {k}={str(v)[:60]}")
            print()


# ---------------------------------------------------------------------------
# Demo 5: Observer + FallbackProvider
# ---------------------------------------------------------------------------


def demo_observer_with_fallback() -> None:
    """Show observer capturing fallback events."""
    print("\n" + "=" * 70)
    print("  Demo 5: Observer with FallbackProvider")
    print("=" * 70 + "\n")

    observer = MetricsObserver()

    fallback_provider = FallbackProvider(
        providers=[MockProvider(), MockProvider()],
        max_failures=3,
        cooldown_seconds=30,
    )

    agent = Agent(
        tools=[track_order],
        provider=fallback_provider,
        config=AgentConfig(
            max_iterations=5,
            observers=[observer],
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Where is ORD-11?")])

    print(f"\n  Events: {len(observer.events)}")
    print(f"  Response: {result.content[:60]}...")

    fallback_events = [e for e in observer.events if e["event"] == "provider_fallback"]
    print(f"  Fallback events: {len(fallback_events)}")
    print(
        "\n  Tip: If the primary provider fails, you'll see on_provider_fallback events "
        "with the provider name and error."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Selectools v0.14.0 — AgentObserver Protocol Demo")
    print("=" * 70)

    demo_custom_observer()
    demo_logging_observer()
    demo_multiple_observers()
    demo_otel_export()
    demo_observer_with_fallback()

    print("\n" + "=" * 70)
    print("  All demos complete!")
    print()
    print("  Key takeaways:")
    print("    - Subclass AgentObserver and override only the events you need")
    print("    - Use LoggingObserver for instant structured JSON logs")
    print("    - Stack multiple observers via AgentConfig(observers=[...])")
    print("    - Export traces to OTel with result.trace.to_otel_spans()")
    print("    - run_id correlates all events within a single run")
    print("    - call_id matches tool start/end in parallel execution")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
