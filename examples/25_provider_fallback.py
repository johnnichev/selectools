#!/usr/bin/env python3
"""
Provider Fallback — Automatic failover between LLM providers.

Demonstrates:
  1. FallbackProvider with priority ordering
  2. Automatic failover on provider failure
  3. Circuit breaker after repeated failures
  4. on_fallback callback for observability
  5. provider_used tracking

No API key needed — uses mock providers.

Prerequisites: pip install selectools
Run: python examples/25_provider_fallback.py
"""

from typing import Any, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role
from selectools.providers.base import ProviderError
from selectools.providers.fallback import FallbackProvider
from selectools.tools import tool
from selectools.types import AgentResult
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class WorkingProvider:
    """Provider that always succeeds."""

    name = "working-provider"
    supports_streaming = False
    supports_async = True

    def __init__(self, label: str = "working") -> None:
        self.name = label
        self.call_count = 0

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
        self.call_count += 1
        return (
            Message(
                role=Role.ASSISTANT, content=f"Response from {self.name} (call #{self.call_count})"
            ),
            UsageStats(100, 50, 150, 0.001, self.name, "mock-model"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class FailingProvider:
    """Provider that always fails with a retriable error."""

    name = "failing-provider"
    supports_streaming = False
    supports_async = True

    def __init__(self, label: str = "failing", error_msg: str = "Connection timeout") -> None:
        self.name = label
        self.call_count = 0
        self.error_msg = error_msg

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.call_count += 1
        raise ProviderError(self.error_msg)

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class IntermittentProvider:
    """Provider that fails N times then succeeds."""

    name = "intermittent"
    supports_streaming = False
    supports_async = True

    def __init__(self, fail_count: int = 2, label: str = "intermittent") -> None:
        self.name = label
        self.call_count = 0
        self.fail_count = fail_count

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise ProviderError("429 rate limit exceeded")
        return (
            Message(role=Role.ASSISTANT, content=f"Response from {self.name} (recovered)"),
            UsageStats(100, 50, 150, 0.001, self.name, "mock-model"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Search for information")
def search(query: str) -> str:
    return f"Results for: {query}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  Provider Fallback Demo")
    print("=" * 70)

    # --- Step 1: Basic fallback - primary fails, secondary succeeds ---
    print("\n--- Step 1: Basic fallback ---\n")

    primary = FailingProvider(label="openai-mock", error_msg="Connection timeout")
    secondary = WorkingProvider(label="anthropic-mock")

    fallback_events: list[str] = []

    def on_fallback(failed: str, next_prov: str, exc: Exception) -> None:
        fallback_events.append(f"{failed} -> {next_prov}: {exc}")

    fallback = FallbackProvider(
        providers=[primary, secondary],
        on_fallback=on_fallback,
    )

    agent = Agent(
        tools=[search],
        provider=fallback,
        config=AgentConfig(max_iterations=1),
    )

    result = agent.ask("Search for Python tutorials")

    print(f"  Primary ({primary.name}) failed: {primary.call_count} call(s)")
    print(f"  Secondary ({secondary.name}) succeeded: {secondary.call_count} call(s)")
    print(f"  provider_used: {fallback.provider_used}")
    print(f"  Fallback events: {fallback_events}")
    print(f"  Response: {result.content[:60]}")

    assert fallback.provider_used == "anthropic-mock"
    assert primary.call_count == 1
    assert secondary.call_count == 1
    assert len(fallback_events) == 1
    print("\n  PASS: Primary failed, fell through to secondary\n")

    # --- Step 2: All providers healthy - uses primary ---
    print("--- Step 2: All providers healthy - uses primary ---\n")

    healthy_primary = WorkingProvider(label="openai-healthy")
    healthy_secondary = WorkingProvider(label="anthropic-healthy")

    fallback_2 = FallbackProvider(providers=[healthy_primary, healthy_secondary])
    agent_2 = Agent(tools=[search], provider=fallback_2, config=AgentConfig(max_iterations=1))

    result_2 = agent_2.ask("Hello")

    print(f"  provider_used: {fallback_2.provider_used}")
    print(f"  Primary calls: {healthy_primary.call_count}")
    print(f"  Secondary calls: {healthy_secondary.call_count}")

    assert fallback_2.provider_used == "openai-healthy"
    assert healthy_primary.call_count == 1
    assert healthy_secondary.call_count == 0
    print("\n  PASS: Used primary when healthy, never touched secondary\n")

    # --- Step 3: Circuit breaker ---
    print("--- Step 3: Circuit breaker after repeated failures ---\n")

    flaky = FailingProvider(label="flaky-provider", error_msg="500 Internal Server Error")
    backup = WorkingProvider(label="backup-provider")

    breaker_events: list[str] = []

    fallback_3 = FallbackProvider(
        providers=[flaky, backup],
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown=30.0,
        on_fallback=lambda f, n, e: breaker_events.append(f"{f} failed"),
    )

    agent_3 = Agent(tools=[search], provider=fallback_3, config=AgentConfig(max_iterations=1))

    for i in range(4):
        agent_3.reset()
        result_i = agent_3.ask(f"Request {i + 1}")
        print(
            f"  Request {i + 1}: provider_used={fallback_3.provider_used}, flaky_calls={flaky.call_count}"
        )

    print(f"\n  Flaky provider was called {flaky.call_count} times (circuit opened after 2)")
    print(f"  Backup handled remaining requests")

    assert flaky.call_count == 2
    assert backup.call_count == 4
    print("\n  PASS: Circuit breaker stopped calling flaky provider after threshold\n")

    # --- Step 4: Three-provider chain ---
    print("--- Step 4: Three-provider fallback chain ---\n")

    p1 = FailingProvider(label="provider-1", error_msg="timeout")
    p2 = FailingProvider(label="provider-2", error_msg="429 rate limit")
    p3 = WorkingProvider(label="provider-3")

    chain_events: list[str] = []
    fallback_4 = FallbackProvider(
        providers=[p1, p2, p3],
        on_fallback=lambda f, n, e: chain_events.append(f"{f}->{n}"),
    )

    agent_4 = Agent(tools=[search], provider=fallback_4, config=AgentConfig(max_iterations=1))
    result_4 = agent_4.ask("Final test")

    print(f"  Chain: {' -> '.join(chain_events)} -> success")
    print(f"  provider_used: {fallback_4.provider_used}")

    assert fallback_4.provider_used == "provider-3"
    assert p1.call_count == 1
    assert p2.call_count == 1
    assert p3.call_count == 1
    print("\n  PASS: Fell through entire chain to provider-3\n")

    print("=" * 70)
    print("  All provider fallback tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
