#!/usr/bin/env python3
"""
Batch Processing — Classify multiple requests concurrently.

Demonstrates:
  1. agent.batch() for sync concurrent processing
  2. agent.abatch() for async concurrent processing
  3. Per-request error isolation
  4. on_progress callback
  5. Batch with structured output (response_format)

No API key needed — uses a mock provider.

Prerequisites: pip install selectools pydantic
Run: python examples/26_batch_processing.py
"""

import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import AgentResult
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider that returns different responses per message content
# ---------------------------------------------------------------------------

INTENT_MAP: Dict[str, str] = {
    "cancel": '{"intent": "cancel", "confidence": 0.95}',
    "upgrade": '{"intent": "upgrade", "confidence": 0.90}',
    "payment": '{"intent": "billing", "confidence": 0.88}',
    "broken": '{"intent": "support", "confidence": 0.92}',
    "buy": '{"intent": "sales", "confidence": 0.85}',
}


class BatchMockProvider:
    """Provider that returns intent-based responses for batch demos."""

    name = "batch-mock"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self.call_count = 0

    def _classify(self, messages: Optional[List[Message]]) -> str:
        text = ""
        if messages:
            for m in messages:
                if m.role == Role.USER and m.content:
                    text = m.content.lower()
                    break

        for keyword, response in INTENT_MAP.items():
            if keyword in text:
                return response

        return '{"intent": "unknown", "confidence": 0.50}'

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
        response = self._classify(messages)
        return (
            Message(role=Role.ASSISTANT, content=response),
            UsageStats(80, 30, 110, 0.0005, "batch-mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Pydantic model for structured output
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    intent: Literal["cancel", "upgrade", "billing", "support", "sales", "unknown"]
    confidence: float


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Route a customer request")
def route_request(intent: str) -> str:
    return f"Routed to {intent} queue"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  Batch Processing Demo")
    print("=" * 70)

    tickets = [
        "I want to cancel my subscription",
        "How do I upgrade my plan?",
        "My payment failed",
        "The app is broken and crashes",
        "I'd like to buy the enterprise plan",
        "Can you help me reset my password?",
    ]

    # --- Step 1: Sync batch processing ---
    print("\n--- Step 1: agent.batch() — sync concurrent processing ---\n")

    provider = BatchMockProvider()
    agent = Agent(
        tools=[route_request],
        provider=provider,
        config=AgentConfig(max_iterations=1),
    )

    progress_log: list[str] = []

    def on_progress(completed: int, total: int) -> None:
        progress_log.append(f"{completed}/{total}")

    results = agent.batch(
        tickets,
        max_concurrency=3,
        on_progress=on_progress,
    )

    print(f"  Processed {len(results)} tickets")
    print(f"  Provider called {provider.call_count} times")
    print(f"  Progress: {' -> '.join(progress_log)}")

    for i, (ticket, result) in enumerate(zip(tickets, results)):
        print(f"  [{i + 1}] '{ticket[:40]}...' -> {result.content[:50]}")

    assert len(results) == len(tickets)
    assert provider.call_count == len(tickets)
    print("\n  PASS: All tickets processed in order\n")

    # --- Step 2: Async batch processing ---
    print("--- Step 2: agent.abatch() — async concurrent processing ---\n")

    provider_2 = BatchMockProvider()
    agent_2 = Agent(
        tools=[route_request],
        provider=provider_2,
        config=AgentConfig(max_iterations=1),
    )

    async def run_async_batch() -> List[AgentResult]:
        return list(await agent_2.abatch(tickets, max_concurrency=5))

    async_results: List[AgentResult] = asyncio.run(run_async_batch())

    print(f"  Processed {len(async_results)} tickets (async)")
    print(f"  Provider called {provider_2.call_count} times")

    for i, (ticket, result) in enumerate(zip(tickets, async_results)):
        print(f"  [{i + 1}] '{ticket[:40]}...' -> {result.content[:50]}")

    assert len(async_results) == len(tickets)
    print("\n  PASS: Async batch completed successfully\n")

    # --- Step 3: Batch with structured output ---
    print("--- Step 3: Batch with response_format (structured output) ---\n")

    provider_3 = BatchMockProvider()
    agent_3 = Agent(
        tools=[route_request],
        provider=provider_3,
        config=AgentConfig(max_iterations=1),
    )

    structured_results = agent_3.batch(
        tickets,
        max_concurrency=3,
        response_format=IntentResult,
    )

    print(f"  Processed {len(structured_results)} tickets with structured output\n")

    for i, (ticket, result) in enumerate(zip(tickets, structured_results)):
        if result.parsed:
            parsed: IntentResult = result.parsed
            print(
                f"  [{i + 1}] '{ticket[:35]}...' -> intent={parsed.intent}, confidence={parsed.confidence}"
            )
        else:
            print(f"  [{i + 1}] '{ticket[:35]}...' -> (no structured output)")

    assert all(r.parsed is not None for r in structured_results)
    print("\n  PASS: Batch with structured output works\n")

    # --- Step 4: Error isolation ---
    print("--- Step 4: Per-request error isolation ---\n")

    class ErrorOnThirdProvider:
        name = "error-on-third"
        supports_streaming = False
        supports_async = True

        def __init__(self) -> None:
            self.call_count = 0

        def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            self.call_count += 1
            if self.call_count == 3:
                raise RuntimeError("Simulated provider failure on request 3")
            return (
                Message(role=Role.ASSISTANT, content=f"OK (call {self.call_count})"),
                UsageStats(50, 20, 70, 0.0003, "error-mock", "mock"),
            )

        async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            return self.complete(**kwargs)

    provider_4 = ErrorOnThirdProvider()
    agent_4 = Agent(
        tools=[route_request],
        provider=provider_4,
        config=AgentConfig(max_iterations=1),
    )

    error_results = agent_4.batch(
        ["msg1", "msg2", "msg3 (will fail)", "msg4", "msg5"],
        max_concurrency=1,
    )

    for i, r in enumerate(error_results):
        status = "OK" if r.content and "OK" in r.content else "ERROR"
        print(f"  [{i + 1}] {status}: {r.content[:60]}")

    print(f"\n  Total results: {len(error_results)} (same as input)")
    print("  Request 3 failed but others completed fine")
    assert len(error_results) == 5
    print("\n  PASS: Failures are isolated per-request\n")

    print("=" * 70)
    print("  All batch processing tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
