"""
Simulation: FallbackProvider with primary failures.

Primary provider fails on the first N calls (simulating a degraded upstream).
Secondary (LocalProvider) always succeeds. Verifies:
- All calls eventually succeed via fallback
- Circuit breaker opens after threshold
- on_fallback callback fires the correct number of times

No API keys required.

Run: pytest tests/simulations/sim_provider_failover.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from selectools.providers.fallback import FallbackProvider
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role


class _FailingProvider:
    """Provider that raises on the first `fail_count` calls, then succeeds."""

    name = "failing"
    supports_async = False
    supports_streaming = False

    def __init__(self, fail_count: int = 3) -> None:
        self.fail_count = fail_count
        self._call_count = 0

    def complete(self, *, model, system_prompt, messages, tools=None, **kwargs):
        self._call_count += 1
        if self._call_count <= self.fail_count:
            raise RuntimeError(f"Simulated 503 service unavailable #{self._call_count}")
        # Should not reach here in tests where fail_count >= total calls
        from selectools.usage import UsageStats

        return Message(role=Role.ASSISTANT, content="primary ok"), UsageStats()

    def stream(self, *args, **kwargs):
        raise NotImplementedError

    def astream(self, *args, **kwargs):
        raise NotImplementedError

    async def acomplete(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.integration
class TestProviderFailover:
    """FallbackProvider must route to secondary when primary is degraded."""

    def test_all_calls_succeed_when_primary_fails(self):
        """
        20 calls with a primary that always fails must all succeed via secondary.
        """
        fallback_events: List[str] = []

        def on_fallback(primary: str, fallback: str, exc: Exception) -> None:
            fallback_events.append(f"{primary} → {fallback}: {type(exc).__name__}")

        primary = _FailingProvider(fail_count=999)  # always fails
        secondary = LocalProvider()
        fp = FallbackProvider(
            providers=[primary, secondary],
            circuit_breaker_threshold=3,
            on_fallback=on_fallback,
        )

        results: List[Message] = []
        for i in range(20):
            msg, _ = fp.complete(
                model="gpt-4o",
                system_prompt="test",
                messages=[Message(role=Role.USER, content=f"call {i}")],
                tools=[],
            )
            results.append(msg)

        assert len(results) == 20
        assert all(r.content for r in results)
        # Fallback must have been triggered
        assert len(fallback_events) > 0

    def test_circuit_breaker_opens_after_threshold(self):
        """
        After `threshold` failures, the circuit breaker must open and skip primary.
        """
        threshold = 3
        primary = _FailingProvider(fail_count=999)
        secondary = LocalProvider()
        fp = FallbackProvider(
            providers=[primary, secondary],
            circuit_breaker_threshold=threshold,
            circuit_breaker_cooldown=999.0,  # never cools down in this test
        )

        # Make threshold calls to open the circuit
        for i in range(threshold + 2):
            fp.complete(
                model="gpt-4o",
                system_prompt="test",
                messages=[Message(role=Role.USER, content=f"warmup {i}")],
                tools=[],
            )

        # After circuit opens, primary._call_count should stop growing
        calls_after_open = primary._call_count
        for i in range(5):
            fp.complete(
                model="gpt-4o",
                system_prompt="test",
                messages=[Message(role=Role.USER, content=f"after_open {i}")],
                tools=[],
            )

        # Primary call count must not have grown significantly after circuit opened
        assert primary._call_count <= calls_after_open + 1

    def test_fallback_call_count_matches_failures(self):
        """
        on_fallback must be called once per failure before circuit opens.
        """
        threshold = 3
        fallback_calls: List[int] = []

        def on_fallback(p, f, e):
            fallback_calls.append(1)

        primary = _FailingProvider(fail_count=threshold)
        secondary = LocalProvider()
        fp = FallbackProvider(
            providers=[primary, secondary],
            circuit_breaker_threshold=threshold,
            on_fallback=on_fallback,
        )

        for i in range(threshold + 3):
            fp.complete(
                model="gpt-4o",
                system_prompt="test",
                messages=[Message(role=Role.USER, content=f"call {i}")],
                tools=[],
            )

        # on_fallback must have fired at least `threshold` times
        assert len(fallback_calls) >= threshold
