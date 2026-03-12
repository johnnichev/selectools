"""
Unit tests for FallbackProvider: complete/acomplete failover, circuit breaker,
_is_retriable(), on_fallback callback.

Previously only covered by E2E tests that were always skipped in CI.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.providers.base import ProviderError
from selectools.providers.fallback import FallbackProvider, _is_retriable
from selectools.types import Message, Role
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")
_OK_MSG = Message(role=Role.ASSISTANT, content="ok")


class _OkProvider:
    name = "ok"
    supports_streaming = False
    supports_async = True

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return _OK_MSG, _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return _OK_MSG, _DUMMY_USAGE


class _FailProvider:
    def __init__(self, name: str = "fail", error: str = "rate limit 429"):
        self.name = name
        self.supports_streaming = False
        self.supports_async = True
        self.error = error

    def complete(self, **kwargs: Any) -> Any:
        raise ProviderError(self.error)

    async def acomplete(self, **kwargs: Any) -> Any:
        raise ProviderError(self.error)


class _NonRetriableFailProvider:
    name = "fatal"
    supports_streaming = False
    supports_async = True

    def complete(self, **kwargs: Any) -> Any:
        raise ProviderError("Authentication failed: invalid API key")

    async def acomplete(self, **kwargs: Any) -> Any:
        raise ProviderError("Authentication failed: invalid API key")


class TestIsRetriable:
    def test_timeout(self) -> None:
        assert _is_retriable(Exception("Request timeout")) is True

    def test_rate_limit(self) -> None:
        assert _is_retriable(Exception("rate limit exceeded")) is True

    def test_429(self) -> None:
        assert _is_retriable(Exception("HTTP 429")) is True

    def test_500(self) -> None:
        assert _is_retriable(Exception("500 Internal Server Error")) is True

    def test_502(self) -> None:
        assert _is_retriable(Exception("502 Bad Gateway")) is True

    def test_connection(self) -> None:
        assert _is_retriable(Exception("Connection refused")) is True

    def test_non_retriable(self) -> None:
        assert _is_retriable(Exception("Invalid API key")) is False

    def test_auth_error_not_retriable(self) -> None:
        assert _is_retriable(Exception("Authentication failed")) is False


class TestCompleteFailover:
    def test_uses_first_provider(self) -> None:
        fb = FallbackProvider(providers=[_OkProvider()])
        msg, usage = fb.complete(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"
        assert fb.provider_used == "ok"

    def test_fails_over_to_second(self) -> None:
        fb = FallbackProvider(providers=[_FailProvider(), _OkProvider()])
        msg, _ = fb.complete(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"
        assert fb.provider_used == "ok"

    def test_all_fail_raises(self) -> None:
        fb = FallbackProvider(providers=[_FailProvider("a"), _FailProvider("b")])
        with pytest.raises(ProviderError, match="All providers exhausted"):
            fb.complete(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            )

    def test_non_retriable_error_propagates(self) -> None:
        fb = FallbackProvider(providers=[_NonRetriableFailProvider(), _OkProvider()])
        with pytest.raises(ProviderError, match="Authentication"):
            fb.complete(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            )


class TestAcompleteFailover:
    @pytest.mark.asyncio
    async def test_uses_first_provider(self) -> None:
        fb = FallbackProvider(providers=[_OkProvider()])
        msg, _ = await fb.acomplete(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"

    @pytest.mark.asyncio
    async def test_fails_over_to_second(self) -> None:
        fb = FallbackProvider(providers=[_FailProvider(), _OkProvider()])
        msg, _ = await fb.acomplete(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"
        assert fb.provider_used == "ok"


class TestCircuitBreaker:
    def test_opens_after_threshold(self) -> None:
        fb = FallbackProvider(
            providers=[_FailProvider("flaky"), _OkProvider()],
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=60.0,
        )
        fb.complete(model="m", system_prompt="s", messages=[Message(role=Role.USER, content="1")])
        fb.complete(model="m", system_prompt="s", messages=[Message(role=Role.USER, content="2")])

        assert fb._failures.get("flaky", 0) >= 2
        assert fb._is_circuit_open("flaky") is True

    def test_records_success_resets_failures(self) -> None:
        ok = _OkProvider()
        fb = FallbackProvider(providers=[ok])
        fb._failures["ok"] = 2
        fb.complete(model="m", system_prompt="s", messages=[Message(role=Role.USER, content="hi")])
        assert fb._failures.get("ok", 0) == 0

    def test_cooldown_recovery(self) -> None:
        fb = FallbackProvider(
            providers=[_FailProvider("flaky"), _OkProvider()],
            circuit_breaker_threshold=1,
            circuit_breaker_cooldown=0.01,
        )
        fb.complete(model="m", system_prompt="s", messages=[Message(role=Role.USER, content="1")])
        assert fb._is_circuit_open("flaky") is True

        time.sleep(0.02)
        assert fb._is_circuit_open("flaky") is False


class TestOnFallbackCallback:
    def test_callback_invoked(self) -> None:
        events: List[Tuple[str, str, str]] = []

        def on_fb(failed: str, next_p: str, exc: Exception) -> None:
            events.append((failed, next_p, str(exc)))

        fb = FallbackProvider(
            providers=[_FailProvider("p1"), _OkProvider()],
            on_fallback=on_fb,
        )
        fb.complete(model="m", system_prompt="s", messages=[Message(role=Role.USER, content="hi")])

        assert len(events) == 1
        assert events[0][0] == "p1"
        assert events[0][1] == "ok"

    def test_callback_exception_swallowed(self) -> None:
        def bad_callback(f: str, n: str, e: Exception) -> None:
            raise RuntimeError("callback crash")

        fb = FallbackProvider(
            providers=[_FailProvider(), _OkProvider()],
            on_fallback=bad_callback,
        )
        msg, _ = fb.complete(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"

    def test_callback_shows_none_when_last(self) -> None:
        events: List[str] = []

        def on_fb(failed: str, next_p: str, exc: Exception) -> None:
            events.append(next_p)

        fb = FallbackProvider(
            providers=[_FailProvider("only")],
            on_fallback=on_fb,
        )
        with pytest.raises(ProviderError):
            fb.complete(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert events[0] == "none"
