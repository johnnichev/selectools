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


# ---------------------------------------------------------------------------
# Regression tests: stream() / astream() correctness
# ---------------------------------------------------------------------------


class _StreamingOkProvider:
    """Provider that supports streaming and yields ToolCall objects."""

    name = "streaming_ok"
    supports_streaming = True
    supports_async = True

    def __init__(self, chunks: Optional[List[Any]] = None):
        from selectools.types import ToolCall

        self._chunks: List[Any] = chunks if chunks is not None else ["hello ", "world"]

    def stream(self, **kwargs: Any) -> Any:
        for c in self._chunks:
            yield c

    async def astream(self, **kwargs: Any) -> Any:
        for c in self._chunks:
            yield c

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return _OK_MSG, _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return _OK_MSG, _DUMMY_USAGE


class _StreamingFailBeforeYield:
    """Provider that fails immediately before yielding anything."""

    name = "stream_fail_before"
    supports_streaming = True
    supports_async = True

    def stream(self, **kwargs: Any) -> Any:
        raise ProviderError("rate limit 429")
        yield  # makes it a generator

    async def astream(self, **kwargs: Any) -> Any:
        raise ProviderError("rate limit 429")
        yield  # makes it an async generator

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise ProviderError("rate limit 429")

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise ProviderError("rate limit 429")


class _StreamingFailAfterYield:
    """Provider that yields one chunk then raises mid-stream."""

    name = "stream_fail_after"
    supports_streaming = True
    supports_async = True

    def stream(self, **kwargs: Any) -> Any:
        yield "partial"
        raise ProviderError("rate limit 429")

    async def astream(self, **kwargs: Any) -> Any:
        yield "partial"
        raise ProviderError("rate limit 429")

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise ProviderError("rate limit 429")

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise ProviderError("rate limit 429")


class TestStreamFallback:
    def test_stream_falls_over_when_no_yield_before_error(self) -> None:
        """Fallback to next provider when first provider errors before any yield."""
        fb = FallbackProvider(
            providers=[_StreamingFailBeforeYield(), _StreamingOkProvider()],
        )
        chunks = list(
            fb.stream(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="hi")]
            )
        )
        assert chunks == ["hello ", "world"]
        assert fb.provider_used == "streaming_ok"

    def test_stream_does_not_fallover_mid_stream(self) -> None:
        """No fallback after partial output — re-raise to avoid data corruption."""
        fb = FallbackProvider(
            providers=[_StreamingFailAfterYield(), _StreamingOkProvider()],
        )
        chunks: List[Any] = []
        with pytest.raises(ProviderError):
            for chunk in fb.stream(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                chunks.append(chunk)
        # Only the partial chunk from first provider; second provider NOT called
        assert "partial" in chunks
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_astream_falls_over_when_no_yield_before_error(self) -> None:
        """astream fallback to next provider when first raises before yield."""
        fb = FallbackProvider(
            providers=[_StreamingFailBeforeYield(), _StreamingOkProvider()],
        )
        chunks = []
        async for chunk in fb.astream(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            chunks.append(chunk)
        assert chunks == ["hello ", "world"]
        assert fb.provider_used == "streaming_ok"

    @pytest.mark.asyncio
    async def test_astream_does_not_fallover_mid_stream(self) -> None:
        """astream does not fall over to next provider after partial output."""
        fb = FallbackProvider(
            providers=[_StreamingFailAfterYield(), _StreamingOkProvider()],
        )
        chunks: List[Any] = []
        with pytest.raises(ProviderError):
            async for chunk in fb.astream(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                chunks.append(chunk)
        # Only the partial chunk from the first provider
        assert "partial" in chunks
        assert len(chunks) == 1

    def test_stream_type_annotation_accepts_toolcall(self) -> None:
        """FallbackProvider.stream() correctly passes through ToolCall objects."""
        from selectools.types import ToolCall

        tc = ToolCall(tool_name="search", parameters={"q": "test"}, id="call_1")
        fb = FallbackProvider(providers=[_StreamingOkProvider(chunks=[tc])])
        chunks = list(
            fb.stream(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="hi")]
            )
        )
        assert len(chunks) == 1
        assert isinstance(chunks[0], ToolCall)
        assert chunks[0].tool_name == "search"


# ---------------------------------------------------------------------------
# Regression tests: _is_retriable() numeric false-positives
# ---------------------------------------------------------------------------


class TestIsRetriableNumerics:
    """Regression: status code checks must use word boundaries to avoid matching
    substrings inside larger numbers (e.g. '15003' contains '500')."""

    def test_500_not_matched_inside_larger_number(self) -> None:
        # '500' appears as a substring of '15003' — must NOT be retriable
        assert _is_retriable(Exception("error code 15003")) is False

    def test_500_not_matched_in_token_count(self) -> None:
        # 'expected 5000 tokens' contains '500' as substring — must NOT match
        assert _is_retriable(Exception("expected 5000 tokens")) is False

    def test_429_not_matched_inside_larger_number(self) -> None:
        assert _is_retriable(Exception("token_count=14290")) is False

    def test_503_not_matched_inside_larger_number(self) -> None:
        assert _is_retriable(Exception("error 15030")) is False

    def test_500_as_standalone_still_retriable(self) -> None:
        assert _is_retriable(Exception("HTTP 500 Internal Server Error")) is True

    def test_429_as_standalone_still_retriable(self) -> None:
        assert _is_retriable(Exception("HTTP 429 Too Many Requests")) is True

    def test_503_as_standalone_still_retriable(self) -> None:
        assert _is_retriable(Exception("HTTP 503 Service Unavailable")) is True

    def test_502_as_standalone_still_retriable(self) -> None:
        assert _is_retriable(Exception("502 Bad Gateway")) is True


# ---------------------------------------------------------------------------
# Regression tests: exception chaining in stream/astream
# ---------------------------------------------------------------------------


class TestStreamExceptionChaining:
    def test_stream_chains_exception_from_last_exc(self) -> None:
        """stream() should chain last_exc with 'from' when all providers fail."""
        fb = FallbackProvider(providers=[_StreamingFailBeforeYield()])
        with pytest.raises(ProviderError) as exc_info:
            list(
                fb.stream(
                    model="m",
                    system_prompt="s",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_astream_chains_exception_from_last_exc(self) -> None:
        """astream() should chain last_exc with 'from' when all providers fail."""
        fb = FallbackProvider(providers=[_StreamingFailBeforeYield()])
        with pytest.raises(ProviderError) as exc_info:
            async for _ in fb.astream(
                model="m",
                system_prompt="s",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Regression tests: circuit-breaker all-open produces clear error
# ---------------------------------------------------------------------------


class _CircuitBrokenProvider:
    """Provider that always raises a retriable error to quickly open the circuit."""

    def __init__(self, name: str = "broken") -> None:
        self.name = name
        self.supports_streaming = True
        self.supports_async = True

    def complete(self, **kwargs: Any) -> Any:
        raise ProviderError("timeout")

    async def acomplete(self, **kwargs: Any) -> Any:
        raise ProviderError("timeout")

    def stream(self, **kwargs: Any) -> Any:
        raise ProviderError("timeout")
        yield  # make it a generator

    async def astream(self, **kwargs: Any) -> Any:
        raise ProviderError("timeout")
        yield  # make it an async generator


class TestAllCircuitsBroken:
    def _make_fb_with_open_circuit(self) -> "FallbackProvider":
        fb = FallbackProvider(
            providers=[_CircuitBrokenProvider("only")],
            circuit_breaker_threshold=1,
            circuit_breaker_cooldown=9999.0,
        )
        # Trigger the circuit to open
        try:
            fb.complete(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="x")]
            )
        except ProviderError:
            pass
        assert fb._is_circuit_open("only")
        return fb

    def test_complete_circuit_broken_error_message(self) -> None:
        fb = self._make_fb_with_open_circuit()
        with pytest.raises(ProviderError, match="circuit-broken"):
            fb.complete(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="x")]
            )

    @pytest.mark.asyncio
    async def test_acomplete_circuit_broken_error_message(self) -> None:
        fb = self._make_fb_with_open_circuit()
        with pytest.raises(ProviderError, match="circuit-broken"):
            await fb.acomplete(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="x")]
            )

    def test_stream_circuit_broken_error_message(self) -> None:
        fb = self._make_fb_with_open_circuit()
        with pytest.raises(ProviderError, match="circuit-broken"):
            list(
                fb.stream(
                    model="m", system_prompt="s", messages=[Message(role=Role.USER, content="x")]
                )
            )

    @pytest.mark.asyncio
    async def test_astream_circuit_broken_error_message(self) -> None:
        fb = self._make_fb_with_open_circuit()
        with pytest.raises(ProviderError, match="circuit-broken"):
            async for _ in fb.astream(
                model="m", system_prompt="s", messages=[Message(role=Role.USER, content="x")]
            ):
                pass
