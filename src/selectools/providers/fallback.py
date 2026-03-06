"""
Fallback provider that tries multiple providers in priority order.

On failure (timeout, server error, rate limit, connection error) the next
provider in the chain is attempted.  A lightweight circuit breaker skips
providers that have failed repeatedly.
"""

from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from ..types import Message, ToolCall
from .base import Provider, ProviderError

if TYPE_CHECKING:
    from ..tools.base import Tool
    from ..usage import UsageStats


_RETRIABLE_SUBSTRINGS = ("timeout", "rate limit", "429", "500", "502", "503", "connection")


def _is_retriable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RETRIABLE_SUBSTRINGS)


class FallbackProvider:
    """Wraps multiple providers with automatic failover.

    Providers are tried in the order given.  If a provider raises a retriable
    error the next one is tried.  A simple circuit breaker skips providers
    that have failed ``circuit_breaker_threshold`` times in a row for
    ``circuit_breaker_cooldown`` seconds.

    Attributes:
        name: Always ``"fallback"``.
        supports_streaming: True if *any* wrapped provider supports streaming.
        supports_async: True if *any* wrapped provider supports async.
        provider_used: Name of the provider that handled the most recent call.
    """

    name: str = "fallback"

    def __init__(
        self,
        providers: List[Any],
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: float = 60.0,
        on_fallback: Optional[Callable[[str, str, Exception], None]] = None,
    ) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider.")

        self.providers = providers
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.on_fallback = on_fallback
        self.provider_used: Optional[str] = None

        self._failures: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    @property
    def supports_streaming(self) -> bool:
        return any(getattr(p, "supports_streaming", False) for p in self.providers)

    @property
    def supports_async(self) -> bool:
        return any(getattr(p, "supports_async", False) for p in self.providers)

    def _is_circuit_open(self, name: str) -> bool:
        until = self._circuit_open_until.get(name, 0.0)
        if until and time.time() < until:
            return True
        if until and time.time() >= until:
            self._circuit_open_until.pop(name, None)
            self._failures[name] = 0
        return False

    def _record_failure(self, name: str) -> None:
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= self.circuit_breaker_threshold:
            self._circuit_open_until[name] = time.time() + self.circuit_breaker_cooldown

    def _record_success(self, name: str) -> None:
        self._failures[name] = 0
        self._circuit_open_until.pop(name, None)

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, "UsageStats"]:
        last_exc: Optional[Exception] = None

        for provider in self.providers:
            pname = getattr(provider, "name", type(provider).__name__)
            if self._is_circuit_open(pname):
                continue

            try:
                result: tuple[Message, "UsageStats"] = provider.complete(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                self._record_success(pname)
                self.provider_used = pname
                return result
            except Exception as exc:
                last_exc = exc
                if _is_retriable(exc):
                    self._record_failure(pname)
                    if self.on_fallback:
                        next_name = self._next_available(pname) or "none"
                        try:
                            self.on_fallback(pname, next_name, exc)
                        except Exception:  # nosec B110
                            pass
                    continue
                raise

        raise ProviderError(f"All providers exhausted. Last error: {last_exc}") from last_exc

    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, "UsageStats"]:
        last_exc: Optional[Exception] = None

        for provider in self.providers:
            pname = getattr(provider, "name", type(provider).__name__)
            if self._is_circuit_open(pname):
                continue

            try:
                aresult: tuple[Message, "UsageStats"]
                if hasattr(provider, "acomplete") and getattr(provider, "supports_async", False):
                    aresult = await provider.acomplete(
                        model=model,
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                else:
                    aresult = provider.complete(
                        model=model,
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                self._record_success(pname)
                self.provider_used = pname
                return aresult
            except Exception as exc:
                last_exc = exc
                if _is_retriable(exc):
                    self._record_failure(pname)
                    if self.on_fallback:
                        next_name = self._next_available(pname) or "none"
                        try:
                            self.on_fallback(pname, next_name, exc)
                        except Exception:  # nosec B110
                            pass
                    continue
                raise

        raise ProviderError(f"All providers exhausted. Last error: {last_exc}") from last_exc

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> Iterable[str]:
        last_exc: Optional[Exception] = None

        for provider in self.providers:
            pname = getattr(provider, "name", type(provider).__name__)
            if self._is_circuit_open(pname):
                continue
            if not getattr(provider, "supports_streaming", False):
                continue

            try:
                gen: Iterable[str] = provider.stream(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                self._record_success(pname)
                self.provider_used = pname
                return gen
            except Exception as exc:
                last_exc = exc
                if _is_retriable(exc):
                    self._record_failure(pname)
                    if self.on_fallback:
                        next_name = self._next_available(pname) or "none"
                        try:
                            self.on_fallback(pname, next_name, exc)
                        except Exception:  # nosec B110
                            pass
                    continue
                raise

        raise ProviderError(f"No streaming provider available. Last error: {last_exc}")

    def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        for provider in self.providers:
            pname = getattr(provider, "name", type(provider).__name__)
            if self._is_circuit_open(pname):
                continue
            if not getattr(provider, "supports_streaming", False):
                continue

            self.provider_used = pname
            stream: AsyncIterable[Union[str, ToolCall]] = provider.astream(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return stream

        raise ProviderError("No async-streaming provider available.")

    def _next_available(self, after_name: str) -> Optional[str]:
        found = False
        for p in self.providers:
            pname = getattr(p, "name", type(p).__name__)
            if pname == after_name:
                found = True
                continue
            if found and not self._is_circuit_open(pname):
                return pname
        return None


__all__ = ["FallbackProvider"]
