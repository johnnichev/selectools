"""
Provider abstraction for model-agnostic tool calling.
"""

from __future__ import annotations

from typing import AsyncIterable, Protocol, runtime_checkable

from ..types import Message


class ProviderError(RuntimeError):
    """Raised when an adapter cannot complete a request."""


@runtime_checkable
class Provider(Protocol):
    """
    Interface every provider adapter must satisfy.
    
    Providers implement synchronous methods (complete, stream) and optionally
    async methods (acomplete, astream) for better performance in async contexts.
    """

    name: str
    supports_streaming: bool
    supports_async: bool = False  # Optional flag to indicate async support

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> str:
        """Return assistant text given conversation state."""
        ...

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        """
        Yield assistant text chunks for providers that support streaming.

        Implementations should raise ProviderError if streaming is not supported
        or fails.
        """
        ...

    # Optional async methods (providers can implement these for better async performance)
    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> str:
        """
        Async version of complete().
        
        Providers can implement this for native async support. If not implemented,
        the agent will fall back to running the sync version in an executor.
        """
        ...

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[str]:
        """
        Async version of stream().
        
        Providers can implement this for native async streaming support.
        """
        ...


__all__ = ["Provider", "ProviderError"]
