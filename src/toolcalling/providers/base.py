"""
Provider abstraction for model-agnostic tool calling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Message


class ProviderError(RuntimeError):
    """Raised when an adapter cannot complete a request."""


@runtime_checkable
class Provider(Protocol):
    """Interface every provider adapter must satisfy."""

    name: str

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """Return assistant text given conversation state."""
        ...


__all__ = ["Provider", "ProviderError"]
