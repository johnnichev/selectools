"""
Local provider for offline testing and development.

This provider doesn't call any external API and simply echoes user messages.
"""

from __future__ import annotations

from typing import Iterable, List

from ..types import Message, Role
from .base import Provider


class LocalProvider(Provider):
    """
    Local fallback provider.

    This does not call a model. It echoes the latest user content and is useful
    for offline/manual testing or as a safe default.
    """

    name = "local"
    supports_streaming = True
    supports_async = False  # LocalProvider is sync-only

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> str:
        last_user = next((m for m in reversed(messages) if m.role == Role.USER), None)
        user_text = last_user.content if last_user else ""
        return f"[local provider: {model}] {user_text or 'No user message provided.'}"

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        text = self.complete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        for token in text.split():
            yield token + " "

    # Async methods not supported for LocalProvider
    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> str:
        raise NotImplementedError("LocalProvider does not support async operations")

    def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        raise NotImplementedError("LocalProvider does not support async operations")
        # Make mypy happy with async generator type
        yield ""  # pragma: no cover


__all__ = ["LocalProvider"]
