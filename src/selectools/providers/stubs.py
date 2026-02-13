"""
Local provider for offline testing and development.

This provider doesn't call any external API and simply echoes user messages.
"""

from __future__ import annotations

from typing import AsyncGenerator, AsyncIterable, Iterable, List

from ..types import Message, Role
from ..usage import UsageStats
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
    ) -> tuple[str, UsageStats]:
        last_user = next((m for m in reversed(messages) if m.role == Role.USER), None)
        user_text = last_user.content if last_user else ""
        response = f"[local provider: {model}] {user_text or 'No user message provided.'}"

        # Create dummy usage stats
        usage_stats = UsageStats(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            model=model,
            provider="local",
        )
        return response, usage_stats

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> Iterable[str]:
        text, _ = self.complete(
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
    ) -> tuple[str, UsageStats]:
        raise NotImplementedError("LocalProvider does not support async operations")

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[str]:
        raise NotImplementedError("LocalProvider does not support async operations")
        # Make mypy happy with async generator type
        if False:
            yield ""  # pragma: no cover


__all__ = ["LocalProvider"]
