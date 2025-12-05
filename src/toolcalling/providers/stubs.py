"""
Stub provider implementations to illustrate the adapter interface.
"""

from __future__ import annotations

from .base import Provider, ProviderError
from ..types import Message


class _UnimplementedProvider(Provider):
    name = "unimplemented"

    def __init__(self, provider_name: str):
        self.name = provider_name

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        raise ProviderError(
            f"{self.name} adapter is a stub. Implement complete() to enable this provider."
        )


class AnthropicProvider(_UnimplementedProvider):
    def __init__(self):
        super().__init__("anthropic")


class GeminiProvider(_UnimplementedProvider):
    def __init__(self):
        super().__init__("gemini")


class LocalProvider(_UnimplementedProvider):
    def __init__(self):
        super().__init__("local")


__all__ = ["AnthropicProvider", "GeminiProvider", "LocalProvider"]
