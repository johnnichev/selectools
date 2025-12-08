"""
Anthropic provider adapter for the tool-calling library.
"""

from __future__ import annotations

import os
from typing import Iterable, List

from ..env import load_default_env
from ..types import Message, Role
from .base import Provider, ProviderError


class AnthropicProvider(Provider):
    """Anthropic Messages API adapter."""

    name = "anthropic"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
        base_url: str | None = None,
    ):
        load_default_env()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set. Set it in env or pass api_key.")

        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError as exc:
            raise ProviderError(
                "anthropic package not installed. Install with `pip install anthropic`."
            ) from exc

        self._client = Anthropic(api_key=self.api_key, base_url=base_url)
        self._async_client = AsyncAnthropic(api_key=self.api_key, base_url=base_url)
        self.default_model = default_model

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
        """
        Call Anthropic's messages API for a non-streaming completion.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            system_prompt: System-level instructions
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Optional request timeout in seconds

        Returns:
            The assistant's response text

        Raises:
            ProviderError: If the API call fails
        """
        payload = self._format_messages(messages)
        request_args = {
            "model": model or self.default_model,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = self._client.messages.create(**request_args)
        except Exception as exc:
            raise ProviderError(f"Anthropic completion failed: {exc}") from exc

        text_chunks = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(text_chunks)

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
        """
        Stream responses from Anthropic's messages API.

        Yields text chunks as they arrive from the API.
        """
        payload = self._format_messages(messages)
        request_args = {
            "model": model or self.default_model,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = self._client.messages.create(**request_args)
        except Exception as exc:
            raise ProviderError(f"Anthropic streaming failed: {exc}") from exc

        for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta else None
                if text:
                    yield text

    def _format_messages(self, messages: List[Message]):
        """
        Format messages for Anthropic's API.

        Anthropic expects messages with explicit content blocks and does not
        support the TOOL role, so we convert TOOL messages to ASSISTANT.
        """
        formatted = []
        for message in messages:
            role = message.role.value
            if role == Role.TOOL.value:
                role = Role.ASSISTANT.value

            # Support vision by checking for image_base64
            content = []
            if message.image_base64:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": message.image_base64,
                        },
                    }
                )
            content.append({"type": "text", "text": message.content})

            formatted.append({"role": role, "content": content})
        return formatted

    # Async methods
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
        """Async version of complete() using AsyncAnthropic client."""
        payload = self._format_messages(messages)
        request_args = {
            "model": model or self.default_model,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = await self._async_client.messages.create(**request_args)
        except Exception as exc:
            raise ProviderError(f"Anthropic async completion failed: {exc}") from exc

        text_chunks = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(text_chunks)

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        """Async version of stream() using AsyncAnthropic client."""
        payload = self._format_messages(messages)
        request_args = {
            "model": model or self.default_model,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = await self._async_client.messages.create(**request_args)
        except Exception as exc:
            raise ProviderError(f"Anthropic async streaming failed: {exc}") from exc

        async for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta else None
                if text:
                    yield text


__all__ = ["AnthropicProvider"]
