"""
OpenAI provider adapter for the tool-calling library.
"""

from __future__ import annotations

import os
from typing import List

from ..env import load_default_env
from ..types import Message, Role
from .base import Provider, ProviderError


class OpenAIProvider(Provider):
    """Adapter that speaks to OpenAI's Chat Completions API."""

    name = "openai"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-4o"):
        load_default_env()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set. Set it in env or pass api_key.")

        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)
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
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)

        try:
            response = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI completion failed: {exc}") from exc

        content = response.choices[0].message.content
        return content or ""

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
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        try:
            response = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI streaming failed: {exc}") from exc

        for chunk in response:
            try:
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue
                content = delta.content
                if isinstance(content, list):
                    content = "".join(
                        [part.text for part in content if getattr(part, "text", None)]
                    )
                yield content
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"OpenAI stream parsing failed: {exc}") from exc

    def _format_messages(self, system_prompt: str, messages: List[Message]):
        payload = [{"role": "system", "content": system_prompt}]
        for message in messages:
            role = message.role.value
            if role == Role.TOOL.value:
                role = Role.ASSISTANT.value
            payload.append(
                {
                    "role": role,
                    "content": self._format_content(message),
                }
            )
        return payload

    def _format_content(self, message: Message):
        if message.image_base64:
            return [
                {"type": "text", "text": message.content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{message.image_base64}"},
                },
            ]
        return message.content

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
        """Async version of complete() using AsyncOpenAI client."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)

        try:
            response = await self._async_client.chat.completions.create(
                model=model or self.default_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI async completion failed: {exc}") from exc

        content = response.choices[0].message.content
        return content or ""

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
        """Async version of stream() using AsyncOpenAI client."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        try:
            response = await self._async_client.chat.completions.create(
                model=model or self.default_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=timeout,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI async streaming failed: {exc}") from exc

        async for chunk in response:
            try:
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue
                content = delta.content
                if isinstance(content, list):
                    content = "".join(
                        [part.text for part in content if getattr(part, "text", None)]
                    )
                yield content
            except Exception as exc:
                raise ProviderError(f"OpenAI async stream parsing failed: {exc}") from exc


__all__ = ["OpenAIProvider"]
