"""
Provider implementations for Anthropic, Gemini, and a local fallback.

These adapters validate configuration and can be mocked or monkeypatched by
callers. They surface clear errors when SDKs or API keys are missing.
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

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-3-5-sonnet-20240620",
        base_url: str | None = None,
    ):
        load_default_env()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set. Set it in env or pass api_key.")

        try:
            from anthropic import Anthropic
        except ImportError as exc:  # noqa: BLE001
            raise ProviderError("anthropic package not installed. Install with `pip install anthropic`.") from exc

        self._client = Anthropic(api_key=self.api_key, base_url=base_url)
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
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic streaming failed: {exc}") from exc

        for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta else None
                if text:
                    yield text

    def _format_messages(self, messages: List[Message]):
        formatted = []
        for message in messages:
            role = message.role.value
            if role == Role.TOOL.value:
                role = Role.ASSISTANT.value
            formatted.append({"role": role, "content": [{"type": "text", "text": message.content}]})
        return formatted


class GeminiProvider(Provider):
    """Google Gemini adapter using google-generativeai SDK."""

    name = "gemini"
    supports_streaming = True

    def __init__(self, api_key: str | None = None, default_model: str = "gemini-1.5-flash"):
        load_default_env()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set. Set it in env or pass api_key.")

        try:
            import google.generativeai as genai
        except ImportError as exc:  # noqa: BLE001
            raise ProviderError(
                "google-generativeai package not installed. Install with `pip install google-generativeai`."
            ) from exc

        genai.configure(api_key=self.api_key)
        self._genai = genai
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
        model_obj = self._genai.GenerativeModel(model or self.default_model)
        prompt_parts = self._build_prompt(system_prompt, messages)
        request_options = {"timeout": timeout} if timeout is not None else None
        try:
            response = model_obj.generate_content(
                prompt_parts,
                temperature=temperature,
                max_output_tokens=max_tokens,
                request_options=request_options,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini completion failed: {exc}") from exc

        return response.text or ""

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
        model_obj = self._genai.GenerativeModel(model or self.default_model)
        prompt_parts = self._build_prompt(system_prompt, messages)
        request_options = {"timeout": timeout} if timeout is not None else None
        try:
            stream = model_obj.generate_content(
                prompt_parts,
                temperature=temperature,
                max_output_tokens=max_tokens,
                request_options=request_options,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini streaming failed: {exc}") from exc

        for chunk in stream:
            if getattr(chunk, "text", None):
                yield chunk.text

    def _build_prompt(self, system_prompt: str, messages: List[Message]):
        conversation = [system_prompt]
        for message in messages:
            prefix = message.role.value.capitalize()
            conversation.append(f"{prefix}: {message.content}")
        return "\n".join(conversation)


class LocalProvider(Provider):
    """
    Local fallback provider.

    This does not call a model. It echoes the latest user content and is useful
    for offline/manual testing or as a safe default.
    """

    name = "local"
    supports_streaming = True

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


__all__ = ["AnthropicProvider", "GeminiProvider", "LocalProvider"]
