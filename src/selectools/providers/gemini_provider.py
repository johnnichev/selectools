"""
Google Gemini provider adapter for the tool-calling library.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

from ..env import load_default_env
from ..types import Message, Role
from .base import Provider, ProviderError


class GeminiProvider(Provider):
    """Google Gemini adapter using google-generativeai SDK."""

    name = "gemini"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = "gemini-1.5-flash"):
        load_default_env()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set. Set it in env or pass api_key.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
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
        """
        Call Gemini's generate_content API for a non-streaming completion.

        Args:
            model: Model name (e.g., "gemini-1.5-flash")
            system_prompt: System-level instructions
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Optional request timeout in seconds

        Returns:
            The model's response text

        Raises:
            ProviderError: If the API call fails
        """
        model_obj = self._genai.GenerativeModel(model or self.default_model)

        # Build a single prompt that includes system instructions and conversation
        prompt_parts = self._build_prompt(system_prompt, messages)

        request_options = {"timeout": timeout} if timeout is not None else None

        try:
            response = model_obj.generate_content(
                prompt_parts,
                temperature=temperature,
                max_output_tokens=max_tokens,
                request_options=request_options,
            )
        except Exception as exc:
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
        """
        Stream responses from Gemini's generate_content API.

        Yields text chunks as they arrive from the API.
        """
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
        except Exception as exc:
            raise ProviderError(f"Gemini streaming failed: {exc}") from exc

        for chunk in stream:
            if getattr(chunk, "text", None):
                yield chunk.text

    def _build_prompt(self, system_prompt: str, messages: List[Message]):
        """
        Build a complete prompt for Gemini including system instructions and messages.

        Gemini doesn't have a system role, so we prepend instructions to the conversation.
        """
        conversation = [system_prompt]
        for message in messages:
            prefix = message.role.value.capitalize()
            conversation.append(f"{prefix}: {message.content}")
        return "\n".join(conversation)

    # Async methods (using ThreadPoolExecutor since Gemini SDK lacks native async)
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
        """
        Async version of complete() using ThreadPoolExecutor.

        Note: Gemini SDK doesn't natively support async, so we run the
        sync method in a thread pool executor.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.complete(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
            )
        return result

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
        """
        Async version of stream() using ThreadPoolExecutor.

        Note: Gemini SDK doesn't natively support async streaming, so we
        run chunks in a thread pool executor.
        """
        loop = asyncio.get_event_loop()

        # Create a sync generator and wrap it for async iteration
        sync_stream = self.stream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        with ThreadPoolExecutor() as executor:
            for chunk in sync_stream:
                # Yield each chunk (already in main thread, no executor needed)
                yield chunk


__all__ = ["GeminiProvider"]
