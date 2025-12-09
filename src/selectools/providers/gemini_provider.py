"""
Google Gemini provider adapter using the new google-genai SDK.

This provider uses the new centralized Client API introduced with Gemini 2.0.
See: https://ai.google.dev/gemini-api/docs/migrate
"""

from __future__ import annotations

import os
from typing import List

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import Gemini as GeminiModels
from ..pricing import calculate_cost
from ..types import Message, Role
from ..usage import UsageStats
from .base import Provider, ProviderError


class GeminiProvider(Provider):
    """
    Google Gemini adapter using the new google-genai SDK.

    This implementation uses the centralized Client API pattern:
    - client.models.generate_content() for non-streaming
    - client.models.generate_content_stream() for streaming
    - client.aio.models.generate_content() for async
    """

    name = "gemini"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = GeminiModels.FLASH_2_0.id):
        load_default_env()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                provider_name="Gemini",
                missing_config="API key",
                env_var="GEMINI_API_KEY",
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError(
                "google-genai package not installed. Install with `pip install google-genai`."
            ) from exc

        # Create the centralized client
        self._client = genai.Client(api_key=self.api_key)
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
    ) -> tuple[str, UsageStats]:
        """
        Call Gemini's generate_content API for a non-streaming completion.

        Args:
            model: Model name (e.g., "gemini-2.0-flash")
            system_prompt: System-level instructions
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Optional request timeout in seconds

        Returns:
            Tuple of (response_text, usage_stats)

        Raises:
            ProviderError: If the API call fails
        """
        from google.genai import types

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini completion failed: {exc}") from exc

        content = response.text or ""

        # Extract usage stats from response
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(model_name, prompt_tokens, completion_tokens),
            model=model_name,
            provider="gemini",
        )

        return content, usage_stats

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
        """
        Stream responses from Gemini's generate_content_stream API.

        Yields text chunks as they arrive from the API.
        """
        from google.genai import types

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            stream = self._client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini streaming failed: {exc}") from exc

        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _format_contents(self, system_prompt: str, messages: List[Message]) -> List:
        """
        Format messages for Gemini's API.

        Converts our Message format to Gemini's content format.
        System instructions are handled separately via config.
        """
        from google.genai import types

        contents = []
        for message in messages:
            role = message.role.value
            # Map our roles to Gemini roles
            if role == Role.ASSISTANT.value or role == Role.TOOL.value:
                role = "model"
            elif role == Role.USER.value:
                role = "user"
            else:
                role = "user"  # Default fallback

            # Build parts
            parts = []
            if message.content:
                parts.append(types.Part(text=message.content))

            # Handle images if present
            if message.image_base64:
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=message.image_base64,
                        )
                    )
                )

            contents.append(types.Content(role=role, parts=parts))

        return contents

    # Async methods using client.aio
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
        """
        Async version of complete() using client.aio.

        Returns:
            Tuple of (response_text, usage_stats)
        """
        from google.genai import types

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini async completion failed: {exc}") from exc

        content = response.text or ""

        # Extract usage stats
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(model_name, prompt_tokens, completion_tokens),
            model=model_name,
            provider="gemini",
        )

        return content, usage_stats

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
        Async version of stream() using client.aio.

        Yields text chunks as they arrive from the API.
        """
        from google.genai import types

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini async streaming failed: {exc}") from exc

        async for chunk in stream:
            if chunk.text:
                yield chunk.text


__all__ = ["GeminiProvider"]
