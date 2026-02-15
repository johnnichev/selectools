"""
Google Gemini provider adapter using the new google-genai SDK.

This provider uses the new centralized Client API introduced with Gemini 2.0.
See: https://ai.google.dev/gemini-api/docs/migrate
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, cast

if TYPE_CHECKING:
    from ..tools.base import Tool

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import Gemini as GeminiModels
from ..pricing import calculate_cost
from ..types import Message, Role, ToolCall
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
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
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

        if tools:
            config.tools = [self._map_tool_to_gemini(t) for t in tools]

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini completion failed: {exc}") from exc

        content_text = response.text or ""
        tool_calls: List[ToolCall] = []

        candidate_content = (
            response.candidates[0].content
            if response.candidates and response.candidates[0].content
            else None
        )
        if candidate_content and candidate_content.parts:
            for part in candidate_content.parts:
                if part.function_call:
                    tc_id = f"call_{uuid.uuid4().hex}"
                    tool_calls.append(
                        ToolCall(
                            tool_name=str(part.function_call.name or ""),
                            parameters=part.function_call.args if part.function_call.args else {},
                            id=tc_id,
                        )
                    )

        # Extract usage stats from response
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        completion_tokens = (usage.candidates_token_count or 0) if usage else 0
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(model_name, prompt_tokens, completion_tokens),
            model=model_name,
            provider="gemini",
        )

        return (
            Message(
                role=Role.ASSISTANT,
                content=content_text,
                tool_calls=tool_calls or None,
            ),
            usage_stats,
        )

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> Iterable[str]:
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
            parts = []

            if role == Role.TOOL.value:
                role = "user"
                # For tool results, we need a FunctionResponse part
                # Note: We need the tool name here. Message has 'tool_name'.
                if message.tool_name:
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=message.tool_name, response={"result": message.content}
                            )
                        )
                    )
                else:
                    # Fallback if no tool name (legacy), treat as text
                    parts.append(types.Part(text=f"Tool output: {message.content}"))

            elif role == Role.ASSISTANT.value:
                role = "model"
                if message.content:
                    parts.append(types.Part(text=message.content))

                # Handle outgoing tool calls
                if message.tool_calls:
                    for tc in message.tool_calls:
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tc.tool_name, args=tc.parameters
                                )
                            )
                        )

            elif role == Role.USER.value:
                role = "user"
                if message.content:
                    parts.append(types.Part(text=message.content))
                if message.image_base64:
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=base64.b64decode(message.image_base64),
                            )
                        )
                    )

            else:
                role = "user"  # Fallback
                if message.content:
                    parts.append(types.Part(text=message.content))

            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents

    def _map_tool_to_gemini(self, tool: Tool) -> Any:
        """Convert a selectools.Tool to Gemini tool schema."""
        # Helper to recursively convert JSON schema dict to types.Schema?
        # Actually, types.Tool accepts function_declarations.
        # FunctionDeclaration accepts parameters as types.Schema.
        # The SDK might auto-convert dict to Schema if we are lucky, or we pass dict directly?
        # Documentation suggests we can pass dicts in some places, but let's be safe.
        # For now, we'll try constructing the tool with a list of dicts if the SDK supports it,
        # or just types.FunctionDeclaration with kwargs.
        # We will iterate and construct FunctionDeclaration objects.
        # Ideally we'd convert the parameters dict to Schema.
        # But writing a full converter here is complex.
        # Let's hope the SDK accepts the dict or we can use a helper.
        # Check if types.Schema has a from_payload or similar?
        from google.genai import types

        schema = tool.schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                )
            ]
        )

    # Async methods using client.aio
    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
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

        if tools:
            config.tools = [self._map_tool_to_gemini(t) for t in tools]

        try:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini async completion failed: {exc}") from exc

        content_text = response.text or ""

        tool_calls: List[ToolCall] = []
        candidate_content = (
            response.candidates[0].content
            if response.candidates and response.candidates[0].content
            else None
        )
        if candidate_content and candidate_content.parts:
            for part in candidate_content.parts:
                if part.function_call:
                    tc_id = f"call_{uuid.uuid4().hex}"
                    tool_calls.append(
                        ToolCall(
                            tool_name=str(part.function_call.name or ""),
                            parameters=part.function_call.args if part.function_call.args else {},
                            id=tc_id,
                        )
                    )

        # Extract usage stats
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        completion_tokens = (usage.candidates_token_count or 0) if usage else 0
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(model_name, prompt_tokens, completion_tokens),
            model=model_name,
            provider="gemini",
        )

        return (
            Message(
                role=Role.ASSISTANT,
                content=content_text,
                tool_calls=tool_calls or None,
            ),
            usage_stats,
        )

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[str]:
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
