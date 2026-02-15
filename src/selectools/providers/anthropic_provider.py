"""
Anthropic provider adapter for the tool-calling library.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, cast

if TYPE_CHECKING:
    from ..tools.base import Tool

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import Anthropic as AnthropicModels
from ..pricing import calculate_cost
from ..types import Message, Role, ToolCall
from ..usage import UsageStats
from .base import Provider, ProviderError


class AnthropicProvider(Provider):
    """Anthropic Messages API adapter."""

    name = "anthropic"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = AnthropicModels.SONNET_4_5.id,
        base_url: str | None = None,
    ):
        load_default_env()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                provider_name="Anthropic",
                missing_config="API key",
                env_var="ANTHROPIC_API_KEY",
            )

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
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
        """
        Call Anthropic's messages API for a non-streaming completion.

        Args:
            model: Model name (e.g., "claude-sonnet-4-5")
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
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = {
            "model": model_name,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            request_args["tools"] = [self._map_tool_to_anthropic(t) for t in tools]

        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = self._client.messages.create(**request_args)  # type: ignore
        except Exception as exc:
            raise ProviderError(f"Anthropic completion failed: {exc}") from exc

        content_text = ""
        tool_calls: List[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_name=block.name,
                        parameters=cast(Dict[str, Any], block.input),
                        id=block.id,
                    )
                )

        # Extract usage stats
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            cost_usd=calculate_cost(
                model_name,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0,
            ),
            model=model_name,
            provider="anthropic",
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
        Stream responses from Anthropic's messages API.

        Yields text chunks as they arrive from the API.
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = {
            "model": model_name,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = self._client.messages.create(**request_args)  # type: ignore
        except Exception as exc:
            raise ProviderError(f"Anthropic streaming failed: {exc}") from exc

        for event in stream:
            event_type = getattr(event, "type", None)

            # Extract text deltas
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta else None
                if text:
                    yield text

    def _format_messages(self, messages: List[Message]) -> List[dict]:
        """
        Format messages for Anthropic's API.

        Anthropic expects messages with explicit content blocks and does not
        support the TOOL role, so we convert TOOL messages to ASSISTANT.
        """
        formatted = []
        for message in messages:
            role = message.role.value
            content: List[Dict[str, Any]] = []

            if role == Role.TOOL.value:
                # Map logical TOOL role to Anthropic USER role with tool_result block
                role = "user"
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "unknown",
                        "content": message.content,
                    }
                )
            else:
                # User or Assistant
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
                if message.content:
                    content.append({"type": "text", "text": message.content})

                # Check for outgoing tool calls (from Assistant)
                if message.tool_calls:
                    for tc in message.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id or "unknown",
                                "name": tc.tool_name,
                                "input": tc.parameters,
                            }
                        )

            formatted.append({"role": role, "content": content})
        return formatted

    def _map_tool_to_anthropic(self, tool: Tool) -> Dict[str, Any]:
        """Convert a selectools.Tool to Anthropic tool schema."""
        schema = tool.schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }

    # Async methods
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
        Async version of complete() using AsyncAnthropic client.

        Returns:
            Tuple of (response_text, usage_stats)
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = {
            "model": model_name,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            request_args["tools"] = [self._map_tool_to_anthropic(t) for t in tools]

        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = await self._async_client.messages.create(**request_args)  # type: ignore
        except Exception as exc:
            raise ProviderError(f"Anthropic async completion failed: {exc}") from exc

        content_text = ""
        tool_calls: List[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_name=block.name,
                        parameters=cast(Dict[str, Any], block.input),
                        id=block.id,
                    )
                )

        # Extract usage stats
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            cost_usd=calculate_cost(
                model_name,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0,
            ),
            model=model_name,
            provider="anthropic",
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
        """Async version of stream() using AsyncAnthropic client."""
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = {
            "model": model_name,
            "system": system_prompt,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = await self._async_client.messages.create(**request_args)  # type: ignore
        except Exception as exc:
            raise ProviderError(f"Anthropic async streaming failed: {exc}") from exc

        async for event in stream:
            event_type = getattr(event, "type", None)

            # Extract text deltas
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta else None
                if text:
                    yield text


__all__ = ["AnthropicProvider"]
