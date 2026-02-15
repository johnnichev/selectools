"""
OpenAI provider adapter for the tool-calling library.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, cast

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import OpenAI as OpenAIModels
from ..pricing import calculate_cost
from ..types import Message, Role, ToolCall
from ..usage import UsageStats

if TYPE_CHECKING:
    from ..tools.base import Tool

from .base import Provider, ProviderError


class OpenAIProvider(Provider):
    """Adapter that speaks to OpenAI's Chat Completions API."""

    name = "openai"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = OpenAIModels.GPT_4O.id):
        load_default_env()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                provider_name="OpenAI",
                missing_config="API key",
                env_var="OPENAI_API_KEY",
            )

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
        tools: List[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        try:
            response = cast(Any, self._client.chat.completions.create(**args))
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI completion failed: {exc}") from exc

        message = response.choices[0].message
        content = message.content
        tool_calls: List[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        parameters=params,
                        id=tc.id,
                    )
                )

        # Extract usage stats
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=calculate_cost(
                model_name,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            model=model_name,
            provider="openai",
        )

        return (
            Message(
                role=Role.ASSISTANT,
                content=content or "",
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
        """Stream response chunks. Note: Does not return usage stats."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = cast(
                Any,
                self._client.chat.completions.create(
                    model=model_name,
                    messages=cast(Any, formatted),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    timeout=timeout,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI streaming failed: {exc}") from exc

        for chunk in response:
            try:
                delta = chunk.choices[0].delta if chunk.choices else None
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

    def _format_messages(self, system_prompt: str, messages: List[Message]) -> List[dict]:
        payload: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            role = message.role.value

            if role == Role.TOOL.value:
                payload.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            elif role == Role.ASSISTANT.value:
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": self._format_content(message),
                }
                if message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.parameters),
                            },
                        }
                        for tc in message.tool_calls
                    ]
                payload.append(msg_dict)
            else:
                # User role
                payload.append(
                    {
                        "role": role,
                        "content": self._format_content(message),
                    }
                )
        return payload

    def _format_content(self, message: Message) -> str | List[Any]:
        if message.image_base64:
            return [
                {"type": "text", "text": message.content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{message.image_base64}"},
                },
            ]
        return message.content

    def _map_tool_to_openai(self, tool: Tool) -> Dict[str, Any]:
        """Convert a selectools.Tool to OpenAI tool schema."""
        return {
            "type": "function",
            "function": tool.schema(),
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
        """Async version of complete() using AsyncOpenAI client."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        try:
            response = cast(Any, await self._async_client.chat.completions.create(**args))
        except Exception as exc:
            raise ProviderError(f"OpenAI async completion failed: {exc}") from exc

        message = response.choices[0].message
        content = message.content
        tool_calls: List[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        parameters=params,
                        id=tc.id,
                    )
                )

        # Extract usage stats
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=calculate_cost(
                model_name,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            model=model_name,
            provider="openai",
        )

        return (
            Message(
                role=Role.ASSISTANT,
                content=content or "",
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
        Async version of stream() using AsyncOpenAI client.

        Note: This is an async generator that yields chunks. Due to Python limitations,
        async generators cannot return values, so usage stats are not returned.
        Use acomplete() if you need usage stats.
        """
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = cast(
                Any,
                await self._async_client.chat.completions.create(
                    model=model_name,
                    messages=cast(Any, formatted),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},  # Get usage stats with streaming
                    timeout=timeout,
                ),
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI async streaming failed: {exc}") from exc

        async for chunk in response:
            try:
                delta = chunk.choices[0].delta if chunk.choices else None
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
