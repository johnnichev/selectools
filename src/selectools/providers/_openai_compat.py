"""
Shared base class for providers that use the OpenAI Python SDK.

Both OpenAI and Ollama speak the same OpenAI Chat Completions API.  This
module extracts the ~95 % of code that is identical between the two into
a single ``_OpenAICompatibleBase`` that each provider subclasses.

Subclasses override a handful of *template methods* to supply
provider-specific behaviour (pricing, error wrapping, token-key
selection, tool-call ID handling).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, Union, cast

from ..types import Message, Role, ToolCall
from ..usage import UsageStats
from .base import Provider, ProviderError

if TYPE_CHECKING:
    from ..tools.base import Tool


class _OpenAICompatibleBase(ABC):
    """Base for providers backed by the OpenAI Python SDK.

    Concrete subclasses must set the following attributes on ``self``
    during ``__init__``:

    * ``_client``       – a synchronous ``openai.OpenAI`` instance
    * ``_async_client`` – an ``openai.AsyncOpenAI`` instance
    * ``default_model`` – the fallback model name (``str``)

    They must also implement the abstract template methods listed below.
    """

    name: str
    supports_streaming: bool = True
    supports_async: bool = True

    # -- template methods that subclasses override ---------------------------

    @abstractmethod
    def _get_token_key(self, model: str) -> str:
        """Return the request kwarg name for the token limit.

        OpenAI returns ``"max_completion_tokens"`` for newer models,
        ``"max_tokens"`` otherwise.  Ollama always returns ``"max_tokens"``.
        """
        ...

    @abstractmethod
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the estimated USD cost for the request."""
        ...

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Short identifier (``"openai"`` or ``"ollama"``)."""
        ...

    @abstractmethod
    def _wrap_error(self, exc: Exception, operation: str) -> ProviderError:
        """Convert a raw SDK exception into a ``ProviderError``."""
        ...

    @abstractmethod
    def _parse_tool_call_id(self, tc: Any) -> str:
        """Extract / generate the tool-call ID from a raw SDK tool-call object."""
        ...

    # -- optional hooks -------------------------------------------------------

    def _parse_tool_call_arguments(self, tc: Any) -> dict:
        """Parse tool-call arguments from the SDK object.

        The default implementation handles the common case where arguments
        are always a JSON string (OpenAI).  Ollama overrides this to also
        handle the case where arguments may already be a ``dict``.
        """
        try:
            return json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            return {}

    def _build_astream_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Allow subclasses to inject extra kwargs for ``astream()``.

        OpenAI adds ``stream_options`` for usage tracking; Ollama does not.
        """
        return args

    # -- shared implementation ------------------------------------------------

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

        token_key = self._get_token_key(model_name)
        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            token_key: max_tokens,
        }
        if timeout is not None:
            args["timeout"] = timeout

        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        try:
            response = cast(Any, self._client.chat.completions.create(**args))
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "completion") from exc

        return self._parse_response(response, model_name)

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
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        token_key = self._get_token_key(model_name)
        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            token_key: max_tokens,
        }
        if timeout is not None:
            args["timeout"] = timeout

        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        try:
            response = cast(Any, await self._async_client.chat.completions.create(**args))
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "async completion") from exc

        return self._parse_response(response, model_name)

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
    ) -> Iterable[Union[str, ToolCall]]:
        """Stream response chunks with tool call support.

        Yields:
            str: Text content deltas.
            ToolCall: Complete tool call objects when all argument chunks arrive.
        """
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        token_key = self._get_token_key(model_name)
        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            token_key: max_tokens,
            "stream": True,
        }
        if timeout is not None:
            args["timeout"] = timeout
        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        try:
            response = cast(Any, self._client.chat.completions.create(**args))
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "streaming") from exc

        tool_call_deltas: Dict[int, Dict[str, Any]] = {}

        try:
            for chunk in response:
                try:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Text content
                    if delta.content:
                        content = delta.content
                        if isinstance(content, list):
                            content = "".join(
                                [part.text for part in content if getattr(part, "text", None)]
                            )
                        yield content

                    # Tool calls
                    if getattr(delta, "tool_calls", None):
                        for tc_delta in delta.tool_calls:
                            index = tc_delta.index
                            if index not in tool_call_deltas:
                                tool_call_deltas[index] = {
                                    "id": self._initial_tool_call_id(tc_delta),
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.id:
                                tool_call_deltas[index]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                tool_call_deltas[index]["name"] = tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_call_deltas[index]["arguments"] += tc_delta.function.arguments

                    # Emit completed tool calls at end of stream
                    finish = chunk.choices[0].finish_reason if chunk.choices else None
                    if finish in ("tool_calls", "stop") and tool_call_deltas:
                        for tc_data in tool_call_deltas.values():
                            try:
                                params = (
                                    json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                                )
                            except json.JSONDecodeError:
                                params = {}
                            yield ToolCall(
                                tool_name=tc_data["name"],
                                parameters=params,
                                id=tc_data["id"],
                            )
                        tool_call_deltas.clear()

                except ProviderError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise self._wrap_error(exc, "stream parsing") from exc
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "streaming failed mid-stream") from exc

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
    ) -> AsyncIterable[Union[str, ToolCall]]:
        """Async streaming with native tool call support."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        token_key = self._get_token_key(model_name)
        args: Dict[str, Any] = {
            "model": model_name,
            "messages": cast(Any, formatted),
            "temperature": temperature,
            token_key: max_tokens,
            "stream": True,
        }
        if timeout is not None:
            args["timeout"] = timeout
        if tools:
            args["tools"] = [self._map_tool_to_openai(t) for t in tools]

        args = self._build_astream_args(args)

        try:
            response = cast(Any, await self._async_client.chat.completions.create(**args))
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "async streaming") from exc

        # Track partial tool calls
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}

        try:
            async for chunk in response:
                try:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # 1. Handle text content
                    if delta.content:
                        yield delta.content

                    # 2. Handle tool calls
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            index = tc_delta.index
                            if index not in tool_call_deltas:
                                tool_call_deltas[index] = {
                                    "id": self._initial_tool_call_id(tc_delta),
                                    "name": "",
                                    "arguments": "",
                                }

                            if tc_delta.id:
                                tool_call_deltas[index]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call_deltas[index]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_call_deltas[index][
                                        "arguments"
                                    ] += tc_delta.function.arguments

                    # Check for finish reason to emit completed tool calls
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason in ("tool_calls", "stop") and tool_call_deltas:
                        for index in sorted(tool_call_deltas.keys()):
                            tc_info = tool_call_deltas[index]
                            try:
                                params = (
                                    json.loads(tc_info["arguments"]) if tc_info["arguments"] else {}
                                )
                            except json.JSONDecodeError:
                                params = {}

                            yield ToolCall(
                                tool_name=tc_info["name"],
                                parameters=params,
                                id=tc_info["id"],
                            )
                        tool_call_deltas = {}  # Clear for next iteration if any

                except ProviderError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise self._wrap_error(exc, "async stream parsing") from exc
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc, "async streaming failed mid-stream") from exc

    # -- message formatting (identical for OpenAI and Ollama) -----------------

    def _format_messages(self, system_prompt: str, messages: List[Message]) -> List[dict]:
        payload: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            role = message.role.value

            if role == Role.TOOL.value:
                payload.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id or "unknown",
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
                            "id": self._format_tool_call_id(tc),
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
                {"type": "text", "text": message.content or ""},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{message.image_base64}"},
                },
            ]
        return message.content or ""

    def _map_tool_to_openai(self, tool: Tool) -> Dict[str, Any]:
        """Convert a selectools.Tool to OpenAI tool schema."""
        return {
            "type": "function",
            "function": tool.schema(),
        }

    # -- response parsing (shared) --------------------------------------------

    def _parse_response(self, response: Any, model_name: str) -> tuple[Message, UsageStats]:
        """Parse an OpenAI-compatible chat completion response."""
        message = response.choices[0].message
        content = message.content
        tool_calls: List[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                params = self._parse_tool_call_arguments(tc)
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        parameters=params,
                        id=self._parse_tool_call_id(tc),
                    )
                )

        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=self._calculate_cost(
                model_name,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            model=model_name,
            provider=self._get_provider_name(),
        )

        return (
            Message(
                role=Role.ASSISTANT,
                content=content or "",
                tool_calls=tool_calls or None,
            ),
            usage_stats,
        )

    # -- tool-call ID helpers -------------------------------------------------

    def _format_tool_call_id(self, tc: ToolCall) -> str:
        """Format a tool-call ID for the messages payload.

        OpenAI always has an ID.  Ollama may not, so the subclass overrides.
        """
        return tc.id or f"call_{id(tc)}"

    def _initial_tool_call_id(self, tc_delta: Any) -> str:
        """Provide the initial tool-call ID for a streaming delta.

        OpenAI always supplies ``tc_delta.id``.  Ollama may not.
        """
        return tc_delta.id


__all__ = ["_OpenAICompatibleBase"]
