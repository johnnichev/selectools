"""
Anthropic provider adapter for the tool-calling library.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, cast

if TYPE_CHECKING:
    from ..tools.base import Tool

from typing import Union

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import Anthropic as AnthropicModels
from ..pricing import calculate_cost
from ..stability import stable
from ..types import Message, Role, ToolCall
from ..usage import UsageStats
from .base import Provider, ProviderError

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Claude-compatible endpoints sometimes emit reasoning inline as <think>
    tags rather than the native thinking content blocks. These must be
    stripped before persisting to conversation history to avoid polluting
    context on subsequent turns (Agno #6878).
    """
    if not text or "<think>" not in text:
        return text
    return _THINK_TAG_RE.sub("", text)


def _consume_think_buffer(buffer: str, in_think_block: bool) -> tuple[str, str, bool]:
    """Consume a streaming text buffer, suppressing <think> reasoning blocks.

    Returns ``(emit, remaining, in_think_block)`` where ``emit`` is the safe
    text to yield to the consumer, ``remaining`` is the unprocessed tail (a
    partial tag prefix or content inside an open block), and
    ``in_think_block`` is the updated state flag.

    The remaining buffer never includes safely emittable text, so the caller
    can yield ``emit`` immediately and re-feed new chunks into ``remaining``.
    """
    emit = ""
    while buffer:
        if in_think_block:
            close_idx = buffer.find(_THINK_CLOSE)
            if close_idx == -1:
                # Still inside the reasoning block; drop everything but
                # keep any suffix that could be the start of </think>.
                return emit, _retain_partial_suffix(buffer, _THINK_CLOSE), True
            buffer = buffer[close_idx + len(_THINK_CLOSE) :]
            in_think_block = False
            continue

        open_idx = buffer.find(_THINK_OPEN)
        if open_idx == -1:
            # No opening tag in buffer. Emit everything except a possible
            # partial-prefix tail (e.g. trailing "<th") that could become
            # a real opening tag once more text arrives.
            safe_len = len(buffer) - _partial_prefix_len(buffer, _THINK_OPEN)
            emit += buffer[:safe_len]
            return emit, buffer[safe_len:], False

        emit += buffer[:open_idx]
        buffer = buffer[open_idx + len(_THINK_OPEN) :]
        in_think_block = True
    return emit, "", in_think_block


def _partial_prefix_len(buffer: str, target: str) -> int:
    """Return length of longest suffix of ``buffer`` that prefixes ``target``.

    Used to hold back text that might be the start of a tag across chunks.
    """
    max_check = min(len(buffer), len(target) - 1)
    for size in range(max_check, 0, -1):
        if target.startswith(buffer[-size:]):
            return size
    return 0


def _retain_partial_suffix(buffer: str, target: str) -> str:
    """Return only the suffix of ``buffer`` that could be a prefix of ``target``.

    Used while inside a <think> block to drop suppressed text but preserve
    bytes that may complete a closing tag in the next chunk.
    """
    n = _partial_prefix_len(buffer, target)
    return buffer[-n:] if n else ""


@stable
class AnthropicProvider(Provider):
    """Anthropic Messages API adapter."""

    name = "anthropic"
    supports_streaming = True
    supports_async = True

    # Class-level defaults so instances created without __init__ (e.g. in
    # tests via __new__) keep the caching-off behavior.
    cache_system: bool = False
    cache_tools: bool = False



    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = AnthropicModels.SONNET_4_6.id,
        base_url: str | None = None,
        cache_system: bool = False,
        cache_tools: bool = False,
    ):
        """
        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY).
            default_model: Model used when calls omit an explicit model.
            base_url: Optional API base URL override.
            cache_system: Opt in to Anthropic prompt caching for the system
                prompt — sends ``system`` in block form with an ephemeral
                ``cache_control`` marker.
            cache_tools: Opt in to Anthropic prompt caching for tools — adds
                an ephemeral ``cache_control`` marker to the last tool, which
                caches the entire tool list.
        """
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
        self.cache_system = cache_system
        self.cache_tools = cache_tools



    def _build_request_args(
        self,
        *,
        model_name: str,
        system_prompt: str,
        payload: List[dict],
        temperature: float,
        max_tokens: int,
        tools: List[Tool] | None,
        timeout: float | None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        request_args: Dict[str, Any] = {
            "model": model_name,
            "system": self._system_param(system_prompt),
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stream:
            request_args["stream"] = True

        if tools:
            request_args["tools"] = self._tools_param(tools)

        if timeout is not None:
            request_args["timeout"] = timeout

        return request_args

    def _message_from_content_blocks(
        self,
        blocks: Any,
    ) -> tuple[str, List[ToolCall]]:
        content_text = ""
        tool_calls: List[ToolCall] = []

        for block in blocks:
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

        return _strip_reasoning_tags(content_text), tool_calls    


    def _usage_from_response(
        self,
        response: Any,
        model_name: str,
    ) -> UsageStats:
        usage = response.usage

        cache_creation_tokens = self._cache_usage_token(
            usage,
            "cache_creation_input_tokens",
        )

        cache_read_tokens = self._cache_usage_token(
            usage,
            "cache_read_input_tokens",
        )

        return UsageStats(
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            cost_usd=calculate_cost(
                model_name,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0,
                cache_read_input_tokens=cache_read_tokens or 0,
                cache_creation_input_tokens=cache_creation_tokens or 0,
            ),
            model=model_name,
            provider="anthropic",
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
        )


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
            Tuple of (response_message, usage_stats)

        Raises:
            ProviderError: If the API call fails
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = self._build_request_args(
            model_name=model_name,
            system_prompt=system_prompt,
            payload=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
        )
        if tools:
            request_args["tools"] = self._tools_param(tools)

        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = self._client.messages.create(**request_args)  # type: ignore[call-overload]
        except Exception as exc:
            raise ProviderError(f"Anthropic completion failed: {exc}") from exc

        content_text, tool_calls = self._message_from_content_blocks(
            response.content
        )

        content_text = _strip_reasoning_tags(content_text)

        # Extract usage stats. Cache tokens are billed separately from
        # input_tokens (reads at 0.1x, 5-min-TTL writes at 1.25x the prompt
        # rate), so they feed into calculate_cost as well.
        usage_stats = self._usage_from_response(
            response,
            model_name,
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
    ) -> Iterable[Union[str, ToolCall]]:
        """
        Stream responses from Anthropic's messages API.

        Yields:
            str: Text content deltas.
            ToolCall: Complete tool call objects when a tool_use block finishes.
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = self._build_request_args(
            model_name=model_name,
            system_prompt=system_prompt,
            payload=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            stream=True,
        )
        if tools:
            request_args["tools"] = self._tools_param(tools)
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = self._client.messages.create(**request_args)  # type: ignore[call-overload]
        except Exception as exc:
            raise ProviderError(f"Anthropic streaming failed: {exc}") from exc

        current_tool_id: str | None = None
        current_tool_name: str = ""
        current_tool_json: str = ""
        text_buffer: str = ""
        in_think_block: bool = False

        try:
            for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if not delta:
                        continue
                    delta_type = getattr(delta, "type", None)

                    if delta_type == "text_delta":
                        text = getattr(delta, "text", None)
                        if text:
                            text_buffer += text
                            emit, text_buffer, in_think_block = _consume_think_buffer(
                                text_buffer, in_think_block
                            )
                            if emit:
                                yield emit
                    elif delta_type == "input_json_delta":
                        partial = getattr(delta, "partial_json", None)
                        if partial:
                            current_tool_json += partial

                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        current_tool_id = getattr(block, "id", None)
                        current_tool_name = getattr(block, "name", "") or ""
                        current_tool_json = ""

                elif event_type == "content_block_stop":
                    if current_tool_name:
                        from ._openai_compat import _parse_tool_args

                        params, parse_error = _parse_tool_args(current_tool_json)
                        yield ToolCall(
                            tool_name=current_tool_name,
                            parameters=params,
                            id=current_tool_id or "",
                            parse_error=parse_error,
                        )
                        current_tool_id = None
                        current_tool_name = ""
                        current_tool_json = ""
            # Flush any trailing buffered text after stream ends.
            if text_buffer and not in_think_block:
                tail = _strip_reasoning_tags(text_buffer)
                if tail:
                    yield tail
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Anthropic streaming failed mid-stream: {exc}") from exc

    def _format_messages(self, messages: List[Message]) -> List[dict]:
        """
        Format messages for Anthropic's API.

        Anthropic expects messages with explicit content blocks and does not
        support the TOOL role, so we convert TOOL messages to user messages
        with ``tool_result`` blocks.

        SYSTEM messages are converted to user messages but are collected
        separately and prepended at the start of the formatted list to
        prevent them from breaking the required assistant(tool_use) ->
        user(tool_result) adjacency that Anthropic enforces.
        """
        formatted = []
        system_converted: List[dict] = []

        for message in messages:
            role = message.role.value
            content: List[Dict[str, Any]] = []

            if role == Role.SYSTEM.value:
                # Anthropic rejects "system" role in messages — it must be the
                # top-level `system` parameter.  Context injections (compressed
                # context, entity memory, knowledge graph) arrive as SYSTEM
                # messages in history; convert to user messages.  We collect
                # them and prepend later so they never sit between an assistant
                # tool_use and the matching user tool_result.
                system_converted.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": message.content or ""}],
                    }
                )
                continue

            if role == Role.TOOL.value:
                # Map logical TOOL role to Anthropic USER role with tool_result block
                role = "user"
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "unknown",
                        "content": message.content or "",
                    }
                )
            else:
                # User or Assistant.
                # Prefer the v0.21.0 multimodal ``content_parts`` path: when
                # the message was built via ``image_message()`` the image
                # lives in a ContentPart (not in the legacy
                # ``message.image_base64`` attribute, which is explicitly
                # None for multimodal messages). Fall back to the legacy
                # path for pre-0.21 callers.
                if getattr(message, "content_parts", None):
                    for cp in message.content_parts:  # type: ignore[union-attr]
                        if cp.type == "text" and cp.text:
                            content.append({"type": "text", "text": cp.text})
                        elif cp.type == "image_url" and cp.image_url:
                            content.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": cp.image_url},
                                }
                            )
                        elif cp.type == "image_base64" and cp.image_base64:
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": cp.media_type or "image/png",
                                        "data": cp.image_base64,
                                    },
                                }
                            )
                else:
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

                # Anthropic API rejects empty content lists for assistant messages
                if not content:
                    content = [{"type": "text", "text": ""}]

            formatted.append({"role": role, "content": content})

        # Prepend converted SYSTEM messages at the start so they never break
        # tool_use -> tool_result adjacency.
        if system_converted:
            formatted = system_converted + formatted

        # Merge consecutive same-role messages (required by Anthropic API).
        # When the assistant triggers multiple parallel tool calls, each TOOL
        # message becomes a separate {"role": "user", "content": [tool_result]}
        # entry.  Anthropic rejects consecutive same-role messages, so we must
        # collapse them into a single message with a combined content list.
        merged: List[dict] = []
        for msg in formatted:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                curr_content = msg["content"]
                if isinstance(prev_content, list) and isinstance(curr_content, list):
                    merged[-1]["content"] = prev_content + curr_content
                elif isinstance(prev_content, str) and isinstance(curr_content, str):
                    merged[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list):
                    if isinstance(curr_content, str):
                        merged[-1]["content"] = prev_content + [
                            {"type": "text", "text": curr_content}
                        ]
                    else:
                        merged[-1]["content"] = prev_content + list(curr_content)
                else:
                    if isinstance(curr_content, list):
                        merged[-1]["content"] = [{"type": "text", "text": prev_content}] + list(
                            curr_content
                        )
                    else:
                        merged[-1]["content"] = [
                            {"type": "text", "text": prev_content},
                            {"type": "text", "text": curr_content},
                        ]
            else:
                merged.append(msg)
        formatted = merged

        return formatted

    def _system_param(self, system_prompt: str) -> Union[str, List[Dict[str, Any]]]:
        """Build the top-level ``system`` field.

        When ``cache_system`` is enabled the prompt is sent in block form
        with an ephemeral ``cache_control`` marker so Anthropic caches it.
        """
        if self.cache_system:
            return [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system_prompt

    def _tools_param(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Map tools to Anthropic schema, marking the last one for caching.

        Anthropic caches everything up to the ``cache_control`` marker, so
        a single marker on the final tool caches the entire tool list.
        """
        mapped = [self._map_tool_to_anthropic(t) for t in tools]
        if self.cache_tools and mapped:
            mapped[-1]["cache_control"] = {"type": "ephemeral"}
        return mapped

    @staticmethod
    def _cache_usage_token(usage: Any, attr: str) -> int | None:
        """Read an optional cache token count from a response usage object."""
        value = getattr(usage, attr, None) if usage else None
        return value if isinstance(value, int) else None

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
            Tuple of (response_message, usage_stats)
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = self._build_request_args(
            model_name=model_name,
            system_prompt=system_prompt,
            payload=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
        )
        if tools:
            request_args["tools"] = self._tools_param(tools)

        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            response = await self._async_client.messages.create(**request_args)  # type: ignore[call-overload]
        except Exception as exc:
            raise ProviderError(f"Anthropic async completion failed: {exc}") from exc

        content_text, tool_calls = self._message_from_content_blocks(
            response.content
        )

        content_text = _strip_reasoning_tags(content_text)

        # Extract usage stats. Cache tokens are billed separately from
        # input_tokens (reads at 0.1x, 5-min-TTL writes at 1.25x the prompt
        # rate), so they feed into calculate_cost as well.
        usage_stats = self._usage_from_response(
            response,
            model_name,
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
    ) -> AsyncIterable[Union[str, ToolCall]]:
        """
        Async streaming with native tool call support.

        Yields:
            str: Text content deltas
            ToolCall: Complete tool call objects when a tool_use block finishes
        """
        payload = self._format_messages(messages)
        model_name = model or self.default_model
        request_args = self._build_request_args(
            model_name=model_name,
            system_prompt=system_prompt,
            payload=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            stream=True,
        )
        if tools:
            request_args["tools"] = self._tools_param(tools)
        if timeout is not None:
            request_args["timeout"] = timeout
        try:
            stream = await self._async_client.messages.create(**request_args)  # type: ignore[call-overload]
        except Exception as exc:
            raise ProviderError(f"Anthropic async streaming failed: {exc}") from exc

        current_tool_id: str | None = None
        current_tool_name: str = ""
        current_tool_json: str = ""
        text_buffer: str = ""
        in_think_block: bool = False

        try:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if not delta:
                        continue
                    delta_type = getattr(delta, "type", None)

                    if delta_type == "text_delta":
                        text = getattr(delta, "text", None)
                        if text:
                            text_buffer += text
                            emit, text_buffer, in_think_block = _consume_think_buffer(
                                text_buffer, in_think_block
                            )
                            if emit:
                                yield emit
                    elif delta_type == "input_json_delta":
                        partial = getattr(delta, "partial_json", None)
                        if partial:
                            current_tool_json += partial

                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        current_tool_id = getattr(block, "id", None)
                        current_tool_name = getattr(block, "name", "") or ""
                        current_tool_json = ""

                elif event_type == "content_block_stop":
                    if current_tool_name:
                        from ._openai_compat import _parse_tool_args

                        params, parse_error = _parse_tool_args(current_tool_json)
                        yield ToolCall(
                            tool_name=current_tool_name,
                            parameters=params,
                            id=current_tool_id or "",
                            parse_error=parse_error,
                        )
                        current_tool_id = None
                        current_tool_name = ""
                        current_tool_json = ""
            # Flush any trailing buffered text after stream ends.
            if text_buffer and not in_think_block:
                tail = _strip_reasoning_tags(text_buffer)
                if tail:
                    yield tail
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Anthropic async streaming failed mid-stream: {exc}") from exc


__stability__ = "stable"

__all__ = ["AnthropicProvider"]
