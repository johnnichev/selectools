"""
Google Gemini provider adapter using the new google-genai SDK.

This provider uses the new centralized Client API introduced with Gemini 2.0.
See: https://ai.google.dev/gemini-api/docs/migrate
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from ..tools.base import Tool

from google.genai import types

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import Gemini as GeminiModels
from ..pricing import calculate_cost_with_cached_input
from ..stability import stable
from ..types import Message, Role, ToolCall
from ..usage import UsageStats
from .base import Provider, ProviderError

logger = logging.getLogger(__name__)


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively adapt a JSON-schema dict to what Gemini's v1beta API accepts.

    BUG-40 (issue #66 investigation): two selectools schema shapes are hard
    400s on the Gemini API (verified live, both flash and flash-lite):

    - ``{"type": "array"}`` without ``items`` -> "parameters.properties[x].items:
      missing field". Bare ``list`` parameters emit exactly this. Gemini requires
      ``items`` for ARRAY, so inject a permissive string-element schema.
    - ``additionalProperties`` (emitted for ``Dict[K, V]`` parameters since
      BUG-29) -> "Unknown name 'additional_properties'". Gemini's Schema proto
      does not support it, so strip it; the parameter degrades to a plain
      OBJECT, which Gemini accepts.
    """
    cleaned: Dict[str, Any] = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if cleaned.get("type") == "array" and "items" not in cleaned:
        cleaned["items"] = {"type": "string"}
    properties = cleaned.get("properties")
    if isinstance(properties, dict):
        cleaned["properties"] = {
            key: _sanitize_schema_for_gemini(value) if isinstance(value, dict) else value
            for key, value in properties.items()
        }
    items = cleaned.get("items")
    if isinstance(items, dict):
        cleaned["items"] = _sanitize_schema_for_gemini(items)
    return cleaned


def _cached_content_tokens(usage: Any) -> Optional[int]:
    """Read ``usage_metadata.cached_content_token_count`` defensively.

    Returns None when the API does not report cache usage, so None (unknown)
    is never conflated with 0 (a real zero-hit count). Pitfall #22.
    """
    value = getattr(usage, "cached_content_token_count", None) if usage is not None else None
    return value if isinstance(value, int) else None


def _candidate_finish_reason(response: Any) -> str:
    """Best-effort extraction of the first candidate's finish_reason as a string."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return "unknown"
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return "unknown"
    return getattr(finish_reason, "name", None) or str(finish_reason)


def _warn_empty_tool_response(model_name: str, finish_reason: str) -> None:
    """
    Surface tool-equipped responses that contain neither text nor tool calls.

    Issue #66: some Gemini models (notably gemini-2.5-flash-lite) can return
    an empty candidate (often finish_reason=MALFORMED_FUNCTION_CALL or
    UNEXPECTED_TOOL_CALL) instead of a function call. Silently returning an
    empty Message makes the agent loop to max_iterations with no signal, so
    log loudly instead.
    """
    logger.warning(
        "Gemini returned neither text nor a tool call for a tool-equipped request "
        "(model=%s, finish_reason=%s). Some Gemini models — notably "
        "gemini-2.5-flash-lite — are known to emit empty candidates (frequently "
        "finish_reason=MALFORMED_FUNCTION_CALL or UNEXPECTED_TOOL_CALL) instead of a "
        "function call. The agent will treat this as an empty response and may loop "
        "to max_iterations. Consider gemini-2.5-flash, simplifying tool schemas, or "
        "retrying. See https://github.com/johnnichev/selectools/issues/66.",
        model_name,
        finish_reason,
    )


@stable
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

    def __init__(
        self, api_key: str | None = None, default_model: str = GeminiModels.FLASH_3_PREVIEW.id
    ):
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

    def _build_config(
        self,
        *,
        system_prompt: str,
        tools: List[Tool] | None,
        temperature: float,
        max_tokens: int,
        timeout: float | None,
    ) -> types.GenerateContentConfig:

        http_options = (
            types.HttpOptions(timeout=int(timeout * 1000)) if timeout is not None else None
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
            http_options=http_options,
        )

        if tools:
            config.tools = [self._map_tool_to_gemini(t) for t in tools]

        return config

    def _toolcall_from_part(
        self,
        part: types.Part,
    ) -> ToolCall:
        tc_id = f"call_{uuid.uuid4().hex}"

        raw_sig = getattr(part, "thought_signature", None)

        sig_str = (
            (
                base64.b64encode(raw_sig).decode("ascii")
                if isinstance(raw_sig, bytes)
                else str(raw_sig)
            )
            if raw_sig
            else None
        )

        function_call = part.function_call
        if function_call is None:  # pragma: no cover - callers guard on part.function_call
            raise ProviderError("Gemini part has no function_call to convert")

        return ToolCall(
            tool_name=str(function_call.name or ""),
            parameters=function_call.args if function_call.args else {},
            id=tc_id,
            thought_signature=sig_str,
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

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini completion failed: {exc}") from exc

        try:
            content_text = response.text or ""
        except ValueError:
            content_text = ""
        tool_calls: List[ToolCall] = []

        candidate_content = (
            response.candidates[0].content
            if response.candidates and response.candidates[0].content
            else None
        )
        if candidate_content and candidate_content.parts:
            for part in candidate_content.parts:
                if part.function_call:
                    tool_calls.append(self._toolcall_from_part(part))

        # Issue #66: a tool-equipped response with neither text nor tool calls
        # would silently no-op the agent loop. Surface it loudly.
        if tools and not content_text and not tool_calls:
            _warn_empty_tool_response(model_name, _candidate_finish_reason(response))

        # Extract usage stats from response
        # BUG-26 / LangChain #36500: use `is not None` guard instead of `or 0`
        # to avoid conflating None (unknown) with 0 (legitimate cached-prompt
        # token count). Pitfall #22.
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = (
            usage.prompt_token_count
            if usage is not None and usage.prompt_token_count is not None
            else 0
        )
        completion_tokens = (
            usage.candidates_token_count
            if usage is not None and usage.candidates_token_count is not None
            else 0
        )
        # Cached input tokens (context-caching hits) are INCLUDED in
        # prompt_token_count but bill at the model's published caching rate.
        cached_tokens = _cached_content_tokens(usage)
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost_with_cached_input(
                model_name, prompt_tokens, completion_tokens, cached_tokens or 0
            ),
            model=model_name,
            provider="gemini",
            cache_read_input_tokens=cached_tokens,
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
        Stream responses from Gemini's generate_content_stream API.

        Yields:
            str: Text content deltas.
            ToolCall: Complete tool call objects when a function_call part arrives.
        """

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        try:
            stream = self._client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini streaming failed: {exc}") from exc

        # BUG-38 / LiteLLM finding: if google-genai streams the same
        # function_call across multiple chunks, we must not emit a fresh
        # ToolCall per chunk. Track seen (name, args) pairs and only yield
        # once per unique tool invocation.
        _seen_tool_calls: set = set()
        # Issue #66: track whether the stream produced anything at all so an
        # empty tool-equipped response is surfaced instead of silently no-oping.
        _yielded_any = False
        _last_finish_reason = "unknown"

        try:
            for chunk in stream:
                chunk_finish_reason = _candidate_finish_reason(chunk)
                if chunk_finish_reason != "unknown":
                    _last_finish_reason = chunk_finish_reason
                try:
                    chunk_text = chunk.text
                except ValueError:
                    chunk_text = None
                if chunk_text:
                    _yielded_any = True
                    yield chunk_text

                candidates = chunk.candidates if hasattr(chunk, "candidates") else None
                if candidates:
                    for candidate in candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.function_call:
                                    # Dedup: skip if we've already emitted this exact call
                                    call_name = str(part.function_call.name or "")
                                    call_args = str(
                                        part.function_call.args if part.function_call.args else {}
                                    )
                                    dedup_key = (call_name, call_args)
                                    if dedup_key in _seen_tool_calls:
                                        continue
                                    _seen_tool_calls.add(dedup_key)

                                    if part.function_call:
                                        _yielded_any = True
                                        yield self._toolcall_from_part(part)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini streaming failed mid-stream: {exc}") from exc

        if tools and not _yielded_any:
            _warn_empty_tool_response(model_name, _last_finish_reason)

    def _format_contents(self, system_prompt: str, messages: List[Message]) -> List:
        """
        Format messages for Gemini's API.

        Converts our Message format to Gemini's content format.
        System instructions are handled separately via config.

        For Gemini 3.x thought signature support:
        - ASSISTANT messages echo thought_signature on function_call parts
        - TOOL messages include the original functionCall (with signature) before
          the functionResponse, as required by the Gemini 3.x API
        """
        from google.genai import types

        # Track the last ASSISTANT message's tool_calls so we can echo them
        # alongside TOOL functionResponse messages (Gemini 3.x requirement)
        last_assistant_tool_calls: dict[str, ToolCall] = {}

        contents: List[Any] = []
        for message in messages:
            role = message.role.value
            parts = []

            if role == Role.TOOL.value:
                role = "user"
                if message.tool_name:
                    # Echo the original functionCall part before the functionResponse
                    # if we have a matching tool_call with thought_signature (Gemini 3.x)
                    matching_tc = last_assistant_tool_calls.get(
                        message.tool_call_id or ""
                    ) or last_assistant_tool_calls.get(f"name:{message.tool_name}")
                    if matching_tc and matching_tc.thought_signature:
                        fc_part = types.Part(
                            function_call=types.FunctionCall(
                                name=matching_tc.tool_name, args=matching_tc.parameters
                            )
                        )
                        fc_part.thought_signature = base64.b64decode(matching_tc.thought_signature)  # type: ignore[assignment]
                        parts.append(fc_part)

                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=message.tool_name,
                                response={"result": message.content or ""},
                            )
                        )
                    )
                else:
                    parts.append(types.Part(text=f"Tool output: {message.content or ''}"))

            elif role == Role.ASSISTANT.value:
                role = "model"
                if message.content:
                    parts.append(types.Part(text=message.content))

                # Handle outgoing tool calls — echo thought_signature if present
                if message.tool_calls:
                    last_assistant_tool_calls = {}
                    for tc in message.tool_calls:
                        # Index by both id and name for lookup from TOOL messages
                        if tc.id:
                            last_assistant_tool_calls[tc.id] = tc
                        last_assistant_tool_calls[f"name:{tc.tool_name}"] = tc

                        fc_part = types.Part(
                            function_call=types.FunctionCall(name=tc.tool_name, args=tc.parameters)
                        )
                        if tc.thought_signature:
                            fc_part.thought_signature = base64.b64decode(tc.thought_signature)  # type: ignore[assignment]
                        parts.append(fc_part)

            elif role == Role.USER.value:
                role = "user"
                # Prefer the v0.21.0 multimodal ``content_parts`` path: when
                # the message was built via ``image_message()`` the image
                # lives in a ContentPart (not in the legacy
                # ``message.image_base64`` attribute, which is explicitly
                # None for multimodal messages). Fall back to the legacy
                # path for pre-0.21 callers.
                if getattr(message, "content_parts", None):
                    for cp in message.content_parts:  # type: ignore[union-attr]
                        if cp.type == "text" and cp.text:
                            parts.append(types.Part(text=cp.text))
                        elif cp.type == "image_url" and cp.image_url:
                            parts.append(
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=cp.image_url,
                                        mime_type=cp.media_type or "image/png",
                                    )
                                )
                            )
                        elif cp.type == "image_base64" and cp.image_base64:
                            parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type=cp.media_type or "image/png",
                                        data=base64.b64decode(cp.image_base64),
                                    )
                                )
                            )
                else:
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

            elif role == Role.SYSTEM.value:
                # Gemini handles system instructions via config, not messages.
                # Context injections (compressed context, entity memory) arrive
                # as SYSTEM messages — convert to user messages.
                role = "user"
                if message.content:
                    parts.append(types.Part(text=message.content))

            else:
                role = "user"  # Fallback
                if message.content:
                    parts.append(types.Part(text=message.content))

            if parts:
                # BUG-39: merge consecutive same-role Content objects so
                # parallel tool results (which are all role="user") don't
                # produce multiple consecutive user messages. Gemini API
                # rejects consecutive same-role entries.
                if contents and contents[-1].role == role:
                    contents[-1].parts.extend(parts)
                else:
                    contents.append(types.Content(role=role, parts=parts))

        return contents

    def _map_tool_to_gemini(self, tool: Tool) -> Any:
        """Convert a selectools.Tool to Gemini tool schema."""
        from google.genai import types

        schema = tool.schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    # BUG-40: Gemini rejects `additionalProperties` and bare
                    # arrays without `items` with hard 400s — sanitize first.
                    # The SDK coerces JSON-schema dicts to types.Schema via
                    # pydantic validation (verified live).
                    parameters=_sanitize_schema_for_gemini(schema["parameters"]),  # type: ignore[arg-type]
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

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini async completion failed: {exc}") from exc

        try:
            content_text = response.text or ""
        except ValueError:
            content_text = ""

        tool_calls: List[ToolCall] = []
        candidate_content = (
            response.candidates[0].content
            if response.candidates and response.candidates[0].content
            else None
        )
        if candidate_content and candidate_content.parts:
            for part in candidate_content.parts:
                if part.function_call:
                    tool_calls.append(self._toolcall_from_part(part))

        # Issue #66: see complete() — surface empty tool-equipped responses.
        if tools and not content_text and not tool_calls:
            _warn_empty_tool_response(model_name, _candidate_finish_reason(response))

        # Extract usage stats (BUG-26: see complete() for context)
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        prompt_tokens = (
            usage.prompt_token_count
            if usage is not None and usage.prompt_token_count is not None
            else 0
        )
        completion_tokens = (
            usage.candidates_token_count
            if usage is not None and usage.candidates_token_count is not None
            else 0
        )
        # Cached input tokens (context-caching hits) are INCLUDED in
        # prompt_token_count but bill at the model's published caching rate.
        cached_tokens = _cached_content_tokens(usage)
        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost_with_cached_input(
                model_name, prompt_tokens, completion_tokens, cached_tokens or 0
            ),
            model=model_name,
            provider="gemini",
            cache_read_input_tokens=cached_tokens,
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
            ToolCall: Complete tool call objects when a function_call part arrives
        """

        model_name = model or self.default_model
        contents = self._format_contents(system_prompt, messages)

        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini async streaming failed: {exc}") from exc

        # BUG-38: same dedup logic as sync stream() — see comment there.
        _seen_tool_calls: set = set()
        # Issue #66: same empty-stream detection as sync stream().
        _yielded_any = False
        _last_finish_reason = "unknown"

        try:
            async for chunk in stream:
                chunk_finish_reason = _candidate_finish_reason(chunk)
                if chunk_finish_reason != "unknown":
                    _last_finish_reason = chunk_finish_reason
                try:
                    chunk_text = chunk.text
                except ValueError:
                    chunk_text = None
                if chunk_text:
                    _yielded_any = True
                    yield chunk_text

                candidates = chunk.candidates if hasattr(chunk, "candidates") else None
                if candidates:
                    for candidate in candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.function_call:
                                    call_name = str(part.function_call.name or "")
                                    call_args = str(
                                        part.function_call.args if part.function_call.args else {}
                                    )
                                    dedup_key = (call_name, call_args)
                                    if dedup_key in _seen_tool_calls:
                                        continue
                                    _seen_tool_calls.add(dedup_key)
                                    if part.function_call:
                                        _yielded_any = True
                                        yield self._toolcall_from_part(part)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini async streaming failed mid-stream: {exc}") from exc

        if tools and not _yielded_any:
            _warn_empty_tool_response(model_name, _last_finish_reason)


__stability__ = "stable"

__all__ = ["GeminiProvider"]
