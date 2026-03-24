"""Mixin providing provider calling methods for the Agent class."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from ..cache import CacheKeyBuilder
from ..providers.base import ProviderError
from ..trace import StepType, TraceStep
from ..types import Message, Role

if TYPE_CHECKING:
    from ..trace import AgentTrace


class _ProviderCallerMixin:
    """Mixin that provides provider calling methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, _history, usage,
    _system_prompt, tools, etc.) which are expected to be provided by the
    Agent class that inherits from this mixin.
    """

    def _call_provider(
        self,
        stream_handler: Optional[Callable[[str], None]] = None,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> Message:
        call_start = time.time()

        cache_key: Optional[str] = None
        if self.config.cache and not (
            self.config.stream and getattr(self.provider, "supports_streaming", False)
        ):
            cache_key = CacheKeyBuilder.build(
                model=self._effective_model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
            )
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                cached_msg = cast(Message, cached[0])
                cached_usage = cached[1]
                self.usage.add_usage(cached_usage, tool_name=None)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self._effective_model,
                        self._system_prompt,
                    )
                    self._notify_observers(
                        "on_llm_end",
                        run_id,
                        cached_msg.content,
                        cached_usage,
                    )
                    self._notify_observers(
                        "on_cache_hit",
                        run_id,
                        self._effective_model,
                        cached_msg.content or "",
                    )
                    self._notify_observers("on_usage", run_id, cached_usage)
                if self.config.verbose:
                    print("[agent] cache hit -- skipping provider call")
                if trace is not None:
                    trace.add(
                        TraceStep(
                            type=StepType.CACHE_HIT,
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self._effective_model,
                            summary=f"Cache hit: {self._effective_model}",
                        )
                    )
                return cached_msg

        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self._effective_model,
                        self._system_prompt,
                    )

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = self._streaming_call(stream_handler=stream_handler)
                    if run_id:
                        self._notify_observers("on_llm_end", run_id, response_text, None)
                    return Message(role=Role.ASSISTANT, content=response_text)

                response_msg, usage_stats = self.provider.complete(
                    model=self._effective_model,
                    system_prompt=self._system_prompt,
                    messages=self._history,
                    tools=self.tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout,
                )
                response_text = response_msg.content or ""

                self.usage.add_usage(usage_stats, tool_name=None)

                if cache_key is not None and self.config.cache:
                    self.config.cache.set(cache_key, (response_msg, usage_stats))

                if run_id:
                    self._notify_observers("on_llm_end", run_id, response_text, usage_stats)
                    self._notify_observers("on_usage", run_id, usage_stats)

                if (
                    self.config.cost_warning_threshold
                    and self.usage.total_cost_usd > self.config.cost_warning_threshold
                ):
                    print(
                        f"\n⚠️  Cost Warning: Total cost ${self.usage.total_cost_usd:.6f} "
                        f"exceeds threshold ${self.config.cost_warning_threshold:.6f}\n"
                    )

                if self.config.verbose:
                    print(
                        f"[agent] tokens: {usage_stats.total_tokens:,} "
                        f"(prompt: {usage_stats.prompt_tokens:,}, completion: {usage_stats.completion_tokens:,}), "
                        f"cost: ${usage_stats.cost_usd:.6f}"
                    )

                if trace is not None:
                    trace.add(
                        TraceStep(
                            type=StepType.LLM_CALL,
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self._effective_model,
                            prompt_tokens=usage_stats.prompt_tokens,
                            completion_tokens=usage_stats.completion_tokens,
                            cost_usd=usage_stats.cost_usd,
                            summary=f"{self._effective_model} → {len(response_text)} chars",
                        )
                    )
                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                backoff = 0.0
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    backoff += self.config.rate_limit_cooldown_seconds * attempts
                if self.config.retry_backoff_seconds:
                    backoff += self.config.retry_backoff_seconds * attempts
                if run_id:
                    self._notify_observers(
                        "on_llm_retry",
                        run_id,
                        attempts,
                        self.config.max_retries,
                        exc,
                        backoff,
                    )
                if backoff > 0:
                    time.sleep(backoff)

        if trace is not None:
            trace.add(
                TraceStep(
                    type=StepType.LLM_CALL,
                    duration_ms=(time.time() - call_start) * 1000,
                    model=self._effective_model,
                    error=last_error,
                    summary=f"Provider error: {last_error}",
                )
            )
        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    def _streaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []
        for chunk in self.provider.stream(
            model=self._effective_model,
            system_prompt=self._system_prompt,
            messages=self._history,
            tools=self.tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
        ):
            if isinstance(chunk, str) and chunk:
                aggregated.append(chunk)
                if stream_handler:
                    stream_handler(chunk)

        return "".join(aggregated)

    def _is_rate_limit_error(self, message: str) -> bool:
        lowered = message.lower()
        return "rate limit" in lowered or "429" in lowered

    async def _acall_provider(
        self,
        stream_handler: Optional[Callable[[str], None]] = None,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> Message:
        """Async version of _call_provider with retry logic."""
        call_start = time.time()

        cache_key: Optional[str] = None
        if self.config.cache and not (
            self.config.stream and getattr(self.provider, "supports_streaming", False)
        ):
            cache_key = CacheKeyBuilder.build(
                model=self._effective_model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
            )
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                cached_msg = cast(Message, cached[0])
                cached_usage = cached[1]
                self.usage.add_usage(cached_usage, tool_name=None)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self._effective_model,
                        self._system_prompt,
                    )
                    self._notify_observers(
                        "on_llm_end",
                        run_id,
                        cached_msg.content,
                        cached_usage,
                    )
                    self._notify_observers(
                        "on_cache_hit",
                        run_id,
                        self._effective_model,
                        cached_msg.content or "",
                    )
                    self._notify_observers("on_usage", run_id, cached_usage)
                if self.config.verbose:
                    print("[agent] cache hit -- skipping provider call")
                if trace is not None:
                    trace.add(
                        TraceStep(
                            type=StepType.CACHE_HIT,
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self._effective_model,
                            summary=f"Cache hit: {self._effective_model}",
                        )
                    )
                return cached_msg

        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self._effective_model,
                        self._system_prompt,
                    )
                    await self._anotify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self._effective_model,
                        self._system_prompt,
                    )

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = await self._astreaming_call(stream_handler=stream_handler)
                    if run_id:
                        self._notify_observers("on_llm_end", run_id, response_text, None)
                        await self._anotify_observers("on_llm_end", run_id, response_text, None)
                    return Message(role=Role.ASSISTANT, content=response_text)

                # Check if provider has async support
                if hasattr(self.provider, "acomplete") and getattr(
                    self.provider, "supports_async", False
                ):
                    response_msg, usage_stats = await self.provider.acomplete(
                        model=self._effective_model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )
                    response_text = response_msg.content or ""
                else:
                    # Fallback to sync in executor
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        response_msg, usage_stats = await loop.run_in_executor(
                            executor,
                            lambda: self.provider.complete(
                                model=self._effective_model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                tools=self.tools,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )
                    response_text = response_msg.content or ""

                self.usage.add_usage(usage_stats, tool_name=None)

                if cache_key is not None and self.config.cache:
                    self.config.cache.set(cache_key, (response_msg, usage_stats))

                if run_id:
                    self._notify_observers("on_llm_end", run_id, response_text, usage_stats)
                    await self._anotify_observers("on_llm_end", run_id, response_text, usage_stats)
                    self._notify_observers("on_usage", run_id, usage_stats)
                    await self._anotify_observers("on_usage", run_id, usage_stats)

                if (
                    self.config.cost_warning_threshold
                    and self.usage.total_cost_usd > self.config.cost_warning_threshold
                ):
                    print(
                        f"\n⚠️  Cost Warning: Total cost ${self.usage.total_cost_usd:.6f} "
                        f"exceeds threshold ${self.config.cost_warning_threshold:.6f}\n"
                    )

                if self.config.verbose:
                    print(
                        f"[agent] tokens: {usage_stats.total_tokens:,} "
                        f"(prompt: {usage_stats.prompt_tokens:,}, completion: {usage_stats.completion_tokens:,}), "
                        f"cost: ${usage_stats.cost_usd:.6f}"
                    )

                if trace is not None:
                    trace.add(
                        TraceStep(
                            type=StepType.LLM_CALL,
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self._effective_model,
                            prompt_tokens=usage_stats.prompt_tokens,
                            completion_tokens=usage_stats.completion_tokens,
                            cost_usd=usage_stats.cost_usd,
                            summary=f"{self._effective_model} → {len(response_text)} chars",
                        )
                    )
                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                backoff = 0.0
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    backoff += self.config.rate_limit_cooldown_seconds * attempts
                if self.config.retry_backoff_seconds:
                    backoff += self.config.retry_backoff_seconds * attempts
                if run_id:
                    self._notify_observers(
                        "on_llm_retry",
                        run_id,
                        attempts,
                        self.config.max_retries,
                        exc,
                        backoff,
                    )
                if backoff > 0:
                    await asyncio.sleep(backoff)

        if trace is not None:
            trace.add(
                TraceStep(
                    type=StepType.LLM_CALL,
                    duration_ms=(time.time() - call_start) * 1000,
                    model=self._effective_model,
                    error=last_error,
                    summary=f"Provider error: {last_error}",
                )
            )
        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    async def _astreaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        """Async version of _streaming_call."""
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []

        if hasattr(self.provider, "astream") and getattr(self.provider, "supports_async", False):
            stream = self.provider.astream(  # type: ignore[attr-defined]
                model=self._effective_model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )
            async for chunk in stream:
                if isinstance(chunk, str) and chunk:
                    aggregated.append(chunk)
                    if stream_handler:
                        stream_handler(chunk)
        else:
            for chunk in self.provider.stream(
                model=self._effective_model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            ):
                if chunk:
                    aggregated.append(str(chunk))
                    if stream_handler:
                        stream_handler(str(chunk))

        return "".join(aggregated)
