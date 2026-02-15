"""
Provider-agnostic agent loop implementing the TOOL_CALL contract.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from ..analytics import AgentAnalytics
from ..parser import ToolCallParser
from ..prompt import PromptBuilder
from ..providers.base import Provider, ProviderError
from ..providers.openai_provider import OpenAIProvider
from ..tools import Tool
from ..types import AgentResult, Message, Role, StreamChunk, ToolCall
from ..usage import AgentUsage
from .config import AgentConfig

if TYPE_CHECKING:
    from ..memory import ConversationMemory


class Agent:
    """
    Provider-agnostic AI agent that iteratively calls tools to accomplish tasks.

    The Agent manages the loop of:
    1. Sending conversation history to an LLM
    2. Parsing the response for tool calls
    3. Executing requested tools
    4. Repeating until the task is complete

    Supports both synchronous and asynchronous execution, conversation memory,
    streaming responses, and multiple LLM providers (OpenAI, Anthropic, Gemini, etc).

    Attributes:
        tools: List of available tools the agent can use.
        provider: LLM provider adapter (OpenAI, Anthropic, Gemini, or custom).
        prompt_builder: Generates system prompts with tool schemas.
        parser: Extracts tool calls from LLM responses.
        config: Configuration options for behavior customization.
        memory: Optional conversation memory for multi-turn dialogues.

    Example:
        >>> # Basic agent
        >>> tools = [search_tool, calculator_tool]
        >>> agent = Agent(tools=tools, provider=OpenAIProvider())
        >>>
        >>> response = agent.run([
        ...     Message(role=Role.USER, content="What's 2+2 and search for Python?")
        ... ])
        >>> print(response.content)

        >>> # Agent with memory for multi-turn conversations
        >>> memory = ConversationMemory(max_messages=20)
        >>> agent = Agent(tools=tools, provider=provider, memory=memory)
        >>>
        >>> # First turn
        >>> agent.run([Message(role=Role.USER, content="My name is Alice")])
        >>>
        >>> # Second turn - context is preserved
        >>> agent.run([Message(role=Role.USER, content="What's my name?")])

        >>> # Async execution for high-performance applications
        >>> response = await agent.arun([Message(...)])
    """

    def __init__(
        self,
        tools: List[Tool],
        provider: Optional[Provider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        parser: Optional[ToolCallParser] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        """
        Initialize a new Agent instance.

        Args:
            tools: List of Tool instances the agent can use (minimum 1 required).
            provider: LLM provider adapter. Defaults to OpenAIProvider().
            prompt_builder: Custom prompt builder. Defaults to PromptBuilder().
            parser: Custom tool call parser. Defaults to ToolCallParser().
            config: Agent configuration options. Defaults to AgentConfig().
            memory: Optional conversation memory for maintaining history across turns.

        Raises:
            ValueError: If tools list is empty.

        Example:
            >>> agent = Agent(
            ...     tools=[search, calculator],
            ...     provider=OpenAIProvider(api_key="sk-..."),
            ...     config=AgentConfig(model="gpt-4o", verbose=True),
            ...     memory=ConversationMemory(max_messages=20)
            ... )
        """
        if not tools:
            raise ValueError("Agent requires at least one tool.")

        self.tools = tools
        self._tools_by_name = {tool.name: tool for tool in tools}
        self.provider = provider or OpenAIProvider()
        if prompt_builder:
            self.prompt_builder = prompt_builder
        elif config and config.system_prompt:
            self.prompt_builder = PromptBuilder(base_instructions=config.system_prompt)
        else:
            self.prompt_builder = PromptBuilder()
        self.parser = parser or ToolCallParser()
        self.config = config or AgentConfig()
        self.memory = memory
        self.usage = AgentUsage()
        self.analytics = AgentAnalytics() if self.config.enable_analytics else None

        self._system_prompt = self.prompt_builder.build(self.tools)
        self._history: List[Message] = []

    def reset(self) -> None:
        """Reset agent state for reuse. Clears history, usage stats, and memory."""
        self._history = []
        self.usage = AgentUsage()
        if self.analytics:
            self.analytics = AgentAnalytics()
        if self.memory:
            self.memory.clear()

    def _call_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Safely call a hook if it exists, swallowing any exceptions."""
        if not self.config.hooks or hook_name not in self.config.hooks:
            return
        try:
            self.config.hooks[hook_name](*args, **kwargs)
        except Exception:  # noqa: BLE001 # nosec B110
            # Silently ignore hook errors to prevent them from breaking agent execution
            pass

    def run(
        self, messages: List[Message], stream_handler: Optional[Callable[[str], None]] = None
    ) -> AgentResult:
        """
        Execute the agent loop with the provided conversation history.

        If the agent was initialized with a ConversationMemory, the new messages
        will be appended to the existing memory history, and the final response
        will be automatically saved to memory.

        Args:
            messages: New messages for this turn (typically a single user message).
            stream_handler: Optional callback for streaming responses.

        Returns:
            AgentResult with the final response and tool call metadata.
        """
        # Call on_agent_start hook
        self._call_hook("on_agent_start", messages)

        try:
            # Load history from memory if available, then append new messages
            if self.memory:
                self._history = self.memory.get_history() + list(messages)
                # Add new user messages to memory
                self.memory.add_many(messages)
            else:
                self._history.extend(messages)

            iteration = 0
            all_tool_calls: List[ToolCall] = []
            last_tool_name: Optional[str] = None
            last_tool_args: Dict[str, Any] = {}

            while iteration < self.config.max_iterations:
                iteration += 1
                self._call_hook("on_iteration_start", iteration, self._history)

                response_msg = self._call_provider(stream_handler=stream_handler)
                response_text = response_msg.content

                # Determine tools to call
                tool_calls_to_execute = []
                if response_msg.tool_calls:
                    # Native tool calls
                    tool_calls_to_execute = response_msg.tool_calls
                else:
                    # Fallback to regex parsing for text-only responses
                    parse_result = self.parser.parse(response_text)
                    if parse_result.tool_call:
                        tool_calls_to_execute.append(parse_result.tool_call)

                if not tool_calls_to_execute:
                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._call_hook("on_iteration_end", iteration, response_text)
                    # Save final response to memory if available
                    if self.memory:
                        self.memory.add(final_response)
                    self._call_hook("on_agent_end", final_response, self.usage)
                    return AgentResult(
                        message=final_response,
                        tool_name=last_tool_name,
                        tool_args=last_tool_args,
                        iterations=iteration,
                        tool_calls=all_tool_calls,
                    )

                if self.config.routing_only and tool_calls_to_execute:
                    # In routing mode, we return the selection without executing.
                    # We do NOT append to history because the action wasn't completed.
                    return AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=iteration,
                        tool_calls=tool_calls_to_execute,
                    )

                # Append the ASSISTANT message (with tool_calls) to history once,
                # then execute all requested tools and append TOOL results.
                self._history.append(response_msg)
                if self.memory:
                    self.memory.add(response_msg)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = self._execute_tools_parallel(
                        tool_calls_to_execute, all_tool_calls, iteration, response_text
                    )
                    last_tool_name = _last_name
                    last_tool_args = _last_args
                else:
                    # Sequential execution (single tool call or parallel disabled)
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        all_tool_calls.append(tool_call)
                        last_tool_name = tool_name
                        last_tool_args = parameters

                        if self.config.verbose:
                            print(
                                f"[agent] Iteration {iteration}: tool={tool_name} params={parameters}"
                            )

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                            self._append_tool_result(error_message, tool_name, tool_call.id)
                            continue

                        try:
                            start_time = time.time()
                            self._call_hook("on_tool_start", tool_name, parameters)

                            chunk_counter = {"count": 0}

                            def chunk_callback(chunk: str) -> None:
                                chunk_counter["count"] += 1
                                self._call_hook("on_tool_chunk", tool_name, chunk)

                            result = self._execute_tool_with_timeout(
                                tool, parameters, chunk_callback
                            )
                            duration = time.time() - start_time
                            self._call_hook("on_tool_end", tool_name, result, duration)

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=True,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter["count"],
                                )
                            if self.usage.iterations:
                                self.usage.tool_usage[tool.name] = (
                                    self.usage.tool_usage.get(tool.name, 0) + 1
                                )
                                self.usage.tool_tokens[tool.name] = (
                                    self.usage.tool_tokens.get(tool.name, 0)
                                    + self.usage.iterations[-1].total_tokens
                                )

                            self._append_tool_result(
                                result, tool_name, tool_call.id, tool_result=result
                            )

                        except Exception as exc:
                            duration = time.time() - start_time
                            self._call_hook("on_tool_error", tool_name, exc, parameters)
                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=False,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=0,
                                )

                            error_message = f"Error executing tool '{tool_name}': {exc}"
                            self._append_tool_result(error_message, tool_name, tool_call.id)

                self._call_hook("on_iteration_end", iteration, response_text)
                continue  # Continue to next iteration

            final_response = Message(
                role=Role.ASSISTANT,
                content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
            )
            # Save final response to memory if available
            if self.memory:
                self.memory.add(final_response)
            self._call_hook("on_agent_end", final_response, self.usage)
            return AgentResult(
                message=final_response,
                tool_name=last_tool_name,
                tool_args=last_tool_args,
                iterations=iteration,
                tool_calls=all_tool_calls,
            )
        except Exception as exc:
            self._call_hook("on_error", exc, {"messages": messages, "iteration": iteration})
            raise

    def _call_provider(self, stream_handler: Optional[Callable[[str], None]] = None) -> Message:
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                self._call_hook("on_llm_start", self._history, self.config.model)

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = self._streaming_call(stream_handler=stream_handler)
                    self._call_hook(
                        "on_llm_end", response_text, None
                    )  # No usage stats for streaming
                    # For streaming, we currently construct a text-only Message
                    return Message(role=Role.ASSISTANT, content=response_text)

                response_msg, usage_stats = self.provider.complete(
                    model=self.config.model,
                    system_prompt=self._system_prompt,
                    messages=self._history,
                    tools=self.tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout,
                )
                response_text = response_msg.content

                # Track usage (tool name will be added later after parsing)
                self.usage.add_usage(usage_stats, tool_name=None)

                # Call on_llm_end hook
                self._call_hook("on_llm_end", response_text, usage_stats)

                # Check cost warning threshold
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

                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    time.sleep(self.config.rate_limit_cooldown_seconds * attempts)
                if self.config.retry_backoff_seconds:
                    time.sleep(self.config.retry_backoff_seconds * attempts)

        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    def _streaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        # Note: Streaming does not return usage stats due to Python generator limitations
        aggregated: List[str] = []
        for chunk in self.provider.stream(
            model=self.config.model,
            system_prompt=self._system_prompt,
            messages=self._history,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
        ):
            if chunk:
                aggregated.append(str(chunk))
                if stream_handler:
                    stream_handler(str(chunk))

        return "".join(aggregated)

    def _append_tool_result(
        self,
        tool_content: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> None:
        """Update history with tool output."""
        tool_msg = Message(
            role=Role.TOOL,
            content=tool_content,
            tool_name=tool_name,
            tool_result=tool_result,
            tool_call_id=tool_call_id,
        )
        self._history.append(tool_msg)
        if self.memory:
            self.memory.add(tool_msg)

    # ------------------------------------------------------------------
    # Parallel tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
    ) -> tuple:
        """Execute multiple tool calls concurrently using ThreadPoolExecutor.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int

        def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            start = time.time()
            self._call_hook("on_tool_start", tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._call_hook("on_tool_chunk", tool_name, chunk)

            try:
                result = self._execute_tool_with_timeout(tool, parameters, chunk_cb)
                dur = time.time() - start
                self._call_hook("on_tool_end", tool_name, result, dur)
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                self._call_hook("on_tool_error", tool_name, exc, parameters)
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        # Submit all tool calls to the thread pool
        with ThreadPoolExecutor(max_workers=len(tool_calls_to_execute)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls_to_execute]
            results = [f.result() for f in futures]  # preserves order

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}

        for r in results:
            all_tool_calls.append(r.tool_call)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )
            if not r.is_error and self.usage.iterations:
                self.usage.tool_usage[r.tool_call.tool_name] = (
                    self.usage.tool_usage.get(r.tool_call.tool_name, 0) + 1
                )
                self.usage.tool_tokens[r.tool_call.tool_name] = (
                    self.usage.tool_tokens.get(r.tool_call.tool_name, 0)
                    + self.usage.iterations[-1].total_tokens
                )

            # Append result to history (in order)
            if r.is_error:
                self._append_tool_result(r.result, r.tool_call.tool_name, r.tool_call.id)
            else:
                self._append_tool_result(
                    r.result, r.tool_call.tool_name, r.tool_call.id, tool_result=r.result
                )

        return last_tool_name, last_tool_args

    async def _aexecute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
    ) -> tuple:
        """Execute multiple tool calls concurrently using asyncio.gather.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int

        async def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            start = time.time()
            self._call_hook("on_tool_start", tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._call_hook("on_tool_chunk", tool_name, chunk)

            try:
                result = await self._aexecute_tool_with_timeout(tool, parameters, chunk_cb)
                dur = time.time() - start
                self._call_hook("on_tool_end", tool_name, result, dur)
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                self._call_hook("on_tool_error", tool_name, exc, parameters)
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        # Run all tool calls concurrently
        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls_to_execute])

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}

        for r in results:
            all_tool_calls.append(r.tool_call)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )

            # Append result to history (in order)
            if r.is_error:
                self._append_tool_result(r.result, r.tool_call.tool_name, r.tool_call.id)
            else:
                self._append_tool_result(
                    r.result, r.tool_call.tool_name, r.tool_call.id, tool_result=r.result
                )

        return last_tool_name, last_tool_args

    def _execute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run tool.execute with optional timeout and chunk callback."""
        if not self.config.tool_timeout_seconds:
            return tool.execute(parameters, chunk_callback=chunk_callback)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.execute, parameters, chunk_callback)
            try:
                return future.result(timeout=self.config.tool_timeout_seconds)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
                )

    def _is_rate_limit_error(self, message: str) -> bool:
        lowered = message.lower()
        return "rate limit" in lowered or "429" in lowered

    # Async methods
    async def astream(
        self, messages: List[Message]
    ) -> AsyncGenerator[Union[StreamChunk, AgentResult], None]:
        """
        Stream the agent's response token-by-token.

        Yields:
            StreamChunk: Intermediate content chunks.
            AgentResult: The final result object (yielded at the very end).
        """
        self._call_hook("on_agent_start", messages)

        try:
            if self.memory:
                self._history = self.memory.get_history() + list(messages)
                self.memory.add_many(messages)
            else:
                self._history.extend(messages)

            iteration = 0
            all_tool_calls: List[ToolCall] = []
            last_tool_name: Optional[str] = None
            last_tool_args: Dict[str, Any] = {}

            while iteration < self.config.max_iterations:
                iteration += 1
                self._call_hook("on_iteration_start", iteration, self._history)

                full_content = ""
                current_tool_calls: List[ToolCall] = []

                has_astream = (
                    hasattr(self.provider, "astream")
                    and self.provider.astream is not None
                    and self.provider.supports_streaming
                )

                if not has_astream:
                    # Fallback: provider doesn't support astream, use acomplete
                    if hasattr(self.provider, "acomplete") and getattr(
                        self.provider, "supports_async", False
                    ):
                        response_msg, _usage = await self.provider.acomplete(
                            model=self.config.model,
                            system_prompt=self._system_prompt,
                            messages=self._history,
                            tools=self.tools,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            timeout=self.config.request_timeout,
                        )
                    else:
                        # Last resort: sync complete in executor
                        loop = asyncio.get_event_loop()
                        response_msg, _usage = await loop.run_in_executor(
                            None,
                            lambda: self.provider.complete(
                                model=self.config.model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                tools=self.tools,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )
                    yield StreamChunk(content=response_msg.content)
                    full_content = response_msg.content
                else:
                    # Real async streaming
                    gen = self.provider.astream(
                        model=self.config.model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )

                    async for item in gen:
                        if isinstance(item, str):
                            yield StreamChunk(content=item)
                            full_content += item
                        elif isinstance(item, ToolCall):
                            current_tool_calls.append(item)
                            yield StreamChunk(tool_calls=[item])

                # Reconstruct the response message
                response_msg = Message(
                    role=Role.ASSISTANT, content=full_content, tool_calls=current_tool_calls or None
                )
                self._history.append(response_msg)
                if self.memory:
                    self.memory.add(response_msg)

                # Tool parsing logic
                tool_calls_to_execute = []
                if response_msg.tool_calls:
                    tool_calls_to_execute = response_msg.tool_calls
                else:
                    parse_result = self.parser.parse(full_content)
                    if parse_result.tool_call:
                        tool_calls_to_execute.append(parse_result.tool_call)

                if not tool_calls_to_execute:
                    final_response = response_msg
                    self._call_hook("on_iteration_end", iteration, full_content)
                    yield AgentResult(
                        message=final_response,
                        tool_name=last_tool_name,
                        tool_args=last_tool_args,
                        iterations=iteration,
                        tool_calls=all_tool_calls,
                    )
                    self._call_hook("on_agent_end", final_response, self.usage)
                    return

                if self.config.routing_only:
                    yield AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=iteration,
                        tool_calls=tool_calls_to_execute,
                    )
                    return

                # Execute tools
                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = await self._aexecute_tools_parallel(
                        tool_calls_to_execute, all_tool_calls, iteration, full_content
                    )
                    last_tool_name = _last_name
                    last_tool_args = _last_args
                else:
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        all_tool_calls.append(tool_call)
                        last_tool_name = tool_name
                        last_tool_args = parameters

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            result = f"Error: Tool {tool_name} not found"
                            self._append_tool_result(result, tool_name, tool_call.id)
                            continue

                        try:
                            result = await self._aexecute_tool_with_timeout(tool, parameters, None)
                            self._append_tool_result(
                                result, tool_name, tool_call.id, tool_result=result
                            )
                        except Exception as exc:
                            error_message = f"Error executing tool '{tool.name}': {exc}"
                            self._append_tool_result(error_message, tool_name, tool_call.id)

                self._call_hook("on_iteration_end", iteration, full_content)

            # Max iterations reached
            final_msg = Message(
                role=Role.ASSISTANT,
                content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
            )
            yield AgentResult(
                message=final_msg,
                tool_name=last_tool_name,
                tool_args=last_tool_args,
                iterations=iteration,
                tool_calls=all_tool_calls,
            )
            if self.memory:
                self.memory.add(final_msg)
            self._call_hook("on_agent_end", final_msg, self.usage)
            return
        except Exception as exc:
            self._call_hook("on_error", exc, {"messages": messages, "iteration": 0})
            raise

    async def arun(
        self, messages: List[Message], stream_handler: Optional[Callable[[str], None]] = None
    ) -> AgentResult:
        """
        Async version of run().

        Execute the agent loop asynchronously with the provided conversation history.
        Uses provider async methods if available, falls back to sync in executor.

        Args:
            messages: New messages for this turn (typically a single user message).
            stream_handler: Optional callback for streaming responses.

        Returns:
            AgentResult with the final response and tool call metadata.
        """
        self._call_hook("on_agent_start", messages)

        try:
            # Load history from memory if available, then append new messages
            if self.memory:
                self._history = self.memory.get_history() + list(messages)
                self.memory.add_many(messages)
            else:
                self._history = list(messages)

            iteration = 0
            all_tool_calls: List[ToolCall] = []
            last_tool_name: Optional[str] = None
            last_tool_args: Dict[str, Any] = {}

            while iteration < self.config.max_iterations:
                iteration += 1
                self._call_hook("on_iteration_start", iteration, self._history)

                response_msg = await self._acall_provider(stream_handler=stream_handler)
                response_text = response_msg.content

                # Determine tools to call
                tool_calls_to_execute = []
                if response_msg.tool_calls:
                    # Native tool calls
                    tool_calls_to_execute = response_msg.tool_calls
                else:
                    # Fallback to regex parsing for text-only responses
                    parse_result = self.parser.parse(response_text)
                    if parse_result.tool_call:
                        tool_calls_to_execute.append(parse_result.tool_call)

                if not tool_calls_to_execute:
                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._call_hook("on_iteration_end", iteration, response_text)
                    if self.memory:
                        self.memory.add(final_response)
                    self._call_hook("on_agent_end", final_response, self.usage)
                    return AgentResult(
                        message=final_response,
                        tool_name=last_tool_name,
                        tool_args=last_tool_args,
                        iterations=iteration,
                        tool_calls=all_tool_calls,
                    )

                if self.config.routing_only and tool_calls_to_execute:
                    return AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=iteration,
                        tool_calls=tool_calls_to_execute,
                    )

                # Execute tool calls
                self._history.append(response_msg)
                if self.memory:
                    self.memory.add(response_msg)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = await self._aexecute_tools_parallel(
                        tool_calls_to_execute, all_tool_calls, iteration, response_text
                    )
                    last_tool_name = _last_name
                    last_tool_args = _last_args
                else:
                    # Sequential execution (single tool call or parallel disabled)
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        all_tool_calls.append(tool_call)
                        last_tool_name = tool_name
                        last_tool_args = parameters

                        if self.config.verbose:
                            print(
                                f"[agent] Iteration {iteration}: tool={tool_name} params={parameters}"
                            )

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                            self._append_tool_result(error_message, tool_name, tool_call.id)
                            continue

                        try:
                            start_time = time.time()
                            self._call_hook("on_tool_start", tool_name, parameters)

                            chunk_counter = {"count": 0}

                            def chunk_callback(chunk: str) -> None:
                                chunk_counter["count"] += 1
                                self._call_hook("on_tool_chunk", tool_name, chunk)

                            result = await self._aexecute_tool_with_timeout(
                                tool, parameters, chunk_callback
                            )
                            duration = time.time() - start_time
                            self._call_hook("on_tool_end", tool_name, result, duration)

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=True,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter["count"],
                                )
                        except Exception as exc:
                            duration = time.time() - start_time
                            self._call_hook("on_tool_error", tool.name, exc, parameters)

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=False,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter.get("count", 0),
                                )

                            error_message = f"Error executing tool '{tool.name}': {exc}"
                            self._append_tool_result(error_message, tool_name, tool_call.id)
                            continue

                        self._append_tool_result(
                            result, tool_name, tool_call.id, tool_result=result
                        )

                self._call_hook("on_iteration_end", iteration, response_text)

            final_response = Message(
                role=Role.ASSISTANT,
                content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
            )
            if self.memory:
                self.memory.add(final_response)
            self._call_hook("on_agent_end", final_response, self.usage)
            return AgentResult(
                message=final_response,
                tool_name=last_tool_name,
                tool_args=last_tool_args,
                iterations=iteration,
                tool_calls=all_tool_calls,
            )
        except Exception as exc:
            self._call_hook("on_error", exc, {"messages": messages, "iteration": iteration})
            raise

    async def _acall_provider(
        self, stream_handler: Optional[Callable[[str], None]] = None
    ) -> Message:
        """Async version of _call_provider with retry logic."""
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                self._call_hook("on_llm_start", self._history, self.config.model)

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = await self._astreaming_call(stream_handler=stream_handler)
                    self._call_hook("on_llm_end", response_text, None)
                    return Message(role=Role.ASSISTANT, content=response_text)

                # Check if provider has async support
                if hasattr(self.provider, "acomplete") and getattr(
                    self.provider, "supports_async", False
                ):
                    response_msg, usage_stats = await self.provider.acomplete(
                        model=self.config.model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )
                    response_text = response_msg.content
                else:
                    # Fallback to sync in executor
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        response_msg, usage_stats = await loop.run_in_executor(
                            executor,
                            lambda: self.provider.complete(
                                model=self.config.model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                tools=self.tools,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )
                    # Sync calls return tuple[Message, UsageStats]
                    response_text = response_msg.content

                # Track usage
                self.usage.add_usage(usage_stats, tool_name=None)

                # Call on_llm_end hook
                self._call_hook("on_llm_end", response_text, usage_stats)

                # Check cost warning threshold
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

                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    await asyncio.sleep(self.config.rate_limit_cooldown_seconds * attempts)
                if self.config.retry_backoff_seconds:
                    await asyncio.sleep(self.config.retry_backoff_seconds * attempts)

        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    async def _astreaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        """Async version of _streaming_call."""
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []

        # Check if provider has async streaming
        if hasattr(self.provider, "astream") and getattr(self.provider, "supports_async", False):
            stream = self.provider.astream(  # type: ignore[attr-defined]
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )
            async for chunk in stream:
                if chunk:
                    aggregated.append(str(chunk))
                    if stream_handler:
                        stream_handler(str(chunk))
        else:
            # Fallback to sync streaming in executor
            for chunk in self.provider.stream(
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            ):
                if chunk:
                    aggregated.append(str(chunk))
                    if stream_handler:
                        stream_handler(str(chunk))

        return "".join(aggregated)

    async def _aexecute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Async version of _execute_tool_with_timeout."""
        if not self.config.tool_timeout_seconds:
            return await tool.aexecute(parameters, chunk_callback=chunk_callback)

        try:
            return await asyncio.wait_for(
                tool.aexecute(parameters, chunk_callback=chunk_callback),
                timeout=self.config.tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )

    # Usage tracking convenience methods
    @property
    def total_cost(self) -> float:
        """Get the total cost in USD for all API calls."""
        return self.usage.total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens used across all API calls."""
        return self.usage.total_tokens

    def get_usage_summary(self) -> str:
        """
        Get a formatted summary of token usage and costs.

        Returns:
            Formatted string with usage statistics including token counts,
            costs, and per-tool breakdown.
        """
        return str(self.usage)

    def get_analytics(self) -> Optional[AgentAnalytics]:
        """Get the analytics tracker if enabled."""
        return self.analytics

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self.usage = AgentUsage()
