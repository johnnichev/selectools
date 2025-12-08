"""
Provider-agnostic agent loop implementing the TOOL_CALL contract.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from .parser import ToolCallParser
from .prompt import PromptBuilder
from .providers.base import Provider, ProviderError
from .providers.openai_provider import OpenAIProvider
from .tools import Tool
from .types import Message, Role
from .usage import AgentUsage

if TYPE_CHECKING:
    from .memory import ConversationMemory


@dataclass
class AgentConfig:
    """
    Configuration options for customizing agent behavior.

    Controls model selection, retry logic, timeouts, and verbosity for debugging.
    Sensible defaults are provided for all options.

    Attributes:
        model: Model identifier to use (e.g., "gpt-4o", "claude-3-5-sonnet-20240620").
        temperature: LLM temperature (0.0 = deterministic, higher = more creative). Default: 0.0.
        max_tokens: Maximum tokens in LLM response. Default: 1000.
        max_iterations: Maximum tool-calling iterations before stopping. Default: 6.
        verbose: Print detailed logs for debugging. Default: False.
        stream: Enable streaming responses (if provider supports it). Default: False.
        request_timeout: Timeout for LLM API requests in seconds. Default: 30.0.
        max_retries: Maximum retry attempts for failed LLM requests. Default: 2.
        retry_backoff_seconds: Seconds to wait between retries (multiplied by attempt number). Default: 1.0.
        rate_limit_cooldown_seconds: Cooldown period after rate limit errors. Default: 5.0.
        tool_timeout_seconds: Maximum execution time for each tool call. None = no timeout. Default: None.
        cost_warning_threshold: Print warning when total cost exceeds this USD amount. None = no warnings. Default: None.

    Example:
        >>> # Production config with retries and timeouts
        >>> config = AgentConfig(
        ...     model="gpt-4o-mini",
        ...     temperature=0.3,
        ...     max_tokens=2000,
        ...     max_iterations=10,
        ...     request_timeout=60.0,
        ...     tool_timeout_seconds=30.0,
        ... )
        >>>
        >>> # Debug config with verbose logging
        >>> config = AgentConfig(verbose=True, stream=True)
    """

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1000
    max_iterations: int = 6
    verbose: bool = False
    stream: bool = False
    request_timeout: Optional[float] = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    rate_limit_cooldown_seconds: float = 5.0
    tool_timeout_seconds: Optional[float] = None
    cost_warning_threshold: Optional[float] = None


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
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ToolCallParser()
        self.config = config or AgentConfig()
        self.memory = memory
        self.usage = AgentUsage()

        self._system_prompt = self.prompt_builder.build(self.tools)
        self._history: List[Message] = []

    def run(
        self, messages: List[Message], stream_handler: Optional[Callable[[str], None]] = None
    ) -> Message:
        """
        Execute the agent loop with the provided conversation history.

        If the agent was initialized with a ConversationMemory, the new messages
        will be appended to the existing memory history, and the final response
        will be automatically saved to memory.

        Args:
            messages: New messages for this turn (typically a single user message).
            stream_handler: Optional callback for streaming responses.

        Returns:
            The final assistant response message.
        """
        # Load history from memory if available, then append new messages
        if self.memory:
            self._history = self.memory.get_history() + list(messages)
            # Add new user messages to memory
            self.memory.add_many(messages)
        else:
            self._history = list(messages)

        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            response_text = self._call_provider(stream_handler=stream_handler)
            parse_result = self.parser.parse(response_text)

            if not parse_result.tool_call:
                final_response = Message(role=Role.ASSISTANT, content=response_text)
                # Save final response to memory if available
                if self.memory:
                    self.memory.add(final_response)
                return final_response

            tool_name = parse_result.tool_call.tool_name
            parameters = parse_result.tool_call.parameters

            if self.config.verbose:
                print(f"[agent] Iteration {iteration}: tool={tool_name} params={parameters}")

            tool = self._tools_by_name.get(tool_name)
            if not tool:
                error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                self._append_assistant_and_tool(response_text, error_message, tool_name)
                continue

            try:
                result = self._execute_tool_with_timeout(tool, parameters)
                # Track tool usage
                if self.usage.iterations:
                    last_iteration = self.usage.iterations[-1]
                    # Update the most recent iteration with the tool name
                    if tool.name not in self.usage.tool_usage:
                        self.usage.tool_usage[tool.name] = 0
                        self.usage.tool_tokens[tool.name] = 0
                    self.usage.tool_usage[tool.name] += 1
                    self.usage.tool_tokens[tool.name] += last_iteration.total_tokens
            except Exception as exc:  # noqa: BLE001
                error_message = f"Error executing tool '{tool.name}': {exc}"
                self._append_assistant_and_tool(response_text, error_message, tool.name)
                continue

            self._append_assistant_and_tool(response_text, result, tool.name, tool_result=result)

        final_response = Message(
            role=Role.ASSISTANT,
            content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
        )
        # Save final response to memory if available
        if self.memory:
            self.memory.add(final_response)
        return final_response

    def _call_provider(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    return self._streaming_call(stream_handler=stream_handler)

                response_text, usage_stats = self.provider.complete(
                    model=self.config.model,
                    system_prompt=self._system_prompt,
                    messages=self._history,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout,
                )

                # Track usage (tool name will be added later after parsing)
                self.usage.add_usage(usage_stats, tool_name=None)

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

                return response_text
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

        return f"Provider error: {last_error or 'unknown error'}"

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

    def _append_assistant_and_tool(
        self,
        assistant_content: str,
        tool_content: str,
        tool_name: str,
        tool_result: Optional[str] = None,
    ) -> None:
        """Update history with assistant response and tool output."""
        assistant_msg = Message(role=Role.ASSISTANT, content=assistant_content)
        tool_msg = Message(
            role=Role.TOOL,
            content=tool_content,
            tool_name=tool_name,
            tool_result=tool_result,
        )

        self._history.append(assistant_msg)
        self._history.append(tool_msg)

        # Also add to memory if available
        if self.memory:
            self.memory.add_many([assistant_msg, tool_msg])

    def _execute_tool_with_timeout(self, tool: Tool, parameters: dict) -> str:
        """Run tool.execute with optional timeout."""
        if not self.config.tool_timeout_seconds:
            return tool.execute(parameters)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.execute, parameters)
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
    async def arun(
        self, messages: List[Message], stream_handler: Optional[Callable[[str], None]] = None
    ) -> Message:
        """
        Async version of run().

        Execute the agent loop asynchronously with the provided conversation history.
        Uses provider async methods if available, falls back to sync in executor.

        Args:
            messages: New messages for this turn (typically a single user message).
            stream_handler: Optional callback for streaming responses.

        Returns:
            The final assistant response message.
        """
        # Load history from memory if available, then append new messages
        if self.memory:
            self._history = self.memory.get_history() + list(messages)
            self.memory.add_many(messages)
        else:
            self._history = list(messages)

        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            response_text = await self._acall_provider(stream_handler=stream_handler)
            parse_result = self.parser.parse(response_text)

            if not parse_result.tool_call:
                final_response = Message(role=Role.ASSISTANT, content=response_text)
                if self.memory:
                    self.memory.add(final_response)
                return final_response

            tool_name = parse_result.tool_call.tool_name
            parameters = parse_result.tool_call.parameters

            if self.config.verbose:
                print(f"[agent] Iteration {iteration}: tool={tool_name} params={parameters}")

            tool = self._tools_by_name.get(tool_name)
            if not tool:
                error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                self._append_assistant_and_tool(response_text, error_message, tool_name)
                continue

            try:
                result = await self._aexecute_tool_with_timeout(tool, parameters)
            except Exception as exc:
                error_message = f"Error executing tool '{tool.name}': {exc}"
                self._append_assistant_and_tool(response_text, error_message, tool.name)
                continue

            self._append_assistant_and_tool(response_text, result, tool.name, tool_result=result)

        final_response = Message(
            role=Role.ASSISTANT,
            content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
        )
        if self.memory:
            self.memory.add(final_response)
        return final_response

    async def _acall_provider(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        """Async version of _call_provider with retry logic."""
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    return await self._astreaming_call(stream_handler=stream_handler)

                # Check if provider has async support
                if hasattr(self.provider, "acomplete") and getattr(
                    self.provider, "supports_async", False
                ):
                    response_text, usage_stats = await self.provider.acomplete(
                        model=self.config.model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )
                else:
                    # Fallback to sync in executor
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        response_text, usage_stats = await loop.run_in_executor(
                            executor,
                            lambda: self.provider.complete(
                                model=self.config.model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )

                # Track usage
                self.usage.add_usage(usage_stats, tool_name=None)

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

                return response_text
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

        return f"Provider error: {last_error or 'unknown error'}"

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

    async def _aexecute_tool_with_timeout(self, tool: Tool, parameters: dict) -> str:
        """Async version of _execute_tool_with_timeout."""
        if not self.config.tool_timeout_seconds:
            return await tool.aexecute(parameters)

        try:
            return await asyncio.wait_for(
                tool.aexecute(parameters), timeout=self.config.tool_timeout_seconds
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

    def reset_usage(self) -> None:
        """
        Reset usage tracking statistics.

        Useful when starting a new conversation or session while reusing
        the same agent instance.
        """
        self.usage = AgentUsage()


__all__ = ["Agent", "AgentConfig"]
