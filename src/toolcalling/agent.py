"""
Provider-agnostic agent loop implementing the TOOL_CALL contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, List, Optional

from .types import Message, Role
from .tools import Tool
from .parser import ToolCallParser
from .prompt import PromptBuilder
from .providers.base import Provider, ProviderError
from .providers.openai_provider import OpenAIProvider


@dataclass
class AgentConfig:
    """Tunable controls for the agent loop."""

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


class Agent:
    """Run an iterative tool-calling loop using a pluggable provider."""

    def __init__(
        self,
        tools: List[Tool],
        provider: Optional[Provider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        parser: Optional[ToolCallParser] = None,
        config: Optional[AgentConfig] = None,
    ):
        if not tools:
            raise ValueError("Agent requires at least one tool.")

        self.tools = tools
        self._tools_by_name = {tool.name: tool for tool in tools}
        self.provider = provider or OpenAIProvider()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ToolCallParser()
        self.config = config or AgentConfig()

        self._system_prompt = self.prompt_builder.build(self.tools)
        self._history: List[Message] = []

    def run(self, messages: List[Message], stream_handler: Optional[Callable[[str], None]] = None) -> Message:
        """Execute the agent loop with the provided conversation history."""
        self._history = list(messages)
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            response_text = self._call_provider(stream_handler=stream_handler)
            parse_result = self.parser.parse(response_text)

            if not parse_result.tool_call:
                return Message(role=Role.ASSISTANT, content=response_text)

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
            except Exception as exc:  # noqa: BLE001
                error_message = f"Error executing tool '{tool.name}': {exc}"
                self._append_assistant_and_tool(response_text, error_message, tool.name)
                continue

            self._append_assistant_and_tool(response_text, result, tool.name, tool_result=result)

        return Message(
            role=Role.ASSISTANT,
            content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
        )

    def _call_provider(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    return self._streaming_call(stream_handler=stream_handler)

                return self.provider.complete(
                    model=self.config.model,
                    system_prompt=self._system_prompt,
                    messages=self._history,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout,
                )
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}")
                if attempts > self.config.max_retries:
                    break
                if self._is_rate_limit_error(last_error) and self.config.rate_limit_cooldown_seconds:
                    time.sleep(self.config.rate_limit_cooldown_seconds * attempts)
                if self.config.retry_backoff_seconds:
                    time.sleep(self.config.retry_backoff_seconds * attempts)

        return f"Provider error: {last_error or 'unknown error'}"

    def _streaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

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
        self._history.append(Message(role=Role.ASSISTANT, content=assistant_content))
        self._history.append(
            Message(
                role=Role.TOOL,
                content=tool_content,
                tool_name=tool_name,
                tool_result=tool_result,
            )
        )

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


__all__ = ["Agent", "AgentConfig"]
