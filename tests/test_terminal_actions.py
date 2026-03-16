"""Tests for terminal action support (FR-001).

Terminal tools stop the agent loop immediately and return the tool result
as the final response.  The ``stop_condition`` callback on ``AgentConfig``
provides a dynamic alternative.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from selectools import Agent, AgentConfig
from selectools.tools import Tool, ToolParameter, tool
from selectools.types import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "greet", *, terminal: bool = False) -> Tool:
    """Create a simple tool that returns a greeting."""
    return Tool(
        name=name,
        description="Say hello",
        parameters=[
            ToolParameter(name="name", param_type=str, description="Who to greet"),
        ],
        function=lambda name: f"Hello, {name}!",
        terminal=terminal,
    )


def _make_agent(
    provider,
    tools: List[Tool],
    *,
    stop_condition=None,
    parallel: bool = False,
) -> Agent:
    return Agent(
        tools=tools,
        provider=provider,
        config=AgentConfig(
            model="fake-model",
            max_iterations=6,
            parallel_tool_execution=parallel,
            stop_condition=stop_condition,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Tool with terminal=True stops the loop after execution
# ---------------------------------------------------------------------------


class TestTerminalToolStopsLoop:
    def test_terminal_tool_stops_loop(self, tool_call_provider):
        """A tool marked terminal=True causes the agent to return its result."""
        terminal_tool = _make_tool("greet", terminal=True)
        non_terminal = _make_tool("other", terminal=False)

        # Provider returns two iterations of tool calls, but only the first
        # should execute because it's terminal.
        provider = tool_call_provider(
            responses=[
                ([ToolCall(tool_name="greet", parameters={"name": "Alice"})], ""),
                ([ToolCall(tool_name="other", parameters={"name": "Bob"})], ""),
            ]
        )

        agent = _make_agent(provider, [terminal_tool, non_terminal])
        result = agent.run("Hi")

        assert result.content == "Hello, Alice!"
        # Only one iteration should have occurred (no second LLM call).
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# 2. stop_condition callback stops the loop when it returns True
# ---------------------------------------------------------------------------


class TestStopConditionStopsLoop:
    def test_stop_condition_true(self, tool_call_provider):
        """stop_condition returning True stops the loop."""
        my_tool = _make_tool("greet", terminal=False)

        provider = tool_call_provider(
            responses=[
                ([ToolCall(tool_name="greet", parameters={"name": "Alice"})], ""),
                ([ToolCall(tool_name="greet", parameters={"name": "Bob"})], ""),
            ]
        )

        def stop_if_alice(tool_name: str, result: str) -> bool:
            return "Alice" in result

        agent = _make_agent(provider, [my_tool], stop_condition=stop_if_alice)
        result = agent.run("Hi")

        assert result.content == "Hello, Alice!"
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# 3. stop_condition returning False does not stop the loop
# ---------------------------------------------------------------------------


class TestStopConditionFalseDoesNotStop:
    def test_stop_condition_false(self, tool_call_provider):
        """stop_condition returning False lets the loop continue."""
        my_tool = _make_tool("greet", terminal=False)

        provider = tool_call_provider(
            responses=[
                ([ToolCall(tool_name="greet", parameters={"name": "Alice"})], ""),
            ]
        )

        agent = _make_agent(provider, [my_tool], stop_condition=lambda name, result: False)
        result = agent.run("Hi")

        # Should have gone through the tool call, then the provider returned "Done"
        assert provider.calls == 2
        assert result.content == "Done"


# ---------------------------------------------------------------------------
# 4. Terminal tool result is in the AgentResult content
# ---------------------------------------------------------------------------


class TestTerminalResultInAgentResult:
    def test_content_matches_tool_result(self, tool_call_provider):
        """AgentResult.content should be the terminal tool's output."""
        terminal_tool = _make_tool("greet", terminal=True)

        provider = tool_call_provider(
            responses=[
                ([ToolCall(tool_name="greet", parameters={"name": "World"})], ""),
            ]
        )

        agent = _make_agent(provider, [terminal_tool])
        result = agent.run("Hi")

        assert result.content == "Hello, World!"
        assert result.tool_name == "greet"
        assert result.tool_args == {"name": "World"}


# ---------------------------------------------------------------------------
# 5. Terminal works in async (arun)
# ---------------------------------------------------------------------------


class TestTerminalAsync:
    @pytest.mark.asyncio
    async def test_terminal_arun(self, tool_call_provider):
        """Terminal tool works with arun()."""
        terminal_tool = _make_tool("greet", terminal=True)

        provider = tool_call_provider(
            responses=[
                ([ToolCall(tool_name="greet", parameters={"name": "Async"})], ""),
                ([ToolCall(tool_name="greet", parameters={"name": "Extra"})], ""),
            ]
        )

        agent = _make_agent(provider, [terminal_tool])
        result = await agent.arun("Hi")

        assert result.content == "Hello, Async!"
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# 6. @tool(terminal=True) decorator creates terminal tool
# ---------------------------------------------------------------------------


class TestToolDecoratorTerminal:
    def test_decorator_terminal_flag(self):
        """The @tool decorator passes terminal through to the Tool instance."""

        @tool(description="Say goodbye", terminal=True)
        def farewell(name: str) -> str:
            return f"Goodbye, {name}!"

        assert farewell.terminal is True
        assert farewell.name == "farewell"

    def test_decorator_terminal_default(self):
        """The @tool decorator defaults terminal to False."""

        @tool(description="Say hello")
        def hello(name: str) -> str:
            return f"Hello, {name}!"

        assert hello.terminal is False


# ---------------------------------------------------------------------------
# 7. Tool terminal defaults to False
# ---------------------------------------------------------------------------


class TestToolTerminalDefault:
    def test_default_false(self):
        """Tool.terminal defaults to False when not specified."""
        t = Tool(
            name="test",
            description="test tool",
            parameters=[],
            function=lambda: "ok",
        )
        assert t.terminal is False

    def test_explicit_true(self):
        """Tool.terminal can be set to True."""
        t = Tool(
            name="test",
            description="test tool",
            parameters=[],
            function=lambda: "ok",
            terminal=True,
        )
        assert t.terminal is True


# ---------------------------------------------------------------------------
# 8. Terminal works with parallel tool execution
# ---------------------------------------------------------------------------


class TestTerminalParallel:
    def test_terminal_in_parallel_batch(self, tool_call_provider):
        """When a terminal tool is in a parallel batch, the loop stops after that batch."""
        terminal_tool = _make_tool("greet", terminal=True)
        normal_tool = Tool(
            name="count",
            description="Count letters",
            parameters=[
                ToolParameter(name="text", param_type=str, description="Text"),
            ],
            function=lambda text: str(len(text)),
        )

        # Provider returns two tools in one response (parallel), then another call.
        # The second LLM call should never happen because "greet" is terminal.
        provider = tool_call_provider(
            responses=[
                (
                    [
                        ToolCall(tool_name="count", parameters={"text": "abc"}),
                        ToolCall(tool_name="greet", parameters={"name": "Parallel"}),
                    ],
                    "",
                ),
                ([ToolCall(tool_name="count", parameters={"text": "xyz"})], ""),
            ]
        )

        agent = _make_agent(provider, [terminal_tool, normal_tool], parallel=True)
        result = agent.run("Hi")

        assert result.content == "Hello, Parallel!"
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_terminal_in_parallel_batch_async(self, tool_call_provider):
        """Terminal parallel execution works with arun() too."""
        terminal_tool = _make_tool("greet", terminal=True)
        normal_tool = Tool(
            name="count",
            description="Count letters",
            parameters=[
                ToolParameter(name="text", param_type=str, description="Text"),
            ],
            function=lambda text: str(len(text)),
        )

        provider = tool_call_provider(
            responses=[
                (
                    [
                        ToolCall(tool_name="count", parameters={"text": "abc"}),
                        ToolCall(tool_name="greet", parameters={"name": "AsyncP"}),
                    ],
                    "",
                ),
                ([ToolCall(tool_name="count", parameters={"text": "xyz"})], ""),
            ]
        )

        agent = _make_agent(provider, [terminal_tool, normal_tool], parallel=True)
        result = await agent.arun("Hi")

        assert result.content == "Hello, AsyncP!"
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# 9. stop_condition with parallel execution
# ---------------------------------------------------------------------------


class TestStopConditionParallel:
    def test_stop_condition_parallel(self, tool_call_provider):
        """stop_condition works with parallel tool execution."""
        tool_a = Tool(
            name="tool_a",
            description="Tool A",
            parameters=[],
            function=lambda: "STOP_NOW",
        )
        tool_b = Tool(
            name="tool_b",
            description="Tool B",
            parameters=[],
            function=lambda: "keep going",
        )

        provider = tool_call_provider(
            responses=[
                (
                    [
                        ToolCall(tool_name="tool_a", parameters={}),
                        ToolCall(tool_name="tool_b", parameters={}),
                    ],
                    "",
                ),
                ([ToolCall(tool_name="tool_b", parameters={})], ""),
            ]
        )

        def stop_on_marker(tool_name: str, result: str) -> bool:
            return "STOP_NOW" in result

        agent = _make_agent(
            provider, [tool_a, tool_b], stop_condition=stop_on_marker, parallel=True
        )
        result = agent.run("Hi")

        assert result.content == "STOP_NOW"
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# 10. Terminal with sequential: second tool not executed
# ---------------------------------------------------------------------------


class TestTerminalSequentialSkipsRemaining:
    def test_sequential_break(self, tool_call_provider):
        """In sequential mode, tools after a terminal tool are not executed."""
        call_log: list = []

        def log_and_greet(name: str) -> str:
            call_log.append(("greet", name))
            return f"Hello, {name}!"

        def log_and_count(text: str) -> str:
            call_log.append(("count", text))
            return str(len(text))

        terminal_tool = Tool(
            name="greet",
            description="Say hello",
            parameters=[
                ToolParameter(name="name", param_type=str, description="Name"),
            ],
            function=log_and_greet,
            terminal=True,
        )
        normal_tool = Tool(
            name="count",
            description="Count",
            parameters=[
                ToolParameter(name="text", param_type=str, description="Text"),
            ],
            function=log_and_count,
        )

        # Two tool calls in one response, sequential mode.
        # "greet" is terminal, so "count" should NOT execute.
        provider = tool_call_provider(
            responses=[
                (
                    [
                        ToolCall(tool_name="greet", parameters={"name": "Alice"}),
                        ToolCall(tool_name="count", parameters={"text": "hello"}),
                    ],
                    "",
                ),
            ]
        )

        agent = _make_agent(provider, [terminal_tool, normal_tool], parallel=False)
        result = agent.run("Hi")

        assert result.content == "Hello, Alice!"
        assert call_log == [("greet", "Alice")]
