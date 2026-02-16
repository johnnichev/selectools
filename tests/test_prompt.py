"""
Comprehensive tests for PromptBuilder (prompt.py).

Tests cover:
- Default system instructions
- Custom base instructions
- build() with empty tools list
- build() with single tool
- build() with multiple tools
- Output format validation (tool schema rendering)
- Tool schema JSON correctness
"""

from __future__ import annotations

import json
from typing import List

import pytest

from selectools.prompt import DEFAULT_SYSTEM_INSTRUCTIONS, PromptBuilder
from selectools.tools import Tool, ToolParameter


@pytest.fixture
def simple_tool() -> Tool:
    """A minimal tool for testing prompt building."""

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return Tool(
        name="greet",
        description="Greet someone by name",
        parameters=[ToolParameter(name="name", param_type=str, description="Person's name")],
        function=greet,
    )


@pytest.fixture
def calculator_tool() -> Tool:
    """A tool with multiple parameters."""

    def add(a: int, b: int) -> str:
        return str(a + b)

    return Tool(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter(name="a", param_type=int, description="First number"),
            ToolParameter(name="b", param_type=int, description="Second number"),
        ],
        function=add,
    )


@pytest.fixture
def optional_param_tool() -> Tool:
    """A tool with optional parameters."""

    def search(query: str, limit: int = 5) -> str:
        return f"Searching: {query} (limit={limit})"

    return Tool(
        name="search",
        description="Search for items",
        parameters=[
            ToolParameter(name="query", param_type=str, description="Search query"),
            ToolParameter(name="limit", param_type=int, description="Result limit", required=False),
        ],
        function=search,
    )


class TestDefaultInstructions:
    """Tests for DEFAULT_SYSTEM_INSTRUCTIONS constant."""

    def test_contains_tool_call_contract(self) -> None:
        assert "TOOL_CALL" in DEFAULT_SYSTEM_INSTRUCTIONS

    def test_contains_json_format(self) -> None:
        assert "tool_name" in DEFAULT_SYSTEM_INSTRUCTIONS
        assert "parameters" in DEFAULT_SYSTEM_INSTRUCTIONS

    def test_is_nonempty_string(self) -> None:
        assert isinstance(DEFAULT_SYSTEM_INSTRUCTIONS, str)
        assert len(DEFAULT_SYSTEM_INSTRUCTIONS.strip()) > 0


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_default_instructions(self) -> None:
        builder = PromptBuilder()
        assert builder.base_instructions == DEFAULT_SYSTEM_INSTRUCTIONS

    def test_custom_instructions(self) -> None:
        custom = "You are a helpful coding assistant."
        builder = PromptBuilder(base_instructions=custom)
        assert builder.base_instructions == custom

    def test_empty_instructions(self) -> None:
        builder = PromptBuilder(base_instructions="")
        assert builder.base_instructions == ""


class TestPromptBuilderBuild:
    """Tests for PromptBuilder.build() method."""

    def test_build_empty_tools(self) -> None:
        """build() with no tools should still produce a valid prompt."""
        builder = PromptBuilder()
        prompt = builder.build([])

        assert "Available tools (JSON schema):" in prompt
        assert "TOOL_CALL" in prompt
        assert "answer directly" in prompt

    def test_build_single_tool(self, simple_tool: Tool) -> None:
        builder = PromptBuilder()
        prompt = builder.build([simple_tool])

        assert "greet" in prompt
        assert "Greet someone by name" in prompt
        assert "name" in prompt

    def test_build_multiple_tools(self, simple_tool: Tool, calculator_tool: Tool) -> None:
        builder = PromptBuilder()
        prompt = builder.build([simple_tool, calculator_tool])

        assert "greet" in prompt
        assert "add" in prompt
        assert "First number" in prompt
        assert "Person's name" in prompt

    def test_build_includes_tool_schemas_as_json(self, simple_tool: Tool) -> None:
        """Tool schemas should be valid JSON blocks in the prompt."""
        builder = PromptBuilder()
        prompt = builder.build([simple_tool])

        lines = prompt.split("Available tools (JSON schema):\n\n")[1]
        json_block = lines.split("\n\nIf a relevant tool exists")[0]
        parsed = json.loads(json_block)

        assert parsed["name"] == "greet"
        assert "parameters" in parsed

    def test_build_tool_schema_properties(self, simple_tool: Tool) -> None:
        """Tool schema should have correct properties section."""
        builder = PromptBuilder()
        prompt = builder.build([simple_tool])

        schema = simple_tool.schema()
        schema_json = json.dumps(schema, indent=2)
        assert schema_json in prompt

    def test_build_with_optional_params(self, optional_param_tool: Tool) -> None:
        builder = PromptBuilder()
        prompt = builder.build([optional_param_tool])

        assert "search" in prompt
        assert "query" in prompt
        assert "limit" in prompt

    def test_build_preserves_custom_instructions(self, simple_tool: Tool) -> None:
        custom = "You are a code review assistant."
        builder = PromptBuilder(base_instructions=custom)
        prompt = builder.build([simple_tool])

        assert prompt.startswith("You are a code review assistant.")

    def test_build_output_structure(self, simple_tool: Tool) -> None:
        """Prompt should follow: instructions + schemas + closing directive."""
        builder = PromptBuilder()
        prompt = builder.build([simple_tool])

        assert "Available tools (JSON schema):" in prompt
        assert "If a relevant tool exists, respond with a TOOL_CALL first." in prompt
        assert "When no tool is useful, answer directly." in prompt

    def test_build_strips_base_instructions(self) -> None:
        """Leading/trailing whitespace in base_instructions should be stripped."""
        builder = PromptBuilder(base_instructions="  hello world  \n\n")
        prompt = builder.build([])

        assert prompt.startswith("hello world\n\n")

    def test_build_multiple_tools_separated(self, simple_tool: Tool, calculator_tool: Tool) -> None:
        """Multiple tool schemas should be separated by double newlines."""
        builder = PromptBuilder()
        prompt = builder.build([simple_tool, calculator_tool])

        schema1 = json.dumps(simple_tool.schema(), indent=2)
        schema2 = json.dumps(calculator_tool.schema(), indent=2)
        combined = f"{schema1}\n\n{schema2}"
        assert combined in prompt

    def test_build_returns_string(self, simple_tool: Tool) -> None:
        builder = PromptBuilder()
        result = builder.build([simple_tool])
        assert isinstance(result, str)


class TestPromptBuilderEdgeCases:
    """Edge case tests for PromptBuilder."""

    def test_tool_with_enum_parameter(self) -> None:
        """Enum values should appear in the rendered schema."""

        def set_mode(mode: str) -> str:
            return mode

        tool = Tool(
            name="set_mode",
            description="Set operating mode",
            parameters=[
                ToolParameter(
                    name="mode",
                    param_type=str,
                    description="The mode",
                    enum=["fast", "slow", "balanced"],
                ),
            ],
            function=set_mode,
        )

        builder = PromptBuilder()
        prompt = builder.build([tool])

        assert "fast" in prompt
        assert "slow" in prompt
        assert "balanced" in prompt

    def test_tool_with_no_parameters(self) -> None:
        """Tool with no params should still render cleanly."""

        def noop() -> str:
            return "done"

        tool = Tool(
            name="noop",
            description="Does nothing useful",
            parameters=[],
            function=noop,
        )

        builder = PromptBuilder()
        prompt = builder.build([tool])

        assert "noop" in prompt
        assert "Does nothing useful" in prompt

    def test_many_tools(self) -> None:
        """Build prompt with many tools to ensure no truncation."""
        tools: List[Tool] = []
        for i in range(20):

            def fn(x: str) -> str:
                return x

            tools.append(
                Tool(
                    name=f"tool_{i}",
                    description=f"Tool number {i}",
                    parameters=[
                        ToolParameter(name="x", param_type=str, description="Input"),
                    ],
                    function=fn,
                )
            )

        builder = PromptBuilder()
        prompt = builder.build(tools)

        for i in range(20):
            assert f"tool_{i}" in prompt
            assert f"Tool number {i}" in prompt
