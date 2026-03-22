"""Tests for pre-execution token estimation (R6)."""

import json

from selectools.token_estimation import TokenEstimate, estimate_run_tokens, estimate_tokens
from selectools.tools.base import Tool
from selectools.types import Message, Role


class TestEstimateTokens:
    """Unit tests for the estimate_tokens function."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_empty_string_returns_zero_with_model(self):
        assert estimate_tokens("", model="claude-3-5-sonnet-20240620") == 0

    def test_short_text_returns_positive(self):
        result = estimate_tokens("Hello, world!")
        assert result > 0

    def test_longer_text_returns_more_tokens(self):
        short = estimate_tokens("Hi")
        long = estimate_tokens("This is a much longer piece of text that should have more tokens")
        assert long > short

    def test_heuristic_roughly_correct(self):
        """Heuristic should be in the right ballpark (~4 chars per token)."""
        text = "a" * 400  # ~100 tokens
        result = estimate_tokens(text, model="unknown-model-xyz")
        assert 80 <= result <= 120


class TestEstimateRunTokens:
    """Tests for estimate_run_tokens."""

    def test_empty_run(self):
        result = estimate_run_tokens(messages=[], tools=[], system_prompt="")
        assert result.total_tokens == 0
        assert result.system_tokens == 0
        assert result.message_tokens == 0
        assert result.tool_schema_tokens == 0

    def test_system_prompt_counted(self):
        result = estimate_run_tokens(
            messages=[], tools=[], system_prompt="You are a helpful assistant."
        )
        assert result.system_tokens > 0
        assert result.total_tokens == result.system_tokens

    def test_messages_counted(self):
        msgs = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
        result = estimate_run_tokens(messages=msgs, tools=[])
        assert result.message_tokens > 0

    def test_tools_counted(self):
        tool = Tool(
            name="search",
            description="Search the web for information",
            parameters=[],
            function=lambda: "result",
        )
        result = estimate_run_tokens(messages=[], tools=[tool])
        assert result.tool_schema_tokens > 0

    def test_total_is_sum(self):
        tool = Tool(name="calc", description="Calculate", parameters=[], function=lambda: "42")
        msgs = [Message(role=Role.USER, content="What is 2+2?")]
        result = estimate_run_tokens(messages=msgs, tools=[tool], system_prompt="Be precise.")
        assert result.total_tokens == (
            result.system_tokens + result.message_tokens + result.tool_schema_tokens
        )

    def test_remaining_tokens_arithmetic(self):
        result = estimate_run_tokens(messages=[], tools=[], system_prompt="x" * 100)
        if result.context_window > 0:
            assert result.remaining_tokens == result.context_window - result.total_tokens

    def test_known_model_has_context_window(self):
        result = estimate_run_tokens(messages=[], tools=[], model="gpt-4o")
        assert result.context_window > 0

    def test_unknown_model_has_zero_context_window(self):
        result = estimate_run_tokens(messages=[], tools=[], model="unknown-model-xyz")
        assert result.context_window == 0
        assert result.remaining_tokens == 0

    def test_method_field_set(self):
        result = estimate_run_tokens(messages=[], tools=[])
        assert result.method in ("tiktoken", "heuristic")

    def test_model_field_set(self):
        result = estimate_run_tokens(messages=[], tools=[], model="claude-3-opus-20240229")
        assert result.model == "claude-3-opus-20240229"

    def test_multiple_tools(self):
        tools = [
            Tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters=[],
                function=lambda: "ok",
            )
            for i in range(5)
        ]
        result = estimate_run_tokens(messages=[], tools=tools)
        assert result.tool_schema_tokens > 0

        # More tools should mean more tokens
        single = estimate_run_tokens(messages=[], tools=[tools[0]])
        assert result.tool_schema_tokens > single.tool_schema_tokens


class TestTokenEstimateDataclass:
    """Verify the TokenEstimate dataclass."""

    def test_fields(self):
        te = TokenEstimate(
            system_tokens=100,
            message_tokens=200,
            tool_schema_tokens=50,
            total_tokens=350,
            context_window=128000,
            remaining_tokens=127650,
            model="gpt-4o",
            method="heuristic",
        )
        assert te.system_tokens == 100
        assert te.total_tokens == 350
        assert te.remaining_tokens == 127650
        assert te.method == "heuristic"
