"""
Comprehensive tests for ToolCallParser (parser.py).

Tests cover:
- ParseResult dataclass
- parse() with TOOL_CALL marker
- parse() with fenced code blocks
- parse() with inline JSON
- parse() with key aliases (tool/name, params)
- _load_json() fallback strategies
- _find_balanced_json() edge cases
- Custom marker and max_payload_chars
- Size limit enforcement
- Deduplication of candidate blocks
- Mixed text with tool calls
"""

from __future__ import annotations

from selectools.parser import ParseResult, ToolCallParser


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_with_tool_call(self) -> None:
        from selectools.types import ToolCall

        tc = ToolCall(tool_name="greet", parameters={"name": "Alice"})
        result = ParseResult(tool_call=tc, raw_text="some text")

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"
        assert result.tool_call.parameters == {"name": "Alice"}
        assert result.raw_text == "some text"

    def test_without_tool_call(self) -> None:
        result = ParseResult(tool_call=None, raw_text="no tool here")
        assert result.tool_call is None
        assert result.raw_text == "no tool here"


class TestToolCallParserInit:
    """Tests for ToolCallParser initialization."""

    def test_default_marker(self) -> None:
        parser = ToolCallParser()
        assert parser.marker == "TOOL_CALL"

    def test_default_max_payload(self) -> None:
        parser = ToolCallParser()
        assert parser.max_payload_chars == 8000

    def test_custom_marker(self) -> None:
        parser = ToolCallParser(marker="FUNCTION_CALL")
        assert parser.marker == "FUNCTION_CALL"

    def test_custom_max_payload(self) -> None:
        parser = ToolCallParser(max_payload_chars=2000)
        assert parser.max_payload_chars == 2000


class TestParseWithToolCallMarker:
    """Tests for parse() with TOOL_CALL marker in text."""

    def test_standard_tool_call(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"
        assert result.tool_call.parameters == {"name": "Alice"}

    def test_tool_call_with_preceding_text(self) -> None:
        parser = ToolCallParser()
        text = (
            "I will greet Alice now.\n"
            'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}'
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_tool_call_with_trailing_text(self) -> None:
        parser = ToolCallParser()
        text = (
            'TOOL_CALL: {"tool_name": "search", "parameters": {"query": "python"}}\n'
            "I will search for Python information."
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search"


class TestParseWithFencedCodeBlocks:
    """Tests for parse() with fenced code blocks."""

    def test_fenced_json_block(self) -> None:
        parser = ToolCallParser()
        text = '```json\nTOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Bob"}}\n```'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_fenced_block_without_marker(self) -> None:
        """Fenced block with tool_name/parameters keys should still be parsed."""
        parser = ToolCallParser()
        text = '```\n{"tool_name": "calc", "parameters": {"x": 5}}\n```'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "calc"

    def test_fenced_block_with_language_tag(self) -> None:
        parser = ToolCallParser()
        text = '```json\n{"tool_name": "search", "parameters": {"q": "test"}}\n```'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search"


class TestParseWithInlineJSON:
    """Tests for parse() with inline JSON (no marker, no fences)."""

    def test_bare_json_object(self) -> None:
        parser = ToolCallParser()
        text = '{"tool_name": "echo", "parameters": {"text": "hello"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "echo"

    def test_json_in_surrounding_text(self) -> None:
        parser = ToolCallParser()
        text = (
            "Let me call a tool: "
            '{"tool_name": "greet", "parameters": {"name": "Charlie"}} '
            "and then respond."
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"


class TestParseKeyAliases:
    """Tests for alternative key names: tool, name, params."""

    def test_tool_key(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool": "greet", "parameters": {"name": "Alice"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_name_key(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"name": "greet", "parameters": {"name": "Alice"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_params_key(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool_name": "greet", "params": {"name": "Alice"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"
        assert result.tool_call.parameters == {"name": "Alice"}

    def test_tool_and_params_keys(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool": "search", "params": {"q": "test"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search"
        assert result.tool_call.parameters == {"q": "test"}


class TestParseNoToolCall:
    """Tests where no tool call should be detected."""

    def test_plain_text(self) -> None:
        parser = ToolCallParser()
        result = parser.parse("Hello, how are you today?")

        assert result.tool_call is None

    def test_empty_string(self) -> None:
        parser = ToolCallParser()
        result = parser.parse("")

        assert result.tool_call is None

    def test_json_without_tool_name(self) -> None:
        parser = ToolCallParser()
        text = '{"data": {"key": "value"}}'
        result = parser.parse(text)

        assert result.tool_call is None

    def test_json_with_unrelated_keys(self) -> None:
        parser = ToolCallParser()
        text = '{"color": "blue", "size": 42}'
        result = parser.parse(text)

        assert result.tool_call is None

    def test_raw_text_preserved(self) -> None:
        parser = ToolCallParser()
        text = "Just a normal response."
        result = parser.parse(text)

        assert result.raw_text == text


class TestParseNoParameters:
    """Tests for tool calls with no or empty parameters."""

    def test_empty_parameters(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool_name": "noop", "parameters": {}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "noop"
        assert result.tool_call.parameters == {}

    def test_missing_parameters_key(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool_name": "noop"}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "noop"
        assert result.tool_call.parameters == {}


class TestSizeLimitEnforcement:
    """Tests for max_payload_chars enforcement."""

    def test_payload_within_limit(self) -> None:
        parser = ToolCallParser(max_payload_chars=500)
        text = 'TOOL_CALL: {"tool_name": "echo", "parameters": {"text": "short"}}'
        result = parser.parse(text)

        assert result.tool_call is not None

    def test_payload_exceeds_limit(self) -> None:
        parser = ToolCallParser(max_payload_chars=50)
        long_value = "x" * 100
        text = f'TOOL_CALL: {{"tool_name": "echo", "parameters": {{"text": "{long_value}"}}}}'
        result = parser.parse(text)

        assert result.tool_call is None

    def test_zero_max_payload_skips_all(self) -> None:
        """When max_payload_chars is 0 (falsy), size check is skipped."""
        parser = ToolCallParser(max_payload_chars=0)
        text = 'TOOL_CALL: {"tool_name": "echo", "parameters": {"text": "hello"}}'
        result = parser.parse(text)

        assert result.tool_call is not None


class TestCustomMarker:
    """Tests with custom marker string."""

    def test_custom_marker_parsed(self) -> None:
        parser = ToolCallParser(marker="FUNCTION_CALL")
        text = 'FUNCTION_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_default_marker_not_matched_with_custom(self) -> None:
        parser = ToolCallParser(marker="FUNCTION_CALL")
        text = 'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}'
        result = parser.parse(text)

        # Should still find it via fallback balanced-json scan
        assert result.tool_call is not None


class TestLoadJsonFallbacks:
    """Tests for _load_json() lenient parsing."""

    def test_single_quotes_fallback(self) -> None:
        parser = ToolCallParser()
        text = "TOOL_CALL: {'tool_name': 'greet', 'parameters': {'name': 'Alice'}}"
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"

    def test_non_dict_json_ignored(self) -> None:
        """JSON arrays or primitives should not be treated as tool calls."""
        parser = ToolCallParser()
        text = 'TOOL_CALL: ["not", "a", "dict"]'
        result = parser.parse(text)

        assert result.tool_call is None


class TestBalancedJsonEdgeCases:
    """Tests for _find_balanced_json() helper."""

    def test_nested_braces(self) -> None:
        parser = ToolCallParser()
        text = 'TOOL_CALL: {"tool_name": "process", "parameters": {"data": {"nested": true}}}'
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "process"
        assert result.tool_call.parameters == {"data": {"nested": True}}

    def test_multiple_json_objects_first_wins(self) -> None:
        parser = ToolCallParser()
        text = (
            'TOOL_CALL: {"tool_name": "first", "parameters": {}}\n'
            '{"tool_name": "second", "parameters": {}}'
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "first"

    def test_no_braces(self) -> None:
        parser = ToolCallParser()
        candidates = parser._find_balanced_json("no braces here")
        assert candidates == []

    def test_unbalanced_braces(self) -> None:
        parser = ToolCallParser()
        candidates = parser._find_balanced_json("{ unbalanced")
        assert candidates == []


class TestMultipleCandidates:
    """Tests for deduplication and multiple candidate handling."""

    def test_deduplication(self) -> None:
        parser = ToolCallParser()
        text = (
            '```json\nTOOL_CALL: {"tool_name": "echo", "parameters": {"text": "hi"}}\n```\n'
            'TOOL_CALL: {"tool_name": "echo", "parameters": {"text": "hi"}}'
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "echo"

    def test_mixed_text_with_multiple_jsons(self) -> None:
        parser = ToolCallParser()
        text = (
            'Here is some context {"irrelevant": true} and now:\n'
            'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}\n'
            "That was the tool call."
        )
        result = parser.parse(text)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "greet"
