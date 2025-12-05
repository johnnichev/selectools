"""
Test Suite for the toolcalling library.

These tests avoid real API calls by using fake providers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from toolcalling.parser import ToolCallParser


class FakeProvider:
    """Minimal provider stub that returns queued responses."""

    name = "fake"

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def complete(self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000):  # noqa: D401
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_role_enum():
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"
    assert Role.TOOL.value == "tool"


def test_message_creation_and_image_encoding():
    image_path = Path(__file__).resolve().parents[1] / "assets" / "dog.png"
    try:
        msg = Message(role=Role.USER, content="What's in this image?", image_path=str(image_path))
        assert msg.image_base64 is not None
    except FileNotFoundError:
        # Skip if asset is missing in CI
        pass

    msg = Message(role=Role.USER, content="Hello")
    formatted = msg.to_dict()
    assert formatted["role"] == "user"
    assert formatted["content"] == "Hello"


def test_tool_schema_and_validation():
    param = ToolParameter(name="query", param_type=str, description="Search query", required=True)
    tool = Tool(name="search", description="Search the web", parameters=[param], function=lambda query: query)

    schema = tool.schema()
    assert schema["name"] == "search"
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["query"]["type"] == "string"

    is_valid, error = tool.validate({"query": "python"})
    assert is_valid is True and error is None

    is_valid, error = tool.validate({})
    assert is_valid is False and "Missing required parameter" in error


def test_parser_handles_fenced_blocks():
    parser = ToolCallParser()
    response = """
    Here you go:
    ```
    TOOL_CALL: {"tool_name": "search", "parameters": {"query": "docs"}}
    ```
    """
    result = parser.parse(response)
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search"
    assert result.tool_call.parameters["query"] == "docs"


def test_agent_executes_tool_and_returns_final_message():
    def add(a: int, b: int) -> str:
        return json.dumps({"sum": a + b})

    add_tool = Tool(
        name="add",
        description="Add two integers",
        parameters=[
            ToolParameter(name="a", param_type=int, description="first"),
            ToolParameter(name="b", param_type=int, description="second"),
        ],
        function=add,
    )

    provider = FakeProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "add", "parameters": {"a": 2, "b": 3}}',
            "The sum is 5.",
        ]
    )

    agent = Agent(
        tools=[add_tool],
        provider=provider,
        config=AgentConfig(max_iterations=3, model="fake-model"),
    )

    result = agent.run([Message(role=Role.USER, content="Add 2 and 3")])
    assert provider.calls == 2
    assert "5" in result.content


if __name__ == "__main__":
    # Simple runner for environments without pytest
    all_tests = [
        test_role_enum,
        test_message_creation_and_image_encoding,
        test_tool_schema_and_validation,
        test_parser_handles_fenced_blocks,
        test_agent_executes_tool_and_returns_final_message,
    ]
    failures = 0
    for test in all_tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as exc:  # noqa: BLE001
            failures += 1
            print(f"✗ {test.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
