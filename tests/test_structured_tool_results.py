"""Tests for structured tool result serialization (R8)."""

import json
from dataclasses import dataclass

import pytest

from selectools.tools.base import Tool, ToolParameter


def _make_tool(func):
    """Helper to create a Tool from a plain function."""
    return Tool(
        name=func.__name__,
        description="test tool",
        parameters=[],
        function=func,
    )


class TestStructuredToolResults:
    """Verify that Tool.execute() auto-serializes non-string returns."""

    def test_dict_returns_json(self):
        def get_data() -> dict:
            return {"users": 42, "revenue": 1000}

        tool = _make_tool(get_data)
        result = tool.execute({})
        parsed = json.loads(result)
        assert parsed == {"users": 42, "revenue": 1000}

    def test_list_returns_json_array(self):
        def get_items() -> list:
            return [1, "two", 3.0]

        tool = _make_tool(get_items)
        result = tool.execute({})
        parsed = json.loads(result)
        assert parsed == [1, "two", 3.0]

    def test_nested_dict_returns_json(self):
        def get_nested() -> dict:
            return {"a": {"b": [1, 2]}, "c": True}

        tool = _make_tool(get_nested)
        result = tool.execute({})
        parsed = json.loads(result)
        assert parsed == {"a": {"b": [1, 2]}, "c": True}

    def test_string_passthrough(self):
        def get_text() -> str:
            return "hello world"

        tool = _make_tool(get_text)
        result = tool.execute({})
        assert result == "hello world"

    def test_int_falls_back_to_str(self):
        def get_number() -> int:
            return 42

        tool = _make_tool(get_number)
        result = tool.execute({})
        assert result == "42"

    def test_dataclass_returns_json(self):
        @dataclass
        class Metrics:
            users: int = 10
            active: bool = True

        def get_metrics() -> Metrics:
            return Metrics(users=42, active=False)

        tool = _make_tool(get_metrics)
        result = tool.execute({})
        parsed = json.loads(result)
        assert parsed == {"users": 42, "active": False}

    def test_pydantic_model_returns_json(self):
        """Test Pydantic v2 model serialization (if pydantic is installed)."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class UserModel(BaseModel):
            name: str = "Alice"
            age: int = 30

        def get_user() -> UserModel:
            return UserModel()

        tool = _make_tool(get_user)
        result = tool.execute({})
        parsed = json.loads(result)
        assert parsed == {"name": "Alice", "age": 30}

    def test_dict_with_non_serializable_values(self):
        """Non-JSON-serializable values use default=str fallback."""
        from datetime import datetime

        def get_data() -> dict:
            return {"ts": datetime(2026, 1, 1)}

        tool = _make_tool(get_data)
        result = tool.execute({})
        parsed = json.loads(result)
        assert "2026" in parsed["ts"]

    @pytest.mark.asyncio
    async def test_async_dict_returns_json(self):
        async def get_data() -> dict:
            return {"key": "value"}

        tool = _make_tool(get_data)
        result = await tool.aexecute({})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    @pytest.mark.asyncio
    async def test_async_string_passthrough(self):
        async def get_text() -> str:
            return "async hello"

        tool = _make_tool(get_text)
        result = await tool.aexecute({})
        assert result == "async hello"
