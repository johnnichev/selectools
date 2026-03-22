"""Tests for the MCP integration module.

Tests that don't require the mcp package (config, bridge schema conversion,
circuit breaker logic) run on any Python. Tests that need the mcp package
are skipped if it's not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.exceptions import MCPConnectionError, MCPError, MCPToolError
from selectools.mcp.config import MCPServerConfig

# ===========================================================================
# MCPServerConfig
# ===========================================================================


class TestMCPServerConfig:
    def test_stdio_basic(self) -> None:
        config = MCPServerConfig(command="python", args=["server.py"])
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.name == "server"  # auto-generated from args

    def test_stdio_no_args(self) -> None:
        config = MCPServerConfig(command="my-server")
        assert config.name == "my-server"

    def test_http_basic(self) -> None:
        config = MCPServerConfig(url="http://localhost:8080/mcp", transport="streamable-http")
        assert config.transport == "streamable-http"
        assert config.name == "localhost"  # auto-generated from URL

    def test_custom_name(self) -> None:
        config = MCPServerConfig(command="python", args=["s.py"], name="search")
        assert config.name == "search"

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError, match="command"):
            MCPServerConfig(transport="stdio")

    def test_http_requires_url(self) -> None:
        with pytest.raises(ValueError, match="url"):
            MCPServerConfig(transport="streamable-http")

    def test_unknown_transport(self) -> None:
        with pytest.raises(ValueError, match="Unknown transport"):
            MCPServerConfig(transport="websocket", command="x")

    def test_defaults(self) -> None:
        config = MCPServerConfig(command="x")
        assert config.timeout == 30.0
        assert config.max_retries == 2
        assert config.auto_reconnect is True
        assert config.circuit_breaker_threshold == 3
        assert config.screen_output is True
        assert config.cache_tools is True

    def test_custom_settings(self) -> None:
        config = MCPServerConfig(
            command="x",
            timeout=60.0,
            max_retries=5,
            circuit_breaker_threshold=10,
            screen_output=False,
        )
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.circuit_breaker_threshold == 10
        assert config.screen_output is False


# ===========================================================================
# Bridge — schema conversion
# ===========================================================================


class TestBridgeSchemaConversion:
    def test_basic_string_param(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            }
        )
        assert len(params) == 1
        assert params[0].name == "query"
        assert params[0].param_type == str
        assert params[0].required is True
        assert params[0].description == "Search query"

    def test_multiple_types(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "score": {"type": "number"},
                    "active": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "metadata": {"type": "object"},
                },
                "required": ["name"],
            }
        )
        assert len(params) == 6
        types = {p.name: p.param_type for p in params}
        assert types["name"] == str
        assert types["age"] == int
        assert types["score"] == float
        assert types["active"] == bool
        assert types["tags"] == list
        assert types["metadata"] == dict

    def test_optional_params(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "string"},
                },
                "required": ["required_param"],
            }
        )
        required_p = next(p for p in params if p.name == "required_param")
        optional_p = next(p for p in params if p.name == "optional_param")
        assert required_p.required is True
        assert optional_p.required is False

    def test_enum_param(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["red", "green", "blue"]},
                },
            }
        )
        assert params[0].enum == ["red", "green", "blue"]

    def test_nullable_type(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "null"]},
                },
            }
        )
        assert params[0].param_type == str

    def test_default_value_in_description(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params(
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
            }
        )
        assert "default: 10" in params[0].description

    def test_empty_schema(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        params = _schema_to_params({"type": "object"})
        assert params == []

    def test_raw_schema_preserved(self) -> None:
        from selectools.mcp.bridge import _schema_to_params

        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            },
        }
        params = _schema_to_params(schema)
        assert hasattr(params[0], "_raw_schema")
        assert params[0]._raw_schema["type"] == "object"


class TestBridgeToolConversion:
    def test_mcp_to_tool(self) -> None:
        from selectools.mcp.bridge import mcp_to_tool

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "search"
        mock_mcp_tool.description = "Search the web"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        mock_mcp_tool.annotations = None

        async def call_fn(name: str, arguments: Dict) -> str:
            return "result"

        tool = mcp_to_tool(mock_mcp_tool, call_fn, server_name="test")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert len(tool.parameters) == 1
        assert tool._mcp_server == "test"

    def test_mcp_to_tool_with_prefix(self) -> None:
        from selectools.mcp.bridge import mcp_to_tool

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        mock_tool.annotations = None

        tool = mcp_to_tool(mock_tool, AsyncMock(), server_name="api", prefix=True)
        assert tool.name == "api_search"

    def test_tool_to_mcp_schema(self) -> None:
        from selectools import tool as tool_decorator
        from selectools.mcp.bridge import tool_to_mcp_schema

        @tool_decorator(description="Get weather")
        def get_weather(city: str) -> str:
            return f"72F in {city}"

        schema = tool_to_mcp_schema(get_weather)
        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get weather"
        assert "properties" in schema["inputSchema"]
        assert "city" in schema["inputSchema"]["properties"]


# ===========================================================================
# Circuit breaker logic
# ===========================================================================


class TestCircuitBreaker:
    def test_circuit_starts_closed(self) -> None:
        from selectools.mcp.client import MCPClient

        config = MCPServerConfig(command="x")
        client = MCPClient(config)
        assert not client.circuit_open

    def test_circuit_opens_after_threshold(self) -> None:
        from selectools.mcp.client import MCPClient

        config = MCPServerConfig(command="x", circuit_breaker_threshold=3)
        client = MCPClient(config)
        client._failure_count = 3
        client._circuit_open_until = 9999999999.0
        assert client.circuit_open

    def test_circuit_closes_after_cooldown(self) -> None:
        import time

        from selectools.mcp.client import MCPClient

        config = MCPServerConfig(command="x", circuit_breaker_cooldown=0.1)
        client = MCPClient(config)
        client._failure_count = 3
        client._circuit_open_until = time.time() - 1  # Already expired
        assert not client.circuit_open
        assert client._failure_count == 0  # Reset


# ===========================================================================
# Exceptions
# ===========================================================================


class TestMCPExceptions:
    def test_hierarchy(self) -> None:
        assert issubclass(MCPConnectionError, MCPError)
        assert issubclass(MCPToolError, MCPError)
        assert issubclass(MCPError, Exception)

    def test_raise_mcp_error(self) -> None:
        with pytest.raises(MCPError):
            raise MCPConnectionError("Connection failed")

    def test_raise_tool_error(self) -> None:
        with pytest.raises(MCPToolError):
            raise MCPToolError("Tool call failed")


# ===========================================================================
# MCPServer (unit tests — no actual server)
# ===========================================================================


class TestMCPServerUnit:
    def test_server_creation(self) -> None:
        """MCPServer can be created without the mcp package installed."""
        from selectools import tool as tool_decorator

        @tool_decorator(description="test")
        def dummy(x: str) -> str:
            return x

        # This should not raise — it's lazy
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[dummy], name="test-server")
        assert server.name == "test-server"
        assert len(server.tools) == 1


# ===========================================================================
# Package imports
# ===========================================================================


class TestMCPImports:
    def test_config_import(self) -> None:
        from selectools.mcp import MCPServerConfig

        assert MCPServerConfig is not None

    def test_mcp_tools_import(self) -> None:
        from selectools.mcp import mcp_tools

        assert callable(mcp_tools)

    def test_lazy_client_import(self) -> None:
        """MCPClient is lazy-loaded to avoid requiring mcp package."""
        from selectools.mcp import MCPClient

        assert MCPClient is not None

    def test_lazy_server_import(self) -> None:
        from selectools.mcp import MCPServer

        assert MCPServer is not None


# ===========================================================================
# Background event loop
# ===========================================================================


class TestBackgroundLoop:
    def test_loop_runs_coroutine(self) -> None:
        import asyncio

        from selectools.mcp._loop import _BackgroundLoop

        loop = _BackgroundLoop()
        try:

            async def add(a: int, b: int) -> int:
                return a + b

            result = loop.run(add(3, 4))
            assert result == 7
        finally:
            loop.stop()

    def test_singleton_loop(self) -> None:
        from selectools.mcp._loop import get_background_loop

        loop1 = get_background_loop()
        loop2 = get_background_loop()
        assert loop1 is loop2
