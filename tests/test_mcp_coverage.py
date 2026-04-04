"""Tests for MCP modules — client, multi, server, bridge, __init__."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.mcp.bridge import _schema_to_params, mcp_to_tool, tool_to_mcp_schema
from selectools.mcp.config import MCPServerConfig
from selectools.tools.base import Tool, ToolParameter

# ===================================================================
# Helpers
# ===================================================================


def _make_config(name: str = "test-server", **kwargs) -> MCPServerConfig:
    """Build a minimal MCPServerConfig for stdio transport."""
    return MCPServerConfig(command="echo", args=["hello"], name=name, **kwargs)


def _make_http_config(name: str = "http-server", **kwargs) -> MCPServerConfig:
    return MCPServerConfig(
        transport="streamable-http",
        url="http://localhost:8080/mcp",
        name=name,
        **kwargs,
    )


@dataclass
class FakeMCPTool:
    """Mimics mcp.types.Tool."""

    name: str
    description: str
    inputSchema: Dict[str, Any]
    annotations: Any = None


class FakeContent:
    """Mimics MCP content objects."""

    def __init__(self, text: str):
        self.text = text


class FakeBinaryContent:
    """Mimics MCP binary content objects."""

    def __init__(self, data: bytes):
        self.data = data


class FakeOtherContent:
    """Mimics unknown MCP content type (no .text or .data)."""

    def __init__(self, value: str):
        self._value = value

    def __str__(self) -> str:
        return self._value


@dataclass
class FakeCallResult:
    content: List[Any]
    isError: bool = False


@dataclass
class FakeListResult:
    tools: List[FakeMCPTool]


class FakeAnnotations:
    """Mimics MCP tool annotations."""

    def __init__(self, readOnlyHint=None, destructiveHint=None, idempotentHint=None):
        self.readOnlyHint = readOnlyHint
        self.destructiveHint = destructiveHint
        self.idempotentHint = idempotentHint


# ===================================================================
# bridge.py tests
# ===================================================================


class TestSchemaToParams:
    """Tests for _schema_to_params()."""

    def test_empty_schema(self):
        params = _schema_to_params({"type": "object", "properties": {}})
        assert params == []

    def test_string_param(self):
        schema = {
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        }
        params = _schema_to_params(schema)
        assert len(params) == 1
        assert params[0].name == "city"
        assert params[0].param_type is str
        assert params[0].required is True
        assert params[0].description == "City name"

    def test_all_types(self):
        schema = {
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
            "required": [],
        }
        params = _schema_to_params(schema)
        type_map = {p.name: p.param_type for p in params}
        assert type_map["s"] is str
        assert type_map["i"] is int
        assert type_map["n"] is float
        assert type_map["b"] is bool
        assert type_map["a"] is list
        assert type_map["o"] is dict

    def test_nullable_type(self):
        schema = {
            "properties": {"city": {"type": ["string", "null"]}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert params[0].param_type is str

    def test_default_in_description(self):
        schema = {
            "properties": {"limit": {"type": "integer", "description": "Max items", "default": 10}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert "default: 10" in params[0].description

    def test_default_no_description(self):
        schema = {
            "properties": {"limit": {"type": "integer", "default": 5}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert "Default: 5" in params[0].description

    def test_enum_support(self):
        schema = {
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert params[0].enum == ["red", "green", "blue"]

    def test_unknown_type_defaults_to_str(self):
        schema = {
            "properties": {"x": {"type": "custom_type"}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert params[0].param_type is str

    def test_missing_type_defaults_to_string(self):
        schema = {
            "properties": {"x": {"description": "no type"}},
            "required": [],
        }
        params = _schema_to_params(schema)
        assert params[0].param_type is str

    def test_raw_schema_stored(self):
        prop = {"type": "string", "minLength": 1}
        schema = {"properties": {"x": prop}, "required": []}
        params = _schema_to_params(schema)
        assert params[0]._raw_schema == prop


class TestMcpToTool:
    """Tests for mcp_to_tool()."""

    def test_basic_conversion(self):
        mcp_tool = FakeMCPTool(
            name="get_weather",
            description="Get current weather",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        call_fn = AsyncMock(return_value="72F")
        tool = mcp_to_tool(mcp_tool, call_fn, server_name="weather")
        assert tool.name == "get_weather"
        assert tool.description == "Get current weather"
        assert len(tool.parameters) == 1
        assert tool._mcp_server == "weather"
        assert tool._mcp_original_name == "get_weather"

    def test_prefix_mode(self):
        mcp_tool = FakeMCPTool(
            name="search",
            description="Search",
            inputSchema={"type": "object", "properties": {}},
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="web", prefix=True)
        assert tool.name == "web_search"

    def test_no_prefix_without_server_name(self):
        mcp_tool = FakeMCPTool(
            name="search",
            description="Search",
            inputSchema={"type": "object", "properties": {}},
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="", prefix=True)
        assert tool.name == "search"

    def test_annotations_read_only(self):
        annotations = FakeAnnotations(readOnlyHint=True)
        mcp_tool = FakeMCPTool(
            name="list_files",
            description="List files",
            inputSchema={"type": "object", "properties": {}},
            annotations=annotations,
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="fs")
        assert tool._mcp_annotations["read_only"] is True

    def test_annotations_destructive(self):
        annotations = FakeAnnotations(destructiveHint=True, idempotentHint=False)
        mcp_tool = FakeMCPTool(
            name="delete_file",
            description="Delete file",
            inputSchema={"type": "object", "properties": {}},
            annotations=annotations,
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="fs")
        assert tool._mcp_annotations["destructive"] is True
        assert tool._mcp_annotations["idempotent"] is False

    def test_no_annotations(self):
        mcp_tool = FakeMCPTool(
            name="foo",
            description="",
            inputSchema={"type": "object", "properties": {}},
            annotations=None,
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="s")
        assert tool._mcp_annotations == {}

    def test_none_description(self):
        mcp_tool = FakeMCPTool(
            name="foo",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="s")
        assert tool.description == ""

    def test_none_input_schema(self):
        mcp_tool = FakeMCPTool(
            name="foo",
            description="x",
            inputSchema=None,
        )
        tool = mcp_to_tool(mcp_tool, AsyncMock(), server_name="s")
        assert tool.parameters == [] or isinstance(tool.parameters, list)

    @pytest.mark.asyncio
    async def test_tool_call_fn_invoked(self):
        call_fn = AsyncMock(return_value="result-text")
        mcp_tool = FakeMCPTool(
            name="do_thing",
            description="Do a thing",
            inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        tool = mcp_to_tool(mcp_tool, call_fn, server_name="s")
        result = await tool.function(x="hello")
        call_fn.assert_called_once_with(name="do_thing", arguments={"x": "hello"})
        assert result == "result-text"


class TestToolToMcpSchema:
    """Tests for tool_to_mcp_schema()."""

    def test_basic_conversion(self):
        tool = Tool(
            name="greet",
            description="Say hello",
            parameters=[
                ToolParameter(name="name", param_type=str, description="Name", required=True),
            ],
            function=lambda name: f"Hello {name}",
        )
        schema = tool_to_mcp_schema(tool)
        assert schema["name"] == "greet"
        assert schema["description"] == "Say hello"
        assert "inputSchema" in schema
        assert "properties" in schema["inputSchema"]


# ===================================================================
# client.py tests
# ===================================================================


class TestMCPClientInit:
    """Tests for MCPClient construction."""

    def test_construction(self):
        from selectools.mcp.client import MCPClient

        config = _make_config()
        client = MCPClient(config)
        assert client.config == config
        assert not client.connected
        assert not client.circuit_open
        assert client._tools_cache is None

    def test_circuit_open_property_default(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        assert client.circuit_open is False

    def test_circuit_open_during_cooldown(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client._circuit_open_until = time.time() + 60
        assert client.circuit_open is True

    def test_circuit_resets_after_cooldown(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client._circuit_open_until = time.time() - 1
        client._failure_count = 5
        assert client.circuit_open is False
        assert client._failure_count == 0
        assert client._circuit_open_until == 0


class TestMCPClientConnect:
    """Tests for MCPClient.connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_unknown_transport(self):
        from selectools.mcp.client import MCPClient

        config = MCPServerConfig.__new__(MCPServerConfig)
        config.transport = "websocket"
        config.command = ""
        config.args = []
        config.env = None
        config.url = ""
        config.headers = None
        config.timeout = 30.0
        config.max_retries = 2
        config.retry_backoff = 1.0
        config.auto_reconnect = True
        config.circuit_breaker_threshold = 3
        config.circuit_breaker_cooldown = 60.0
        config.screen_output = True
        config.cache_tools = True
        config.name = "test"
        config.cwd = None

        client = MCPClient(config)
        with pytest.raises(ValueError, match="Unknown transport"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        from selectools.mcp.client import MCPClient

        config = _make_config()
        client = MCPClient(config)

        with patch.dict("sys.modules", {"mcp": None}):
            with pytest.raises(ImportError, match="mcp"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client._connected = True
        client._session = MagicMock()
        client._session_cm = AsyncMock()
        client._session_cm.__aexit__ = AsyncMock()
        client._cm = AsyncMock()
        client._cm.__aexit__ = AsyncMock()
        client._tools_cache = [MagicMock()]

        await client.disconnect()
        assert not client.connected
        assert client._session is None
        assert client._tools_cache is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_exceptions(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client._connected = True
        client._session_cm = AsyncMock()
        client._session_cm.__aexit__ = AsyncMock(side_effect=RuntimeError("oops"))
        client._cm = AsyncMock()
        client._cm.__aexit__ = AsyncMock(side_effect=RuntimeError("oops"))

        await client.disconnect()
        assert not client.connected


class TestMCPClientListTools:
    """Tests for MCPClient.list_tools() and _fetch_tools()."""

    @pytest.mark.asyncio
    async def test_list_tools_not_connected_raises(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_list_tools_returns_cached(self):
        from selectools.mcp.client import MCPClient

        config = _make_config(cache_tools=True)
        client = MCPClient(config)
        cached_tools = [MagicMock(spec=Tool)]
        client._tools_cache = cached_tools
        result = await client.list_tools()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_tools_converts(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = FakeListResult(
            tools=[
                FakeMCPTool(
                    name="greet",
                    description="Say hi",
                    inputSchema={"type": "object", "properties": {"name": {"type": "string"}}},
                )
            ]
        )
        client._session = mock_session

        tools = await client._fetch_tools()
        assert len(tools) == 1
        assert tools[0].name == "greet"
        assert client._tools_cache is not None

    @pytest.mark.asyncio
    async def test_refresh_tools_clears_cache(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = FakeListResult(tools=[])
        client._session = mock_session
        client._tools_cache = [MagicMock()]

        await client.refresh_tools()
        assert client._tools_cache == []


class TestMCPClientCallTool:
    """Tests for MCPClient._call_tool()."""

    @pytest.mark.asyncio
    async def test_call_tool_circuit_open(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client._circuit_open_until = time.time() + 60

        with pytest.raises(ConnectionError, match="circuit breaker"):
            await client._call_tool("foo", {})

    @pytest.mark.asyncio
    async def test_call_tool_success_text(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(
            content=[FakeContent("Hello!")], isError=False
        )
        client._session = mock_session

        result = await client._call_tool("greet", {"name": "World"})
        assert result == "Hello!"
        assert client._failure_count == 0

    @pytest.mark.asyncio
    async def test_call_tool_binary_content(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(
            content=[FakeBinaryContent(b"\x00\x01")], isError=False
        )
        client._session = mock_session

        result = await client._call_tool("get_data", {})
        assert "Binary content" in result

    @pytest.mark.asyncio
    async def test_call_tool_other_content(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(
            content=[FakeOtherContent("custom-value")], isError=False
        )
        client._session = mock_session

        result = await client._call_tool("get_data", {})
        assert result == "custom-value"

    @pytest.mark.asyncio
    async def test_call_tool_error_result(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(
            content=[FakeContent("not found")], isError=True
        )
        client._session = mock_session

        result = await client._call_tool("missing", {})
        assert "[MCP Error]" in result

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(content=[], isError=False)
        client._session = mock_session

        result = await client._call_tool("empty", {})
        assert result == ""

    @pytest.mark.asyncio
    async def test_call_tool_not_connected_auto_reconnect_disabled(self):
        from selectools.mcp.client import MCPClient

        config = _make_config(auto_reconnect=False, max_retries=0)
        client = MCPClient(config)
        client._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            await client._call_tool("foo", {})

    @pytest.mark.asyncio
    async def test_call_tool_retries_and_circuit_breaker(self):
        from selectools.mcp.client import MCPClient

        config = _make_config(
            max_retries=1,
            retry_backoff=0.01,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=60.0,
            auto_reconnect=False,
        )
        client = MCPClient(config)
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = RuntimeError("server down")
        client._session = mock_session

        with pytest.raises(RuntimeError, match="server down"):
            await client._call_tool("foo", {})

        assert client._failure_count >= 2
        assert client._circuit_open_until > 0

    @pytest.mark.asyncio
    async def test_call_tool_multiple_text_content(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = FakeCallResult(
            content=[FakeContent("line1"), FakeContent("line2")],
            isError=False,
        )
        client._session = mock_session

        result = await client._call_tool("multi", {})
        assert result == "line1\nline2"


class TestMCPClientConnectWithMockedMCP:
    """Tests for MCPClient.connect() with mocked mcp imports."""

    @pytest.mark.asyncio
    async def test_connect_stdio_full_path(self):
        """Test full stdio connect path with mocked mcp modules."""
        from selectools.mcp.client import MCPClient

        config = _make_config(cache_tools=False)
        client = MCPClient(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=("read", "write", "extra"))
        mock_cm.__aexit__ = AsyncMock()

        fake_mcp = MagicMock()
        fake_mcp.StdioServerParameters = MagicMock(return_value="params")
        fake_mcp.ClientSession = MagicMock(return_value=mock_session_cm)

        fake_stdio = MagicMock()
        fake_stdio.stdio_client = MagicMock(return_value=mock_cm)

        with patch.dict(
            "sys.modules",
            {
                "mcp": fake_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.stdio": fake_stdio,
            },
        ):
            await client.connect()

        assert client.connected
        assert client._session is mock_session

    @pytest.mark.asyncio
    async def test_connect_http_transport(self):
        """Test streamable-http connect path."""
        from selectools.mcp.client import MCPClient

        config = _make_http_config(cache_tools=False)
        client = MCPClient(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=("read", "write"))

        fake_mcp = MagicMock()
        fake_mcp.ClientSession = MagicMock(return_value=mock_session_cm)

        fake_http = MagicMock()
        fake_http.streamablehttp_client = MagicMock(return_value=mock_cm)

        with patch.dict(
            "sys.modules",
            {
                "mcp": fake_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.streamable_http": fake_http,
            },
        ):
            await client.connect()

        assert client.connected

    @pytest.mark.asyncio
    async def test_connect_tuple_one_element(self):
        """Test connect with tuple result of length 1."""
        from selectools.mcp.client import MCPClient

        config = _make_config(cache_tools=False)
        client = MCPClient(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=("read_only",))

        fake_mcp = MagicMock()
        fake_mcp.StdioServerParameters = MagicMock()
        fake_mcp.ClientSession = MagicMock(return_value=mock_session_cm)

        fake_stdio = MagicMock()
        fake_stdio.stdio_client = MagicMock(return_value=mock_cm)

        with patch.dict(
            "sys.modules",
            {
                "mcp": fake_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.stdio": fake_stdio,
            },
        ):
            await client.connect()

        assert client._read == "read_only"
        assert client._write is None

    @pytest.mark.asyncio
    async def test_connect_non_tuple_result(self):
        """Test connect when transport returns non-tuple."""
        from selectools.mcp.client import MCPClient

        config = _make_config(cache_tools=False)
        client = MCPClient(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value="single_stream")

        fake_mcp = MagicMock()
        fake_mcp.StdioServerParameters = MagicMock()
        fake_mcp.ClientSession = MagicMock(return_value=mock_session_cm)

        fake_stdio = MagicMock()
        fake_stdio.stdio_client = MagicMock(return_value=mock_cm)

        with patch.dict(
            "sys.modules",
            {
                "mcp": fake_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.stdio": fake_stdio,
            },
        ):
            await client.connect()

        assert client._read == "single_stream"
        assert client._write is None

    @pytest.mark.asyncio
    async def test_connect_with_cache_tools(self):
        """Test that connect() pre-fetches tools when cache_tools=True."""
        from selectools.mcp.client import MCPClient

        config = _make_config(cache_tools=True)
        client = MCPClient(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=FakeListResult(tools=[]))

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=("r", "w", "x"))

        fake_mcp = MagicMock()
        fake_mcp.StdioServerParameters = MagicMock()
        fake_mcp.ClientSession = MagicMock(return_value=mock_session_cm)

        fake_stdio = MagicMock()
        fake_stdio.stdio_client = MagicMock(return_value=mock_cm)

        with patch.dict(
            "sys.modules",
            {
                "mcp": fake_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.stdio": fake_stdio,
            },
        ):
            await client.connect()

        assert client._tools_cache is not None


class TestMCPClientRetryWithReconnect:
    """Tests for _call_tool retry with auto_reconnect."""

    @pytest.mark.asyncio
    async def test_call_tool_reconnects_on_retry(self):
        from selectools.mcp.client import MCPClient

        config = _make_config(
            max_retries=1,
            retry_backoff=0.001,
            auto_reconnect=True,
            circuit_breaker_threshold=10,
        )
        client = MCPClient(config)

        call_count = 0

        async def fake_call_tool(name, args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("temp failure")
            return FakeCallResult(content=[FakeContent("ok")], isError=False)

        mock_session = AsyncMock()
        mock_session.call_tool = fake_call_tool
        client._session = mock_session
        client.disconnect = AsyncMock()
        client.connect = AsyncMock()

        result = await client._call_tool("test", {})
        assert result == "ok"
        client.disconnect.assert_called_once()
        client.connect.assert_called_once()


class TestMCPClientSyncWrappers:
    """Tests for sync context manager and sync tool listing."""

    def test_sync_list_tools(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_tools = [MagicMock(spec=Tool)]

        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=mock_tools)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            result = client.list_tools_sync()
            assert result == mock_tools

    def test_sync_call_tool(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value="tool-result")

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            result = client.call_tool_sync("greet", {"name": "World"})
            assert result == "tool-result"

    def test_sync_enter(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=None)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            result = client.__enter__()
            assert result is client

    def test_sync_exit(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=None)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            client.__exit__(None, None, None)
            mock_bg_loop.run.assert_called_once()


class TestMCPClientContextManager:
    """Tests for MCPClient async context managers."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        from selectools.mcp.client import MCPClient

        client = MCPClient(_make_config())
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        async with client as c:
            assert c is client
            client.connect.assert_called_once()

        client.disconnect.assert_called_once()


# ===================================================================
# multi.py tests
# ===================================================================


class TestMultiMCPClient:
    """Tests for MultiMCPClient."""

    def test_construction(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1"), _make_config("s2")]
        client = MultiMCPClient(configs)
        assert len(client.configs) == 2
        assert client.prefix_tools is True
        assert client.active_servers == []
        assert client.failed_servers == []

    @pytest.mark.asyncio
    async def test_connect_all_success(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1"), _make_config("s2")]
        client = MultiMCPClient(configs)

        with patch("selectools.mcp.multi.MCPClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.connected = True
            MockClient.return_value = mock_instance

            await client.connect_all()
            assert len(client._clients) == 2
            assert client.active_servers == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_connect_all_partial_failure(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1"), _make_config("s2")]
        client = MultiMCPClient(configs)

        call_count = 0

        async def mock_connect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("s1 down")

        with patch("selectools.mcp.multi.MCPClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.connect = mock_connect
            mock_instance.connected = True
            MockClient.return_value = mock_instance

            await client.connect_all()
            assert "s1" in client.failed_servers
            # s2 connected
            assert len(client._clients) == 1

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)

        mock_client = AsyncMock()
        mock_client.connected = True
        client._clients = {"s1": mock_client}
        client._failed_servers = ["s2"]

        await client.disconnect_all()
        assert len(client._clients) == 0
        assert client.failed_servers == []

    @pytest.mark.asyncio
    async def test_disconnect_all_handles_errors(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)

        mock_client = AsyncMock()
        mock_client.disconnect.side_effect = RuntimeError("oops")
        client._clients = {"s1": mock_client}

        await client.disconnect_all()
        assert len(client._clients) == 0

    @pytest.mark.asyncio
    async def test_list_all_tools_with_prefix(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("search")]
        client = MultiMCPClient(configs, prefix_tools=True)

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "web_search"

        mock_client = AsyncMock()
        mock_client.connected = True
        mock_client.list_tools.return_value = [mock_tool]
        client._clients = {"search": mock_client}

        tools = await client.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_web_search"

    @pytest.mark.asyncio
    async def test_list_all_tools_no_prefix(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs, prefix_tools=False)

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "unique_tool"

        mock_client = AsyncMock()
        mock_client.connected = True
        mock_client.list_tools.return_value = [mock_tool]
        client._clients = {"s1": mock_client}

        tools = await client.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "unique_tool"

    @pytest.mark.asyncio
    async def test_list_all_tools_collision_raises(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1"), _make_config("s2")]
        client = MultiMCPClient(configs, prefix_tools=False)

        mock_tool1 = MagicMock(spec=Tool)
        mock_tool1.name = "search"
        mock_tool2 = MagicMock(spec=Tool)
        mock_tool2.name = "search"

        mock_client1 = AsyncMock()
        mock_client1.connected = True
        mock_client1.list_tools.return_value = [mock_tool1]

        mock_client2 = AsyncMock()
        mock_client2.connected = True
        mock_client2.list_tools.return_value = [mock_tool2]

        client._clients = {"s1": mock_client1, "s2": mock_client2}

        with pytest.raises(ValueError, match="Tool name collision"):
            await client.list_all_tools()

    @pytest.mark.asyncio
    async def test_list_all_tools_skips_disconnected(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)

        mock_client = AsyncMock()
        mock_client.connected = False
        client._clients = {"s1": mock_client}

        tools = await client.list_all_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_all_tools_server_error_marks_failed(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)

        mock_client = AsyncMock()
        mock_client.connected = True
        mock_client.list_tools.side_effect = RuntimeError("server error")
        client._clients = {"s1": mock_client}

        tools = await client.list_all_tools()
        assert tools == []
        assert "s1" in client.failed_servers

    @pytest.mark.asyncio
    async def test_refresh_all_tools(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "t1"

        mock_client = AsyncMock()
        mock_client.connected = True
        mock_client.refresh_tools = AsyncMock()
        mock_client.list_tools.return_value = [mock_tool]
        client._clients = {"s1": mock_client}

        tools = await client.refresh_all_tools()
        mock_client.refresh_tools.assert_called_once()
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)
        client.connect_all = AsyncMock()
        client.disconnect_all = AsyncMock()

        async with client as c:
            assert c is client
            client.connect_all.assert_called_once()

        client.disconnect_all.assert_called_once()


class TestMultiMCPClientSync:
    """Tests for MultiMCPClient sync wrappers."""

    def test_sync_enter(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=None)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            result = client.__enter__()
            assert result is client

    def test_sync_exit(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=None)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            client.__exit__(None, None, None)
            mock_bg_loop.run.assert_called_once()

    def test_sync_list_all_tools(self):
        from selectools.mcp.multi import MultiMCPClient

        configs = [_make_config("s1")]
        client = MultiMCPClient(configs)
        mock_tools = [MagicMock(spec=Tool)]
        mock_bg_loop = MagicMock()
        mock_bg_loop.run = MagicMock(return_value=mock_tools)

        with patch("selectools.mcp._loop.get_background_loop", return_value=mock_bg_loop):
            result = client.list_all_tools_sync()
            assert result == mock_tools


# ===================================================================
# server.py tests
# ===================================================================


class TestMCPServer:
    """Tests for MCPServer."""

    def test_construction(self):
        from selectools.mcp.server import MCPServer

        tool = Tool(
            name="greet",
            description="Say hello",
            parameters=[],
            function=lambda: "hello",
        )
        server = MCPServer(tools=[tool], name="my-server", version="1.0.0")
        assert server.name == "my-server"
        assert server.version == "1.0.0"
        assert len(server.tools) == 1

    def test_default_version_uses_package(self):
        from selectools import __version__
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        assert server.version == __version__

    def test_build_fastmcp_import_error(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        with patch.dict(
            "sys.modules", {"mcp": None, "mcp.server": None, "mcp.server.fastmcp": None}
        ):
            with pytest.raises(ImportError, match="mcp"):
                server._build_fastmcp()

    def test_build_fastmcp_registers_tools(self):
        from selectools.mcp.server import MCPServer

        tool1 = Tool(
            name="greet",
            description="Say hello",
            parameters=[],
            function=lambda: "hello",
        )
        tool2 = Tool(
            name="farewell",
            description="Say bye",
            parameters=[],
            function=lambda: "bye",
        )

        server = MCPServer(tools=[tool1, tool2])

        mock_fastmcp = MagicMock()
        with patch("selectools.mcp.server.FastMCP", mock_fastmcp, create=True):
            with patch.object(server, "_register_tool") as mock_register:
                # Simulate mcp import
                import sys

                fake_mcp = MagicMock()
                fake_mcp.server.fastmcp.FastMCP = mock_fastmcp
                with patch.dict(
                    sys.modules,
                    {
                        "mcp": fake_mcp,
                        "mcp.server": fake_mcp.server,
                        "mcp.server.fastmcp": fake_mcp.server.fastmcp,
                    },
                ):
                    server._build_fastmcp()
                    assert mock_register.call_count == 2

    def test_register_tool_sync_function(self):
        from selectools.mcp.server import MCPServer

        def my_sync_tool(x: str) -> str:
            return f"result: {x}"

        tool = Tool(
            name="my_tool",
            description="A tool",
            parameters=[
                ToolParameter(name="x", param_type=str, description="input", required=True)
            ],
            function=my_sync_tool,
        )

        server = MCPServer(tools=[tool])
        mock_mcp = MagicMock()

        server._register_tool(mock_mcp, tool)
        mock_mcp.add_tool.assert_called_once()
        call_args = mock_mcp.add_tool.call_args
        assert call_args.kwargs["name"] == "my_tool"
        assert call_args.kwargs["description"] == "A tool"
        # The function should be an async wrapper (not the original sync one)
        fn = call_args.args[0]
        assert asyncio.iscoroutinefunction(fn)

    def test_register_tool_async_function(self):
        from selectools.mcp.server import MCPServer

        async def my_async_tool(x: str) -> str:
            return f"result: {x}"

        tool = Tool(
            name="my_async_tool",
            description="Async tool",
            parameters=[],
            function=my_async_tool,
            _skip_validation=True,
        )

        server = MCPServer(tools=[tool])
        mock_mcp = MagicMock()

        server._register_tool(mock_mcp, tool)
        mock_mcp.add_tool.assert_called_once()
        # The function should be the original async one
        fn = mock_mcp.add_tool.call_args.args[0]
        assert asyncio.iscoroutinefunction(fn)

    def test_get_fastmcp_lazy(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])

        def fake_build():
            server._fastmcp = "mock-fastmcp"
            return "mock-fastmcp"

        server._build_fastmcp = fake_build

        result = server.get_fastmcp()
        assert result == "mock-fastmcp"

    def test_get_fastmcp_cached(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        server._fastmcp = "existing"
        server._build_fastmcp = MagicMock()

        result = server.get_fastmcp()
        server._build_fastmcp.assert_not_called()
        assert result == "existing"

    def test_serve_unknown_transport(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        server._fastmcp = MagicMock()

        with pytest.raises(ValueError, match="Unknown transport"):
            server.serve(transport="websocket")

    def test_serve_stdio(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        mock_fastmcp = MagicMock()
        server._fastmcp = mock_fastmcp

        server.serve(transport="stdio")
        mock_fastmcp.run.assert_called_once_with(transport="stdio")

    def test_serve_http(self):
        from selectools.mcp.server import MCPServer

        server = MCPServer(tools=[])
        mock_fastmcp = MagicMock()
        server._fastmcp = mock_fastmcp

        server.serve(transport="streamable-http", host="0.0.0.0", port=9090)
        mock_fastmcp.run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",
            port=9090,
        )


# ===================================================================
# __init__.py tests
# ===================================================================


class TestMCPInit:
    """Tests for selectools.mcp.__init__ lazy imports."""

    def test_mcp_server_config_direct(self):
        from selectools.mcp import MCPServerConfig

        assert MCPServerConfig is not None

    def test_lazy_mcp_client(self):
        from selectools.mcp import MCPClient

        assert MCPClient is not None
        from selectools.mcp.client import MCPClient as DirectClient

        assert MCPClient is DirectClient

    def test_lazy_multi_mcp_client(self):
        from selectools.mcp import MultiMCPClient

        assert MultiMCPClient is not None
        from selectools.mcp.multi import MultiMCPClient as DirectMulti

        assert MultiMCPClient is DirectMulti

    def test_lazy_mcp_server(self):
        from selectools.mcp import MCPServer

        assert MCPServer is not None
        from selectools.mcp.server import MCPServer as DirectServer

        assert MCPServer is DirectServer

    def test_unknown_attribute_raises(self):
        import selectools.mcp as mcp_mod

        with pytest.raises(AttributeError, match="no attribute"):
            _ = mcp_mod.NonExistentClass

    def test_mcp_tools_function(self):
        from selectools.mcp import mcp_tools

        assert callable(mcp_tools)

    def test_all_exports(self):
        from selectools.mcp import __all__

        expected = {"MCPServerConfig", "MCPClient", "MultiMCPClient", "MCPServer", "mcp_tools"}
        assert set(__all__) == expected


class TestMCPToolsContext:
    """Tests for the _MCPToolsContext / mcp_tools() wrapper."""

    def test_mcp_tools_returns_context(self):
        from selectools.mcp import _MCPToolsContext, mcp_tools

        config = _make_config()
        ctx = mcp_tools(config)
        assert isinstance(ctx, _MCPToolsContext)

    @pytest.mark.asyncio
    async def test_async_context_manager_enter(self):
        from selectools.mcp import _MCPToolsContext

        config = _make_config()

        mock_tools = [MagicMock(spec=Tool)]

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.list_tools = AsyncMock(return_value=mock_tools)

        mock_cls = MagicMock(return_value=mock_client_instance)

        ctx = _MCPToolsContext(config)
        with patch("selectools.mcp.client.MCPClient", mock_cls):
            tools = await ctx.__aenter__()
            assert len(tools) == 1
            assert ctx._client is mock_client_instance

    @pytest.mark.asyncio
    async def test_async_context_manager_exit(self):
        from selectools.mcp import _MCPToolsContext

        config = _make_config()
        ctx = _MCPToolsContext(config)

        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock(return_value=None)
        ctx._client = mock_client

        await ctx.__aexit__(None, None, None)
        mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_exit_no_client(self):
        from selectools.mcp import _MCPToolsContext

        config = _make_config()
        ctx = _MCPToolsContext(config)
        ctx._client = None
        await ctx.__aexit__(None, None, None)  # should not raise

    def test_sync_exit_no_client(self):
        from selectools.mcp import _MCPToolsContext

        config = _make_config()
        ctx = _MCPToolsContext(config)
        ctx._client = None
        ctx.__exit__(None, None, None)  # should not raise
