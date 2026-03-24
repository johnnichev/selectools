"""MCPClient — connect to a single MCP server."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..tools.base import Tool
from .bridge import mcp_to_tool
from .config import MCPServerConfig


class MCPClient:
    """Connect to a single MCP server and discover its tools.

    Supports stdio (local subprocess) and Streamable HTTP (remote) transports.
    MCP tools are converted to selectools Tool objects automatically.

    Example::

        async with MCPClient(MCPServerConfig(command="python", args=["server.py"])) as client:
            tools = await client.list_tools()
            agent = Agent(provider=p, tools=tools, config=c)

    For sync usage::

        with MCPClient(config) as client:
            tools = client.list_tools_sync()
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._session: Any = None
        self._read: Any = None
        self._write: Any = None
        self._cm: Any = None  # transport context manager
        self._session_cm: Any = None
        self._tools_cache: Optional[List[Tool]] = None
        self._connected = False

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open_until: float = 0

    @property
    def connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    @property
    def circuit_open(self) -> bool:
        """Whether the circuit breaker is open (server is considered down)."""
        if self._circuit_open_until > 0 and time.time() < self._circuit_open_until:
            return True
        if self._circuit_open_until > 0 and time.time() >= self._circuit_open_until:
            # Cooldown expired, reset
            self._circuit_open_until = 0
            self._failure_count = 0
        return False

    async def connect(self) -> None:
        """Initialize the MCP session."""
        try:
            if self.config.transport == "stdio":
                from mcp import StdioServerParameters  # type: ignore[import-untyped]
                from mcp.client.stdio import stdio_client  # type: ignore[import-untyped]

                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                )
                self._cm = stdio_client(params)
            elif self.config.transport == "streamable-http":
                from mcp.client.streamable_http import (
                    streamablehttp_client,  # type: ignore[import-untyped]
                )

                self._cm = streamablehttp_client(
                    self.config.url,
                    headers=self.config.headers or {},
                    timeout=self.config.timeout,
                )
            else:
                raise ValueError(f"Unknown transport: {self.config.transport}")

            result = await self._cm.__aenter__()
            if isinstance(result, tuple):
                if len(result) == 3:
                    self._read, self._write, _ = result
                elif len(result) == 2:
                    self._read, self._write = result
                else:
                    self._read = result[0]
                    self._write = result[1] if len(result) > 1 else None
            else:
                self._read = result
                self._write = None

            from mcp import ClientSession  # type: ignore[import-untyped]

            self._session_cm = ClientSession(self._read, self._write)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
            self._connected = True
            self._failure_count = 0

            # Pre-fetch tools if caching is enabled
            if self.config.cache_tools:
                await self._fetch_tools()

        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install selectools[mcp]"
            )

    async def disconnect(self) -> None:
        """Close the MCP session."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:  # nosec B110
                pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:  # nosec B110
                pass
        self._connected = False
        self._session = None
        self._tools_cache = None

    async def list_tools(self) -> List[Tool]:
        """Discover tools from the MCP server.

        Returns selectools Tool objects that proxy calls to the MCP server.
        Results are cached if ``cache_tools=True`` in config.
        """
        if self._tools_cache is not None and self.config.cache_tools:
            return list(self._tools_cache)
        return await self._fetch_tools()

    async def _fetch_tools(self) -> List[Tool]:
        """Fetch tools from the server and convert to selectools Tools."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._session.list_tools()
        tools: List[Tool] = []
        for mcp_tool in result.tools:
            tool = mcp_to_tool(
                mcp_tool,
                call_fn=self._call_tool,
                server_name=self.config.name,
            )
            tools.append(tool)

        self._tools_cache = tools
        return list(tools)

    async def refresh_tools(self) -> List[Tool]:
        """Re-fetch tools from the server, bypassing cache."""
        self._tools_cache = None
        return await self._fetch_tools()

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the text result."""
        if self.circuit_open:
            raise ConnectionError(
                f"MCP server '{self.config.name}' circuit breaker is open. "
                f"Server will be retried after cooldown."
            )

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if not self._session:
                    if self.config.auto_reconnect:
                        await self.connect()
                    else:
                        raise RuntimeError("Not connected and auto_reconnect is disabled.")

                result = await self._session.call_tool(name, arguments)
                self._failure_count = 0  # Reset on success

                # Extract text from result content
                texts: List[str] = []
                for content in result.content:
                    if hasattr(content, "text"):
                        texts.append(content.text)
                    elif hasattr(content, "data"):
                        texts.append(f"[Binary content: {type(content).__name__}]")
                    else:
                        texts.append(str(content))

                if result.isError:
                    return f"[MCP Error] {' '.join(texts)}"

                return "\n".join(texts) if texts else ""

            except Exception as e:
                last_error = e
                self._failure_count += 1

                # Check circuit breaker threshold
                if self._failure_count >= self.config.circuit_breaker_threshold:
                    self._circuit_open_until = time.time() + self.config.circuit_breaker_cooldown

                if attempt < self.config.max_retries:
                    backoff = self.config.retry_backoff * (2**attempt)
                    await asyncio.sleep(backoff)
                    # Try reconnecting
                    if self.config.auto_reconnect:
                        try:
                            await self.disconnect()
                            await self.connect()
                        except Exception:  # nosec B110
                            pass

        raise last_error or RuntimeError(f"MCP call to '{name}' failed after retries")

    # Context manager support

    async def __aenter__(self) -> "MCPClient":  # noqa: D105
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:  # noqa: D105
        await self.disconnect()

    def __enter__(self) -> "MCPClient":  # noqa: D105
        from ._loop import get_background_loop

        get_background_loop().run(self.connect())
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: D105
        from ._loop import get_background_loop

        get_background_loop().run(self.disconnect())

    def list_tools_sync(self) -> List[Tool]:
        """Synchronous version of list_tools()."""
        from ._loop import get_background_loop

        return get_background_loop().run(self.list_tools())

    def call_tool_sync(self, name: str, arguments: Dict[str, Any]) -> str:
        """Synchronous version of _call_tool()."""
        from ._loop import get_background_loop

        return get_background_loop().run(self._call_tool(name, arguments))
