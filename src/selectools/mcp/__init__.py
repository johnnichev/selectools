"""MCP (Model Context Protocol) integration for selectools.

Connect to any MCP-compatible tool server and expose selectools tools
as MCP servers. Requires: ``pip install selectools[mcp]``
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, List

from .config import MCPServerConfig


def mcp_tools(config: MCPServerConfig) -> "_MCPToolsContext":
    """Connect to an MCP server and get selectools Tool objects.

    Use as a context manager to manage the connection lifecycle::

        from selectools.mcp import mcp_tools, MCPServerConfig

        with mcp_tools(MCPServerConfig(command="python", args=["server.py"])) as tools:
            agent = Agent(provider=p, tools=tools, config=c)
            result = agent.run("query")

    Args:
        config: MCP server connection configuration.

    Returns:
        Context manager that yields a list of Tool objects.
    """
    return _MCPToolsContext(config)


class _MCPToolsContext:
    """Context manager wrapper for mcp_tools()."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._client: Any = None

    def __enter__(self) -> List[Any]:
        from .client import MCPClient

        self._client = MCPClient(self._config)
        self._client.__enter__()
        return self._client.list_tools_sync()

    def __exit__(self, *args: Any) -> None:
        if self._client:
            self._client.__exit__(*args)

    async def __aenter__(self) -> List[Any]:
        from .client import MCPClient

        self._client = MCPClient(self._config)
        await self._client.__aenter__()
        return await self._client.list_tools()

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.__aexit__(*args)


# Lazy imports to avoid requiring mcp package at import time
def __getattr__(name: str) -> Any:
    if name == "MCPClient":
        from .client import MCPClient

        return MCPClient
    if name == "MultiMCPClient":
        from .multi import MultiMCPClient

        return MultiMCPClient
    if name == "MCPServer":
        from .server import MCPServer

        return MCPServer
    raise AttributeError(f"module 'selectools.mcp' has no attribute {name}")


__all__ = [
    "MCPServerConfig",
    "MCPClient",
    "MultiMCPClient",
    "MCPServer",
    "mcp_tools",
]
