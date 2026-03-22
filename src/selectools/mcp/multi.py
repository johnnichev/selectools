"""MultiMCPClient — manage connections to multiple MCP servers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..tools.base import Tool
from .client import MCPClient
from .config import MCPServerConfig


class MultiMCPClient:
    """Manage connections to multiple MCP servers.

    Aggregates tools from all servers with optional name prefixing
    to avoid collisions. Supports graceful degradation — if one server
    goes down, tools from other servers continue working.

    Example::

        async with MultiMCPClient([
            MCPServerConfig(command="python", args=["search.py"], name="search"),
            MCPServerConfig(url="http://api.example.com/mcp",
                            transport="streamable-http", name="api"),
        ]) as client:
            tools = await client.list_all_tools()
            agent = Agent(provider=p, tools=tools, config=c)
    """

    def __init__(
        self,
        servers: List[MCPServerConfig],
        *,
        prefix_tools: bool = True,
    ) -> None:
        self.configs = servers
        self.prefix_tools = prefix_tools
        self._clients: Dict[str, MCPClient] = {}
        self._failed_servers: List[str] = []

    @property
    def active_servers(self) -> List[str]:
        """Names of currently connected servers."""
        return [name for name, client in self._clients.items() if client.connected]

    @property
    def failed_servers(self) -> List[str]:
        """Names of servers that failed to connect."""
        return list(self._failed_servers)

    async def connect_all(self) -> None:
        """Connect to all configured servers.

        Servers that fail to connect are recorded in ``failed_servers``
        but do not prevent other servers from connecting.
        """
        self._failed_servers.clear()
        for config in self.configs:
            client = MCPClient(config)
            try:
                await client.connect()
                self._clients[config.name] = client
            except Exception:  # nosec B110
                self._failed_servers.append(config.name)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:  # nosec B110
                pass
        self._clients.clear()
        self._failed_servers.clear()

    async def list_all_tools(self) -> List[Tool]:
        """Aggregate tools from all connected servers.

        If ``prefix_tools=True``, tool names are prefixed with the server name
        (e.g., ``search_web_search``). If ``prefix_tools=False`` and there are
        name collisions, a ``ValueError`` is raised.
        """
        all_tools: List[Tool] = []
        seen_names: Set[str] = set()

        for name, client in self._clients.items():
            if not client.connected:
                continue
            try:
                server_tools = await client.list_tools()
                for tool in server_tools:
                    if self.prefix_tools:
                        # Re-create with prefix
                        from .bridge import mcp_to_tool

                        prefixed_name = f"{name}_{tool.name}"
                        tool.name = prefixed_name
                    else:
                        if tool.name in seen_names:
                            raise ValueError(
                                f"Tool name collision: '{tool.name}' exists in multiple "
                                f"MCP servers. Use prefix_tools=True to avoid conflicts."
                            )
                    seen_names.add(tool.name)
                    all_tools.append(tool)
            except Exception as e:
                if not isinstance(e, ValueError):
                    # Server failed during tool listing — mark as failed
                    self._failed_servers.append(name)
                else:
                    raise

        return all_tools

    async def refresh_all_tools(self) -> List[Tool]:
        """Re-fetch tools from all connected servers."""
        for client in self._clients.values():
            if client.connected:
                await client.refresh_tools()
        return await self.list_all_tools()

    # Context manager support

    async def __aenter__(self) -> "MultiMCPClient":  # noqa: D105
        await self.connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:  # noqa: D105
        await self.disconnect_all()

    def __enter__(self) -> "MultiMCPClient":  # noqa: D105
        from ._loop import get_background_loop

        get_background_loop().run(self.connect_all())
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: D105
        from ._loop import get_background_loop

        get_background_loop().run(self.disconnect_all())

    def list_all_tools_sync(self) -> List[Tool]:
        """Synchronous version of list_all_tools()."""
        from ._loop import get_background_loop

        return get_background_loop().run(self.list_all_tools())
