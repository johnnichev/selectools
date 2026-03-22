"""MCPServer — expose selectools tools as an MCP-compliant server."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from .. import __version__
from ..tools.base import Tool


class MCPServer:
    """Expose selectools @tool functions as an MCP server.

    Uses the official ``mcp.server.fastmcp.FastMCP`` under the hood.
    Any selectools Tool can be exposed to MCP clients like Claude Desktop,
    Cursor, VS Code, or other selectools agents.

    Example::

        from selectools import tool
        from selectools.mcp import MCPServer

        @tool(description="Get weather")
        def get_weather(city: str) -> str:
            return f"72F in {city}"

        server = MCPServer(tools=[get_weather])
        server.serve(transport="stdio")
    """

    def __init__(
        self,
        tools: List[Tool],
        *,
        name: str = "selectools",
        version: str = "",
    ) -> None:
        self.tools = tools
        self.name = name
        self.version = version or __version__
        self._fastmcp: Any = None

    def _build_fastmcp(self) -> Any:
        """Create and configure the FastMCP instance."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install selectools[mcp]"
            )

        mcp = FastMCP(self.name)

        for tool_obj in self.tools:
            self._register_tool(mcp, tool_obj)

        self._fastmcp = mcp
        return mcp

    def _register_tool(self, mcp: Any, tool_obj: Tool) -> None:
        """Register a selectools Tool as an MCP tool handler.

        Uses the original function (which has proper parameter names)
        so FastMCP can introspect the signature correctly.
        """
        fn = tool_obj.function

        # If the function is sync, wrap it for FastMCP (which is async)
        if not tool_obj.is_async:
            original_fn = fn

            async def async_wrapper(**kwargs: Any) -> str:
                result = await asyncio.to_thread(original_fn, **kwargs)
                return str(result) if result is not None else ""

            # Preserve the original signature for FastMCP introspection
            import functools

            functools.update_wrapper(async_wrapper, original_fn)
            fn = async_wrapper

        mcp.add_tool(fn, name=tool_obj.name, description=tool_obj.description)

    def get_fastmcp(self) -> Any:
        """Return the underlying FastMCP instance for advanced customization."""
        if self._fastmcp is None:
            self._build_fastmcp()
        return self._fastmcp

    def serve(
        self,
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """Start the MCP server.

        Args:
            transport: ``"stdio"`` or ``"streamable-http"``.
            host: Host to bind to (HTTP only).
            port: Port to bind to (HTTP only).
        """
        mcp = self.get_fastmcp()

        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "streamable-http":
            mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
            )
        else:
            raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'streamable-http'")
