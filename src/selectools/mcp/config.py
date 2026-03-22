"""Configuration for MCP server connections."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    Supports two transports:
    - ``stdio``: spawn a local subprocess (requires ``command``)
    - ``streamable-http``: connect to a remote HTTP endpoint (requires ``url``)

    Example::

        # Local subprocess
        MCPServerConfig(command="python", args=["server.py"])

        # Remote HTTP
        MCPServerConfig(url="http://localhost:8080/mcp", transport="streamable-http")
    """

    # Identity
    name: str = ""

    # Transport
    transport: str = "stdio"  # "stdio" or "streamable-http"

    # stdio transport
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

    # HTTP transport
    url: str = ""
    headers: Optional[Dict[str, str]] = None

    # Connection
    timeout: float = 30.0
    max_retries: int = 2
    retry_backoff: float = 1.0
    auto_reconnect: bool = True

    # Circuit breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown: float = 60.0

    # Security
    screen_output: bool = True

    # Caching
    cache_tools: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio transport requires 'command'")
        if self.transport == "streamable-http" and not self.url:
            raise ValueError("streamable-http transport requires 'url'")
        if self.transport not in ("stdio", "streamable-http"):
            raise ValueError(
                f"Unknown transport: {self.transport}. Use 'stdio' or 'streamable-http'"
            )
        if not self.name:
            self.name = self._auto_name()

    def _auto_name(self) -> str:
        """Generate a name from the command or URL."""
        if self.command:
            parts = self.args[:1] if self.args else [self.command]
            basename = os.path.basename(parts[0])
            return basename.replace(".py", "").replace(".js", "")
        if self.url:
            parsed = urlparse(self.url)
            return parsed.hostname or "mcp-server"
        return "mcp-server"
