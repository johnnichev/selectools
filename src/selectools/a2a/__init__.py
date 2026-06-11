"""
A2A protocol — agent-to-agent communication (Google-backed standard).

Serving::

    from selectools.serve import A2AServer   # or selectools.a2a

    server = A2AServer(agent=my_agent, auth_token="sk-...")
    server.serve(port=8000)

Consuming::

    from selectools.a2a import A2AClient

    client = A2AClient("https://other-agent.example.com")
    card = await client.discover()       # reads /.well-known/agent.json
    result = await client.send_task("Research quantum computing trends")

The server requires ``starlette`` and the client requires ``httpx``
(both ship with ``pip install selectools[serve]``); imports are lazy so
this package loads without them.
"""

from typing import Any

from .types import A2AError, A2ATask, AgentCard, AgentSkill, TaskState

__stability__ = "beta"

__all__ = [
    "A2AClient",
    "A2AError",
    "A2AServer",
    "A2ATask",
    "AgentCard",
    "AgentSkill",
    "TaskState",
]


def __getattr__(name: str) -> Any:
    """Lazily import the server (starlette) and client (httpx)."""
    if name == "A2AServer":
        try:
            from .server import A2AServer
        except ImportError as exc:
            raise ImportError(
                "A2AServer requires the 'starlette' package. "
                "Install it with: pip install selectools[serve]"
            ) from exc
        return A2AServer
    if name == "A2AClient":
        from .client import A2AClient

        return A2AClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
