"""
Serve agents as REST APIs with one command.

Usage::

    # CLI
    selectools serve agent.yaml
    selectools serve agent.yaml --port 8000 --playground --builder
    selectools serve --builder   # builder only, no agent needed

    # Python
    from selectools.serve import AgentRouter, BuilderServer, create_app

    router = AgentRouter(agent)
    app = create_app(agent, playground=True, builder=True)
    app.run(port=8000)

    # Builder-only (no agent)
    BuilderServer(port=8080).serve()

    # Production REST API (requires: pip install selectools[serve])
    from selectools.serve import AgentAPI
    app = AgentAPI(agents=[agent], auth_key="sk-...")
    # uvicorn app:app --port 8000
"""

from typing import Any

from .app import AgentRouter, BuilderServer, create_app

__all__ = ["A2AServer", "AgentAPI", "AgentRouter", "BuilderServer", "create_app"]


def __getattr__(name: str) -> Any:
    """Lazily import Starlette-backed apps so this package imports without starlette."""
    if name == "AgentAPI":
        try:
            from .api import AgentAPI
        except ImportError as exc:
            raise ImportError(
                "AgentAPI requires the 'starlette' package. "
                "Install it with: pip install selectools[serve]"
            ) from exc
        return AgentAPI
    if name == "A2AServer":
        try:
            from ..a2a.server import A2AServer
        except ImportError as exc:
            raise ImportError(
                "A2AServer requires the 'starlette' package. "
                "Install it with: pip install selectools[serve]"
            ) from exc
        return A2AServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
