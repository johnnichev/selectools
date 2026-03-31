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
"""

from .app import AgentRouter, BuilderServer, create_app

__all__ = ["AgentRouter", "BuilderServer", "create_app"]
