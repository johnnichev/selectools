"""
Serve agents as REST APIs with one command.

Usage::

    # CLI
    selectools serve agent.yaml
    selectools serve agent.yaml --port 8000 --playground

    # Python
    from selectools.serve import AgentRouter, create_app

    router = AgentRouter(agent)
    app = create_app(agent, playground=True)
    app.run(port=8000)
"""

from .app import AgentRouter, create_app

__all__ = ["AgentRouter", "create_app"]
