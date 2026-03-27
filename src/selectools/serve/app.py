"""
Agent serving via lightweight HTTP.

Uses Python's built-in http.server for zero dependencies.
For production, use the FastAPI or Flask integrations.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from ..agent.core import Agent

from .. import __version__
from .models import HealthResponse, InvokeResponse
from .playground import PLAYGROUND_HTML


class AgentRouter:
    """Routes HTTP requests to agent methods.

    Works with any WSGI-compatible framework. Also usable standalone
    via ``create_app()``.

    Endpoints:
        POST /invoke  — single prompt → JSON response
        POST /stream  — single prompt → SSE stream
        GET  /health  — health check
        GET  /schema  — tool schemas
        GET  /playground — chat UI
    """

    def __init__(
        self,
        agent: "Agent",
        prefix: str = "",
        enable_playground: bool = True,
    ) -> None:
        self.agent = agent
        self.prefix = prefix.rstrip("/")
        self.enable_playground = enable_playground

    def handle_invoke(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle POST /invoke."""
        prompt = body.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        try:
            result = self.agent.run(prompt)
        except Exception as exc:
            return {"error": str(exc), "type": type(exc).__name__}

        response = InvokeResponse(
            content=result.content or "",
            tool_calls=[
                {"tool_name": tc.tool_name, "parameters": tc.parameters}
                for tc in (result.tool_calls or [])
            ],
            reasoning=result.reasoning,
            iterations=result.iterations,
            tokens=result.usage.total_tokens if result.usage else 0,
            cost_usd=result.usage.total_cost_usd if result.usage else 0.0,
            run_id=result.trace.run_id if result.trace else "",
        )
        return response.to_dict()

    def handle_stream(self, body: Dict[str, Any]):
        """Handle POST /stream as SSE. Yields SSE-formatted strings."""
        prompt = body.get("prompt", "")
        if not prompt:
            yield 'data: {"error": "prompt is required"}\n\n'
            return

        async def _stream():
            chunks = []
            async for item in self.agent.astream(prompt):
                from ..types import AgentResult, StreamChunk

                if isinstance(item, str):
                    chunks.append(item)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': item})}\n\n"
                elif hasattr(item, "content") and hasattr(item, "iterations"):
                    # AgentResult
                    yield f"data: {json.dumps({'type': 'result', 'content': item.content or '', 'iterations': item.iterations})}\n\n"
            yield "data: [DONE]\n\n"

        loop = asyncio.new_event_loop()
        gen = _stream()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def handle_health(self) -> Dict[str, Any]:
        """Handle GET /health."""
        response = HealthResponse(
            status="ok",
            version=__version__,
            model=self.agent.config.model,
            provider=getattr(self.agent.provider, "name", "unknown"),
            tools=[t.name for t in self.agent.tools],
        )
        return {
            "status": response.status,
            "version": response.version,
            "model": response.model,
            "provider": response.provider,
            "tools": response.tools,
        }

    def handle_schema(self) -> Dict[str, Any]:
        """Handle GET /schema."""
        return {
            "tools": [t.schema() for t in self.agent.tools],
            "model": self.agent.config.model,
        }


def create_app(
    agent: "Agent",
    prefix: str = "",
    playground: bool = True,
    host: str = "0.0.0.0",  # nosec B104
    port: int = 8000,
) -> "AgentServer":
    """Create a standalone HTTP server for an agent.

    Usage::

        app = create_app(agent, playground=True)
        app.serve(port=8000)  # Blocking
    """
    return AgentServer(agent, prefix=prefix, playground=playground, host=host, port=port)


class AgentServer:
    """Lightweight HTTP server wrapping an AgentRouter.

    Zero dependencies — uses Python stdlib http.server.
    For production, use FastAPI or Flask integrations instead.
    """

    def __init__(
        self,
        agent: "Agent",
        prefix: str = "",
        playground: bool = True,
        host: str = "0.0.0.0",  # nosec B104
        port: int = 8000,
    ) -> None:
        self.router = AgentRouter(agent, prefix=prefix, enable_playground=playground)
        self.host = host
        self.port = port

    def serve(self, port: Optional[int] = None) -> None:
        """Start the HTTP server (blocking)."""
        actual_port = port or self.port
        router = self.router

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                if path == f"{router.prefix}/health" or path == "/health":
                    self._json_response(router.handle_health())
                elif path == f"{router.prefix}/schema" or path == "/schema":
                    self._json_response(router.handle_schema())
                elif (
                    path == f"{router.prefix}/playground" or path == "/playground" or path == "/"
                ) and router.enable_playground:
                    self._html_response(PLAYGROUND_HTML)
                else:
                    self._json_response({"error": "not found"}, 404)

            def do_POST(self):  # noqa: N802
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length else b"{}"
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    self._json_response({"error": "invalid JSON"}, 400)
                    return

                path = urlparse(self.path).path.rstrip("/")

                if path == f"{router.prefix}/invoke" or path == "/invoke":
                    result = router.handle_invoke(body)
                    self._json_response(result)
                elif path == f"{router.prefix}/stream" or path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    for chunk in router.handle_stream(body):
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                else:
                    self._json_response({"error": "not found"}, 404)

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def _json_response(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def _html_response(self, html):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            def log_message(self, format, *args):
                pass  # Suppress default logging

        server = HTTPServer((self.host, actual_port), Handler)
        print(f"Selectools agent serving at http://{self.host}:{actual_port}")
        print(f"  POST /invoke   — single prompt")
        print(f"  POST /stream   — SSE streaming")
        print(f"  GET  /health   — health check")
        print(f"  GET  /schema   — tool schemas")
        if router.enable_playground:
            print(f"  GET  /playground — chat UI")
        print(f"\nPress Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
