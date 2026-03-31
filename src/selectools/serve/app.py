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
from .builder import BUILDER_HTML
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
        enable_builder: bool = False,
    ) -> None:
        self.agent = agent
        self.prefix = prefix.rstrip("/")
        self.enable_playground = enable_playground
        self.enable_builder = enable_builder

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
    builder: bool = False,
    host: str = "0.0.0.0",  # nosec B104
    port: int = 8000,
) -> "AgentServer":
    """Create a standalone HTTP server for an agent.

    Usage::

        app = create_app(agent, playground=True, builder=True)
        app.serve(port=8000)  # Blocking
    """
    return AgentServer(
        agent, prefix=prefix, playground=playground, builder=builder, host=host, port=port
    )


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
        builder: bool = False,
        host: str = "0.0.0.0",  # nosec B104
        port: int = 8000,
    ) -> None:
        self.router = AgentRouter(
            agent, prefix=prefix, enable_playground=playground, enable_builder=builder
        )
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
                    path == f"{router.prefix}/builder" or path == "/builder"
                ) and router.enable_builder:
                    self._html_response(BUILDER_HTML)
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
        if router.enable_builder:
            print(f"  GET  /builder    — visual agent builder")
        print(f"\nPress Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()


def _builder_run_mock(
    nodes_data: List[Dict[str, Any]],
    input_msg: str,
    emit: Any,
) -> None:
    """Execute a mock graph run (no API keys required)."""
    agent_nodes = [n for n in nodes_data if n.get("type") == "agent"]
    if not agent_nodes:
        emit({"type": "error", "message": "No agent nodes in graph."})
        return

    total_tokens = 0
    for n in agent_nodes:
        node_id = n.get("id", "?")
        node_name = n.get("name", node_id)
        model = n.get("model", "gpt-4o-mini")
        provider = n.get("provider", "openai")
        emit({"type": "node_start", "node_id": node_id, "node_name": node_name})

        tools_str = n.get("tools", "")
        if tools_str:
            for tool_name in [t.strip() for t in tools_str.split(",") if t.strip()][:2]:
                emit(
                    {
                        "type": "tool_call",
                        "node_id": node_id,
                        "tool": tool_name,
                        "args": {"query": input_msg[:40]},
                    }
                )
                emit(
                    {
                        "type": "tool_result",
                        "node_id": node_id,
                        "tool": tool_name,
                        "result": f"[mock result from {tool_name}]",
                    }
                )

        mock_text = (
            f"[MOCK] {node_name} processed your request using {provider}/{model}. "
            f"In live mode this calls the real API. Input: {input_msg[:60]}"
        )
        for word in mock_text.split():
            emit({"type": "chunk", "node_id": node_id, "content": word + " "})
            time.sleep(0.025)

        node_tokens = 45
        total_tokens += node_tokens
        emit({"type": "node_end", "node_id": node_id, "tokens": node_tokens, "cost": 0.0})

    emit({"type": "run_end", "total_tokens": total_tokens, "total_cost": 0.0})


def _builder_run_live(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    input_msg: str,
    api_key: str,
    emit: Any,
) -> None:
    """Execute a live graph run using the real provider."""
    from ..agent.config import AgentConfig
    from ..agent.core import Agent
    from ..toolbox import get_all_tools

    # Determine provider from first agent node
    provider_name = "openai"
    for n in nodes_data:
        if n.get("type") == "agent":
            provider_name = n.get("provider", "openai")
            break

    from ..providers.base import Provider as _Provider

    live_provider: _Provider
    try:
        if provider_name == "anthropic":
            from ..providers.anthropic_provider import AnthropicProvider

            live_provider = AnthropicProvider(api_key=api_key)
        elif provider_name == "gemini":
            from ..providers.gemini_provider import GeminiProvider

            live_provider = GeminiProvider(api_key=api_key)
        elif provider_name == "ollama":
            from ..providers.ollama_provider import OllamaProvider

            live_provider = OllamaProvider()
        else:
            from ..providers.openai_provider import OpenAIProvider

            live_provider = OpenAIProvider(api_key=api_key)
    except Exception as exc:
        emit({"type": "error", "message": f"Provider init failed: {exc}"})
        return

    toolbox = get_all_tools()
    agent_nodes = [n for n in nodes_data if n.get("type") == "agent"]
    if not agent_nodes:
        emit({"type": "error", "message": "No agent nodes in graph."})
        return

    # Follow edges from START to determine execution order
    start_node = next((n for n in nodes_data if n.get("type") == "start"), None)
    ordered: List[Dict[str, Any]] = []
    visited: set = set()

    def _walk(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        n = next((x for x in nodes_data if x["id"] == nid), None)
        if n and n.get("type") == "agent":
            ordered.append(n)
        for e in edges_data:
            if e.get("from") == nid:
                _walk(e.get("to", ""))

    if start_node:
        _walk(start_node["id"])
    for n in agent_nodes:
        if n["id"] not in visited:
            ordered.append(n)

    total_tokens = 0
    total_cost = 0.0
    current_input = input_msg

    for n in ordered:
        node_id = n.get("id", "?")
        node_name = n.get("name", node_id)
        model = n.get("model", "gpt-4o-mini")
        system_prompt = n.get("system_prompt", "")
        emit({"type": "node_start", "node_id": node_id, "node_name": node_name})
        try:
            config = AgentConfig(
                name=node_name,
                model=model,
                system_prompt=system_prompt or None,
            )
            agent = Agent(toolbox, provider=live_provider, config=config)
            result = agent.run(current_input)
            content = result.content or ""
            for word in content.split():
                emit({"type": "chunk", "node_id": node_id, "content": word + " "})
            usage = result.usage
            node_tokens = usage.total_tokens if usage else 0
            node_cost = float(usage.total_cost_usd) if usage else 0.0
            total_tokens += node_tokens
            total_cost += node_cost
            emit({"type": "node_end", "node_id": node_id, "tokens": node_tokens, "cost": node_cost})
            current_input = content or current_input
        except Exception as exc:
            emit({"type": "error", "message": f"{node_name}: {exc}"})

    emit({"type": "run_end", "total_tokens": total_tokens, "total_cost": total_cost})


class BuilderServer:
    """Standalone visual builder server — no agent required.

    Usage::

        from selectools.serve.app import BuilderServer
        BuilderServer(port=8080).serve()

    Or via CLI::

        selectools serve --builder

    Endpoints:
        GET  /builder  — visual builder UI
        GET  /health   — health check
        POST /run      — execute graph (mock or live via SSE)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # nosec B104
        port: int = 8000,
    ) -> None:
        self.host = host
        self.port = port

    def serve(self, port: Optional[int] = None) -> None:
        """Start the builder-only HTTP server (blocking)."""
        actual_port = port or self.port

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                if path in ("/builder", ""):
                    self._html(BUILDER_HTML)
                elif path == "/health":
                    self._json({"status": "ok", "mode": "builder"})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                if path != "/run":
                    self._json({"error": "not found"}, 404)
                    return
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length else b"{}"
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    self._json({"error": "invalid JSON"}, 400)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def emit(event: Dict[str, Any]) -> None:
                    line = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()

                try:
                    input_msg = body.get("input", "Hello")
                    nodes_data = body.get("nodes", [])
                    edges_data = body.get("edges", [])
                    api_key = body.get("api_key", "").strip()
                    mock_mode = not api_key

                    emit({"type": "run_start", "mock": mock_mode})
                    if mock_mode:
                        _builder_run_mock(nodes_data, input_msg, emit)
                    else:
                        _builder_run_live(nodes_data, edges_data, input_msg, api_key, emit)
                except Exception as exc:
                    emit({"type": "error", "message": str(exc)})
                finally:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def _json(self, data: Dict[str, Any], status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _html(self, html: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
                pass

        server = HTTPServer((self.host, actual_port), Handler)
        print(f"Visual agent builder at http://{self.host}:{actual_port}/builder")
        print(f"  POST /run  — execute graph (mock or live via SSE)")
        print(f"\nPress Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
