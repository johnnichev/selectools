"""
Agent serving via lightweight HTTP.

Uses Python's built-in http.server for zero dependencies.
For production, use the FastAPI or Flask integrations.
"""

from __future__ import annotations

import asyncio
import hmac
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

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>selectools — login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f172a;color:#e2e8f0;font-family:ui-monospace,monospace;
  display:flex;align-items:center;justify-content:center;height:100vh}
.box{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:32px;width:340px}
h2{color:#22d3ee;font-size:16px;margin-bottom:20px}
input{width:100%;padding:10px 12px;background:#0f172a;color:#e2e8f0;
  border:1px solid #334155;border-radius:6px;font:13px ui-monospace,monospace;outline:none}
input:focus{border-color:#22d3ee}
button{margin-top:14px;width:100%;padding:10px;background:#22d3ee;color:#0f172a;
  border:none;border-radius:6px;font:13px ui-monospace,monospace;font-weight:700;cursor:pointer}
button:hover{background:#38bdf8}
.err{margin-top:10px;color:#f87171;font-size:12px}
</style></head>
<body><div class="box">
<h2>selectools builder</h2>
<form method="POST" action="/login">
<input type="password" name="token" placeholder="Enter access token" autofocus autocomplete="current-password">
<button type="submit">Unlock Builder</button>
<p class="err"></p>
</form>
</div></body></html>"""

LOGIN_HTML_ERROR = LOGIN_HTML.replace(
    '<p class="err"></p>', '<p class="err">Invalid token &mdash; try again.</p>'
)


def _resolve_auth_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve auth token from CLI flag, env var, or ~/.selectools/auth_token file.

    Priority: CLI flag > BUILDER_AUTH_TOKEN env var > ~/.selectools/auth_token file.
    Returns None if no token is configured (auth disabled).
    """
    if cli_token:
        return cli_token
    env = os.environ.get("BUILDER_AUTH_TOKEN")
    if env:
        return env
    dotfile = os.path.expanduser("~/.selectools/auth_token")
    if os.path.isfile(dotfile):
        try:
            token = open(dotfile).read().strip()  # noqa: WPS515
            if token:
                return token
        except OSError:
            pass
    return None


def _make_session_cookie(auth_token: str) -> str:
    """Return the HMAC-signed session cookie value for a given auth token."""
    return hmac.new(auth_token.encode(), b"builder_ok", "sha256").hexdigest()


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
    auth_token: Optional[str] = None,
) -> "AgentServer":
    """Create a standalone HTTP server for an agent.

    Usage::

        app = create_app(agent, playground=True, builder=True)
        app.serve(port=8000)  # Blocking
    """
    return AgentServer(
        agent,
        prefix=prefix,
        playground=playground,
        builder=builder,
        host=host,
        port=port,
        auth_token=auth_token,
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
        auth_token: Optional[str] = None,
    ) -> None:
        self.router = AgentRouter(
            agent, prefix=prefix, enable_playground=playground, enable_builder=builder
        )
        self.host = host
        self.port = port
        self.auth_token = auth_token

    def serve(self, port: Optional[int] = None) -> None:
        """Start the HTTP server (blocking)."""
        actual_port = port or self.port
        router = self.router
        _auth_token = self.auth_token

        class Handler(BaseHTTPRequestHandler):
            def _is_authed(self) -> bool:
                if not _auth_token:
                    return True
                cookie_header = self.headers.get("Cookie", "")
                expected = _make_session_cookie(_auth_token)
                for part in cookie_header.split(";"):
                    k, _, v = part.strip().partition("=")
                    if k == "builder_session" and hmac.compare_digest(v.strip(), expected):
                        return True
                return False

            def _redirect_login(self) -> None:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                if path in ("/health", f"{router.prefix}/health"):
                    self._json_response(router.handle_health())
                    return
                if path == "/login":
                    self._html_response(LOGIN_HTML)
                    return
                if not self._is_authed():
                    self._redirect_login()
                    return
                if path == f"{router.prefix}/schema" or path == "/schema":
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
                path = urlparse(self.path).path.rstrip("/")
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length else b"{}"
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    self._json_response({"error": "invalid JSON"}, 400)
                    return

                if path == "/login":
                    token = body.get("token", "")
                    if _auth_token and hmac.compare_digest(token, _auth_token):
                        cookie_val = _make_session_cookie(_auth_token)
                        self.send_response(302)
                        self.send_header(
                            "Set-Cookie",
                            f"builder_session={cookie_val}; HttpOnly; SameSite=Strict; Path=/",
                        )
                        self.send_header("Location", "/builder")
                        self.end_headers()
                    else:
                        self._html_response(LOGIN_HTML_ERROR)
                    return

                if not self._is_authed():
                    self._redirect_login()
                    return

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


def _run_builder_evals(text: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """Run deterministic eval checks on node output. No API key required."""
    results: List[Dict[str, Any]] = []

    # not_empty: output must be non-blank
    results.append({"name": "not_empty", "pass": bool(text.strip())})

    # no_apology: output should not open with an apology
    lower = text.lower().strip()
    no_apology = not any(
        lower.startswith(p) for p in ("i'm sorry", "i am sorry", "i apologize", "sorry,")
    )
    results.append({"name": "no_apology", "pass": no_apology})

    # eval_assertion: user-defined keyword/phrase must appear in output
    assertion = (node.get("eval_assertion") or "").strip()
    if assertion:
        results.append(
            {"name": f"contains({assertion[:20]})", "pass": assertion.lower() in text.lower()}
        )

    overall = all(r["pass"] for r in results)
    return {"pass": overall, "results": results}


def _builder_run_mock(
    nodes_data: List[Dict[str, Any]],
    input_msg: str,
    emit: Any,
) -> None:
    """Execute a mock graph run (no API keys required)."""
    _t0 = time.time()
    _base_emit = emit

    def emit(ev: Dict[str, Any]) -> None:  # type: ignore[misc, no-redef]
        ev["ts"] = int((time.time() - _t0) * 1000)
        _base_emit(ev)

    agent_nodes = [n for n in nodes_data if n.get("type") == "agent"]
    hitl_nodes = [n for n in nodes_data if n.get("type") == "hitl"]
    if not agent_nodes and not hitl_nodes:
        emit({"type": "error", "message": "No agent nodes in graph."})
        return

    total_tokens = 0
    for n in nodes_data:
        if n.get("type") not in ("agent", "hitl"):
            continue
        node_id = n.get("id", "?")
        node_name = n.get("name", node_id)
        node_type = n.get("type", "agent")
        emit(
            {
                "type": "node_start",
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
            }
        )

        if node_type == "hitl":
            # Emit pause event; auto-resolve with first option after brief delay
            opts_raw = n.get("options", "approve, reject")
            opts = [o.strip() for o in opts_raw.split(",") if o.strip()]
            auto_choice = opts[0] if opts else "approve"
            emit(
                {
                    "type": "hitl_pause",
                    "node_id": node_id,
                    "node_name": node_name,
                    "options": opts_raw,
                }
            )
            time.sleep(0.5)
            emit({"type": "hitl_auto", "node_id": node_id, "choice": auto_choice})
            emit({"type": "node_end", "node_id": node_id, "tokens": 0, "cost": 0.0})
            continue

        model = n.get("model", "gpt-4o-mini")
        provider = n.get("provider", "openai")
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

        eval_data = _run_builder_evals(mock_text, n)
        emit(
            {
                "type": "eval_result",
                "node_id": node_id,
                "pass": eval_data["pass"],
                "results": eval_data["results"],
            }
        )

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
    _t0 = time.time()
    _base_emit = emit

    def emit(ev: Dict[str, Any]) -> None:  # type: ignore[misc, no-redef]
        ev["ts"] = int((time.time() - _t0) * 1000)
        _base_emit(ev)

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


def _ai_build_fallback(description: str) -> Dict[str, Any]:
    """Deterministic graph builder from keyword detection — no API key required."""
    desc = description.lower()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seq = 1

    def mkid(t: str) -> str:
        nonlocal seq
        i = seq
        seq += 1
        return f"{t}_{i}"

    start_id = mkid("start")
    end_id = mkid("end")
    nodes.append({"id": start_id, "type": "start", "name": "START"})

    agent_specs: List[tuple] = []
    if any(w in desc for w in ["research", "search", "find", "look up"]):
        agent_specs.append(("Researcher", "Search for information and summarise findings."))
    if any(w in desc for w in ["write", "draft", "compose", "generate"]):
        agent_specs.append(("Writer", "Write clear, well-structured content based on the input."))
    if any(w in desc for w in ["review", "critic", "evaluat", "check", "assess"]):
        agent_specs.append(
            ("Critic", "Review the output. Respond with 'approved' or 'revise: <feedback>'.")
        )
    if any(w in desc for w in ["classif", "categor", "sort", "route"]):
        agent_specs.append(("Classifier", "Classify the input and route it appropriately."))
    if any(w in desc for w in ["summariz", "summar", "condense"]):
        agent_specs.append(("Summarizer", "Summarize the input concisely."))
    if not agent_specs:
        agent_specs = [("Agent", "Process the input and produce a helpful response.")]

    prev_id = start_id
    for name, prompt in agent_specs:
        aid = mkid("agent")
        nodes.append(
            {
                "id": aid,
                "type": "agent",
                "name": name,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": prompt,
                "tools": "",
                "frozen": False,
                "eval_assertion": "",
            }
        )
        edges.append({"id": mkid("edge"), "from": prev_id, "to": aid, "label": ""})
        prev_id = aid

    nodes.append({"id": end_id, "type": "end", "name": "END"})
    edges.append({"id": mkid("edge"), "from": prev_id, "to": end_id, "label": ""})
    return {"nodes": nodes, "edges": edges}


def _ai_build_live(description: str, api_key: str) -> Dict[str, Any]:
    """Call real LLM to generate graph nodes and edges from a description."""
    system_prompt = (
        "You are a selectools agent graph designer. Given a workflow description, "
        'output a JSON object with "nodes" and "edges" arrays.\n\n'
        "Node types: start (required, one only), end (required, one only), agent (main type), "
        "hitl (human-in-the-loop), loop (iteration).\n"
        'Agent node fields: id (string), type ("agent"), name (string), system_prompt (string), '
        'provider ("openai"), model ("gpt-4o-mini"), tools (""), frozen (false), eval_assertion ("").\n'
        "Edge fields: id (string), from (node id), to (node id), label (string, empty or routing condition).\n\n"
        "Rules:\n"
        "- Always include exactly one start and one end node.\n"
        "- IDs: start_1, end_1, agent_1, agent_2, etc.\n"
        "- Start node connects to the first agent.\n"
        "- Last agent connects to end.\n"
        "- Keep graphs simple: 2-5 agent nodes unless the description clearly needs more.\n"
        "- If the workflow has a review/revision loop, add a labelled edge back to an earlier node.\n"
        "- Output ONLY the JSON object. No explanation, no markdown fences."
    )

    try:
        provider: Any = None
        if api_key.startswith("sk-ant"):
            from ..providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key=api_key)
        else:
            from ..providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key=api_key)

        from ..types import Message, Role

        messages = [Message(role=Role.USER, content=description)]
        result = provider.complete(messages, system_prompt=system_prompt, model="gpt-4o-mini")
        raw = (result.content or "").strip()
        # Strip markdown fences if model added them
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        parsed = json.loads(raw)
        if "nodes" not in parsed or "edges" not in parsed:
            raise ValueError("Missing nodes or edges")
        return parsed
    except Exception:
        return _ai_build_fallback(description)


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
        auth_token: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.auth_token = auth_token

    def serve(self, port: Optional[int] = None) -> None:
        """Start the builder-only HTTP server (blocking)."""
        actual_port = port or self.port
        _auth_token = self.auth_token

        class Handler(BaseHTTPRequestHandler):
            def _is_authed(self) -> bool:
                if not _auth_token:
                    return True
                cookie_header = self.headers.get("Cookie", "")
                expected = _make_session_cookie(_auth_token)
                for part in cookie_header.split(";"):
                    k, _, v = part.strip().partition("=")
                    if k == "builder_session" and hmac.compare_digest(v.strip(), expected):
                        return True
                return False

            def _redirect_login(self) -> None:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                if path == "/health":
                    self._json({"status": "ok", "mode": "builder"})
                    return
                if path == "/login":
                    self._html(LOGIN_HTML)
                    return
                if not self._is_authed():
                    self._redirect_login()
                    return
                if path in ("/builder", ""):
                    self._html(BUILDER_HTML)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path.rstrip("/")
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length else b"{}"
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    self._json({"error": "invalid JSON"}, 400)
                    return

                if path == "/login":
                    token = body.get("token", "")
                    if _auth_token and hmac.compare_digest(token, _auth_token):
                        cookie_val = _make_session_cookie(_auth_token)
                        self.send_response(302)
                        self.send_header(
                            "Set-Cookie",
                            f"builder_session={cookie_val}; HttpOnly; SameSite=Strict; Path=/",
                        )
                        self.send_header("Location", "/builder")
                        self.end_headers()
                    else:
                        self._html(LOGIN_HTML_ERROR)
                    return

                if not self._is_authed():
                    self._redirect_login()
                    return

                if path == "/ai-build":
                    description = body.get("description", "").strip()
                    api_key = body.get("api_key", "").strip()
                    if not description:
                        self._json({"error": "description required"}, 400)
                        return
                    if api_key:
                        result = _ai_build_live(description, api_key)
                    else:
                        result = _ai_build_fallback(description)
                    self._json(result)
                    return

                if path != "/run":
                    self._json({"error": "not found"}, 404)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.close_connection = True

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
