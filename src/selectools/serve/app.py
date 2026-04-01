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
    host: str = "0.0.0.0",
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
        host: str = "0.0.0.0",
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
    pinned_ports: Optional[Dict[str, Any]] = None,
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
        parsed: Dict[str, Any] = json.loads(raw)
        if "nodes" not in parsed or "edges" not in parsed:
            raise ValueError("Missing nodes or edges")
        return parsed
    except Exception:
        return _ai_build_fallback(description)


# ─── Feature 10: Data Pinning ────────────────────────────────────────────────
def _apply_pinned_ports(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    pinned_ports: Dict[str, Any],
    last_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Return node_inputs dict respecting pinned port overrides."""
    inputs: Dict[str, Any] = {}
    for edge in edges_data:
        src = edge.get("source") or edge.get("from", "")
        src_handle = edge.get("sourceHandle", "output")
        tgt_handle = edge.get("targetHandle", "input")
        key = f"{src}::{src_handle}"
        if key in pinned_ports:
            inputs[tgt_handle] = pinned_ports[key]
        else:
            inputs[tgt_handle] = last_outputs.get(src)
    return inputs


# ─── Feature 14: AI Copilot ───────────────────────────────────────────────────
def _ai_refine_live(
    current_graph: Dict[str, Any],
    selected_node_id: Optional[str],
    message: str,
    history: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Any]:
    """Call LLM to generate an iterative graph patch from the current state."""
    system_prompt = (
        "You are an AI assistant for a visual agent builder. "
        "The user has a graph of AI agents with nodes and edges. "
        "Given their current graph and a natural-language request, return a JSON patch. "
        'Patch format: {"type": "update_node|add_node|remove_node|add_edge", '
        '"node_id": "...", "changes": {...}}. '
        "For update_node, changes is a dict of node properties to update. "
        "For add_node, changes contains the full new node definition. "
        'Return ONLY a JSON object with keys: "patch", "explanation", "suggested_follow_up".'
    )
    n_nodes = len(current_graph.get("nodes", []))
    context = f"Current graph has {n_nodes} nodes.\n"
    if selected_node_id:
        node = next(
            (n for n in current_graph.get("nodes", []) if n.get("id") == selected_node_id), None
        )
        if node:
            context += f"Selected node: {json.dumps(node)}\n"
    from ..types import Message, Role

    messages: List[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
    for m in history[-6:]:
        try:
            messages.append(Message(role=Role(m["role"]), content=m["content"]))
        except (KeyError, ValueError):
            pass
    messages.append(Message(role=Role.USER, content=context + message))
    try:
        from ..providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key=api_key)
        resp_msg, _ = provider.complete(
            messages=messages, model="gpt-4o-mini", system_prompt="", max_tokens=600
        )
        raw = (resp_msg.content or "").strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        data: Dict[str, Any] = json.loads(raw)
        if "patch" not in data:
            raise ValueError("missing patch")
        return data
    except Exception as exc:
        return {"error": str(exc), "patch": None, "explanation": "", "suggested_follow_up": ""}


# ─── Feature 15: HITL Form Builder ───────────────────────────────────────────
def _render_hitl_form(node_data: Dict[str, Any]) -> str:
    """Render HITL form fields as HTML snippet for the /wait page."""
    fields = node_data.get("form_fields", [])
    if not fields:
        # fallback: old options buttons
        opts_raw = node_data.get("options", "approve, reject")
        opts = [o.strip() for o in opts_raw.split(",") if o.strip()]
        buttons = "".join(
            f'<button name="choice" value="{o}" style="padding:8px 18px;margin:4px;border:1px solid #f59e0b;border-radius:6px;background:rgba(245,158,11,0.1);color:#f59e0b;cursor:pointer;font-size:13px">{o}</button>'
            for o in opts
        )
        return buttons
    html_fields: List[str] = []
    for f in fields:
        label = f.get("label", "")
        ftype = f.get("type", "text")
        fid = f.get("id", "")
        placeholder = f.get("placeholder", "")
        required = "required" if f.get("required") else ""
        style = "width:100%;padding:8px;margin:4px 0;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;font-family:inherit;font-size:13px"
        if ftype == "text":
            html_fields.append(
                f'<label style="font-size:12px;color:#94a3b8">{label}</label>'
                f'<input type="text" name="{fid}" placeholder="{placeholder}" {required} style="{style}"><br>'
            )
        elif ftype == "textarea":
            html_fields.append(
                f'<label style="font-size:12px;color:#94a3b8">{label}</label>'
                f'<textarea name="{fid}" placeholder="{placeholder}" {required} style="{style};height:80px;resize:vertical"></textarea><br>'
            )
        elif ftype == "number":
            html_fields.append(
                f'<label style="font-size:12px;color:#94a3b8">{label}</label>'
                f'<input type="number" name="{fid}" {required} style="{style}"><br>'
            )
        elif ftype == "select":
            opts2 = "".join(f"<option>{o}</option>" for o in f.get("options", []))
            html_fields.append(
                f'<label style="font-size:12px;color:#94a3b8">{label}</label>'
                f'<select name="{fid}" {required} style="{style}">{opts2}</select><br>'
            )
        elif ftype == "checkbox":
            html_fields.append(
                f'<label style="font-size:12px;color:#94a3b8">'
                f'<input type="checkbox" name="{fid}" {required} style="margin-right:6px"> {label}</label><br>'
            )
    return "\n".join(html_fields)


# ─── Feature 16: Bidirectional file sync ─────────────────────────────────────
def _parse_python_to_graph(source: str) -> Dict[str, Any]:
    """Parse a selectools Python agent file into {nodes, edges} using AST."""
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return {"nodes": [], "edges": []}
    nodes_out: List[Dict[str, Any]] = []
    edges_out: List[Dict[str, Any]] = []
    i = 0
    for stmt in _ast.walk(tree):
        if not (
            isinstance(stmt, _ast.Expr)
            and isinstance(stmt.value, _ast.Call)
            and isinstance(stmt.value.func, _ast.Attribute)
        ):
            continue
        call = stmt.value  # type: ignore[union-attr]
        method = call.func.attr  # type: ignore[attr-defined]
        if method == "add_node" and call.args:
            try:
                node_id = _ast.literal_eval(call.args[0])
                nodes_out.append(
                    {"id": node_id, "label": node_id, "type": "agent", "x": i * 180 + 60, "y": 200}
                )
                i += 1
            except (ValueError, TypeError):
                pass
        elif method == "add_edge" and len(call.args) >= 2:
            try:
                src = _ast.literal_eval(call.args[0])
                dst = _ast.literal_eval(call.args[1])
                edges_out.append(
                    {"id": f"e_{src}_{dst}", "source": src, "target": dst, "from": src, "to": dst}
                )
            except (ValueError, TypeError):
                pass
    return {"nodes": nodes_out, "edges": edges_out}


# ─── Feature 17: Agent-as-Tool node ──────────────────────────────────────────
def _build_agent_tool_from_node(
    node_data: Dict[str, Any], graph_nodes: List[Dict[str, Any]], api_key: str
) -> Any:
    """Create a selectools Tool that runs a nested agent node."""
    from ..agent.config import AgentConfig
    from ..agent.core import Agent
    from ..tools.base import Tool, ToolParameter

    target_id = node_data.get("tool_target_node")
    target_node = next((n for n in graph_nodes if n.get("id") == target_id), None)
    tool_name = node_data.get("tool_name", "nested_agent")
    tool_desc = node_data.get("tool_description", "A nested agent")
    input_param = node_data.get("tool_input_param", "query")
    max_tokens = int(node_data.get("tool_max_tokens", 500))

    def _run_nested(**kwargs: Any) -> str:
        inp = kwargs.get(input_param, "")
        if not target_node:
            return f"[agent_tool error: target node {target_id} not found]"
        from ..providers.openai_provider import OpenAIProvider

        cfg = AgentConfig(
            model=target_node.get("model", "gpt-4o-mini"),
            system_prompt=target_node.get("system_prompt", ""),
            max_tokens=max_tokens,
        )
        provider = OpenAIProvider(api_key=api_key)
        agent = Agent(tools=[], config=cfg, provider=provider)
        result = agent.run(str(inp))
        return result.content or ""

    return Tool(
        name=tool_name,
        description=tool_desc,
        function=_run_nested,
        parameters=[
            ToolParameter(name=input_param, param_type=str, description="Input to the nested agent")
        ],
    )


# ─── Feature 18: Multi-user auth + RBAC ──────────────────────────────────────
def _resolve_users() -> Dict[str, Any]:
    """Load user token→role map from BUILDER_USERS env or ~/.selectools/users.json."""
    raw = os.environ.get("BUILDER_USERS")
    if not raw:
        dotfile = os.path.expanduser("~/.selectools/users.json")
        if os.path.isfile(dotfile):
            try:
                raw = open(dotfile).read()  # noqa: WPS515
            except OSError:
                raw = None
    if raw:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


ROLES: Dict[str, set] = {
    "admin": {"view", "edit", "run", "export", "delete", "manage_users"},
    "editor": {"view", "edit", "run", "export"},
    "viewer": {"view"},
}


def _has_permission(role: str, action: str) -> bool:
    return action in ROLES.get(role, set())


def _check_graph_permission(
    graph_id: str, username: str, role: str, action: str, graphs_dir: Optional[str] = None
) -> bool:
    if role == "admin":
        return True
    base = graphs_dir or os.path.expanduser("~/.selectools/graphs")
    path = os.path.join(base, f"{graph_id}.json")
    if not os.path.isfile(path):
        return False
    try:
        g = json.loads(open(path).read())  # noqa: WPS515
    except Exception:
        return False
    if g.get("owner") == username:
        return True
    for entry in g.get("acl", []):
        if entry.get("user") == username:
            return _has_permission(entry.get("permission", "viewer"), action)
    return False


# ─── Feature 19: Online production eval ──────────────────────────────────────
import queue as _queue
import threading as _threading

_eval_queue: "_queue.Queue[Optional[Dict[str, Any]]]" = _queue.Queue()


def _log_run(run_data: Dict[str, Any]) -> None:
    """Append a completed run record to ~/.selectools/runs/<date>.jsonl."""
    log_dir = os.path.expanduser("~/.selectools/runs")
    os.makedirs(log_dir, exist_ok=True)
    import datetime

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(log_dir, f"{today}.jsonl")
    try:
        with open(path, "a") as f:
            f.write(json.dumps(run_data) + "\n")
    except OSError:
        pass


def _run_evals_on_run(job: Dict[str, Any]) -> None:
    """Run configured evaluators on a completed production run (background worker)."""
    try:
        config = job.get("eval_config", {})
        evaluator_names = config.get("evaluators", [])
        if not evaluator_names:
            return
        from ..evals.evaluators import DEFAULT_EVALUATORS

        _eval_registry = {type(e).__name__: type(e) for e in DEFAULT_EVALUATORS}
        results = []
        for name in evaluator_names:
            ev_cls = _eval_registry.get(name)
            if ev_cls:
                try:
                    from ..evals.types import TestCase

                    case = TestCase(input=job.get("input", ""))
                    r = ev_cls().evaluate(case)
                    results.append({"name": name, "pass": r.pass_, "score": r.score})
                except Exception:
                    pass
        scores = [r["score"] for r in results if r.get("score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            threshold = config.get("alert_threshold", 0.0)
            if avg < threshold:
                _fire_eval_alert(job, avg, threshold)
    except Exception:
        pass


def _fire_eval_alert(job: Dict[str, Any], score: float, threshold: float) -> None:
    """Fire webhook alert if eval score drops below threshold."""
    import urllib.request as _ureq

    webhook = job.get("eval_config", {}).get("webhook_url")
    if not webhook:
        return
    payload = {
        "graph_id": job.get("graph_id", ""),
        "run_id": job.get("run_id", ""),
        "score": score,
        "threshold": threshold,
        "input": str(job.get("input", ""))[:200],
    }
    try:
        _ureq.urlopen(  # nosec B310 — webhook URL is user-configured in eval_config
            _ureq.Request(
                webhook,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=5,
        )
    except Exception:
        pass


def _route_experiment(graph_id: str, run_id: str, experiments_dir: Optional[str] = None) -> str:
    """Return which graph variant to use for A/B experiment routing."""
    import hashlib

    base = experiments_dir or os.path.expanduser("~/.selectools/experiments.json")
    experiments: List[Dict[str, Any]] = []
    if os.path.isfile(base):
        try:
            experiments = json.loads(open(base).read())  # noqa: WPS515
        except Exception:
            pass
    exp = next(
        (
            e
            for e in experiments
            if e.get("active")
            and (e.get("variant_a") == graph_id or e.get("variant_b") == graph_id)
        ),
        None,
    )
    if not exp:
        return graph_id
    h = int(hashlib.md5(run_id.encode()).hexdigest(), 16)
    if (h % 1000) / 1000.0 < float(exp.get("split", 0.5)):
        return str(exp.get("variant_a", graph_id))
    return str(exp.get("variant_b", graph_id))


def _eval_worker() -> None:
    """Background thread that processes eval jobs from the queue."""
    while True:
        job = _eval_queue.get()
        if job is None:
            break
        try:
            _run_evals_on_run(job)
        except Exception:
            pass
        _eval_queue.task_done()


_eval_worker_thread = _threading.Thread(target=_eval_worker, daemon=True)
_eval_worker_thread.start()


# ─── Feature 20: Multi-provider smart routing ─────────────────────────────────
_provider_health: Dict[str, Dict[str, Any]] = {
    "openai": {"status": "unknown", "latency_ms": None, "last_check": 0, "error": None},
    "anthropic": {"status": "unknown", "latency_ms": None, "last_check": 0, "error": None},
    "gemini": {"status": "unknown", "latency_ms": None, "last_check": 0, "error": None},
    "ollama": {"status": "unknown", "latency_ms": None, "last_check": 0, "error": None},
}

CAPABILITY_TIERS: Dict[str, List[str]] = {
    "simple": ["gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash"],
    "standard": ["gpt-4o", "claude-sonnet-4-6", "gemini-2.5-pro"],
    "advanced": ["o3", "claude-opus-4-6", "gemini-2.5-pro"],
}


def _estimate_task_tier(prompt: str, system_prompt: str) -> str:
    """Heuristic task tier selection from keyword signals."""
    combined = (prompt + " " + system_prompt).lower()
    advanced_signals = ["analyze", "reason", "complex", "multi-step", "evaluate", "critique"]
    simple_signals = ["summarize", "extract", "classify", "yes or no", "format", "translate"]
    adv_count = sum(1 for s in advanced_signals if s in combined)
    simple_count = sum(1 for s in simple_signals if s in combined)
    if adv_count >= 2:
        return "advanced"
    if simple_count >= 2:
        return "simple"
    return "standard"


def _smart_route(
    prompt: str,
    system_prompt: str,
    available_providers: Optional[List[str]] = None,
    budget_usd: Optional[float] = None,
) -> str:
    """Select the cheapest capable model given health and optional budget."""
    tier = _estimate_task_tier(prompt, system_prompt)
    candidates = list(CAPABILITY_TIERS.get(tier, CAPABILITY_TIERS["standard"]))
    healthy = {name for name, h in _provider_health.items() if h["status"] == "ok"}
    if available_providers:
        healthy &= set(available_providers)

    def _provider_for_model(model_id: str) -> str:
        if "claude" in model_id:
            return "anthropic"
        if "gemini" in model_id:
            return "gemini"
        if "llama" in model_id or "mistral" in model_id:
            return "ollama"
        return "openai"

    def _model_cost(model_id: str) -> float:
        try:
            from ..models import ALL_MODELS

            m = next((x for x in ALL_MODELS if x.id == model_id), None)
            if m:
                return (m.prompt_cost or 0.0) + (m.completion_cost or 0.0)
        except Exception:
            pass
        return 0.0

    available = (
        [m for m in candidates if _provider_for_model(m) in healthy] if healthy else candidates
    )
    available.sort(key=_model_cost)
    if budget_usd is not None:
        est_tokens = len(prompt.split()) * 1.3 + 200
        available = [m for m in available if _model_cost(m) * est_tokens / 1000 <= budget_usd]
    return available[0] if available else candidates[0]


def _make_provider(model_id: str, api_key: str) -> Any:
    """Instantiate the correct provider for model_id."""
    if "claude" in model_id:
        from ..providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key)
    if "gemini" in model_id:
        from ..providers.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=api_key)
    if "llama" in model_id or "mistral" in model_id:
        from ..providers.ollama_provider import OllamaProvider

        return OllamaProvider()
    from ..providers.openai_provider import OpenAIProvider

    return OpenAIProvider(api_key=api_key)


def _run_eval_sample(
    model_id: str,
    system_prompt: str,
    cases: List[Dict[str, Any]],
    api_key: str,
) -> float:
    """Run up to 3 eval cases against model_id; return accuracy [0.0, 1.0]."""
    from ..types import Message, Role

    sample = cases[:3]
    if not sample:
        return 1.0
    passed = 0
    for case in sample:
        inp = str(case.get("input", ""))
        expected = str(case.get("expected_output") or case.get("expect_contains") or "")
        try:
            provider = _make_provider(model_id, api_key)
            msgs = [Message(role=Role.USER, content=inp)]
            resp, _ = provider.complete(
                messages=msgs, model=model_id, system_prompt=system_prompt, max_tokens=200
            )
            output = (resp.content or "").strip().lower()
            if not expected or expected.lower() in output:
                passed += 1
        except Exception:
            pass
    return round(passed / len(sample), 2)


def _eval_route(
    prompt: str,
    system_prompt: str,
    eval_cases: List[Dict[str, Any]],
    threshold: float = 0.7,
    budget_usd: Optional[float] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    """Return cheapest model in tier that passes eval_cases at threshold.

    Evaluates candidates cheapest-first (satisficing) to minimise latency.
    Falls back to heuristic _smart_route if no API key or no eval cases.
    """
    if not eval_cases or not api_key:
        return {
            "model": _smart_route(prompt, system_prompt, None, budget_usd),
            "scores": {},
            "method": "heuristic",
        }
    tier = _estimate_task_tier(prompt, system_prompt)
    candidates = list(CAPABILITY_TIERS.get(tier, CAPABILITY_TIERS["standard"]))
    healthy = {name for name, h in _provider_health.items() if h["status"] == "ok"}

    def _prov(mid: str) -> str:
        if "claude" in mid:
            return "anthropic"
        if "gemini" in mid:
            return "gemini"
        if "llama" in mid or "mistral" in mid:
            return "ollama"
        return "openai"

    def _cost(mid: str) -> float:
        try:
            from ..models import ALL_MODELS

            m = next((x for x in ALL_MODELS if x.id == mid), None)
            if m:
                return (m.prompt_cost or 0.0) + (m.completion_cost or 0.0)
        except Exception:
            pass
        return 0.0

    available = [m for m in candidates if _prov(m) in healthy] if healthy else list(candidates)
    available.sort(key=_cost)
    scores: Dict[str, float] = {}
    for model_id in available:
        if budget_usd is not None:
            if _cost(model_id) * (len(prompt.split()) * 1.3 + 200) / 1000 > budget_usd:
                continue
        score = _run_eval_sample(model_id, system_prompt, eval_cases, api_key)
        scores[model_id] = score
        if score >= threshold:
            return {
                "model": model_id,
                "scores": scores,
                "method": "eval-validated",
                "threshold": threshold,
                "tier": tier,
            }
    best = (
        max(scores, key=lambda m: scores[m])
        if scores
        else (available[0] if available else candidates[0])
    )
    return {
        "model": best,
        "scores": scores,
        "method": "best-available",
        "threshold": threshold,
        "tier": tier,
    }


def _estimate_run_cost(nodes_data: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """Estimate total tokens and cost for a graph run."""
    total_tokens = 0
    total_cost = 0.0
    try:
        from ..models import ALL_MODELS
        from ..token_estimation import estimate_tokens
    except Exception:
        return {"total_tokens": 0, "total_cost_usd": 0.0}
    for node in nodes_data:
        if node.get("type") != "agent":
            continue
        model_id = node.get("model", "gpt-4o-mini")
        sp = node.get("system_prompt", "")
        try:
            est_in = estimate_tokens(sp + input_text)
        except Exception:
            est_in = len((sp + input_text).split())
        est_out = min(int(node.get("max_tokens", 500)), 500)
        m = next((x for x in ALL_MODELS if x.id == model_id), None)
        if m:
            cost = (est_in * (m.prompt_cost or 0.0) + est_out * (m.completion_cost or 0.0)) / 1000
            total_cost += cost
            total_tokens += est_in + est_out
    return {"total_tokens": total_tokens, "total_cost_usd": round(total_cost, 6)}


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
        host: str = "0.0.0.0",
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
                elif path == "/provider-health":
                    self._json(_provider_health)
                elif path == "/eval-dashboard":
                    self._html(
                        "<html><body style='background:#0f172a;color:#e2e8f0;font-family:monospace;padding:24px'>"
                        "<h2 style='color:#22d3ee'>Eval Dashboard</h2>"
                        "<p>Production eval dashboard — coming soon.</p></body></html>"
                    )
                elif path == "/auth/github":
                    import urllib.parse as _uparse

                    gh_client_id = os.environ.get("GITHUB_CLIENT_ID", "")
                    gh_params = _uparse.urlencode(
                        {
                            "client_id": gh_client_id,
                            "redirect_uri": f"http://{self.headers.get('Host','localhost')}/auth/github/callback",
                            "scope": "read:user user:email",
                        }
                    )
                    self.send_response(302)
                    self.send_header(
                        "Location",
                        f"https://github.com/login/oauth/authorize?{gh_params}",
                    )
                    self.end_headers()
                elif path == "/auth/github/callback":
                    from urllib.parse import parse_qs as _pqs
                    from urllib.parse import urlparse as _up

                    qs = _pqs(_up(self.path).query)
                    code = qs.get("code", [""])[0]
                    import urllib.parse as _uparse2
                    import urllib.request as _ureq2

                    gh_client_id = os.environ.get("GITHUB_CLIENT_ID", "")
                    gh_client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
                    try:
                        data = _uparse2.urlencode(
                            {
                                "client_id": gh_client_id,
                                "client_secret": gh_client_secret,
                                "code": code,
                            }
                        ).encode()
                        req = _ureq2.Request(
                            "https://github.com/login/oauth/access_token",
                            data=data,
                            headers={"Accept": "application/json"},
                        )
                        resp_data = json.loads(_ureq2.urlopen(req, timeout=10).read())
                        access_token = resp_data.get("access_token", "")
                        req2 = _ureq2.Request(
                            "https://api.github.com/user",
                            headers={
                                "Authorization": f"token {access_token}",
                                "User-Agent": "selectools",
                            },
                        )
                        user_info = json.loads(_ureq2.urlopen(req2, timeout=10).read())
                        login = user_info.get("login", "unknown")
                        users = _resolve_users()
                        role = users.get(login, {}).get("role", "viewer")  # type: ignore[attr-defined]
                        session_val = hmac.new(
                            gh_client_secret.encode(), login.encode(), "sha256"
                        ).hexdigest()
                        self.send_response(302)
                        self.send_header(
                            "Set-Cookie",
                            f"builder_session={login}:{role}:{session_val}; HttpOnly; SameSite=Strict; Path=/",
                        )
                        self.send_header("Location", "/builder")
                        self.end_headers()
                    except Exception:
                        self._redirect_login()
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

                if path == "/ai-refine":
                    current_graph = body.get("current_graph", {})
                    selected_node_id = body.get("selected_node_id")
                    message = body.get("message", "").strip()
                    history = body.get("history", [])
                    api_key = body.get("api_key", "").strip()
                    if not message:
                        self._json({"error": "message required"}, 400)
                        return
                    if api_key:
                        result2 = _ai_refine_live(
                            current_graph, selected_node_id, message, history, api_key
                        )
                    else:
                        result2 = {
                            "patch": None,
                            "explanation": "No API key — provide an API key to use AI Copilot.",
                            "suggested_follow_up": "",
                        }
                    self._json(result2)
                    return

                if path == "/estimate-run-cost":
                    nodes_d = body.get("nodes", [])
                    input_t = body.get("input", "")
                    self._json(_estimate_run_cost(nodes_d, input_t))
                    return

                if path == "/smart-route":
                    prompt_t = body.get("prompt", "")
                    sys_t = body.get("system_prompt", "")
                    avail = body.get("available_providers") or None
                    budget = body.get("budget_usd") or None
                    self._json({"model": _smart_route(prompt_t, sys_t, avail, budget)})
                    return

                if path == "/eval-route":
                    self._json(
                        _eval_route(
                            prompt=body.get("prompt", ""),
                            system_prompt=body.get("system_prompt", ""),
                            eval_cases=body.get("eval_cases", []),
                            threshold=float(body.get("threshold", 0.7)),
                            budget_usd=body.get("budget_usd") or None,
                            api_key=body.get("api_key", ""),
                        )
                    )
                    return

                if path == "/watch-file":
                    watch_path = body.get("path", "")
                    if not os.path.isfile(watch_path):
                        self._json({"error": "File not found"}, 404)
                        return
                    import hashlib as _hl

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    last_hash = ""
                    for _ in range(600):
                        try:
                            content = open(watch_path).read()  # noqa: WPS515
                            h = _hl.md5(content.encode()).hexdigest()
                            if h != last_hash:
                                last_hash = h
                                ev_data = json.dumps({"type": "file_changed", "content": content})
                                self.wfile.write(f"data: {ev_data}\n\n".encode("utf-8"))
                                self.wfile.flush()
                        except Exception:
                            pass
                        time.sleep(1)
                    self.wfile.write(b'data: {"type": "timeout"}\n\n')
                    self.wfile.flush()
                    return

                if path == "/sync-to-file":
                    sync_path = body.get("path", "")
                    patch = body.get("patch", {})
                    if not os.path.isfile(sync_path):
                        self._json({"error": "File not found"}, 404)
                        return
                    try:
                        source = open(sync_path).read()  # noqa: WPS515
                        # Simple regex-based patch for update_node
                        if patch.get("type") == "update_node":
                            node_id = patch.get("node_id", "")
                            changes = patch.get("changes", {})
                            if "system_prompt" in changes:
                                import re as _re

                                source = _re.sub(
                                    rf'(add_node\(["\']){_re.escape(node_id)}',
                                    rf"\g<1>{node_id}",
                                    source,
                                )
                        with open(sync_path, "w") as fw:
                            fw.write(source)
                        self._json({"ok": True})
                    except Exception as ex:
                        self._json({"error": str(ex)}, 500)
                    return

                if path == "/runs":
                    log_dir = os.path.expanduser("~/.selectools/runs")
                    runs: List[Dict[str, Any]] = []
                    if os.path.isdir(log_dir):
                        for fname in sorted(os.listdir(log_dir), reverse=True)[:7]:
                            fpath = os.path.join(log_dir, fname)
                            try:
                                with open(fpath) as f:
                                    for line in f:
                                        line = line.strip()
                                        if line:
                                            runs.append(json.loads(line))
                            except Exception:
                                pass
                    self._json({"runs": runs[:100]})
                    return

                if path == "/feedback":
                    run_id = body.get("run_id", "")
                    score = body.get("score", 0)
                    _log_run({"run_id": run_id, "feedback": score, "ts": time.time()})
                    self._json({"ok": True})
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
                    pinned_ports_data = body.get("pinned_ports", {})
                    mock_mode = not api_key

                    emit({"type": "run_start", "mock": mock_mode})
                    if mock_mode:
                        _builder_run_mock(nodes_data, input_msg, emit, pinned_ports_data)
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
