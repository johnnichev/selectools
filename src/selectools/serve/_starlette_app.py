"""
Starlette ASGI application for selectools serve --builder.

All heavy business logic lives in app.py (module-level functions).
This module is a thin HTTP adapter — routing + request/response only.

Usage::

    uvicorn selectools.serve._starlette_app:create_builder_app --factory --port 8000

Or programmatically::

    from selectools.serve._starlette_app import create_builder_app
    app = create_builder_app(auth_token="secret")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, AsyncIterator, Dict, Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .. import __version__
from .app import (
    LOGIN_HTML,
    LOGIN_HTML_ERROR,
    _ai_build_fallback,
    _ai_build_live,
    _ai_refine_live,
    _builder_run_live,
    _builder_run_mock,
    _estimate_run_cost,
    _eval_route,
    _log_run,
    _make_session_cookie,
    _provider_health,
    _resolve_users,
    _smart_route,
)
from .builder import BUILDER_HTML

# ── auth helpers ─────────────────────────────────────────────────────────────


def _is_authed(request: Request, auth_token: Optional[str]) -> bool:
    if not auth_token:
        return True
    cookie = request.cookies.get("builder_session", "")
    expected = _make_session_cookie(auth_token)
    return hmac.compare_digest(cookie, expected)


def _login_redirect() -> Response:
    return Response(status_code=302, headers={"Location": "/login"})


def _builder_redirect() -> Response:
    return Response(status_code=302, headers={"Location": "/builder"})


# ── route factories ───────────────────────────────────────────────────────────


def _make_routes(auth_token: Optional[str]) -> list:  # type: ignore[type-arg]
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "mode": "builder", "version": __version__})

    async def login_get(_: Request) -> HTMLResponse:
        return HTMLResponse(LOGIN_HTML)

    async def login_post(request: Request) -> Response:
        form = await request.form()
        token = str(form.get("token", ""))
        if auth_token and hmac.compare_digest(token, auth_token):
            cookie_val = _make_session_cookie(auth_token)
            resp = _builder_redirect()
            resp.set_cookie(
                "builder_session", cookie_val, httponly=True, samesite="strict", path="/"
            )
            return resp
        return HTMLResponse(LOGIN_HTML_ERROR)

    async def builder(request: Request) -> Response:
        if not _is_authed(request, auth_token):
            return _login_redirect()
        return HTMLResponse(BUILDER_HTML)

    async def provider_health(request: Request) -> Response:
        if not _is_authed(request, auth_token):
            return _login_redirect()
        return JSONResponse(_provider_health)

    async def eval_dashboard(request: Request) -> Response:
        if not _is_authed(request, auth_token):
            return _login_redirect()
        return HTMLResponse(
            "<html><body style='background:#0f172a;color:#e2e8f0;"
            "font-family:monospace;padding:24px'>"
            "<h2 style='color:#22d3ee'>Eval Dashboard</h2>"
            "<p>Production eval dashboard — coming soon.</p></body></html>"
        )

    async def auth_github(request: Request) -> Response:
        import urllib.parse as _up

        gh_client_id = os.environ.get("GITHUB_CLIENT_ID", "")
        params = _up.urlencode(
            {
                "client_id": gh_client_id,
                "redirect_uri": f"http://{request.headers.get('host', 'localhost')}/auth/github/callback",
                "scope": "read:user user:email",
            }
        )
        return Response(
            status_code=302,
            headers={"Location": f"https://github.com/login/oauth/authorize?{params}"},
        )

    async def auth_github_callback(request: Request) -> Response:
        import urllib.parse as _up
        import urllib.request as _ur

        code = request.query_params.get("code", "")
        gh_client_id = os.environ.get("GITHUB_CLIENT_ID", "")
        gh_client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
        try:
            data = _up.urlencode(
                {
                    "client_id": gh_client_id,
                    "client_secret": gh_client_secret,
                    "code": code,
                }
            ).encode()
            req = _ur.Request(
                "https://github.com/login/oauth/access_token",
                data=data,
                headers={"Accept": "application/json"},
            )
            token_data = json.loads(_ur.urlopen(req, timeout=10).read())
            access_token = token_data.get("access_token", "")
            req2 = _ur.Request(
                "https://api.github.com/user",
                headers={"Authorization": f"token {access_token}", "User-Agent": "selectools"},
            )
            user_info = json.loads(_ur.urlopen(req2, timeout=10).read())
            login = user_info.get("login", "unknown")
            users = _resolve_users()
            role = users.get(login, {}).get("role", "viewer")  # type: ignore[attr-defined]
            session_val = hmac.new(gh_client_secret.encode(), login.encode(), "sha256").hexdigest()
            resp = _builder_redirect()
            resp.set_cookie(
                "builder_session",
                f"{login}:{role}:{session_val}",
                httponly=True,
                samesite="strict",
                path="/",
            )
            return resp
        except Exception:
            return _login_redirect()

    async def ai_build(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        description = body.get("description", "").strip()
        api_key = body.get("api_key", "").strip()
        if not description:
            return JSONResponse({"error": "description required"}, status_code=400)
        result = (
            _ai_build_live(description, api_key) if api_key else _ai_build_fallback(description)
        )
        return JSONResponse(result)

    async def ai_refine(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "message required"}, status_code=400)
        api_key = body.get("api_key", "").strip()
        if api_key:
            result = _ai_refine_live(
                body.get("current_graph", {}),
                body.get("selected_node_id"),
                message,
                body.get("history", []),
                api_key,
            )
        else:
            result = {
                "patch": None,
                "explanation": "No API key — provide an API key to use AI Copilot.",
                "suggested_follow_up": "",
            }
        return JSONResponse(result)

    async def estimate_run_cost(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        return JSONResponse(_estimate_run_cost(body.get("nodes", []), body.get("input", "")))

    async def smart_route(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        return JSONResponse(
            {
                "model": _smart_route(
                    body.get("prompt", ""),
                    body.get("system_prompt", ""),
                    body.get("available_providers") or None,
                    body.get("budget_usd") or None,
                )
            }
        )

    async def eval_route(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        return JSONResponse(
            _eval_route(
                prompt=body.get("prompt", ""),
                system_prompt=body.get("system_prompt", ""),
                eval_cases=body.get("eval_cases", []),
                threshold=float(body.get("threshold", 0.7)),
                budget_usd=body.get("budget_usd") or None,
                api_key=body.get("api_key", ""),
            )
        )

    async def watch_file(request: Request) -> Response:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        watch_path = body.get("path", "")
        if not os.path.isfile(watch_path):
            return JSONResponse({"error": "File not found"}, status_code=404)

        async def _stream() -> AsyncIterator[bytes]:
            last_hash = ""
            for _ in range(600):
                try:
                    content = open(watch_path).read()  # noqa: WPS515
                    h = hashlib.md5(content.encode()).hexdigest()
                    if h != last_hash:
                        last_hash = h
                        ev = json.dumps({"type": "file_changed", "content": content})
                        yield f"data: {ev}\n\n".encode()
                except Exception:
                    pass
                import asyncio

                await asyncio.sleep(1)
            yield b'data: {"type": "timeout"}\n\n'

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def sync_to_file(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        sync_path = body.get("path", "")
        if not os.path.isfile(sync_path):
            return JSONResponse({"error": "File not found"}, status_code=404)
        try:
            source = open(sync_path).read()  # noqa: WPS515
            patch = body.get("patch", {})
            if patch.get("type") == "update_node":
                import re as _re

                node_id = patch.get("node_id", "")
                changes = patch.get("changes", {})
                if "system_prompt" in changes:
                    source = _re.sub(
                        rf'(add_node\(["\']){_re.escape(node_id)}',
                        rf"\g<1>{node_id}",
                        source,
                    )
            with open(sync_path, "w") as fw:
                fw.write(source)
            return JSONResponse({"ok": True})
        except Exception as ex:
            return JSONResponse({"error": str(ex)}, status_code=500)

    async def runs(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        log_dir = os.path.expanduser("~/.selectools/runs")
        run_list: list = []  # type: ignore[type-arg]
        if os.path.isdir(log_dir):
            for fname in sorted(os.listdir(log_dir), reverse=True)[:7]:
                fpath = os.path.join(log_dir, fname)
                try:
                    with open(fpath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                run_list.append(json.loads(line))
                except Exception:
                    pass
        return JSONResponse({"runs": run_list[:100]})

    async def feedback(request: Request) -> JSONResponse:
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        _log_run(
            {"run_id": body.get("run_id", ""), "feedback": body.get("score", 0), "ts": time.time()}
        )
        return JSONResponse({"ok": True})

    async def run_sse(request: Request) -> Response:
        """POST /run — SSE stream of graph execution events."""
        if not _is_authed(request, auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        input_msg = body.get("input", "Hello")
        nodes_data = body.get("nodes", [])
        edges_data = body.get("edges", [])
        api_key = body.get("api_key", "").strip()
        pinned_ports_data = body.get("pinned_ports", {})
        mock_mode = not api_key

        events: list = []  # type: ignore[type-arg]

        def collect(event: Dict[str, Any]) -> None:
            events.append(event)

        if mock_mode:
            _builder_run_mock(nodes_data, input_msg, collect, pinned_ports_data)
        else:
            _builder_run_live(nodes_data, edges_data, input_msg, api_key, collect)

        async def _stream() -> AsyncIterator[bytes]:
            yield f"data: {json.dumps({'type': 'run_start', 'mock': mock_mode})}\n\n".encode()
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

    return [
        Route("/health", health),
        Route("/login", login_get, methods=["GET"]),
        Route("/login", login_post, methods=["POST"]),
        Route("/builder", builder),
        Route("/", builder),
        Route("/provider-health", provider_health),
        Route("/eval-dashboard", eval_dashboard),
        Route("/auth/github", auth_github),
        Route("/auth/github/callback", auth_github_callback),
        Route("/ai-build", ai_build, methods=["POST"]),
        Route("/ai-refine", ai_refine, methods=["POST"]),
        Route("/estimate-run-cost", estimate_run_cost, methods=["POST"]),
        Route("/smart-route", smart_route, methods=["POST"]),
        Route("/eval-route", eval_route, methods=["POST"]),
        Route("/watch-file", watch_file, methods=["POST"]),
        Route("/sync-to-file", sync_to_file, methods=["POST"]),
        Route("/runs", runs, methods=["POST"]),
        Route("/feedback", feedback, methods=["POST"]),
        Route("/run", run_sse, methods=["POST"]),
    ]


# ── app factory ───────────────────────────────────────────────────────────────


def create_builder_app(auth_token: Optional[str] = None) -> Starlette:
    """Return a Starlette ASGI app for the visual builder.

    Args:
        auth_token: Optional bearer token to protect the builder.
                    When set, users must authenticate at /login before
                    accessing any other route.

    Example::

        import uvicorn
        from selectools.serve._starlette_app import create_builder_app
        uvicorn.run(create_builder_app(), host="0.0.0.0", port=8000)
    """
    app = Starlette(routes=_make_routes(auth_token))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    return app
