"""
Agent-as-API: auto-generated production REST endpoints for any Agent.

One line turns an :class:`~selectools.agent.core.Agent` (or several) into a
production-ready Starlette ASGI app with standardized JSON schemas, session
persistence, per-user isolation, and optional bearer-token auth.

Usage::

    from selectools.serve import AgentAPI

    app = AgentAPI(agents=[my_agent, my_other_agent], auth_key="sk-...")
    # Run with: uvicorn app:app --port 8000

Or via CLI::

    selectools serve agent.yaml --api --port 8000

Endpoints:
    POST   /v1/chat           — single-turn completion (JSON)
    POST   /v1/chat/stream    — streaming completion (SSE)
    POST   /v1/sessions       — create session
    GET    /v1/sessions/{id}  — get session history
    DELETE /v1/sessions/{id}  — delete session
    GET    /v1/health         — health check (never requires auth)

Request schema (chat)::

    {"input": "...", "session_id": "optional", "agent": "optional name"}

Response schema (chat)::

    {"output": "...", "session_id": "...", "agent": "...", "usage": {...}}

Errors::

    {"error": {"message": "...", "type": "unauthorized|not_found|validation_error"}}
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .. import __version__
from ..memory import ConversationMemory
from ..sessions import SessionMetadata, SessionSearchResult, SessionStore, _make_key
from ..stability import beta
from ..types import AgentResult, Message, Role, StreamChunk

if TYPE_CHECKING:
    from ..agent.core import Agent

__all__ = ["AgentAPI"]

logger = logging.getLogger(__name__)


# ─── In-memory session store (default backend) ───────────────────────────────


class _InMemorySessionStore:
    """Dict-backed SessionStore used when no ``session_store`` is provided.

    Implements the :class:`~selectools.sessions.SessionStore` protocol.
    Snapshots memory via ``to_dict()``/``from_dict()`` so callers cannot
    mutate stored sessions through aliased references.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        key = _make_key(session_id, namespace)
        now = time.time()
        with self._lock:
            existing = self._data.get(key)
            created_at = existing["created_at"] if existing else now
            self._data[key] = {
                "session_id": session_id,
                "memory": memory.to_dict(),
                "created_at": created_at,
                "updated_at": now,
            }

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        key = _make_key(session_id, namespace)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            return ConversationMemory.from_dict(entry["memory"])

    def list(self) -> List[SessionMetadata]:
        with self._lock:
            return [
                SessionMetadata(
                    session_id=entry["session_id"],
                    message_count=entry["memory"].get("message_count", 0),
                    created_at=entry["created_at"],
                    updated_at=entry["updated_at"],
                )
                for entry in self._data.values()
            ]

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        key = _make_key(session_id, namespace)
        with self._lock:
            return self._data.pop(key, None) is not None

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        key = _make_key(session_id, namespace)
        with self._lock:
            return key in self._data

    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Search is not supported by the default in-memory store.

        Pass a persistent ``SessionStore`` backend (SQLite, JSON file,
        Redis, Supabase) to ``AgentAPI(session_store=...)`` for
        cross-session search.
        """
        raise NotImplementedError(
            "_InMemorySessionStore does not support search(); "
            "use a persistent SessionStore backend (e.g. SQLiteSessionStore)."
        )

    def branch(self, source_id: str, new_id: str) -> None:
        memory = self.load(source_id)
        if memory is None:
            raise ValueError(f"Session {source_id!r} not found")
        self.save(new_id, memory)


# ─── Response helpers ─────────────────────────────────────────────────────────


def _error(message: str, error_type: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": error_type}}, status_code=status)


def _usage_dict(result: Optional[AgentResult]) -> Dict[str, Any]:
    usage = getattr(result, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "total_prompt_tokens", 0),
        "completion_tokens": getattr(usage, "total_completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "cost_usd": getattr(usage, "total_cost_usd", 0.0),
    }


def _serialize_message(msg: Message) -> Dict[str, Any]:
    role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
    return {"role": role, "content": msg.content or ""}


async def _read_json(request: Request) -> Optional[Dict[str, Any]]:
    """Parse the request body as a JSON object, or return ``None`` if invalid."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return body


# ─── AgentAPI ─────────────────────────────────────────────────────────────────


@beta
class AgentAPI:
    """Auto-generated production REST API for one or more agents.

    The instance is itself a Starlette ASGI application — pass it directly
    to uvicorn, hypercorn, or any ASGI server.

    Args:
        agents: A single :class:`Agent` or a list of agents. Each agent is
            addressable by its ``config.name``; the first agent is the
            default when a chat request omits the ``"agent"`` field.
        auth_key: Optional API key. When set, every route except
            ``/v1/health`` requires ``Authorization: Bearer <auth_key>``.
        session_store: Optional :class:`~selectools.sessions.SessionStore`
            backend (JSON file, SQLite, Redis, Supabase). Defaults to an
            in-memory store (sessions are lost on restart).
        cors: Add a permissive CORS middleware (default ``True``).

    Per-user isolation:
        Clients may send a ``user_id`` header (or ``x-user-id``). Sessions
        are namespaced per user — a user can never read, write, or delete
        another user's sessions. Requests without the header share the
        ``"default"`` user namespace.

    Example::

        from selectools.serve import AgentAPI

        app = AgentAPI(agents=[support_agent, billing_agent], auth_key="sk-...")
        # uvicorn app:app --port 8000
    """

    def __init__(
        self,
        agents: Union["Agent", List["Agent"]],
        auth_key: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        cors: bool = True,
    ) -> None:
        agent_list = agents if isinstance(agents, list) else [agents]
        if not agent_list:
            raise ValueError("AgentAPI requires at least one agent")
        self._agents: Dict[str, "Agent"] = {}
        for agent in agent_list:
            name = agent.config.name
            if name in self._agents:
                raise ValueError(
                    f"Duplicate agent name {name!r}: give each agent a unique "
                    f"AgentConfig(name=...) to route between them"
                )
            self._agents[name] = agent
        self._default_name = agent_list[0].config.name
        self._auth_key = auth_key
        self._store: SessionStore = session_store or _InMemorySessionStore()
        self._app = Starlette(routes=self._build_routes())
        if cors:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization", "user_id", "x-user-id"],
            )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI entry point — delegates to the internal Starlette app."""
        await self._app(scope, receive, send)

    @property
    def agents(self) -> Dict[str, "Agent"]:
        """Read-only view of the registered agents by name."""
        return dict(self._agents)

    # ── request helpers ──────────────────────────────────────────────────────

    def _check_auth(self, request: Request) -> Optional[JSONResponse]:
        """Return a 401 response when bearer auth fails, else ``None``."""
        if not self._auth_key:
            return None
        header = request.headers.get("authorization", "")
        expected = f"Bearer {self._auth_key}"
        if hmac.compare_digest(header, expected):
            return None
        return _error("Missing or invalid Authorization bearer token", "unauthorized", 401)

    @staticmethod
    def _namespace(request: Request) -> str:
        """Derive the session namespace from the user_id header."""
        user_id = request.headers.get("user_id") or request.headers.get("x-user-id") or "default"
        return f"user:{user_id}"

    def _resolve_agent(
        self, body: Dict[str, Any]
    ) -> Tuple[Optional["Agent"], Optional[str], Optional[JSONResponse]]:
        """Resolve the target agent from the request body.

        Returns ``(agent, name, None)`` on success or ``(None, None, error)``.
        """
        name = body.get("agent")
        if name is None:
            name = self._default_name
        elif not isinstance(name, str):
            return None, None, _error("'agent' must be a string", "validation_error", 422)
        agent = self._agents.get(name)
        if agent is None:
            available = ", ".join(sorted(self._agents))
            return (
                None,
                None,
                _error(f"Unknown agent {name!r}. Available agents: {available}", "not_found", 404),
            )
        return agent, name, None

    def _resolve_session(
        self, body: Dict[str, Any], namespace: str
    ) -> Tuple[Optional[str], Optional[ConversationMemory], Optional[JSONResponse]]:
        """Load (or create) the session for a chat request.

        Returns ``(session_id, memory, None)`` on success or an error response.
        """
        session_id = body.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                return (
                    None,
                    None,
                    _error("'session_id' must be a non-empty string", "validation_error", 422),
                )
            memory = self._store.load(session_id, namespace=namespace)
            if memory is None:
                return None, None, _error(f"Session {session_id!r} not found", "not_found", 404)
            return session_id, memory, None
        return uuid.uuid4().hex, ConversationMemory(), None

    def _validate_chat_body(
        self, body: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[JSONResponse]]:
        """Validate the chat body, returning ``(input_text, None)`` or an error."""
        if body is None:
            return None, _error("Request body must be a JSON object", "validation_error", 422)
        input_text = body.get("input")
        if not isinstance(input_text, str) or not input_text.strip():
            return None, _error(
                "'input' is required and must be a non-empty string", "validation_error", 422
            )
        return input_text, None

    def _persist_turn(
        self,
        session_id: str,
        namespace: str,
        memory: ConversationMemory,
        user_msg: Message,
        assistant_msg: Message,
    ) -> None:
        memory.add(user_msg)
        memory.add(assistant_msg)
        self._store.save(session_id, memory, namespace=namespace)

    # ── endpoint handlers ────────────────────────────────────────────────────

    async def _health(self, _: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__, "agents": list(self._agents)})

    async def _chat(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        body = await _read_json(request)
        input_text, invalid = self._validate_chat_body(body)
        if invalid:
            return invalid
        assert body is not None and input_text is not None  # narrowed above  # nosec B101
        agent, agent_name, agent_err = self._resolve_agent(body)
        if agent_err:
            return agent_err
        assert agent is not None  # nosec B101
        namespace = self._namespace(request)
        session_id, memory, session_err = self._resolve_session(body, namespace)
        if session_err:
            return session_err
        assert session_id is not None and memory is not None  # nosec B101

        user_msg = Message(role=Role.USER, content=input_text)
        messages = memory.get_history() + [user_msg]
        # Run on an isolated clone: Agent.run appends every caller's turn to
        # the shared _history, cross-contaminating subsequent requests. The
        # session history is passed in explicitly, so the clone (fresh
        # history/usage, memory dropped) keeps behavior identical minus the
        # leak. _clone_for_isolation is underscore-private today; promoting
        # it to a public API is a follow-up.
        runner = agent._clone_for_isolation()
        try:
            result: AgentResult = await run_in_threadpool(runner.run, messages)
        except Exception as exc:
            # Only the exception type reaches the client — exception messages
            # can contain internal URLs/paths. Full detail goes to the log.
            logger.warning(
                "Agent %r run failed: %s: %s", agent_name, type(exc).__name__, exc, exc_info=True
            )
            return _error(f"Agent execution failed ({type(exc).__name__})", type(exc).__name__, 500)

        output = result.content or ""
        self._persist_turn(
            session_id,
            namespace,
            memory,
            user_msg,
            Message(role=Role.ASSISTANT, content=output),
        )
        return JSONResponse(
            {
                "output": output,
                "session_id": session_id,
                "agent": agent_name,
                "usage": _usage_dict(result),
            }
        )

    async def _chat_stream(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        body = await _read_json(request)
        input_text, invalid = self._validate_chat_body(body)
        if invalid:
            return invalid
        assert body is not None and input_text is not None  # nosec B101
        agent, agent_name, agent_err = self._resolve_agent(body)
        if agent_err:
            return agent_err
        assert agent is not None  # nosec B101
        namespace = self._namespace(request)
        session_id, memory, session_err = self._resolve_session(body, namespace)
        if session_err:
            return session_err
        assert session_id is not None and memory is not None  # nosec B101

        user_msg = Message(role=Role.USER, content=input_text)
        messages = memory.get_history() + [user_msg]
        # Same per-request isolation as _chat: never run on the shared agent.
        runner = agent._clone_for_isolation()

        async def _events() -> AsyncIterator[bytes]:
            chunks: List[str] = []
            final: Optional[AgentResult] = None
            try:
                if getattr(runner.provider, "supports_async", True):
                    async for item in runner.astream(messages):
                        if isinstance(item, AgentResult):
                            final = item
                        elif isinstance(item, StreamChunk) and item.content:
                            chunks.append(item.content)
                            payload = {"type": "chunk", "content": item.content}
                            yield f"data: {json.dumps(payload)}\n\n".encode()
                else:
                    # Sync-only provider: run in a thread and emit one chunk.
                    final = await run_in_threadpool(runner.run, messages)
                    if final.content:
                        chunks.append(final.content)
                        payload = {"type": "chunk", "content": final.content}
                        yield f"data: {json.dumps(payload)}\n\n".encode()
            except Exception as exc:
                logger.warning(
                    "Agent %r stream failed: %s: %s",
                    agent_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                err_payload = {
                    "type": "error",
                    "error": {
                        "message": f"Agent execution failed ({type(exc).__name__})",
                        "type": type(exc).__name__,
                    },
                }
                yield f"data: {json.dumps(err_payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            output = (final.content or "") if final is not None else "".join(chunks)
            self._persist_turn(
                session_id,
                namespace,
                memory,
                user_msg,
                Message(role=Role.ASSISTANT, content=output),
            )
            result_payload = {
                "type": "result",
                "output": output,
                "session_id": session_id,
                "agent": agent_name,
                "usage": _usage_dict(final),
            }
            yield f"data: {json.dumps(result_payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _session_create(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        namespace = self._namespace(request)
        body = await _read_json(request) or {}
        session_id = body.get("session_id")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            return _error("'session_id' must be a non-empty string", "validation_error", 422)
        if session_id is None:
            session_id = uuid.uuid4().hex
        elif self._store.exists(session_id, namespace=namespace):
            return _error(f"Session {session_id!r} already exists", "conflict", 409)
        self._store.save(session_id, ConversationMemory(), namespace=namespace)
        return JSONResponse({"session_id": session_id}, status_code=201)

    async def _session_get(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        session_id = request.path_params["session_id"]
        namespace = self._namespace(request)
        memory = self._store.load(session_id, namespace=namespace)
        if memory is None:
            return _error(f"Session {session_id!r} not found", "not_found", 404)
        return JSONResponse(
            {
                "session_id": session_id,
                "messages": [_serialize_message(m) for m in memory.get_history()],
            }
        )

    async def _session_delete(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        session_id = request.path_params["session_id"]
        namespace = self._namespace(request)
        if not self._store.delete(session_id, namespace=namespace):
            return _error(f"Session {session_id!r} not found", "not_found", 404)
        return JSONResponse({"deleted": True, "session_id": session_id})

    # ── routing ──────────────────────────────────────────────────────────────

    def _build_routes(self) -> List[Route]:
        return [
            Route("/v1/health", self._health, methods=["GET"]),
            Route("/v1/chat", self._chat, methods=["POST"]),
            Route("/v1/chat/stream", self._chat_stream, methods=["POST"]),
            Route("/v1/sessions", self._session_create, methods=["POST"]),
            Route("/v1/sessions/{session_id}", self._session_get, methods=["GET"]),
            Route("/v1/sessions/{session_id}", self._session_delete, methods=["DELETE"]),
        ]
