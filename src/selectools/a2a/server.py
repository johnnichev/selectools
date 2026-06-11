"""
A2A server — expose a selectools Agent to other agents over the A2A protocol.

Two routes on a Starlette ASGI app:

    GET  /.well-known/agent.json  — Agent Card (never requires auth)
    POST /a2a                     — JSON-RPC 2.0 message handler

Usage::

    from selectools.serve import A2AServer

    server = A2AServer(agent=my_agent, auth_token="sk-...")
    server.serve(port=8000)            # blocking, runs uvicorn
    # or: uvicorn module:server --port 8000  (the instance is an ASGI app)

Task lifecycle: ``submitted`` → ``working`` → ``input-required`` →
``completed`` / ``failed`` / ``canceled``. This synchronous v1 runs the
agent inside the request, so a task goes submitted→working→completed (or
failed) before the response is sent; the state field is modeled anyway so
async backends can slot in later without changing the wire format.

v1 is text-first: ``file`` and ``data`` parts are accepted (data parts are
appended to the prompt as JSON), but binary file content is not forwarded
to the agent.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .. import __version__
from ..stability import beta
from ..types import Message, Role
from .types import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    TaskState,
)

if TYPE_CHECKING:
    from ..agent.core import Agent

__stability__ = "beta"

__all__ = ["A2AServer"]

logger = logging.getLogger(__name__)

_RequestId = Union[str, int, None]

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rpc_result(request_id: _RequestId, result: Any) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _rpc_error(
    request_id: _RequestId, code: int, message: str, data: Optional[Any] = None
) -> JSONResponse:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": error})


def _extract_prompt(parts: List[Dict[str, Any]]) -> str:
    """Build the agent prompt from message parts (text-first v1).

    Text parts are concatenated; ``data`` parts are appended as JSON blocks;
    ``file`` parts are accepted but their content is not forwarded.
    """
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            # Defensive: validation rejects non-dict parts up front, but a
            # stray list element must never crash prompt assembly.
            continue
        kind = part.get("kind")
        if kind == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
        elif kind == "data" and part.get("data") is not None:
            chunks.append(json.dumps(part["data"], ensure_ascii=False, default=str))
    return "\n".join(chunks)


@beta
class A2AServer:
    """Serve a selectools Agent over the A2A protocol (Agent Card + JSON-RPC).

    The instance is itself a Starlette ASGI application — pass it directly
    to uvicorn, hypercorn, or any ASGI server, or call :meth:`serve`.

    Supported JSON-RPC methods: ``message/send`` (and the legacy
    ``tasks/send`` alias), ``tasks/get``, ``tasks/cancel``.

    Trust model (v1):
        - **Single shared tenant.** One bearer token grants full access:
          any authenticated caller can read every stored task (including
          other callers' inputs and outputs) via ``tasks/get``. Do not
          share one server between mutually untrusting parties.
        - **Do not run unauthenticated on a public interface.** Without
          ``auth_token`` anyone who can reach the port can run the agent
          and read all tasks. :meth:`serve` logs a warning in that case.
        - **Per-request agent isolation.** Each ``message/send`` runs on
          an isolated clone of the agent (fresh history and usage, no
          memory), so callers never see each other's conversation context.
          A2A v1 has no session model, so agent memory is intentionally
          dropped: every request starts from a clean slate.

    Args:
        agent: The :class:`Agent` to expose. Its ``config.name`` and tools
            populate the Agent Card.
        auth_token: Optional bearer token. When set, ``POST /a2a`` requires
            ``Authorization: Bearer <auth_token>``. The Agent Card route
            never requires auth (discovery must stay public).
        name: Agent Card name override (defaults to ``agent.config.name``).
        description: Agent Card description. Defaults to a generic string —
            the system prompt is intentionally NOT used, since the card is
            served unauthenticated.
        url: Public URL of the A2A endpoint advertised in the Agent Card.
        version: Agent version advertised in the card (defaults to the
            selectools package version).
        max_tasks: Maximum number of completed tasks kept in the in-memory
            store (default 2000). Oldest tasks are evicted first (FIFO);
            an evicted task id returns ``-32001`` from ``tasks/get``.
        max_body_bytes: Maximum accepted ``POST /a2a`` body size in bytes
            (default 1 MiB). Larger bodies are rejected with JSON-RPC
            ``-32600`` before parsing.

    Example::

        from selectools.serve import A2AServer

        server = A2AServer(agent=my_agent, auth_token="sk-...")
        server.serve(port=8000)
    """

    def __init__(
        self,
        agent: "Agent",
        auth_token: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        url: str = "",
        version: Optional[str] = None,
        max_tasks: int = 2000,
        max_body_bytes: int = 1_048_576,
    ) -> None:
        self._agent = agent
        self._auth_token = auth_token
        self._name = name or agent.config.name
        self._description = description or f"selectools agent {self._name!r} served over A2A"
        self._url = url
        self._version = version or __version__
        self._max_tasks = max_tasks
        self._max_body_bytes = max_body_bytes
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._tasks_lock = threading.Lock()
        self._app = Starlette(
            routes=[
                Route("/.well-known/agent.json", self._agent_card, methods=["GET"]),
                Route("/a2a", self._handle_rpc, methods=["POST"]),
            ]
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI entry point — delegates to the internal Starlette app."""
        await self._app(scope, receive, send)

    def serve(self, port: int = 8000, host: str = "0.0.0.0") -> None:  # nosec B104
        """Run the server with uvicorn (blocking)."""
        if self._auth_token is None and host not in _LOOPBACK_HOSTS:
            logger.warning(
                "A2AServer is starting unauthenticated on non-loopback host %r: "
                "anyone who can reach this port can run the agent and read all "
                "stored tasks. Pass auth_token=... to require a bearer token.",
                host,
            )
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "A2AServer.serve() requires uvicorn. Install it with: pip install selectools[serve]"
            ) from exc
        uvicorn.run(self, host=host, port=port)

    # ── Agent Card ───────────────────────────────────────────────────────────

    def build_agent_card(self) -> Dict[str, Any]:
        """Build the Agent Card payload from the wrapped agent's metadata."""
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "name": self._name,
            "description": self._description,
            "url": self._url,
            "version": self._version,
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [
                {
                    "id": tool.name,
                    "name": tool.name,
                    "description": tool.description,
                    "tags": ["tool"],
                }
                for tool in self._agent.tools
            ],
        }

    async def _agent_card(self, _: Request) -> JSONResponse:
        return JSONResponse(self.build_agent_card())

    # ── JSON-RPC handler ─────────────────────────────────────────────────────

    def _check_auth(self, request: Request) -> Optional[Response]:
        if not self._auth_token:
            return None
        header = request.headers.get("authorization", "")
        expected = f"Bearer {self._auth_token}"
        if hmac.compare_digest(header, expected):
            return None
        return JSONResponse(
            {"error": {"message": "Missing or invalid Authorization bearer token"}},
            status_code=401,
        )

    async def _handle_rpc(self, request: Request) -> Response:
        denied = self._check_auth(request)
        if denied:
            return denied
        raw = await request.body()
        if len(raw) > self._max_body_bytes:
            return _rpc_error(
                None,
                INVALID_REQUEST,
                f"Invalid request: body exceeds {self._max_body_bytes} bytes",
            )
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _rpc_error(None, PARSE_ERROR, "Parse error: request body is not valid JSON")
        if not isinstance(body, dict):
            return _rpc_error(None, INVALID_REQUEST, "Invalid request: body must be a JSON object")
        request_id: _RequestId = body.get("id")
        if body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            return _rpc_error(
                request_id,
                INVALID_REQUEST,
                "Invalid request: 'jsonrpc' must be '2.0' and 'method' must be a string",
            )
        method = body["method"]
        params = body.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _rpc_error(
                request_id, INVALID_REQUEST, "Invalid request: 'params' must be an object"
            )

        if method in ("message/send", "tasks/send"):
            return await self._message_send(request_id, params)
        if method == "tasks/get":
            return self._tasks_get(request_id, params)
        if method == "tasks/cancel":
            return self._tasks_cancel(request_id, params)
        return _rpc_error(request_id, METHOD_NOT_FOUND, f"Method not found: {method!r}")

    # ── methods ──────────────────────────────────────────────────────────────

    def _validate_message(
        self, params: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        message = params.get("message")
        if not isinstance(message, dict):
            return None, "'message' is required and must be an object"
        parts = message.get("parts")
        if not isinstance(parts, list) or not parts:
            return None, "'message.parts' must be a non-empty array"
        if not all(isinstance(p, dict) for p in parts):
            return None, "every entry in 'message.parts' must be an object"
        if not any(
            isinstance(p, dict) and p.get("kind") == "text" and isinstance(p.get("text"), str)
            for p in parts
        ):
            return None, "at least one part of kind 'text' is required (v1 is text-first)"
        return message, None

    async def _message_send(self, request_id: _RequestId, params: Dict[str, Any]) -> JSONResponse:
        message, invalid = self._validate_message(params)
        if invalid:
            return _rpc_error(request_id, INVALID_PARAMS, f"Invalid params: {invalid}")
        assert message is not None  # narrowed above  # nosec B101
        prompt = _extract_prompt(message["parts"])
        task_id = uuid.uuid4().hex
        context_id = message.get("contextId") or params.get("contextId") or uuid.uuid4().hex

        # Synchronous v1: submitted → working → completed/failed within
        # this request. States are recorded so the wire format already
        # matches what an async backend would emit.
        task: Dict[str, Any] = {
            "id": task_id,
            "contextId": context_id,
            "kind": "task",
            "status": {"state": TaskState.WORKING, "timestamp": _now_iso()},
            "history": [message],
            "artifacts": [],
        }
        try:
            # Run on an isolated clone: Agent.run mutates shared _history (and
            # memory), which would leak caller A's conversation into caller B's
            # provider context and race under concurrency. clone_for_isolation
            # (same mechanism run_batch uses) shares tools/provider/config but
            # gives fresh history/usage and drops memory — correct for A2A v1,
            # which has no session model.
            clone = self._agent.clone_for_isolation()
            result = await run_in_threadpool(clone.run, [Message(role=Role.USER, content=prompt)])
        except Exception as exc:
            # Agent failure is a task-level outcome, not a transport error.
            # Only the exception type goes to the remote caller — exception
            # messages can contain internal URLs/paths. Full detail is logged.
            logger.warning(
                "A2A task %s failed: %s: %s", task_id, type(exc).__name__, exc, exc_info=True
            )
            task["status"] = {
                "state": TaskState.FAILED,
                "timestamp": _now_iso(),
                "message": {
                    "role": "agent",
                    "parts": [
                        {"kind": "text", "text": f"Agent execution failed ({type(exc).__name__})"}
                    ],
                    "messageId": uuid.uuid4().hex,
                    "kind": "message",
                },
            }
            self._store_task(task)
            return _rpc_result(request_id, task)

        output = result.content or ""
        task["status"] = {"state": TaskState.COMPLETED, "timestamp": _now_iso()}
        task["artifacts"] = [
            {
                "artifactId": uuid.uuid4().hex,
                "name": "response",
                "parts": [{"kind": "text", "text": output}],
            }
        ]
        self._store_task(task)
        return _rpc_result(request_id, task)

    def _store_task(self, task: Dict[str, Any]) -> None:
        with self._tasks_lock:
            self._tasks[task["id"]] = task
            # Bounded FIFO: dicts preserve insertion order, so the first key
            # is always the oldest task. Caps memory (tasks retain the full
            # inbound message + output) at max_tasks entries.
            while len(self._tasks) > self._max_tasks:
                self._tasks.pop(next(iter(self._tasks)))

    def _tasks_get(self, request_id: _RequestId, params: Dict[str, Any]) -> JSONResponse:
        task_id = params.get("id")
        with self._tasks_lock:
            task = self._tasks.get(task_id) if isinstance(task_id, str) else None
        if task is None:
            return _rpc_error(request_id, TASK_NOT_FOUND, f"Task not found: {task_id!r}")
        return _rpc_result(request_id, task)

    def _tasks_cancel(self, request_id: _RequestId, params: Dict[str, Any]) -> JSONResponse:
        task_id = params.get("id")
        with self._tasks_lock:
            task = self._tasks.get(task_id) if isinstance(task_id, str) else None
            if task is None:
                return _rpc_error(request_id, TASK_NOT_FOUND, f"Task not found: {task_id!r}")
            # Synchronous v1: every stored task is already terminal.
            if task["status"]["state"] in TaskState.TERMINAL:
                return _rpc_error(
                    request_id, TASK_NOT_CANCELABLE, f"Task {task_id!r} is in a terminal state"
                )
            # State check + mutation stay under the lock so a concurrent
            # cancel/store cannot interleave between them.
            task["status"] = {  # pragma: no cover
                "state": TaskState.CANCELED,
                "timestamp": _now_iso(),
            }
            return _rpc_result(request_id, task)  # pragma: no cover
