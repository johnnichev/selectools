"""
A2A client — discover and talk to remote A2A agents.

Usage::

    from selectools.a2a import A2AClient

    client = A2AClient("https://other-agent.example.com")
    card = await client.discover()      # reads /.well-known/agent.json
    result = await client.send_task("Research quantum computing trends")
    print(result.text)

    # Sync code can use the *_sync wrappers:
    card = client.discover_sync()

Requires ``httpx`` (ships with ``pip install selectools[serve]``); the
import is lazy so this module loads without it.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .._async_utils import run_sync
from ..stability import beta
from .types import A2AError, A2ATask, AgentCard

__stability__ = "beta"

__all__ = ["A2AClient"]


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "A2AClient requires the 'httpx' package. Install it with: pip install selectools[serve]"
        ) from exc
    return httpx


@beta
class A2AClient:
    """Client for a remote A2A agent (Agent Card discovery + task sending).

    Args:
        base_url: Root URL of the remote agent, e.g.
            ``"https://other-agent.example.com"``. The Agent Card is read
            from ``{base_url}/.well-known/agent.json`` and tasks are posted
            to ``{base_url}/a2a``.
        auth_token: Optional bearer token sent as
            ``Authorization: Bearer <auth_token>`` on every request.
        timeout: Request timeout in seconds (default 30).
        transport: Optional ``httpx`` transport — pass
            ``httpx.ASGITransport(app=server)`` to talk to an in-process
            :class:`~selectools.a2a.server.A2AServer` without a socket.

    Raises:
        A2AError: On transport failures (non-2xx responses) and JSON-RPC
            protocol errors. Agent failures are NOT raised — they come back
            as an :class:`A2ATask` with ``state == TaskState.FAILED``.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[Any] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._transport = transport

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _async_client(self) -> Any:
        httpx = _httpx()
        kwargs: Dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    @staticmethod
    def _check_http(response: Any) -> None:
        if response.status_code >= 400:
            raise A2AError(f"HTTP {response.status_code} from A2A server: {response.text[:500]}")

    # ── discovery ────────────────────────────────────────────────────────────

    async def discover(self) -> AgentCard:
        """Fetch and parse the remote Agent Card from /.well-known/agent.json."""
        async with self._async_client() as client:
            response = await client.get(
                f"{self._base_url}/.well-known/agent.json", headers=self._headers()
            )
        self._check_http(response)
        return AgentCard.from_dict(response.json())

    def discover_sync(self) -> AgentCard:
        """Synchronous wrapper around :meth:`discover`."""
        return run_sync(self.discover())

    # ── tasks ────────────────────────────────────────────────────────────────

    async def _call(self, method: str, params: Dict[str, Any]) -> Any:
        """Issue one JSON-RPC call against POST /a2a and return its result."""
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        async with self._async_client() as client:
            response = await client.post(
                f"{self._base_url}/a2a", json=payload, headers=self._headers()
            )
        self._check_http(response)
        body = response.json()
        if "error" in body:
            error = body["error"] or {}
            raise A2AError(
                str(error.get("message", "Unknown A2A error")),
                code=error.get("code"),
                data=error.get("data"),
            )
        return body.get("result")

    async def send_task(
        self,
        text: str,
        context_id: Optional[str] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> A2ATask:
        """Send a text task via ``message/send`` and return the final task.

        Args:
            text: The task prompt (becomes a part of kind ``"text"``).
            context_id: Optional conversation context id to group tasks.
            extra_parts: Optional additional A2A parts (``file``/``data``)
                appended after the text part. The v1 server is text-first.
        """
        parts: List[Dict[str, Any]] = [{"kind": "text", "text": text}]
        if extra_parts:
            parts.extend(extra_parts)
        message: Dict[str, Any] = {
            "role": "user",
            "parts": parts,
            "messageId": uuid.uuid4().hex,
            "kind": "message",
        }
        if context_id:
            message["contextId"] = context_id
        result = await self._call("message/send", {"message": message})
        return A2ATask.from_dict(result or {})

    def send_task_sync(
        self,
        text: str,
        context_id: Optional[str] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> A2ATask:
        """Synchronous wrapper around :meth:`send_task`."""
        return run_sync(self.send_task(text, context_id=context_id, extra_parts=extra_parts))

    async def get_task(self, task_id: str) -> A2ATask:
        """Fetch a previously created task via ``tasks/get``."""
        result = await self._call("tasks/get", {"id": task_id})
        return A2ATask.from_dict(result or {})
