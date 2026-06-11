"""
A2A protocol types — Agent Card, task lifecycle states, and errors.

Plain dataclasses with no third-party dependencies so they can be imported
without ``starlette`` or ``httpx`` installed. Field names follow the A2A
JSON-RPC wire format (camelCase on the wire, snake_case on the dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..stability import beta

__stability__ = "beta"

__all__ = ["A2AError", "AgentCard", "AgentSkill", "A2ATask", "TaskState"]

# A2A JSON-RPC protocol version advertised in the Agent Card.
PROTOCOL_VERSION = "0.2.6"

# JSON-RPC 2.0 standard error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# A2A-specific error codes (per the A2A specification).
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002


@beta
class TaskState:
    """Task lifecycle states defined by the A2A protocol.

    Lifecycle: ``submitted`` → ``working`` → ``input-required`` →
    ``completed`` / ``failed`` / ``canceled``. The synchronous v1 server
    moves a task submitted→working→completed (or failed) within a single
    request; the state field exists so async backends can slot in later.
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    TERMINAL = frozenset({COMPLETED, FAILED, CANCELED})


@beta
class A2AError(Exception):
    """A JSON-RPC or transport error returned by an A2A server.

    Attributes:
        message: Human-readable error description.
        code: JSON-RPC error code when the failure happened at the
            protocol level, ``None`` for transport-level failures.
        data: Optional structured error payload from the server.
    """

    def __init__(
        self, message: str, code: Optional[int] = None, data: Optional[Any] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


@beta
@dataclass
class AgentSkill:
    """One skill advertised in an Agent Card (mapped from an agent tool)."""

    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSkill":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            tags=list(data.get("tags") or []),
        )


@beta
@dataclass
class AgentCard:
    """Parsed Agent Card from ``/.well-known/agent.json``.

    The full wire payload (including fields this dataclass does not model)
    is preserved in :attr:`raw`.
    """

    name: str
    description: str = ""
    url: str = ""
    version: str = ""
    protocol_version: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    skills: List[AgentSkill] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            url=str(data.get("url", "")),
            version=str(data.get("version", "")),
            protocol_version=str(data.get("protocolVersion", "")),
            capabilities=dict(data.get("capabilities") or {}),
            skills=[AgentSkill.from_dict(s) for s in data.get("skills") or []],
            raw=data,
        )


@beta
@dataclass
class A2ATask:
    """Result of an A2A task returned by ``message/send``.

    Attributes:
        id: Server-generated task id.
        context_id: Conversation context id (echoed from the request when
            provided, otherwise server-generated).
        state: Final :class:`TaskState` value.
        text: Concatenated text of all text parts in the task artifacts
            (the agent's answer). Empty when the task failed.
        error: Failure detail when ``state == TaskState.FAILED``.
        raw: Full Task object as returned on the wire.
    """

    id: str
    context_id: str = ""
    state: str = TaskState.SUBMITTED
    text: str = ""
    error: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ATask":
        status = data.get("status") or {}
        text_chunks: List[str] = []
        for artifact in data.get("artifacts") or []:
            for part in artifact.get("parts") or []:
                if part.get("kind") == "text" and part.get("text"):
                    text_chunks.append(part["text"])
        error = ""
        status_message = status.get("message") or {}
        for part in status_message.get("parts") or []:
            if part.get("kind") == "text" and part.get("text"):
                error = part["text"]
                break
        return cls(
            id=str(data.get("id", "")),
            context_id=str(data.get("contextId", "")),
            state=str(status.get("state", TaskState.SUBMITTED)),
            text="".join(text_chunks),
            error=error,
            raw=data,
        )
