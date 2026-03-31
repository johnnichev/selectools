"""
AuditLogger — JSONL append-only audit log with privacy controls.

Provides a structured, immutable record of every agent action for
compliance, debugging, and cost analysis.  Integrates with the agent
via the :class:`AgentObserver` protocol.

Features:
- JSONL format (one JSON object per line, easy to ingest in any log system)
- Privacy controls: hash argument values, log only key names, or full values
- Optional daily file rotation
- Thread-safe file writes
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .observer import AgentObserver
from .stability import stable
from .types import AgentResult, Message
from .usage import UsageStats


@stable
class PrivacyLevel(str, Enum):
    """Controls how tool arguments are recorded in audit logs."""

    FULL = "full"
    KEYS_ONLY = "keys_only"
    HASHED = "hashed"
    NONE = "none"


@stable
class AuditLogger(AgentObserver):
    """JSONL audit logger that implements the AgentObserver protocol.

    Every agent lifecycle event is appended as a single JSON line to a
    log file.  The logger is thread-safe and supports daily rotation.

    Args:
        log_dir: Directory for audit log files.  Created if missing.
        privacy: How to handle tool argument values.

            - ``full``: log all argument values verbatim.
            - ``keys_only``: log argument key names only, values replaced with ``"<redacted>"``.
            - ``hashed``: log SHA-256 hashes of argument values.
            - ``none``: omit tool_args entirely.

        daily_rotation: Create a new file each day (``audit-YYYY-MM-DD.jsonl``).
            When ``False``, writes to ``audit.jsonl``.  Default: ``True``.
        include_content: Log LLM response content.  Set ``False`` for
            strict privacy.  Default: ``False``.
    """

    def __init__(
        self,
        log_dir: str = "./audit",
        *,
        privacy: PrivacyLevel = PrivacyLevel.KEYS_ONLY,
        daily_rotation: bool = True,
        include_content: bool = False,
    ) -> None:
        # Resolve symlinks and ``..`` components so the log path is unambiguous
        # and callers can inspect ``_log_dir`` to see the canonical location.
        self._log_dir = os.path.realpath(os.path.abspath(log_dir))
        self._privacy = privacy
        self._daily_rotation = daily_rotation
        self._include_content = include_content
        self._lock = threading.Lock()
        os.makedirs(self._log_dir, exist_ok=True)

    def _log_path(self) -> str:
        if self._daily_rotation:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return os.path.join(self._log_dir, f"audit-{date_str}.jsonl")
        return os.path.join(self._log_dir, "audit.jsonl")

    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._privacy == PrivacyLevel.FULL:
            return args
        if self._privacy == PrivacyLevel.NONE:
            return {}
        if self._privacy == PrivacyLevel.KEYS_ONLY:
            return {k: "<redacted>" for k in args}
        if self._privacy == PrivacyLevel.HASHED:
            return {k: hashlib.sha256(str(v).encode()).hexdigest()[:16] for k, v in args.items()}
        return args

    def _write(self, entry: Dict[str, Any]) -> None:
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------
    # AgentObserver implementation
    # ------------------------------------------------------------------

    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self._write(
            {
                "event": "run_start",
                "run_id": run_id,
                "message_count": len(messages),
            }
        )

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        entry: Dict[str, Any] = {
            "event": "run_end",
            "run_id": run_id,
            "iterations": result.iterations,
            "tool_name": result.tool_name,
        }
        if result.usage:
            entry["total_cost_usd"] = getattr(result.usage, "total_cost_usd", None)
        self._write(entry)

    def on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self._write(
            {
                "event": "tool_start",
                "run_id": run_id,
                "call_id": call_id,
                "tool_name": tool_name,
                "tool_args": self._sanitize_args(tool_args),
            }
        )

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        entry: Dict[str, Any] = {
            "event": "tool_end",
            "run_id": run_id,
            "call_id": call_id,
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": True,
        }
        if self._include_content:
            entry["result_length"] = len(result) if result is not None else 0
        self._write(entry)

    def on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        self._write(
            {
                "event": "tool_error",
                "run_id": run_id,
                "call_id": call_id,
                "tool_name": tool_name,
                "error": str(error),
                "error_type": type(error).__name__,
                "duration_ms": round(duration_ms, 2),
                "success": False,
                "tool_args": self._sanitize_args(tool_args),
            }
        )

    def on_llm_end(self, run_id: str, response: str, usage: Optional[UsageStats]) -> None:
        entry: Dict[str, Any] = {
            "event": "llm_end",
            "run_id": run_id,
        }
        if usage:
            entry["model"] = usage.model
            entry["prompt_tokens"] = usage.prompt_tokens
            entry["completion_tokens"] = usage.completion_tokens
            entry["cost_usd"] = usage.cost_usd
        if self._include_content:
            entry["response_length"] = len(response) if response else 0
        self._write(entry)

    def on_policy_decision(
        self,
        run_id: str,
        tool_name: str,
        decision: str,
        reason: str,
        tool_args: Dict[str, Any],
    ) -> None:
        self._write(
            {
                "event": "policy_decision",
                "run_id": run_id,
                "tool_name": tool_name,
                "decision": decision,
                "reason": reason,
                "tool_args": self._sanitize_args(tool_args),
            }
        )

    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._write(
            {
                "event": "error",
                "run_id": run_id,
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )


__all__ = ["AuditLogger", "PrivacyLevel"]
