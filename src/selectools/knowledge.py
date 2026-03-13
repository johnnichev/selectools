"""
Cross-session knowledge memory — daily logs and persistent facts.

Provides durable memory that persists across agent sessions using
file-based storage with daily log files and a persistent MEMORY.md
for long-term facts.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


class KnowledgeMemory:
    """Maintains cross-session knowledge with daily logs and persistent facts.

    Stores two kinds of information:
    - **Daily logs**: Time-stamped entries in per-day files (auto-pruned).
    - **Persistent facts**: Long-lived entries in ``MEMORY.md`` that survive pruning.

    The ``build_context()`` method produces a prompt-injectable block combining
    recent daily logs and persistent facts.

    Args:
        directory: Base directory for knowledge files.  Created if absent.
        recent_days: Number of recent days to include in context.  Default: 2.
        max_context_chars: Maximum characters to include in context output.
    """

    def __init__(
        self,
        directory: str = "./memory",
        recent_days: int = 2,
        max_context_chars: int = 5000,
    ) -> None:
        self._directory = directory
        self._recent_days = recent_days
        self._max_context_chars = max_context_chars
        os.makedirs(directory, exist_ok=True)

    @property
    def directory(self) -> str:
        """Base directory for knowledge files."""
        return self._directory

    def remember(
        self,
        content: str,
        category: str = "general",
        persistent: bool = False,
    ) -> str:
        """Store a piece of knowledge.

        Args:
            content: The text to remember.
            category: Category tag for the entry (e.g. "preference", "fact").
            persistent: If True, also writes to MEMORY.md for long-term retention.

        Returns:
            Confirmation message.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{category}] {content}"

        # Write to daily log
        today = now.strftime("%Y-%m-%d")
        log_path = os.path.join(self._directory, f"{today}.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

        # Optionally write to persistent memory
        if persistent:
            mem_path = os.path.join(self._directory, "MEMORY.md")
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(f"- [{category}] {content}\n")

        return f"Remembered: {content}"

    def get_recent_logs(self, days: Optional[int] = None) -> str:
        """Read recent daily log entries.

        Args:
            days: Number of recent days to read.  Defaults to ``recent_days``.

        Returns:
            Combined text from recent daily log files.
        """
        days = days or self._recent_days
        lines: List[str] = []

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            log_path = os.path.join(self._directory, f"{date}.log")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        lines.append(f"=== {date} ===")
                        lines.append(content)

        return "\n".join(lines)

    def get_persistent_facts(self) -> str:
        """Read persistent facts from MEMORY.md.

        Returns:
            Contents of MEMORY.md, or empty string if not found.
        """
        mem_path = os.path.join(self._directory, "MEMORY.md")
        if not os.path.exists(mem_path):
            return ""
        with open(mem_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def build_context(self) -> str:
        """Build a context string for prompt injection.

        Combines persistent facts and recent daily logs, truncated to
        ``max_context_chars``.

        Returns:
            A formatted context block with ``[Long-term Memory]`` and
            ``[Recent Memory]`` sections, or empty string if no data.
        """
        parts: List[str] = []

        persistent = self.get_persistent_facts()
        if persistent:
            parts.append("[Long-term Memory]")
            parts.append(persistent)

        recent = self.get_recent_logs()
        if recent:
            if parts:
                parts.append("")
            parts.append("[Recent Memory]")
            parts.append(recent)

        if not parts:
            return ""

        context = "\n".join(parts)
        if len(context) > self._max_context_chars:
            suffix = "\n... (truncated)"
            context = context[: self._max_context_chars - len(suffix)] + suffix
        return context

    def prune_old_logs(self, keep_days: Optional[int] = None) -> int:
        """Remove daily log files older than ``keep_days``.

        Args:
            keep_days: Number of days to keep.  Defaults to ``recent_days``.

        Returns:
            Number of log files removed.
        """
        keep_days = keep_days or self._recent_days
        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0

        for filename in os.listdir(self._directory):
            if not filename.endswith(".log"):
                continue
            date_str = filename[:-4]
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    os.remove(os.path.join(self._directory, filename))
                    removed += 1
            except ValueError:
                continue

        return removed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self._directory,
            "recent_days": self._recent_days,
            "max_context_chars": self._max_context_chars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeMemory":
        return cls(
            directory=data.get("directory", "./memory"),
            recent_days=data.get("recent_days", 2),
            max_context_chars=data.get("max_context_chars", 5000),
        )


__all__ = ["KnowledgeMemory"]
