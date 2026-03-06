"""
Execution tracing for agent runs.

Captures a structured timeline of every step in the agent loop:
LLM calls, tool selections, tool executions, cache hits, and errors.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional

StepType = Literal[
    "llm_call",
    "tool_selection",
    "tool_execution",
    "cache_hit",
    "error",
    "structured_retry",
]


@dataclass
class TraceStep:
    """A single step in the agent execution timeline."""

    type: StepType
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None

    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    reasoning: Optional[str] = None

    error: Optional[str] = None

    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class AgentTrace:
    """Ordered list of trace steps from a single agent run."""

    steps: List[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add(self, step: TraceStep) -> None:
        self.steps.append(step)

    def filter(self, *, type: Optional[StepType] = None) -> List[TraceStep]:
        if type is None:
            return list(self.steps)
        return [s for s in self.steps if s.type == type]

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    @property
    def llm_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps if s.type == "llm_call")

    @property
    def tool_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps if s.type == "tool_execution")

    def timeline(self) -> str:
        """Human-readable timeline string."""
        lines = []
        for i, s in enumerate(self.steps, 1):
            summary = s.summary or s.type
            lines.append(f"  {i}. [{s.type:18s}] {s.duration_ms:7.1f}ms  {summary}")
        total = self.total_duration_ms
        lines.append(
            f"  Total: {total:.1f}ms (LLM: {self.llm_duration_ms:.1f}ms, Tools: {self.tool_duration_ms:.1f}ms)"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "total_duration_ms": self.total_duration_ms,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __len__(self) -> int:
        """Return the number of steps in this trace."""
        return len(self.steps)

    def __iter__(self) -> Iterator[TraceStep]:
        """Iterate over the steps in this trace."""
        return iter(self.steps)

    def __repr__(self) -> str:
        """Return a concise string representation of this trace."""
        return f"AgentTrace(steps={len(self.steps)}, total_ms={self.total_duration_ms:.1f})"


__all__ = ["TraceStep", "AgentTrace", "StepType"]
