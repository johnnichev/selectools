"""
Execution tracing for agent runs.

Captures a structured timeline of every step in the agent loop:
LLM calls, tool selections, tool executions, cache hits, and errors.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class StepType(str, Enum):
    """Enumeration of trace step types.

    Inherits from ``str`` so that ``StepType.LLM_CALL == "llm_call"`` is ``True``,
    preserving backward compatibility with code that compares against string literals.
    """

    LLM_CALL = "llm_call"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    CACHE_HIT = "cache_hit"
    ERROR = "error"
    STRUCTURED_RETRY = "structured_retry"
    GUARDRAIL = "guardrail"
    COHERENCE_CHECK = "coherence_check"
    OUTPUT_SCREENING = "output_screening"
    SESSION_LOAD = "session_load"
    SESSION_SAVE = "session_save"
    MEMORY_SUMMARIZE = "memory_summarize"
    ENTITY_EXTRACTION = "entity_extraction"
    KG_EXTRACTION = "kg_extraction"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    PROMPT_COMPRESSED = "prompt_compressed"
    GRAPH_NODE_START = "graph_node_start"
    GRAPH_NODE_END = "graph_node_end"
    GRAPH_ROUTING = "graph_routing"
    GRAPH_CHECKPOINT = "graph_checkpoint"
    GRAPH_INTERRUPT = "graph_interrupt"
    GRAPH_RESUME = "graph_resume"
    GRAPH_PARALLEL_START = "graph_parallel_start"
    GRAPH_PARALLEL_END = "graph_parallel_end"
    GRAPH_STALL = "graph_stall"
    GRAPH_LOOP_DETECTED = "graph_loop_detected"


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
    cost_usd: Optional[float] = None

    reasoning: Optional[str] = None

    error: Optional[str] = None

    summary: Optional[str] = None

    node_name: Optional[str] = None
    step_number: Optional[int] = None
    checkpoint_id: Optional[str] = None
    interrupt_key: Optional[str] = None
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    children: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def _new_run_id() -> str:
    return uuid.uuid4().hex


@dataclass
class AgentTrace:
    """Ordered list of trace steps from a single agent run.

    Attributes:
        steps: The ordered list of ``TraceStep`` objects.
        start_time: Unix timestamp when the run started.
        run_id: Unique identifier for this run (auto-generated UUID).
        parent_run_id: Optional ID of the parent run for nested/chained agents.
        metadata: Arbitrary key-value pairs attached by the caller
            (e.g. ``user_id``, ``request_id``, ``environment``).
    """

    steps: List[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    run_id: str = field(default_factory=_new_run_id)
    parent_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
            type_val = s.type.value if hasattr(s.type, "value") else s.type
            summary = s.summary or type_val
            lines.append(f"  {i}. [{type_val:18s}] {s.duration_ms:7.1f}ms  {summary}")
        total = self.total_duration_ms
        lines.append(
            f"  Total: {total:.1f}ms (LLM: {self.llm_duration_ms:.1f}ms, Tools: {self.tool_duration_ms:.1f}ms)"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "total_duration_ms": self.total_duration_ms,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.parent_run_id:
            d["parent_run_id"] = self.parent_run_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentTrace":
        """Reconstruct an AgentTrace from a dict produced by to_dict()."""
        trace = cls(
            metadata=d.get("metadata", {}),
        )
        trace.run_id = d.get("run_id", trace.run_id)
        trace.start_time = d.get("start_time", trace.start_time)
        trace.parent_run_id = d.get("parent_run_id")
        for step_dict in d.get("steps", []):
            step_type_str = step_dict.get("type", "error")
            try:
                step_type = StepType(step_type_str)
            except (ValueError, KeyError):
                step_type = StepType.ERROR
            step = TraceStep(type=step_type)
            for k, v in step_dict.items():
                if k != "type" and hasattr(step, k):
                    setattr(step, k, v)
            trace.steps.append(step)
        return trace

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # OpenTelemetry export
    # ------------------------------------------------------------------

    def to_otel_spans(self) -> List[Dict[str, Any]]:
        """Convert trace steps into OpenTelemetry-compatible span dicts.

        Each span follows the `OTel data model
        <https://opentelemetry.io/docs/specs/otel/trace/api/>`_ with fields
        that can be fed directly into ``opentelemetry.sdk.trace.ReadableSpan``
        or serialised to OTLP JSON.

        Returns a list of span dicts.  The first span is the root span
        representing the entire agent run; child spans represent individual
        steps (LLM calls, tool executions, etc.).

        No ``opentelemetry`` dependency is required — the output is plain
        dicts that any OTel SDK or collector can consume.
        """
        trace_id = self.run_id
        root_start_ns = int(self.start_time * 1e9)
        root_end_ns = root_start_ns + int(self.total_duration_ms * 1e6)

        root_span: Dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": _span_id(),
            "name": "agent.run",
            "kind": "INTERNAL",
            "start_time_unix_nano": root_start_ns,
            "end_time_unix_nano": root_end_ns,
            "attributes": {
                "selectools.run_id": self.run_id,
                "selectools.step_count": len(self.steps),
            },
            "status": {"code": "OK"},
        }

        if self.parent_run_id:
            root_span["parent_span_id"] = self.parent_run_id
            root_span["attributes"]["selectools.parent_run_id"] = self.parent_run_id

        if self.metadata:
            for k, v in self.metadata.items():
                root_span["attributes"][f"selectools.metadata.{k}"] = str(v)

        spans: List[Dict[str, Any]] = [root_span]

        for step in self.steps:
            start_ns = int(step.timestamp * 1e9)
            end_ns = start_ns + int(step.duration_ms * 1e6)
            type_val = step.type.value if hasattr(step.type, "value") else step.type
            attrs: Dict[str, Any] = {"selectools.step_type": type_val}

            if step.type == "llm_call":
                name = f"llm.{step.model or 'unknown'}"
                if step.prompt_tokens is not None:
                    attrs["llm.prompt_tokens"] = step.prompt_tokens
                if step.completion_tokens is not None:
                    attrs["llm.completion_tokens"] = step.completion_tokens
                if step.model:
                    attrs["llm.model"] = step.model
            elif step.type in ("tool_execution", "tool_selection"):
                name = f"tool.{step.tool_name or 'unknown'}"
                if step.tool_name:
                    attrs["tool.name"] = step.tool_name
            elif step.type == "cache_hit":
                name = "cache.hit"
                if step.model:
                    attrs["llm.model"] = step.model
            elif step.type == "error":
                name = f"error.{step.tool_name or 'agent'}"
                if step.error:
                    attrs["error.message"] = step.error[:500]
            elif step.type == "structured_retry":
                name = "structured.retry"
                if step.error:
                    attrs["error.message"] = step.error[:500]
            else:
                name = step.type.value if hasattr(step.type, "value") else step.type

            if step.summary:
                attrs["selectools.summary"] = step.summary[:200]

            status = {"code": "ERROR"} if step.type == "error" else {"code": "OK"}

            child: Dict[str, Any] = {
                "trace_id": trace_id,
                "span_id": _span_id(),
                "parent_span_id": root_span["span_id"],
                "name": name,
                "kind": "INTERNAL",
                "start_time_unix_nano": start_ns,
                "end_time_unix_nano": end_ns,
                "attributes": attrs,
                "status": status,
            }
            spans.append(child)

        return spans

    def __len__(self) -> int:
        """Return the number of steps in this trace."""
        return len(self.steps)

    def __iter__(self) -> Iterator[TraceStep]:
        """Iterate over the steps in this trace."""
        return iter(self.steps)

    def __repr__(self) -> str:
        """Return a concise string representation of this trace."""
        return f"AgentTrace(steps={len(self.steps)}, total_ms={self.total_duration_ms:.1f})"


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


__all__ = ["TraceStep", "AgentTrace", "StepType"]
