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

from selectools.stability import beta, stable


@stable
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


@stable
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


@stable
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


# ---------------------------------------------------------------------------
# HTML trace viewer
# ---------------------------------------------------------------------------

_STEP_COLORS: Dict[str, str] = {
    "llm_call": "#3b82f6",
    "tool_execution": "#8b5cf6",
    "tool_selection": "#a78bfa",
    "cache_hit": "#4ade80",
    "error": "#f87171",
    "guardrail": "#fbbf24",
    "output_screening": "#fbbf24",
    "coherence_check": "#fbbf24",
    "graph_node_start": "#06b6d4",
    "graph_node_end": "#06b6d4",
    "graph_routing": "#06b6d4",
    "graph_checkpoint": "#06b6d4",
    "graph_interrupt": "#06b6d4",
    "graph_resume": "#06b6d4",
    "graph_parallel_start": "#06b6d4",
    "graph_parallel_end": "#06b6d4",
    "graph_stall": "#f97316",
    "graph_loop_detected": "#f97316",
}
_DEFAULT_COLOR = "#64748b"


@beta
def trace_to_html(trace: "AgentTrace") -> str:
    """Render an AgentTrace as a standalone HTML waterfall timeline.

    Returns a self-contained HTML string (no external dependencies).
    The caller is responsible for writing it to disk if desired::

        from selectools import trace_to_html
        Path("trace.html").write_text(trace_to_html(result.trace))

    Args:
        trace: The ``AgentTrace`` to visualise.

    Returns:
        A complete HTML document as a string.
    """
    import html as _html

    max_dur = max((s.duration_ms for s in trace.steps), default=1.0)
    max_dur = max(max_dur, 1.0)  # guard against all-zero durations

    total_ms = trace.total_duration_ms
    llm_ms = trace.llm_duration_ms
    tool_ms = trace.tool_duration_ms
    llm_pct = f"{llm_ms / max(total_ms, 1) * 100:.0f}%"
    tool_pct = f"{tool_ms / max(total_ms, 1) * 100:.0f}%"

    def _esc(v: Any) -> str:
        return _html.escape(str(v)) if v is not None else ""

    def _detail(step: "TraceStep") -> str:
        parts = []
        if step.model:
            parts.append(f"<b>model:</b> {_esc(step.model)}")
        if step.prompt_tokens is not None:
            parts.append(f"<b>prompt tokens:</b> {step.prompt_tokens}")
        if step.completion_tokens is not None:
            parts.append(f"<b>completion tokens:</b> {step.completion_tokens}")
        if step.cost_usd is not None:
            parts.append(f"<b>cost:</b> ${step.cost_usd:.6f}")
        if step.tool_name:
            parts.append(f"<b>tool:</b> {_esc(step.tool_name)}")
        if step.tool_args:
            parts.append(f"<b>args:</b> <code>{_esc(json.dumps(step.tool_args))}</code>")
        if step.tool_result:
            truncated = str(step.tool_result)[:300]
            parts.append(f"<b>result:</b> <code>{_esc(truncated)}</code>")
        if step.error:
            parts.append(f"<b>error:</b> <span style='color:#f87171'>{_esc(step.error)}</span>")
        if step.node_name:
            parts.append(f"<b>node:</b> {_esc(step.node_name)}")
        if step.from_node or step.to_node:
            parts.append(f"<b>route:</b> {_esc(step.from_node)} → {_esc(step.to_node)}")
        if step.summary:
            parts.append(f"<b>summary:</b> {_esc(step.summary)}")
        if step.reasoning:
            parts.append(f"<b>reasoning:</b> {_esc(step.reasoning[:200])}")
        return "<br>".join(parts) if parts else "<i>no details</i>"

    rows = []
    for i, step in enumerate(trace.steps):
        type_val = step.type.value if hasattr(step.type, "value") else str(step.type)
        color = _STEP_COLORS.get(type_val, _DEFAULT_COLOR)
        bar_pct = step.duration_ms / max_dur * 100
        label = step.summary or type_val
        detail_html = _detail(step)
        rows.append(
            f"""
      <tr onclick="toggleDetail('d{i}')" style="cursor:pointer">
        <td style="color:#94a3b8;width:32px;text-align:right;padding-right:12px">{i + 1}</td>
        <td style="width:160px">
          <span style="background:{color};color:#0f172a;font-size:11px;font-weight:600;
                       padding:2px 7px;border-radius:4px">{_esc(type_val)}</span>
        </td>
        <td style="width:200px;padding:0 12px">
          <div style="background:#1e3a5f;border-radius:3px;height:10px;width:100%">
            <div style="background:{color};border-radius:3px;height:10px;width:{bar_pct:.1f}%"></div>
          </div>
        </td>
        <td style="color:#94a3b8;width:70px;text-align:right;padding-right:12px">
          {step.duration_ms:.1f}ms
        </td>
        <td style="color:#e2e8f0">{_esc(label)}</td>
      </tr>
      <tr id="d{i}" style="display:none">
        <td colspan="5" style="padding:8px 16px 12px 44px;color:#94a3b8;
                               font-size:12px;background:#0f2035;border-bottom:1px solid #1e3a5f">
          {detail_html}
        </td>
      </tr>"""
        )

    rows_html = "\n".join(rows)
    run_id_display = _esc(trace.run_id[:16]) + "…" if len(trace.run_id) > 16 else _esc(trace.run_id)
    step_count = len(trace.steps)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Trace — {run_id_display}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'JetBrains Mono', 'Fira Code', monospace;
          font-size: 13px; padding: 24px; }}
  h1 {{ font-size: 16px; font-weight: 600; color: #f8fafc; margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 12px; margin-bottom: 24px; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat b {{ color: #e2e8f0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  tr:hover td {{ background: #1a2744; }}
  td {{ padding: 7px 4px; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
  code {{ background: #1e293b; padding: 1px 4px; border-radius: 3px; font-size: 11px; }}
</style>
</head>
<body>
<h1>Agent Trace</h1>
<div class="meta">
  <span class="stat"><b>run:</b> {run_id_display}</span>
  <span class="stat"><b>steps:</b> {step_count}</span>
  <span class="stat"><b>total:</b> {total_ms:.1f}ms</span>
  <span class="stat"><b>llm:</b> {llm_ms:.1f}ms ({llm_pct})</span>
  <span class="stat"><b>tools:</b> {tool_ms:.1f}ms ({tool_pct})</span>
</div>
<table>
  <thead>
    <tr>
      <th style="color:#475569;text-align:right;padding-right:12px">#</th>
      <th style="color:#475569;text-align:left">type</th>
      <th style="color:#475569;padding:0 12px">duration</th>
      <th style="color:#475569;text-align:right;padding-right:12px">ms</th>
      <th style="color:#475569;text-align:left">label</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
<script>
function toggleDetail(id) {{
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
}}
</script>
</body>
</html>"""


__all__ = ["TraceStep", "AgentTrace", "StepType", "trace_to_html"]
