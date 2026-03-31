"""Tests for trace_to_html."""

import time

import pytest

from selectools import trace_to_html
from selectools.trace import AgentTrace, StepType, TraceStep


def _make_trace() -> AgentTrace:
    trace = AgentTrace()
    trace.steps = [
        TraceStep(
            type=StepType.LLM_CALL,
            duration_ms=120.5,
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            summary="Generate response",
        ),
        TraceStep(type=StepType.TOOL_SELECTION, duration_ms=0.2, tool_name="search"),
        TraceStep(
            type=StepType.TOOL_EXECUTION,
            duration_ms=45.0,
            tool_name="search",
            tool_args={"query": "python agents"},
            tool_result="Found 5 results",
        ),
        TraceStep(type=StepType.CACHE_HIT, duration_ms=0.1, model="gpt-4o"),
        TraceStep(type=StepType.ERROR, duration_ms=1.0, error="Connection timeout"),
        TraceStep(type=StepType.GUARDRAIL, duration_ms=3.0, summary="PII check passed"),
        TraceStep(type=StepType.GRAPH_NODE_START, duration_ms=0.5, node_name="writer"),
    ]
    return trace


def test_returns_html_string():
    trace = _make_trace()
    html = trace_to_html(trace)
    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html


def test_contains_run_id():
    trace = AgentTrace()
    html = trace_to_html(trace)
    assert trace.run_id[:16] in html


def test_contains_step_types():
    trace = _make_trace()
    html = trace_to_html(trace)
    assert "llm_call" in html
    assert "tool_execution" in html
    assert "cache_hit" in html
    assert "error" in html
    assert "guardrail" in html


def test_contains_step_count():
    trace = _make_trace()
    html = trace_to_html(trace)
    assert f"<b>steps:</b> {len(trace.steps)}" in html


def test_contains_total_duration():
    trace = _make_trace()
    html = trace_to_html(trace)
    assert "total:" in html
    assert "ms" in html


def test_step_colors_present():
    trace = _make_trace()
    html = trace_to_html(trace)
    assert "#3b82f6" in html  # llm_call
    assert "#8b5cf6" in html  # tool_execution
    assert "#4ade80" in html  # cache_hit
    assert "#f87171" in html  # error
    assert "#fbbf24" in html  # guardrail


def test_empty_trace_renders():
    trace = AgentTrace()
    html = trace_to_html(trace)
    assert "<html" in html
    assert "steps:</b> 0" in html


def test_zero_duration_no_division_error():
    trace = AgentTrace()
    trace.steps = [TraceStep(type=StepType.LLM_CALL, duration_ms=0.0)]
    html = trace_to_html(trace)  # must not raise ZeroDivisionError
    assert "<html" in html


def test_tool_args_included():
    trace = AgentTrace()
    trace.steps = [
        TraceStep(
            type=StepType.TOOL_EXECUTION,
            tool_name="calc",
            tool_args={"x": 1},
            tool_result="42",
            duration_ms=5.0,
        )
    ]
    html = trace_to_html(trace)
    assert "calc" in html


def test_error_message_included():
    trace = AgentTrace()
    trace.steps = [TraceStep(type=StepType.ERROR, error="Oops!", duration_ms=1.0)]
    html = trace_to_html(trace)
    assert "Oops!" in html


def test_does_not_write_files(tmp_path, monkeypatch):
    """trace_to_html is a pure function — it must not write any files."""
    import builtins

    original_open = builtins.open

    written = []

    def patched_open(path, mode="r", **kw):
        if "w" in str(mode):
            written.append(path)
        return original_open(path, mode, **kw)

    monkeypatch.setattr(builtins, "open", patched_open)

    trace = _make_trace()
    trace_to_html(trace)

    assert written == [], f"trace_to_html wrote files unexpectedly: {written}"


def test_importable_from_selectools():
    from selectools import trace_to_html as t2h  # noqa: F401

    assert callable(t2h)


def test_xss_escaping():
    """User-controlled content must be HTML-escaped."""
    trace = AgentTrace()
    trace.steps = [
        TraceStep(
            type=StepType.TOOL_EXECUTION, tool_name="<script>alert(1)</script>", duration_ms=1.0
        )
    ]
    html = trace_to_html(trace)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
