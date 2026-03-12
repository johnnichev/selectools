"""
Unit tests for AgentTrace and TraceStep: add(), filter(), timeline(),
to_dict(), to_json(), to_otel_spans(), duration properties.

Previously only covered by E2E tests that were always skipped in CI.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest

from selectools.trace import AgentTrace, TraceStep


class TestTraceStep:
    def test_to_dict_excludes_none(self) -> None:
        step = TraceStep(type="llm_call", duration_ms=100.0, model="gpt-4o")
        d = step.to_dict()
        assert d["type"] == "llm_call"
        assert d["model"] == "gpt-4o"
        assert "tool_name" not in d
        assert "error" not in d

    def test_to_dict_includes_all_set_fields(self) -> None:
        step = TraceStep(
            type="tool_execution",
            duration_ms=50.0,
            tool_name="search",
            tool_args={"q": "test"},
            tool_result="found it",
            summary="search → found it",
        )
        d = step.to_dict()
        assert d["tool_name"] == "search"
        assert d["tool_args"] == {"q": "test"}
        assert d["tool_result"] == "found it"


class TestAgentTraceAdd:
    def test_add_step(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        assert len(trace.steps) == 1

    def test_len(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="tool_execution", duration_ms=50.0))
        assert len(trace) == 2

    def test_iter(self) -> None:
        trace = AgentTrace()
        s1 = TraceStep(type="llm_call", duration_ms=100.0)
        s2 = TraceStep(type="tool_execution", duration_ms=50.0)
        trace.add(s1)
        trace.add(s2)
        assert list(trace) == [s1, s2]


class TestAgentTraceFilter:
    def test_filter_by_type(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="tool_execution", duration_ms=50.0))
        trace.add(TraceStep(type="llm_call", duration_ms=200.0))
        trace.add(TraceStep(type="error", duration_ms=10.0))

        llm_steps = trace.filter(type="llm_call")
        assert len(llm_steps) == 2
        assert all(s.type == "llm_call" for s in llm_steps)

    def test_filter_none_returns_all(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="error", duration_ms=10.0))
        assert len(trace.filter(type=None)) == 2

    def test_filter_no_match(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        assert len(trace.filter(type="cache_hit")) == 0


class TestAgentTraceDurations:
    def test_total_duration(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="tool_execution", duration_ms=50.0))
        assert trace.total_duration_ms == 150.0

    def test_llm_duration(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="tool_execution", duration_ms=50.0))
        trace.add(TraceStep(type="llm_call", duration_ms=200.0))
        assert trace.llm_duration_ms == 300.0

    def test_tool_duration(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        trace.add(TraceStep(type="tool_execution", duration_ms=50.0))
        assert trace.tool_duration_ms == 50.0

    def test_empty_trace(self) -> None:
        trace = AgentTrace()
        assert trace.total_duration_ms == 0.0
        assert trace.llm_duration_ms == 0.0
        assert trace.tool_duration_ms == 0.0


class TestAgentTraceTimeline:
    def test_timeline_format(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0, summary="GPT-4o → 50 chars"))
        trace.add(TraceStep(type="tool_execution", duration_ms=30.0, summary="search → result"))
        timeline = trace.timeline()

        assert "llm_call" in timeline
        assert "tool_execution" in timeline
        assert "100.0ms" in timeline
        assert "Total:" in timeline

    def test_timeline_empty(self) -> None:
        trace = AgentTrace()
        timeline = trace.timeline()
        assert "Total: 0.0ms" in timeline


class TestAgentTraceToDict:
    def test_basic_fields(self) -> None:
        trace = AgentTrace(run_id="test-run-123")
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        d = trace.to_dict()

        assert d["run_id"] == "test-run-123"
        assert d["step_count"] == 1
        assert len(d["steps"]) == 1
        assert "total_duration_ms" in d

    def test_parent_run_id_included(self) -> None:
        trace = AgentTrace(parent_run_id="parent-123")
        d = trace.to_dict()
        assert d["parent_run_id"] == "parent-123"

    def test_parent_run_id_excluded_when_none(self) -> None:
        trace = AgentTrace()
        d = trace.to_dict()
        assert "parent_run_id" not in d

    def test_metadata_included(self) -> None:
        trace = AgentTrace(metadata={"user_id": "u42", "env": "prod"})
        d = trace.to_dict()
        assert d["metadata"] == {"user_id": "u42", "env": "prod"}

    def test_metadata_excluded_when_empty(self) -> None:
        trace = AgentTrace()
        d = trace.to_dict()
        assert "metadata" not in d


class TestAgentTraceToJson:
    def test_writes_valid_json(self) -> None:
        trace = AgentTrace(run_id="json-test")
        trace.add(TraceStep(type="llm_call", duration_ms=50.0, model="gpt-4o"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            trace.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == "json-test"
            assert len(data["steps"]) == 1
        finally:
            os.unlink(path)


class TestAgentTraceToOtelSpans:
    def test_root_span_created(self) -> None:
        trace = AgentTrace(run_id="otel-test")
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        spans = trace.to_otel_spans()

        assert len(spans) == 2  # root + 1 child
        root = spans[0]
        assert root["name"] == "agent.run"
        assert root["trace_id"] == "otel-test"
        assert root["attributes"]["selectools.run_id"] == "otel-test"

    def test_child_spans_have_parent(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0, model="gpt-4o"))
        spans = trace.to_otel_spans()

        root_id = spans[0]["span_id"]
        child = spans[1]
        assert child["parent_span_id"] == root_id

    def test_llm_span_attributes(self) -> None:
        trace = AgentTrace()
        trace.add(
            TraceStep(
                type="llm_call",
                duration_ms=100.0,
                model="gpt-4o",
                prompt_tokens=500,
                completion_tokens=100,
            )
        )
        spans = trace.to_otel_spans()
        child = spans[1]

        assert child["name"] == "llm.gpt-4o"
        assert child["attributes"]["llm.model"] == "gpt-4o"
        assert child["attributes"]["llm.prompt_tokens"] == 500

    def test_tool_span_attributes(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="tool_execution", duration_ms=30.0, tool_name="search"))
        spans = trace.to_otel_spans()
        child = spans[1]

        assert child["name"] == "tool.search"
        assert child["attributes"]["tool.name"] == "search"

    def test_error_span_status(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="error", duration_ms=5.0, error="Timeout"))
        spans = trace.to_otel_spans()
        child = spans[1]

        assert child["status"]["code"] == "ERROR"
        assert child["attributes"]["error.message"] == "Timeout"

    def test_cache_hit_span(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="cache_hit", duration_ms=1.0, model="gpt-4o"))
        spans = trace.to_otel_spans()
        child = spans[1]
        assert child["name"] == "cache.hit"

    def test_parent_run_id_in_root_span(self) -> None:
        trace = AgentTrace(parent_run_id="parent-abc")
        spans = trace.to_otel_spans()
        root = spans[0]
        assert root["parent_span_id"] == "parent-abc"

    def test_metadata_in_root_span(self) -> None:
        trace = AgentTrace(metadata={"user_id": "u42"})
        spans = trace.to_otel_spans()
        root = spans[0]
        assert root["attributes"]["selectools.metadata.user_id"] == "u42"

    def test_summary_truncated(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=10.0, summary="x" * 300))
        spans = trace.to_otel_spans()
        child = spans[1]
        assert len(child["attributes"]["selectools.summary"]) == 200


class TestAgentTraceRepr:
    def test_repr(self) -> None:
        trace = AgentTrace()
        trace.add(TraceStep(type="llm_call", duration_ms=100.0))
        r = repr(trace)
        assert "steps=1" in r
        assert "100.0" in r
