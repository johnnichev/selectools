"""Tests for OTelObserver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestOTelObserver:
    def _make_observer(self):
        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.set_span_in_context = MagicMock(return_value=None)
        with patch.dict(
            "sys.modules", {"opentelemetry": MagicMock(), "opentelemetry.trace": mock_trace}
        ):
            from selectools.observe.otel import OTelObserver

            obs = OTelObserver.__new__(OTelObserver)
            obs._tracer = mock_tracer
            obs._trace_mod = mock_trace
            obs._spans = {}
            obs._llm_starts = {}
        return obs, mock_tracer

    def test_import_error(self):
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            with pytest.raises(ImportError, match="opentelemetry-api"):
                # Force fresh import
                import importlib

                import selectools.observe.otel as mod

                importlib.reload(mod)
                mod.OTelObserver()

    def test_run_start_creates_span(self):
        obs, tracer = self._make_observer()
        obs.on_run_start("run1", [], "sys prompt")
        tracer.start_span.assert_called_once()
        assert "run1" in obs._spans

    def test_run_end_ends_span(self):
        obs, tracer = self._make_observer()
        mock_span = MagicMock()
        obs._spans["run1"] = mock_span
        result = MagicMock()
        result.usage.prompt_tokens = 100
        result.usage.completion_tokens = 50
        result.usage.total_tokens = 150
        result.usage.total_cost_usd = 0.005
        result.iterations = 3
        obs.on_run_end("run1", result)
        mock_span.end.assert_called_once()
        assert "run1" not in obs._spans

    def test_run_end_missing_run_id(self):
        obs, _ = self._make_observer()
        obs.on_run_end("nonexistent", MagicMock())  # Should not raise

    def test_llm_start_creates_child_span(self):
        obs, tracer = self._make_observer()
        obs._spans["run1"] = MagicMock()
        obs.on_llm_start("run1", [], "gpt-4o", "prompt")
        assert "run1:llm" in obs._spans

    def test_llm_end_ends_child_span(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._spans["run1:llm"] = mock_span
        obs._llm_starts["run1:llm"] = 1000.0
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 25
        obs.on_llm_end("run1", "response text", usage)
        mock_span.end.assert_called_once()

    def test_tool_start_creates_span(self):
        obs, tracer = self._make_observer()
        obs._spans["run1"] = MagicMock()
        obs.on_tool_start("run1", "call1", "search", {"q": "test"})
        assert "run1:tool:call1" in obs._spans

    def test_tool_end_ends_span(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._spans["run1:tool:call1"] = mock_span
        obs.on_tool_end("run1", "call1", "search", "results", 42.0)
        mock_span.end.assert_called_once()

    def test_tool_error_records_error(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._spans["run1:tool:call1"] = mock_span
        obs.on_tool_error("run1", "call1", "search", Exception("timeout"), {"q": "test"}, 100.0)
        mock_span.set_attribute.assert_any_call("error", True)
        mock_span.end.assert_called_once()

    def test_stability_marker(self):
        obs, _ = self._make_observer()
        assert getattr(type(obs), "__stability__", None) == "beta"
