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
            obs._llm_counter = 0
            obs._tool_counter = 0
            import threading

            obs._lock = threading.Lock()
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
        assert "run1:llm:1" in obs._spans

    def test_llm_end_ends_child_span(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._spans["run1:llm:1"] = mock_span
        obs._llm_starts["run1:llm:1"] = 1000.0
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

    def test_multi_iteration_llm_no_overwrite(self):
        """Regression: Bug 7 — multiple LLM calls must not overwrite spans."""
        obs, tracer = self._make_observer()
        span1 = MagicMock()
        span2 = MagicMock()
        tracer.start_span.side_effect = [span1, span2]
        obs._spans["run1"] = MagicMock()

        obs.on_llm_start("run1", [], "gpt-4o", "prompt")
        # First span should be stored
        assert "run1:llm:1" in obs._spans
        assert obs._spans["run1:llm:1"] is span1

        # End the first LLM call
        obs.on_llm_end("run1", "response 1", None)
        span1.end.assert_called_once()

        # Start a second LLM call — should NOT overwrite the first key
        obs.on_llm_start("run1", [], "gpt-4o", "prompt")
        assert "run1:llm:2" in obs._spans
        assert obs._spans["run1:llm:2"] is span2

        obs.on_llm_end("run1", "response 2", None)
        span2.end.assert_called_once()

    def test_concurrent_llm_spans_resolved_correctly(self):
        """Regression: Bug 7 — on_llm_end picks the highest-numbered span."""
        obs, tracer = self._make_observer()
        span1 = MagicMock()
        span2 = MagicMock()
        tracer.start_span.side_effect = [span1, span2]
        obs._spans["run1"] = MagicMock()

        # Start two LLM calls without ending either
        obs.on_llm_start("run1", [], "gpt-4o", "prompt")
        obs.on_llm_start("run1", [], "gpt-4o", "prompt")

        # End should close the most recent (span2)
        obs.on_llm_end("run1", "response", None)
        span2.end.assert_called_once()
        span1.end.assert_not_called()
        assert "run1:llm:1" in obs._spans  # span1 still open

    def test_run_end_cleans_up_orphaned_spans(self):
        """Regression: Bug 17 — orphaned spans cleaned up on run end."""
        obs, _ = self._make_observer()
        run_span = MagicMock()
        orphan_llm = MagicMock()
        orphan_tool = MagicMock()
        obs._spans["run1"] = run_span
        obs._spans["run1:llm:1"] = orphan_llm
        obs._spans["run1:tool:call99"] = orphan_tool
        obs._llm_starts["run1:llm:1"] = 1000.0

        result = MagicMock()
        result.usage = None
        del result.iterations  # no iterations attr
        obs.on_run_end("run1", result)

        # Orphans should be ended
        orphan_llm.end.assert_called_once()
        orphan_tool.end.assert_called_once()
        # Run span also ended
        run_span.end.assert_called_once()
        # All cleaned up
        assert not any(k.startswith("run1") for k in obs._spans)
        assert "run1:llm:1" not in obs._llm_starts

    def test_run_end_orphan_cleanup_does_not_affect_other_runs(self):
        """Orphan cleanup must only touch spans for the given run_id."""
        obs, _ = self._make_observer()
        run1_span = MagicMock()
        run2_llm = MagicMock()
        obs._spans["run1"] = run1_span
        obs._spans["run2:llm:1"] = run2_llm

        result = MagicMock()
        result.usage = None
        del result.iterations
        obs.on_run_end("run1", result)

        # run2's span should be untouched
        run2_llm.end.assert_not_called()
        assert "run2:llm:1" in obs._spans

    def test_stability_marker(self):
        obs, _ = self._make_observer()
        assert getattr(type(obs), "__stability__", None) == "beta"
