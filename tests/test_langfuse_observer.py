"""Tests for LangfuseObserver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLangfuseObserver:
    def _make_observer(self):
        mock_langfuse_mod = MagicMock()
        mock_client = MagicMock()
        mock_langfuse_mod.Langfuse.return_value = mock_client
        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from selectools.observe.langfuse import LangfuseObserver

            obs = LangfuseObserver.__new__(LangfuseObserver)
            obs._langfuse = mock_client
            obs._traces = {}
            obs._generations = {}
        return obs, mock_client

    def test_import_error(self):
        with patch.dict("sys.modules", {"langfuse": None}):
            with pytest.raises(ImportError, match="langfuse"):
                import importlib

                import selectools.observe.langfuse as mod

                importlib.reload(mod)
                mod.LangfuseObserver()

    def test_run_start_creates_trace(self):
        obs, client = self._make_observer()
        mock_trace = MagicMock()
        client.trace.return_value = mock_trace
        obs.on_run_start("run1", [], "system prompt")
        client.trace.assert_called_once()
        assert "run1" in obs._traces

    def test_run_end_updates_and_flushes(self):
        obs, client = self._make_observer()
        mock_trace = MagicMock()
        obs._traces["run1"] = mock_trace
        result = MagicMock()
        result.content = "response"
        result.usage.total_tokens = 100
        result.usage.total_cost_usd = 0.003
        result.iterations = 2
        obs.on_run_end("run1", result)
        mock_trace.update.assert_called_once()
        client.flush.assert_called_once()
        assert "run1" not in obs._traces

    def test_run_end_missing_trace(self):
        obs, client = self._make_observer()
        obs.on_run_end("nonexistent", MagicMock())
        client.flush.assert_not_called()

    def test_run_end_flush_error_handled(self):
        obs, client = self._make_observer()
        mock_trace = MagicMock()
        obs._traces["run1"] = mock_trace
        client.flush.side_effect = Exception("network error")
        obs.on_run_end("run1", MagicMock())  # Should not raise

    def test_llm_start_creates_generation(self):
        obs, _ = self._make_observer()
        mock_trace = MagicMock()
        mock_gen = MagicMock()
        mock_trace.generation.return_value = mock_gen
        obs._traces["run1"] = mock_trace
        obs.on_llm_start("run1", [{"role": "user", "content": "hi"}], "gpt-4o", "prompt")
        mock_trace.generation.assert_called_once()
        assert "run1:llm" in obs._generations

    def test_llm_start_no_trace(self):
        obs, _ = self._make_observer()
        obs.on_llm_start("nonexistent", [], "gpt-4o", "prompt")  # Should not raise

    def test_llm_end_updates_generation(self):
        obs, _ = self._make_observer()
        mock_gen = MagicMock()
        obs._generations["run1:llm"] = mock_gen
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 30
        usage.total_tokens = 80
        obs.on_llm_end("run1", "response", usage)
        mock_gen.update.assert_called_once()

    def test_llm_end_no_usage(self):
        obs, _ = self._make_observer()
        mock_gen = MagicMock()
        obs._generations["run1:llm"] = mock_gen
        obs.on_llm_end("run1", "response", None)
        mock_gen.update.assert_called_once()

    def test_tool_start_creates_span(self):
        obs, _ = self._make_observer()
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        obs._traces["run1"] = mock_trace
        obs.on_tool_start("run1", "call1", "search", {"q": "test"})
        assert "run1:tool:call1" in obs._generations

    def test_tool_end_updates_span(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._generations["run1:tool:call1"] = mock_span
        obs.on_tool_end("run1", "call1", "search", "results", 42.0)
        mock_span.update.assert_called_once()

    def test_tool_error_records_error(self):
        obs, _ = self._make_observer()
        mock_span = MagicMock()
        obs._generations["run1:tool:call1"] = mock_span
        obs.on_tool_error("run1", "call1", "search", Exception("timeout"), {"q": "test"}, 100.0)
        mock_span.update.assert_called_once()
        call_kwargs = mock_span.update.call_args[1]
        assert "ERROR" in call_kwargs.get("output", "") or call_kwargs.get("level") == "ERROR"

    def test_shutdown_flushes(self):
        obs, client = self._make_observer()
        obs.shutdown()
        client.flush.assert_called_once()

    def test_shutdown_error_handled(self):
        obs, client = self._make_observer()
        client.flush.side_effect = Exception("fail")
        obs.shutdown()  # Should not raise

    def test_stability_marker(self):
        obs, _ = self._make_observer()
        assert getattr(type(obs), "__stability__", None) == "beta"
