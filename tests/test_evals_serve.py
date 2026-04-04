"""Tests for evals/serve.py — live eval dashboard."""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from selectools.evals.serve import _DASHBOARD_HTML, _DashboardHandler, serve_eval

# ---------------------------------------------------------------------------
# _DashboardHandler
# ---------------------------------------------------------------------------


class TestDashboardHandler:
    def _make_handler(self, path="/"):
        """Create a _DashboardHandler with a mock request."""
        handler = _DashboardHandler.__new__(_DashboardHandler)
        handler.path = path
        handler.wfile = io.BytesIO()
        handler.requestline = f"GET {path} HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.server = MagicMock()
        handler.close_connection = True
        handler.headers = {}
        handler._headers_buffer = []

        # Mock send_response/send_header/end_headers
        sent_headers = {}

        def mock_send_response(code):
            sent_headers["status"] = code

        def mock_send_header(key, value):
            sent_headers[key] = value

        def mock_end_headers():
            pass

        def mock_send_error(code):
            sent_headers["status"] = code

        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers
        handler.send_error = mock_send_error
        handler._sent_headers = sent_headers

        return handler

    def test_serve_dashboard_root(self):
        handler = self._make_handler("/")
        handler.do_GET()
        assert handler._sent_headers.get("status") == 200
        assert handler._sent_headers.get("Content-Type") == "text/html; charset=utf-8"
        content = handler.wfile.getvalue()
        assert b"Selectools Eval" in content

    def test_serve_dashboard_index(self):
        handler = self._make_handler("/index.html")
        handler.do_GET()
        assert handler._sent_headers.get("status") == 200

    def test_serve_api_state(self):
        _DashboardHandler.dashboard_state = {
            "status": "running",
            "completed": 5,
            "total_cases": 10,
        }
        handler = self._make_handler("/api/state")
        handler.do_GET()
        assert handler._sent_headers.get("status") == 200
        assert handler._sent_headers.get("Content-Type") == "application/json"
        data = json.loads(handler.wfile.getvalue().decode())
        assert data["status"] == "running"
        assert data["completed"] == 5

    def test_serve_404(self):
        handler = self._make_handler("/nonexistent")
        handler.do_GET()
        assert handler._sent_headers.get("status") == 404

    def test_log_message_suppressed(self):
        handler = self._make_handler("/")
        # Should not raise
        handler.log_message("test %s", "arg")

    def test_dashboard_html_is_valid(self):
        assert "<!DOCTYPE html>" in _DASHBOARD_HTML
        assert "Selectools Eval" in _DASHBOARD_HTML
        assert "poll()" in _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# serve_eval (integration-ish — mocked suite)
# ---------------------------------------------------------------------------


class TestServeEval:
    def test_serve_eval_updates_state(self):
        """serve_eval updates dashboard state and returns a report."""
        mock_suite = MagicMock()
        mock_suite.name = "test-suite"
        mock_suite.cases = [MagicMock(), MagicMock()]
        mock_suite.on_progress = None

        # Mock the report returned by suite.run()
        mock_case_result = MagicMock()
        mock_case_result.case.name = "case1"
        mock_case_result.case.input = "test input"
        mock_case_result.verdict.value = "pass"
        mock_case_result.latency_ms = 100.0
        mock_case_result.cost_usd = 0.001
        mock_case_result.failures = []

        mock_report = MagicMock()
        mock_report.case_results = [mock_case_result]
        mock_report.accuracy = 1.0
        mock_report.pass_count = 1
        mock_report.fail_count = 0
        mock_report.error_count = 0
        mock_report.total_cost = 0.001
        mock_report.total_tokens = 50
        mock_report.latency_p50 = 100.0
        mock_report.latency_p95 = 100.0
        mock_report.metadata.duration_ms = 200.0

        mock_suite.run.return_value = mock_report

        # Patch server and keyboard interrupt so serve_eval doesn't block
        with (
            patch("selectools.evals.serve.HTTPServer") as mock_server_cls,
            patch("selectools.evals.serve.webbrowser", create=True),
            patch("builtins.print"),
        ):

            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server

            # Make server_thread.join() raise KeyboardInterrupt immediately
            mock_thread = MagicMock()
            mock_thread.join.side_effect = KeyboardInterrupt
            with patch("selectools.evals.serve.threading.Thread", return_value=mock_thread):
                report = serve_eval(mock_suite, port=9999, open_browser=False)

        assert report is mock_report
        mock_suite.run.assert_called_once()

    def test_serve_eval_restores_on_progress(self):
        """serve_eval restores the original on_progress callback."""
        original_cb = MagicMock()
        mock_suite = MagicMock()
        mock_suite.name = "test"
        mock_suite.cases = []
        mock_suite.on_progress = original_cb

        mock_report = MagicMock()
        mock_report.case_results = []
        mock_report.accuracy = 0.0
        mock_report.pass_count = 0
        mock_report.fail_count = 0
        mock_report.error_count = 0
        mock_report.total_cost = 0.0
        mock_report.total_tokens = 0
        mock_report.latency_p50 = 0.0
        mock_report.latency_p95 = 0.0
        mock_report.metadata.duration_ms = 0.0
        mock_suite.run.return_value = mock_report

        with patch("selectools.evals.serve.HTTPServer") as mock_srv, patch("builtins.print"):
            mock_srv.return_value = MagicMock()
            mock_thread = MagicMock()
            mock_thread.join.side_effect = KeyboardInterrupt
            with patch("selectools.evals.serve.threading.Thread", return_value=mock_thread):
                serve_eval(mock_suite, open_browser=False)

        assert mock_suite.on_progress is original_cb
