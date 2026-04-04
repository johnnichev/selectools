"""Tests for evals/html.py — HTML report generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from selectools.evals.html import _donut_svg, _histogram_svg, _trend_svg, render_html_report
from selectools.evals.report import EvalReport
from selectools.evals.types import CaseResult, CaseVerdict, EvalFailure, EvalMetadata, TestCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata():
    return EvalMetadata(
        suite_name="test-suite",
        model="gpt-4o",
        provider="openai",
        timestamp=1700000000.0,
        run_id="abc123",
        total_cases=3,
        duration_ms=1500.0,
        selectools_version="0.20.0",
    )


def _make_case_result(
    name="test case",
    verdict=CaseVerdict.PASS,
    latency_ms=100.0,
    cost_usd=0.001,
    tokens=50,
    tags=None,
    failures=None,
    error=None,
    tool_calls=None,
):
    return CaseResult(
        case=TestCase(input="test input", name=name, tags=tags or []),
        verdict=verdict,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        tokens=tokens,
        failures=failures or [],
        error=error,
        tool_calls=tool_calls or [],
    )


def _make_report(case_results=None):
    meta = _make_metadata()
    if case_results is None:
        case_results = [
            _make_case_result("pass case", CaseVerdict.PASS),
            _make_case_result(
                "fail case",
                CaseVerdict.FAIL,
                failures=[EvalFailure("contains", "hello", "world", "missing 'hello'")],
            ),
            _make_case_result("error case", CaseVerdict.ERROR, error="boom"),
        ]
    meta.total_cases = len(case_results)
    return EvalReport(metadata=meta, case_results=case_results)


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


class TestDonutSvg:
    def test_empty_returns_empty(self):
        assert _donut_svg(0, 0, 0, 0) == ""

    def test_all_pass_returns_svg(self):
        result = _donut_svg(10, 0, 0, 0)
        assert "<svg" in result
        assert "viewBox" in result
        # Full circle path
        assert "<path" in result

    def test_mixed_returns_svg(self):
        result = _donut_svg(5, 3, 1, 1)
        assert "<svg" in result
        assert result.count("<path") == 4  # One per segment

    def test_single_fail_full_circle(self):
        result = _donut_svg(0, 10, 0, 0)
        assert "<svg" in result
        assert "#f87171" in result  # fail color


class TestHistogramSvg:
    def test_empty_returns_empty(self):
        assert _histogram_svg([]) == ""

    def test_single_value(self):
        result = _histogram_svg([100.0])
        assert "<svg" in result
        assert "<rect" in result

    def test_uniform_values(self):
        result = _histogram_svg([50.0, 50.0, 50.0])
        assert "<svg" in result

    def test_varied_values(self):
        result = _histogram_svg([10.0, 20.0, 30.0, 40.0, 100.0, 200.0])
        assert "<svg" in result
        assert "Latency Distribution" in result


class TestTrendSvg:
    def test_empty_returns_empty(self):
        assert _trend_svg([]) == ""

    def test_single_point_returns_empty(self):
        assert _trend_svg([0.9]) == ""

    def test_two_points_returns_svg(self):
        result = _trend_svg([0.8, 0.9])
        assert "<svg" in result
        assert "Accuracy Trend" in result

    def test_improving_trend_green(self):
        result = _trend_svg([0.5, 0.6, 0.7, 0.8])
        assert "#4ade80" in result  # green

    def test_declining_trend_red(self):
        result = _trend_svg([0.9, 0.8, 0.7, 0.6])
        assert "#f87171" in result  # red

    def test_flat_trend(self):
        result = _trend_svg([0.5, 0.5, 0.5])
        assert "<svg" in result


# ---------------------------------------------------------------------------
# render_html_report
# ---------------------------------------------------------------------------


class TestRenderHtmlReport:
    def test_basic_report(self, tmp_path):
        report = _make_report()
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        assert filepath.exists()
        content = filepath.read_text()
        assert "<!DOCTYPE html>" in content
        assert "test-suite" in content
        assert "gpt-4o" in content
        assert "pass case" in content
        assert "fail case" in content
        assert "error case" in content

    def test_report_contains_stats(self, tmp_path):
        report = _make_report()
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "Accuracy" in content
        assert "Pass" in content
        assert "Fail" in content
        assert "Latency" in content
        assert "Cost" in content

    def test_report_with_failures(self, tmp_path):
        case_results = [
            _make_case_result(
                "failed test",
                CaseVerdict.FAIL,
                failures=[
                    EvalFailure("contains", "expected", "actual", "missing keyword"),
                    EvalFailure("length", 100, 5, "too short"),
                ],
            ),
        ]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "Failures by Evaluator" in content
        assert "contains" in content
        assert "length" in content

    def test_report_with_tags(self, tmp_path):
        case_results = [
            _make_case_result("tagged", CaseVerdict.PASS, tags=["safety", "perf"]),
            _make_case_result("other", CaseVerdict.PASS, tags=["safety"]),
        ]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "safety" in content
        assert "perf" in content
        assert "filter-btn" in content

    def test_report_with_tool_calls(self, tmp_path):
        case_results = [
            _make_case_result(
                "tools test",
                CaseVerdict.PASS,
                tool_calls=["search", "calculator"],
            ),
        ]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "search" in content
        assert "calculator" in content

    def test_report_with_error(self, tmp_path):
        case_results = [
            _make_case_result("error test", CaseVerdict.ERROR, error="Connection timeout"),
        ]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "Connection timeout" in content

    def test_empty_report(self, tmp_path):
        report = _make_report(case_results=[])
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        assert filepath.exists()
        content = filepath.read_text()
        assert "<!DOCTYPE html>" in content

    def test_report_with_history_trend(self, tmp_path):
        report = _make_report()
        filepath = tmp_path / "report.html"

        class FakeHistory:
            accuracy_trend = [0.7, 0.8, 0.85, 0.9]

        render_html_report(report, filepath, history=FakeHistory())

        content = filepath.read_text()
        assert "Accuracy Trend" in content

    def test_report_with_short_history(self, tmp_path):
        """History with < 2 data points should not render a trend chart."""
        report = _make_report()
        filepath = tmp_path / "report.html"

        class ShortHistory:
            accuracy_trend = [0.9]

        render_html_report(report, filepath, history=ShortHistory())

        content = filepath.read_text()
        assert "Accuracy Trend" not in content

    def test_report_no_tags_still_has_filter_bar(self, tmp_path):
        case_results = [_make_case_result("no tags", CaseVerdict.PASS)]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "filter-btn" in content
        assert "Failures" in content

    def test_report_skip_verdict(self, tmp_path):
        case_results = [
            _make_case_result("skipped", CaseVerdict.SKIP, latency_ms=0.0),
        ]
        report = _make_report(case_results)
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "skip" in content.lower()

    def test_report_atomic_write(self, tmp_path):
        """Render uses atomic write (tmp file then replace)."""
        report = _make_report()
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)
        # The temp file should not be left behind
        tmp_file = filepath.with_suffix(".html.tmp")
        assert not tmp_file.exists()
        assert filepath.exists()

    def test_report_javascript_included(self, tmp_path):
        report = _make_report()
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "toggleDetail" in content
        assert "filterByTag" in content
        assert "filterByVerdict" in content

    def test_report_with_agent_result(self, tmp_path):
        """CaseResult with agent_result shows output and reasoning."""

        class FakeAgentResult:
            content = "This is the agent output"
            reasoning = "Because reasons"

        cr = _make_case_result("with result", CaseVerdict.PASS)
        cr.agent_result = FakeAgentResult()
        report = _make_report(case_results=[cr])
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "agent output" in content
        assert "Because reasons" in content

    def test_report_escapes_html(self, tmp_path):
        """Verify HTML characters in input/output are escaped."""
        cr = _make_case_result(
            name="<script>alert(1)</script>",
            verdict=CaseVerdict.PASS,
        )
        cr.case.input = '<img onerror="alert(1)">'
        report = _make_report(case_results=[cr])
        filepath = tmp_path / "report.html"
        render_html_report(report, filepath)

        content = filepath.read_text()
        assert "<script>alert(1)</script>" not in content
        assert "&lt;script&gt;" in content
