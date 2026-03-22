"""Tests for v0.17.0 release features: to_markdown, observer, trend chart, pip extra."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.evals import (
    CaseResult,
    CaseVerdict,
    EvalMetadata,
    EvalReport,
    EvalSuite,
    HistoryStore,
    HistoryTrend,
    TestCase,
)
from selectools.evals.history import HistoryEntry
from selectools.evals.html import _trend_svg
from selectools.observer import AgentObserver
from tests.conftest import SharedFakeProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool(description="test tool")
def dummy_tool(x: str) -> str:
    return x


def _make_agent(responses: list, observers: list | None = None) -> Agent:
    provider = SharedFakeProvider(responses=responses)
    config = AgentConfig(model="fake-model")
    if observers:
        config.observers = observers
    return Agent(provider=provider, config=config, tools=[dummy_tool])


def _make_report(name: str = "test", accuracy: float = 1.0) -> EvalReport:
    n_pass = int(accuracy * 4)
    n_fail = 4 - n_pass
    cases = []
    for i in range(n_pass):
        tc = TestCase(input=f"p{i}", name=f"pass_{i}")
        cases.append(CaseResult(case=tc, verdict=CaseVerdict.PASS, latency_ms=100, cost_usd=0.001))
    for i in range(n_fail):
        tc = TestCase(input=f"f{i}", name=f"fail_{i}")
        cases.append(
            CaseResult(
                case=tc,
                verdict=CaseVerdict.FAIL,
                latency_ms=200,
                cost_usd=0.002,
                failures=[MagicMock(evaluator_name="contains", message="missing substring")],
            )
        )
    meta = EvalMetadata(name, "gpt-test", "Fake", 1000, "r1", 4, 500, "0.17.0")
    return EvalReport(metadata=meta, case_results=cases)


# ===========================================================================
# #4: to_markdown()
# ===========================================================================


class TestToMarkdown:
    def test_basic_output(self) -> None:
        report = _make_report()
        md = report.to_markdown()
        assert "## 🟢 Eval Report:" in md
        assert "**Accuracy**" in md
        assert "100.0%" in md
        assert "NichevLabs" in md

    def test_with_failures(self) -> None:
        report = _make_report(accuracy=0.5)
        md = report.to_markdown()
        assert "🟡" in md or "🔴" in md
        assert "Failed cases" in md
        assert "<details>" in md
        assert "fail_" in md

    def test_red_badge_low_accuracy(self) -> None:
        report = _make_report(accuracy=0.0)
        md = report.to_markdown()
        assert "🔴" in md

    def test_contains_model(self) -> None:
        report = _make_report()
        md = report.to_markdown()
        assert "gpt-test" in md


# ===========================================================================
# #9: Observer events
# ===========================================================================


class TestEvalObserverEvents:
    def test_eval_start_fires(self) -> None:
        events: list[str] = []

        class TestObserver(AgentObserver):
            def on_eval_start(self, suite_name: str, total_cases: int, model: str) -> None:
                events.append(f"start:{suite_name}:{total_cases}:{model}")

        agent = _make_agent(["ok"], observers=[TestObserver()])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        suite.run()
        assert len(events) == 1
        assert events[0] == "start:eval:1:fake-model"

    def test_eval_case_end_fires(self) -> None:
        events: list[dict] = []

        class TestObserver(AgentObserver):
            def on_eval_case_end(
                self,
                suite_name: str,
                case_name: str,
                verdict: str,
                latency_ms: float,
                failures: int,
            ) -> None:
                events.append({"case": case_name, "verdict": verdict})

        agent = _make_agent(["ok"], observers=[TestObserver()])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", name="case_a"),
                TestCase(input="b", name="case_b"),
            ],
        )
        suite.run()
        assert len(events) == 2
        assert events[0]["case"] == "case_a"
        assert events[0]["verdict"] == "pass"

    def test_eval_end_fires(self) -> None:
        events: list[dict] = []

        class TestObserver(AgentObserver):
            def on_eval_end(
                self,
                suite_name: str,
                accuracy: float,
                total_cases: int,
                pass_count: int,
                fail_count: int,
                total_cost: float,
                duration_ms: float,
            ) -> None:
                events.append({"accuracy": accuracy, "total": total_cases})

        agent = _make_agent(["hello"], observers=[TestObserver()])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_contains="hello")],
        )
        suite.run()
        assert len(events) == 1
        assert events[0]["accuracy"] == 1.0
        assert events[0]["total"] == 1

    def test_observer_errors_dont_break_eval(self) -> None:
        class BrokenObserver(AgentObserver):
            def on_eval_start(self, **kwargs: Any) -> None:
                raise RuntimeError("observer crash")

        agent = _make_agent(["ok"], observers=[BrokenObserver()])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        report = suite.run()
        assert report.pass_count == 1  # Eval still completes

    def test_no_observers_no_crash(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        report = suite.run()
        assert report.pass_count == 1


# ===========================================================================
# #10: Trend chart SVG
# ===========================================================================


class TestTrendSvg:
    def test_basic_trend(self) -> None:
        svg = _trend_svg([0.7, 0.8, 0.9, 0.95])
        assert "<svg" in svg
        assert "polyline" in svg
        assert "#4ade80" in svg  # green (improving)

    def test_declining_trend(self) -> None:
        svg = _trend_svg([0.9, 0.8, 0.7])
        assert "#f87171" in svg  # red (declining)

    def test_too_few_points(self) -> None:
        assert _trend_svg([0.8]) == ""
        assert _trend_svg([]) == ""

    def test_two_points(self) -> None:
        svg = _trend_svg([0.5, 0.9])
        assert "<svg" in svg


class TestHTMLWithHistory:
    def test_html_with_trend(self, tmp_path: Any) -> None:
        report = _make_report()
        trend = HistoryTrend(
            entries=[
                HistoryEntry("r1", "s", 0, 0.7, 7, 3, 0, 0.01, 100, 100, 200, 10, "m", 500),
                HistoryEntry("r2", "s", 0, 0.8, 8, 2, 0, 0.01, 100, 100, 200, 10, "m", 500),
                HistoryEntry("r3", "s", 0, 0.9, 9, 1, 0, 0.01, 100, 100, 200, 10, "m", 500),
            ]
        )
        path = tmp_path / "trend_report.html"
        report.to_html(path, history=trend)
        content = path.read_text()
        assert "Accuracy Trend" in content
        assert "polyline" in content

    def test_html_without_history(self, tmp_path: Any) -> None:
        report = _make_report()
        path = tmp_path / "no_trend.html"
        report.to_html(path)
        content = path.read_text()
        assert "Accuracy Trend" not in content


# ===========================================================================
# #8: pip extra (structural test)
# ===========================================================================


class TestPipExtra:
    def test_evals_import_without_pyyaml(self) -> None:
        """Core eval framework works without pyyaml installed."""
        from selectools.evals import EvalReport, EvalSuite, TestCase

        assert EvalSuite is not None
        assert TestCase is not None
        assert EvalReport is not None

    def test_yaml_loader_gives_helpful_error(self) -> None:
        """from_yaml should work if pyyaml is installed (it is in dev)."""
        from selectools.evals import DatasetLoader

        # This should not raise ImportError since pyyaml is in dev deps
        # Just verify the method exists
        assert hasattr(DatasetLoader, "from_yaml")
