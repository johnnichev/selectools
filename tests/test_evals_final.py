"""Tests for templates, cost estimation, history tracking, and advanced example."""

from __future__ import annotations

import json
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.evals import (
    EvalReport,
    EvalSuite,
    HistoryEntry,
    HistoryStore,
    HistoryTrend,
    TestCase,
    code_quality_suite,
    customer_support_suite,
    rag_quality_suite,
    safety_suite,
)
from selectools.evals.types import CaseResult, CaseVerdict, EvalMetadata
from tests.conftest import SharedFakeProvider

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@tool(description="Search knowledge base")
def search(query: str) -> str:
    return f"Results for: {query}"


@tool(description="Cancel subscription")
def cancel_subscription(user_id: str) -> str:
    return f"Cancelled {user_id}"


def _make_agent(responses: list) -> Agent:
    provider = SharedFakeProvider(responses=responses)
    return Agent(
        provider=provider,
        config=AgentConfig(model="fake-model"),
        tools=[search, cancel_subscription],
    )


def _make_report(accuracy: float = 1.0, name: str = "test") -> EvalReport:
    n_pass = int(accuracy * 10)
    n_fail = 10 - n_pass
    cases = []
    for i in range(n_pass):
        tc = TestCase(input=f"pass_{i}", name=f"pass_{i}")
        cases.append(
            CaseResult(
                case=tc, verdict=CaseVerdict.PASS, latency_ms=100, cost_usd=0.001, tokens=100
            )
        )
    for i in range(n_fail):
        tc = TestCase(input=f"fail_{i}", name=f"fail_{i}")
        cases.append(
            CaseResult(
                case=tc, verdict=CaseVerdict.FAIL, latency_ms=200, cost_usd=0.002, tokens=150
            )
        )
    meta = EvalMetadata(name, "gpt-test", "Fake", 1000.0, "run1", 10, 500, "0.17.0")
    return EvalReport(metadata=meta, case_results=cases)


# ===========================================================================
# Templates
# ===========================================================================


class TestTemplates:
    def test_customer_support_defaults(self) -> None:
        agent = _make_agent(["I can help you with that."])
        suite = customer_support_suite(agent)
        assert suite.name == "customer-support"
        assert len(suite.cases) == 6
        assert any(c.name == "greeting" for c in suite.cases)
        assert any(c.name == "harmful_request" for c in suite.cases)

    def test_customer_support_custom_cases(self) -> None:
        agent = _make_agent(["ok"])
        cases = [TestCase(input="custom", name="custom_case")]
        suite = customer_support_suite(agent, cases=cases)
        assert len(suite.cases) == 1
        assert suite.cases[0].name == "custom_case"

    def test_rag_quality_defaults(self) -> None:
        agent = _make_agent(["The refund policy is..."])
        suite = rag_quality_suite(agent)
        assert suite.name == "rag-quality"
        assert len(suite.cases) == 4
        assert any(c.name == "basic_retrieval" for c in suite.cases)

    def test_safety_defaults(self) -> None:
        agent = _make_agent(["I cannot help with that."])
        suite = safety_suite(agent)
        assert suite.name == "safety"
        assert len(suite.cases) == 6
        assert any(c.name == "injection_jailbreak" for c in suite.cases)
        assert any(c.name == "benign_control" for c in suite.cases)

    def test_code_quality_defaults(self) -> None:
        agent = _make_agent(["def hello(): pass"])
        suite = code_quality_suite(agent)
        assert suite.name == "code-quality"
        assert len(suite.cases) == 4
        assert any("python" in c.tags for c in suite.cases)
        assert any("sql" in c.tags for c in suite.cases)

    def test_template_runs(self) -> None:
        agent = _make_agent(["ok"])
        suite = customer_support_suite(agent)
        report = suite.run()
        assert report.metadata.suite_name == "customer-support"
        assert report.metadata.total_cases == 6

    def test_template_with_concurrency(self) -> None:
        agent = _make_agent(["ok"])
        suite = safety_suite(agent, max_concurrency=3)
        report = suite.run()
        assert report.metadata.total_cases == 6


# ===========================================================================
# Cost Estimation
# ===========================================================================


class TestCostEstimation:
    def test_basic_estimate(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x") for _ in range(5)],
        )
        estimate = suite.estimate_cost()
        assert estimate["cases"] == 5
        assert estimate["model"] == "fake-model"
        assert "estimated_cost_usd" in estimate
        assert "estimated_input_tokens" in estimate
        assert "estimated_output_tokens" in estimate
        assert estimate["estimated_input_tokens"] == 1000  # 5 * 200
        assert estimate["estimated_output_tokens"] == 1500  # 5 * 300

    def test_estimate_with_llm_evaluators(self) -> None:
        agent = _make_agent(["ok"])
        mock_eval = MagicMock()
        mock_eval.provider = MagicMock()  # has provider = LLM evaluator
        mock_eval.name = "llm_judge"

        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x") for _ in range(3)],
            evaluators=[mock_eval],
        )
        estimate = suite.estimate_cost()
        assert estimate["llm_evaluators"] == 1

    def test_estimate_no_llm_evaluators(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        estimate = suite.estimate_cost()
        assert estimate["llm_evaluators"] == 0

    def test_estimate_unknown_model(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        estimate = suite.estimate_cost()
        assert estimate["pricing_available"] is False


# ===========================================================================
# History Tracking
# ===========================================================================


class TestHistoryStore:
    def test_record_and_trend(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        report = _make_report(accuracy=0.8, name="my-suite")
        store.record(report)

        trend = store.trend("my-suite")
        assert len(trend.entries) == 1
        assert trend.entries[0].accuracy == 0.8
        assert trend.entries[0].suite_name == "my-suite"

    def test_multiple_records(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        store.record(_make_report(accuracy=0.7, name="suite"))
        store.record(_make_report(accuracy=0.8, name="suite"))
        store.record(_make_report(accuracy=0.9, name="suite"))

        trend = store.trend("suite")
        assert len(trend.entries) == 3
        assert trend.accuracy_trend == [0.7, 0.8, 0.9]

    def test_trend_last_n(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        for acc in [0.5, 0.6, 0.7, 0.8, 0.9]:
            store.record(_make_report(accuracy=acc, name="suite"))

        trend = store.trend("suite", last_n=3)
        assert len(trend.entries) == 3
        assert trend.accuracy_trend == [0.7, 0.8, 0.9]

    def test_empty_trend(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        trend = store.trend("nonexistent")
        assert len(trend.entries) == 0
        assert trend.accuracy_delta == 0.0

    def test_list_suites(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        store.record(_make_report(name="suite-a"))
        store.record(_make_report(name="suite-b"))

        suites = store.list_suites()
        assert set(suites) == {"suite-a", "suite-b"}

    def test_list_suites_empty(self, tmp_path: Any) -> None:
        store = HistoryStore(tmp_path / "history")
        assert store.list_suites() == []


class TestHistoryTrend:
    def _entry(self, accuracy: float) -> HistoryEntry:
        return HistoryEntry(
            run_id="r",
            suite_name="s",
            timestamp=0,
            accuracy=accuracy,
            pass_count=int(accuracy * 10),
            fail_count=10 - int(accuracy * 10),
            error_count=0,
            total_cost=0.01,
            total_tokens=100,
            latency_p50=100,
            latency_p95=200,
            total_cases=10,
            model="m",
            duration_ms=500,
        )

    def test_is_improving(self) -> None:
        trend = HistoryTrend(entries=[self._entry(0.7), self._entry(0.8), self._entry(0.9)])
        assert trend.is_improving
        assert not trend.is_degrading

    def test_is_degrading(self) -> None:
        trend = HistoryTrend(entries=[self._entry(0.9), self._entry(0.8), self._entry(0.7)])
        assert trend.is_degrading
        assert not trend.is_improving

    def test_stable(self) -> None:
        trend = HistoryTrend(entries=[self._entry(0.8), self._entry(0.8), self._entry(0.8)])
        assert not trend.is_degrading

    def test_accuracy_delta(self) -> None:
        trend = HistoryTrend(entries=[self._entry(0.8), self._entry(0.9)])
        assert trend.accuracy_delta == pytest.approx(0.1)

    def test_cost_delta(self) -> None:
        e1 = self._entry(0.8)
        e2 = self._entry(0.9)
        e1.total_cost = 0.01
        e2.total_cost = 0.02
        trend = HistoryTrend(entries=[e1, e2])
        assert trend.cost_delta == pytest.approx(0.01)

    def test_summary(self) -> None:
        trend = HistoryTrend(entries=[self._entry(0.8), self._entry(0.9)])
        s = trend.summary()
        assert "2 runs" in s
        assert "90.0%" in s
        assert "improving" in s.lower() or "Trend" in s

    def test_empty_summary(self) -> None:
        trend = HistoryTrend()
        assert "No history" in trend.summary()

    def test_cost_trend(self) -> None:
        e1 = self._entry(0.8)
        e2 = self._entry(0.9)
        e1.total_cost = 0.01
        e2.total_cost = 0.02
        trend = HistoryTrend(entries=[e1, e2])
        assert trend.cost_trend == [0.01, 0.02]

    def test_latency_trend(self) -> None:
        e1 = self._entry(0.8)
        e2 = self._entry(0.9)
        e1.latency_p50 = 100
        e2.latency_p50 = 150
        trend = HistoryTrend(entries=[e1, e2])
        assert trend.latency_trend == [100, 150]


class TestHistoryStoreCorruptLine:
    """Regression tests: HistoryStore.trend() must tolerate corrupted JSONL lines."""

    def test_corrupt_line_skipped(self, tmp_path: Any) -> None:
        """Regression: json.JSONDecodeError on a corrupt JSONL line must not crash trend()."""
        store = HistoryStore(tmp_path / "hist")
        store._dir.mkdir(parents=True, exist_ok=True)
        valid_line = json.dumps(
            {
                "run_id": "abc",
                "suite_name": "s",
                "timestamp": 1.0,
                "accuracy": 0.9,
                "pass_count": 9,
                "fail_count": 1,
                "error_count": 0,
                "total_cost": 0.01,
                "total_tokens": 100,
                "latency_p50": 50.0,
                "latency_p95": 100.0,
                "total_cases": 10,
                "model": "gpt-4",
                "duration_ms": 500.0,
            }
        )
        (store._dir / "s.jsonl").write_text(valid_line + "\n" + "CORRUPT JSON\n")
        trend = store.trend("s")
        # Valid line is loaded; corrupt line is skipped
        assert len(trend.entries) == 1
        assert trend.entries[0].accuracy == 0.9

    def test_all_corrupt_returns_empty(self, tmp_path: Any) -> None:
        """All-corrupt JSONL file returns empty HistoryTrend without crashing."""
        store = HistoryStore(tmp_path / "hist")
        store._dir.mkdir(parents=True, exist_ok=True)
        (store._dir / "s.jsonl").write_text("NOT JSON\nALSO NOT JSON\n")
        trend = store.trend("s")
        assert len(trend.entries) == 0
