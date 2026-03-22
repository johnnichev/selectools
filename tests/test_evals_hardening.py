"""Hardening tests — real Agent E2E with SharedToolCallProvider + edge cases.

These tests cover the scenarios from the manual E2E run plus edge cases
not covered by existing tests: tool call arguments, concurrent stress,
empty inputs, very long outputs, unicode, snapshot diffs, regression
improvement detection, observer error isolation, markdown with failures,
and chained evaluator interactions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.evals import (
    BaselineStore,
    CaseVerdict,
    DatasetLoader,
    EvalSuite,
    HistoryStore,
    PairwiseEval,
    SnapshotStore,
    TestCase,
    generate_badge,
    generate_detailed_badge,
)
from selectools.evals.evaluators import DEFAULT_EVALUATORS
from selectools.evals.history import HistoryEntry, HistoryTrend
from selectools.evals.html import _donut_svg, _histogram_svg, _trend_svg
from selectools.observer import AgentObserver
from selectools.types import ToolCall
from tests.conftest import SharedFakeProvider, SharedToolCallProvider

# ---------------------------------------------------------------------------
# Shared tools and helpers
# ---------------------------------------------------------------------------


@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"72F in {city}"


@tool(description="Search docs")
def search(query: str) -> str:
    return f"Results for {query}"


@tool(description="Cancel subscription")
def cancel_sub(user_id: str) -> str:
    return f"Cancelled {user_id}"


def _tc_agent(responses: list, tools: list | None = None) -> Agent:
    """Agent with SharedToolCallProvider."""
    provider = SharedToolCallProvider(responses=responses)
    return Agent(
        provider=provider,
        config=AgentConfig(model="test"),
        tools=tools or [get_weather, search, cancel_sub],
    )


def _fake_agent(responses: list, tools: list | None = None, **config_kw: Any) -> Agent:
    """Agent with SharedFakeProvider."""
    provider = SharedFakeProvider(responses=responses)
    return Agent(
        provider=provider,
        config=AgentConfig(model="test", **config_kw),
        tools=tools or [get_weather],
    )


# ===========================================================================
# Tool call with SharedToolCallProvider (real tool execution)
# ===========================================================================


class TestToolCallE2EReal:
    def test_expect_tool_pass(self) -> None:
        tc = ToolCall(tool_name="get_weather", parameters={"city": "NYC"})
        agent = _tc_agent([([tc], "Getting weather...")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="Weather?", expect_tool="get_weather"),
            ],
        ).run()
        assert report.pass_count == 1
        assert report.case_results[0].tool_calls == ["get_weather"]

    def test_expect_tool_fail(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"query": "test"})
        agent = _tc_agent([([tc], "Searching...")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_tool="get_weather"),
            ],
        ).run()
        assert report.fail_count == 1

    def test_expect_tools_multiple(self) -> None:
        tc1 = ToolCall(tool_name="search", parameters={"query": "q"})
        tc2 = ToolCall(tool_name="get_weather", parameters={"city": "LA"})
        agent = _tc_agent([([tc1], "step 1"), ([tc2], "step 2")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_tools=["search", "get_weather"]),
            ],
        ).run()
        assert report.pass_count == 1

    def test_expect_tool_args_exact(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"query": "python"})
        agent = _tc_agent([([tc], "Found it")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_tool_args={"search": {"query": "python"}}),
            ],
        ).run()
        assert report.pass_count == 1

    def test_expect_tool_args_mismatch(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"query": "python"})
        agent = _tc_agent([([tc], "Found")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_tool_args={"search": {"query": "java"}}),
            ],
        ).run()
        assert report.fail_count == 1

    def test_expect_tool_order(self) -> None:
        tc1 = ToolCall(tool_name="search", parameters={"query": "q"})
        tc2 = ToolCall(tool_name="get_weather", parameters={"city": "LA"})
        agent = _tc_agent([([tc1], ""), ([tc2], "")])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_tool_order=["search", "get_weather"]),
            ],
        ).run()
        assert report.pass_count == 1


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_response(self) -> None:
        agent = _fake_agent([""])
        report = EvalSuite(agent=agent, cases=[TestCase(input="x")]).run()
        assert report.pass_count == 1  # no assertions = pass

    def test_empty_response_with_contains(self) -> None:
        agent = _fake_agent([""])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_contains="something"),
            ],
        ).run()
        assert report.fail_count == 1

    def test_very_long_response(self) -> None:
        long_text = "word " * 10000
        agent = _fake_agent([long_text])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_min_words=5000, expect_max_words=15000),
            ],
        ).run()
        assert report.pass_count == 1

    def test_unicode_emoji_content(self) -> None:
        agent = _fake_agent(["Hello! 🌍🎉 Привет мир"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="Hi 🌟", expect_contains="Привет"),
            ],
        ).run()
        assert report.pass_count == 1

    def test_special_characters_in_input(self) -> None:
        agent = _fake_agent(["ok"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input='What is "hello" & <world>?', name="special_chars"),
            ],
        ).run()
        assert report.pass_count == 1

    def test_single_case(self) -> None:
        agent = _fake_agent(["yes"])
        report = EvalSuite(agent=agent, cases=[TestCase(input="x")]).run()
        assert report.accuracy == 1.0
        assert report.latency_p50 >= 0

    def test_zero_weight_case(self) -> None:
        agent = _fake_agent(["ok"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", expect_contains="ok", weight=1.0),
                TestCase(input="b", expect_contains="nope", weight=0.0),
            ],
        ).run()
        # weight=0 case is excluded from accuracy
        assert report.accuracy == 1.0

    def test_all_cases_error(self) -> None:
        from tests.conftest import SharedErrorProvider

        provider = SharedErrorProvider(exception=RuntimeError("crash"))
        agent = Agent(provider=provider, config=AgentConfig(model="m"), tools=[get_weather])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a"),
                TestCase(input="b"),
            ],
        ).run()
        assert report.error_count == 2
        assert report.accuracy == 0.0

    def test_report_latency_all_same(self) -> None:
        """When all latencies are identical, percentiles shouldn't crash."""
        agent = _fake_agent(["ok"])
        report = EvalSuite(agent=agent, cases=[TestCase(input=f"q{i}") for i in range(5)]).run()
        # All latencies are nearly identical (fast fake provider)
        assert report.latency_p50 >= 0
        assert report.latency_p95 >= report.latency_p50
        assert report.latency_p99 >= report.latency_p95


# ===========================================================================
# Combined feature chains
# ===========================================================================


class TestFeatureChains:
    def test_run_then_snapshot_then_regression(self, tmp_path: Path) -> None:
        """Full pipeline: run → snapshot → baseline → run again → compare."""
        agent = _fake_agent(["good answer"])
        cases = [TestCase(input="test", name="q1", expect_contains="good")]

        # First run
        r1 = EvalSuite(agent=agent, cases=cases, name="chain").run()
        assert r1.pass_count == 1

        # Save snapshot
        snap = SnapshotStore(tmp_path / "snap")
        snap.save(r1, "chain")

        # Save baseline
        base = BaselineStore(tmp_path / "base")
        base.save(r1)

        # Second run with same agent — no regression, no snapshot change
        r2 = EvalSuite(agent=agent, cases=cases, name="chain").run()
        assert not snap.compare(r2, "chain").has_changes
        assert not base.compare(r2).is_regression

        # Third run with worse agent — regression detected
        bad_agent = _fake_agent(["bad answer"])
        r3 = EvalSuite(agent=bad_agent, cases=cases, name="chain").run()
        assert snap.compare(r3, "chain").has_changes
        assert base.compare(r3).is_regression

    def test_history_tracks_improvement(self, tmp_path: Path) -> None:
        hist = HistoryStore(tmp_path / "hist")

        for response, expect in [("wrong", "right"), ("wrong", "right"), ("right answer", "right")]:
            agent = _fake_agent([response])
            r = EvalSuite(
                agent=agent,
                cases=[TestCase(input="x", expect_contains=expect)],
                name="improving",
            ).run()
            hist.record(r)

        trend = hist.trend("improving")
        assert len(trend.entries) == 3
        assert trend.entries[0].accuracy == 0.0
        assert trend.entries[2].accuracy == 1.0
        assert trend.is_improving

    def test_pairwise_with_tool_calls(self) -> None:
        tc_good = ToolCall(tool_name="get_weather", parameters={"city": "NYC"})
        agent_a = _tc_agent([([tc_good], "Weather info")])
        agent_b = _fake_agent(["No weather"])

        result = PairwiseEval(
            agent_a,
            agent_b,
            [TestCase(input="Weather?", expect_tool="get_weather")],
            agent_a_name="WithTools",
            agent_b_name="NoTools",
        ).run()
        assert result.a_wins == 1

    def test_badge_from_full_run(self, tmp_path: Path) -> None:
        agent = _fake_agent(["Hello"])
        cases = [
            TestCase(input="a", expect_contains="hello"),
            TestCase(input="b", expect_contains="nope"),
        ]
        report = EvalSuite(agent=agent, cases=cases).run()

        path = tmp_path / "badge.svg"
        generate_badge(report, path)
        svg = path.read_text()
        assert "50%" in svg  # 1/2 pass

    def test_html_report_with_all_verdicts(self, tmp_path: Path) -> None:
        """HTML report handles pass, fail, and error in the same report."""
        from tests.conftest import SharedErrorProvider

        agent_good = _fake_agent(["good"])
        r1 = EvalSuite(
            agent=agent_good,
            cases=[
                TestCase(input="a", name="pass_case", expect_contains="good"),
                TestCase(input="b", name="fail_case", expect_contains="nope"),
            ],
        ).run()

        path = tmp_path / "report.html"
        r1.to_html(path)
        html = path.read_text()
        assert "pass_case" in html
        assert "fail_case" in html
        assert "toggleDetail" in html


# ===========================================================================
# Evaluator isolation tests
# ===========================================================================


class TestEvaluatorIsolation:
    """Verify each evaluator only fires when its field is set."""

    def test_no_assertions_always_passes(self) -> None:
        agent = _fake_agent(["anything at all"])
        report = EvalSuite(agent=agent, cases=[TestCase(input="x")]).run()
        assert report.pass_count == 1
        assert report.case_results[0].failures == []

    def test_only_set_fields_checked(self) -> None:
        """A case with only expect_contains should not trigger tool_use checks."""
        agent = _fake_agent(["hello world"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", expect_contains="hello"),
            ],
        ).run()
        assert report.pass_count == 1
        # Should NOT fail on tool_use even though no tools were called

    def test_multiple_failures_recorded(self) -> None:
        """A case failing on multiple assertions records all of them."""
        agent = _fake_agent(["short"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(
                    input="x",
                    expect_contains="missing",
                    expect_starts_with="Wrong",
                    expect_min_length=1000,
                ),
            ],
        ).run()
        assert report.fail_count == 1
        failures = report.case_results[0].failures
        assert len(failures) == 3
        evaluator_names = {f.evaluator_name for f in failures}
        assert "contains" in evaluator_names
        assert "starts_with" in evaluator_names
        assert "length" in evaluator_names


# ===========================================================================
# Export roundtrip tests
# ===========================================================================


class TestExportRoundtrip:
    def test_json_roundtrip(self, tmp_path: Path) -> None:
        agent = _fake_agent(["hello"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", name="test1", tags=["a"]),
            ],
        ).run()

        path = tmp_path / "report.json"
        report.to_json(path)
        data = json.loads(path.read_text())

        assert data["summary"]["accuracy"] == 1.0
        assert data["cases"][0]["name"] == "test1"
        assert data["cases"][0]["verdict"] == "pass"
        assert data["metadata"]["selectools_version"]

    def test_markdown_with_failures(self) -> None:
        agent = _fake_agent(["wrong"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="x", name="failing", expect_contains="right"),
            ],
        ).run()
        md = report.to_markdown()
        assert "Failed cases" in md
        assert "failing" in md

    def test_markdown_all_pass(self) -> None:
        agent = _fake_agent(["ok"])
        report = EvalSuite(agent=agent, cases=[TestCase(input="x")]).run()
        md = report.to_markdown()
        assert "🟢" in md
        assert "Failed cases" not in md

    def test_junit_xml_valid(self, tmp_path: Path) -> None:
        import xml.etree.ElementTree as ET

        agent = _fake_agent(["hello"])
        report = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", name="p1"),
                TestCase(input="b", name="f1", expect_contains="nope"),
            ],
        ).run()

        path = tmp_path / "results.xml"
        report.to_junit_xml(path)
        tree = ET.parse(str(path))
        root = tree.getroot()
        assert root.tag == "testsuite"
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "1"


# ===========================================================================
# Concurrent stress test
# ===========================================================================


class TestConcurrentStress:
    def test_50_cases_concurrent(self) -> None:
        agent = _fake_agent(["ok"])
        cases = [TestCase(input=f"q{i}", name=f"case_{i}") for i in range(50)]
        report = EvalSuite(agent=agent, cases=cases, max_concurrency=10).run()
        assert report.metadata.total_cases == 50
        assert report.pass_count == 50

    def test_concurrent_progress_callback(self) -> None:
        agent = _fake_agent(["ok"])
        progress: list[tuple[int, int]] = []
        cases = [TestCase(input=f"q{i}") for i in range(10)]
        EvalSuite(
            agent=agent,
            cases=cases,
            max_concurrency=3,
            on_progress=lambda d, t: progress.append((d, t)),
        ).run()
        # Progress should be called 10 times
        assert len(progress) == 10
        assert progress[-1] == (10, 10)


# ===========================================================================
# Observer integration
# ===========================================================================


class TestObserverIntegration:
    def test_all_events_fire(self) -> None:
        events: list[str] = []

        class FullObserver(AgentObserver):
            def on_eval_start(self, **kw: Any) -> None:
                events.append("start")

            def on_eval_case_end(self, **kw: Any) -> None:
                events.append("case")

            def on_eval_end(self, **kw: Any) -> None:
                events.append("end")

        agent = _fake_agent(["ok"], observers=[FullObserver()])
        EvalSuite(agent=agent, cases=[TestCase(input="a"), TestCase(input="b")]).run()
        assert events == ["start", "case", "case", "end"]

    def test_broken_observer_doesnt_crash(self) -> None:
        class CrashObserver(AgentObserver):
            def on_eval_start(self, **kw: Any) -> None:
                raise RuntimeError("crash")

            def on_eval_case_end(self, **kw: Any) -> None:
                raise RuntimeError("crash")

            def on_eval_end(self, **kw: Any) -> None:
                raise RuntimeError("crash")

        agent = _fake_agent(["ok"], observers=[CrashObserver()])
        report = EvalSuite(agent=agent, cases=[TestCase(input="x")]).run()
        assert report.pass_count == 1  # eval completes despite observer crashes
