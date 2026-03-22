"""E2E tests for the selectools eval framework.

Tests every feature end-to-end using real Agent instances with SharedFakeProvider
and SharedToolCallProvider — no mocks. Covers all gaps identified in the coverage audit.
"""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.evals import (
    BaselineStore,
    CaseResult,
    CaseVerdict,
    DatasetLoader,
    EvalReport,
    EvalSuite,
    PairwiseEval,
    SnapshotStore,
    TestCase,
    generate_badge,
    generate_cases,
    generate_detailed_badge,
    serve_eval,
)
from selectools.evals.html import _donut_svg, _histogram_svg, render_html_report
from selectools.evals.junit import render_junit_xml
from selectools.evals.llm_evaluators import (
    BiasEvaluator,
    CoherenceEvaluator,
    CompletenessEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
    LLMJudgeEvaluator,
    RelevanceEvaluator,
    SummaryEvaluator,
    ToxicityEvaluator,
)
from selectools.evals.pairwise import PairwiseReport
from selectools.evals.snapshot import SnapshotDiff, SnapshotResult
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# Import shared test providers from conftest
from tests.conftest import SharedFakeProvider, SharedToolCallProvider

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@tool(description="Get the weather for a city")
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"


@tool(description="Search the knowledge base")
def search_docs(query: str) -> str:
    return f"Results for: {query}"


@tool(description="Cancel a subscription")
def cancel_sub(user_id: str) -> str:
    return f"Subscription {user_id} cancelled"


def _make_agent(responses: list, tools: list | None = None) -> Agent:
    """Create a real Agent with SharedFakeProvider."""
    provider = SharedFakeProvider(responses=responses)
    return Agent(
        provider=provider,
        config=AgentConfig(model="fake-model"),
        tools=tools or [get_weather, search_docs, cancel_sub],
    )


def _make_tool_call_agent(
    responses: list[tuple[list[ToolCall], str]], tools: list | None = None
) -> Agent:
    """Create an Agent with SharedToolCallProvider for tool call scenarios."""
    provider = SharedToolCallProvider(responses=responses)
    return Agent(
        provider=provider,
        config=AgentConfig(model="fake-model"),
        tools=tools or [get_weather, search_docs, cancel_sub],
    )


# ===========================================================================
# E2E: EvalSuite with real Agent
# ===========================================================================


class TestEvalSuiteE2E:
    """Full end-to-end eval suite execution with real Agent instances."""

    def test_basic_run(self) -> None:
        agent = _make_agent(["The weather in NYC is great."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Weather in NYC?", expect_contains="weather")],
        )
        report = suite.run()
        assert report.accuracy == 1.0
        assert report.pass_count == 1
        assert report.metadata.model == "fake-model"

    def test_tool_call_eval(self) -> None:
        tc = ToolCall(tool_name="get_weather", parameters={"city": "NYC"})
        agent = _make_tool_call_agent([([tc], "Getting weather...")])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Weather?", expect_tool="get_weather")],
        )
        report = suite.run()
        assert report.pass_count == 1

    def test_multiple_cases_mixed_results(self) -> None:
        agent = _make_agent(["Hello world"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="Say hello", expect_contains="hello"),
                TestCase(input="Say goodbye", expect_contains="goodbye"),
            ],
        )
        report = suite.run()
        assert report.pass_count == 1
        assert report.fail_count == 1
        assert report.accuracy == 0.5

    def test_empty_cases(self) -> None:
        agent = _make_agent(["hi"])
        suite = EvalSuite(agent=agent, cases=[])
        report = suite.run()
        assert report.accuracy == 0.0
        assert report.metadata.total_cases == 0

    def test_concurrent_execution(self) -> None:
        agent = _make_agent(["response"])
        cases = [TestCase(input=f"q{i}") for i in range(10)]
        suite = EvalSuite(agent=agent, cases=cases, max_concurrency=4)
        report = suite.run()
        assert report.metadata.total_cases == 10
        assert report.pass_count == 10

    def test_progress_callback(self) -> None:
        agent = _make_agent(["ok"])
        progress: list[tuple[int, int]] = []
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="a"), TestCase(input="b"), TestCase(input="c")],
            on_progress=lambda d, t: progress.append((d, t)),
        )
        suite.run()
        assert progress == [(1, 3), (2, 3), (3, 3)]

    def test_error_case(self) -> None:
        """Agent that raises an exception produces ERROR verdict."""
        from tests.conftest import SharedErrorProvider

        provider = SharedErrorProvider(exception=RuntimeError("API down"))
        agent = Agent(
            provider=provider,
            config=AgentConfig(model="fake"),
            tools=[get_weather],
        )
        suite = EvalSuite(agent=agent, cases=[TestCase(input="crash")])
        report = suite.run()
        assert report.error_count == 1
        assert report.accuracy == 0.0
        assert "API down" in report.case_results[0].error

    def test_suite_name_and_tags(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x")],
            name="prod-suite",
            tags={"env": "staging", "version": "2.0"},
        )
        report = suite.run()
        assert report.metadata.suite_name == "prod-suite"
        assert report.metadata.tags == {"env": "staging", "version": "2.0"}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_run(self) -> None:
        agent = _make_agent(["async response"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="async?", expect_contains="async")],
        )
        report = await suite.arun()
        assert report.pass_count == 1

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_concurrent(self) -> None:
        agent = _make_agent(["ok"])
        cases = [TestCase(input=f"q{i}") for i in range(5)]
        suite = EvalSuite(agent=agent, cases=cases, max_concurrency=3)
        report = await suite.arun()
        assert report.metadata.total_cases == 5


# ===========================================================================
# E2E: All deterministic evaluators with real Agent
# ===========================================================================


class TestDeterministicEvaluatorsE2E:
    def test_expect_contains_pass(self) -> None:
        agent = _make_agent(["The balance is $500."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Balance?", expect_contains="balance")],
        )
        assert suite.run().pass_count == 1

    def test_expect_not_contains_pass(self) -> None:
        agent = _make_agent(["Everything is fine."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Status?", expect_not_contains="error")],
        )
        assert suite.run().pass_count == 1

    def test_expect_output_exact(self) -> None:
        agent = _make_agent(["42"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="2+2?", expect_output="42")],
        )
        assert suite.run().pass_count == 1

    def test_expect_output_regex(self) -> None:
        agent = _make_agent(["Call 555-1234 for support"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Phone?", expect_output_regex=r"\d{3}-\d{4}")],
        )
        assert suite.run().pass_count == 1

    def test_expect_json(self) -> None:
        agent = _make_agent(['{"key": "value"}'])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="JSON?", expect_json=True)],
        )
        assert suite.run().pass_count == 1

    def test_expect_json_fail(self) -> None:
        agent = _make_agent(["not json"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="JSON?", expect_json=True)],
        )
        assert suite.run().fail_count == 1

    def test_expect_length(self) -> None:
        agent = _make_agent(["Hello world, this is a test"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_min_length=5, expect_max_length=100)],
        )
        assert suite.run().pass_count == 1

    def test_expect_starts_with(self) -> None:
        agent = _make_agent(["Hello, how can I help?"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_starts_with="Hello")],
        )
        assert suite.run().pass_count == 1

    def test_expect_ends_with(self) -> None:
        agent = _make_agent(["Here is your answer."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_ends_with=".")],
        )
        assert suite.run().pass_count == 1

    def test_expect_no_pii_pass(self) -> None:
        agent = _make_agent(["Your account is active."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Account?", expect_no_pii=True)],
        )
        assert suite.run().pass_count == 1

    def test_expect_no_pii_fail(self) -> None:
        agent = _make_agent(["Your SSN is 123-45-6789"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="SSN?", expect_no_pii=True)],
        )
        report = suite.run()
        assert report.fail_count == 1

    def test_expect_no_injection_pass(self) -> None:
        agent = _make_agent(["Here is your answer."])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_no_injection=True)],
        )
        assert suite.run().pass_count == 1

    def test_expect_iterations(self) -> None:
        agent = _make_agent(["done"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_iterations_lte=5)],
        )
        assert suite.run().pass_count == 1

    def test_expect_latency(self) -> None:
        agent = _make_agent(["fast"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_latency_ms_lte=10000)],
        )
        assert suite.run().pass_count == 1

    def test_expect_cost(self) -> None:
        agent = _make_agent(["cheap"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_cost_usd_lte=1.0)],
        )
        assert suite.run().pass_count == 1

    def test_custom_evaluator(self) -> None:
        agent = _make_agent(["Please help me."])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(
                    input="x",
                    custom_evaluator=lambda r: "please" in r.content.lower(),
                    custom_evaluator_name="politeness",
                )
            ],
        )
        assert suite.run().pass_count == 1

    def test_weighted_accuracy(self) -> None:
        agent = _make_agent(["hello"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", expect_contains="hello", weight=3.0),
                TestCase(input="b", expect_contains="goodbye", weight=1.0),
            ],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(0.75)

    def test_multiple_assertions_same_case(self) -> None:
        agent = _make_agent(["Hello world"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(
                    input="x",
                    expect_contains="hello",
                    expect_starts_with="Hello",
                    expect_min_length=5,
                    expect_max_length=100,
                    expect_no_pii=True,
                    expect_no_injection=True,
                )
            ],
        )
        assert suite.run().pass_count == 1

    def test_unicode_content(self) -> None:
        agent = _make_agent(["Bonjour le monde! 🌍"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Salut!", expect_contains="bonjour")],
        )
        assert suite.run().pass_count == 1

    def test_tags_filtering(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", tags=["billing"]),
                TestCase(input="b", tags=["support"]),
                TestCase(input="c", tags=["billing", "support"]),
            ],
        )
        report = suite.run()
        billing = report.filter_by_tag("billing")
        assert len(billing) == 2
        support = report.filter_by_tag("support")
        assert len(support) == 2


# ===========================================================================
# E2E: HTML report rendering
# ===========================================================================


class TestHTMLReportE2E:
    def test_full_html_report(self, tmp_path: Path) -> None:
        agent = _make_agent(["Balance is $500", "Error occurred"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(
                    input="Balance?", name="balance", expect_contains="balance", tags=["billing"]
                ),
                TestCase(input="Fail?", name="fail_case", expect_contains="success"),
            ],
        )
        report = suite.run()
        path = tmp_path / "report.html"
        report.to_html(path)

        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "balance" in content
        assert "fail_case" in content
        assert "NichevLabs" in content
        assert "toggleDetail" in content  # JS function
        assert "filterByTag" in content  # JS function
        assert "filterByVerdict" in content  # JS function
        assert "billing" in content  # tag pill

    def test_donut_svg(self) -> None:
        svg = _donut_svg(8, 2, 1, 0)
        assert "<svg" in svg
        assert "#4ade80" in svg  # green for pass
        assert "#f87171" in svg  # red for fail

    def test_donut_svg_empty(self) -> None:
        assert _donut_svg(0, 0, 0, 0) == ""

    def test_histogram_svg(self) -> None:
        svg = _histogram_svg([100, 200, 150, 300, 250, 180])
        assert "<svg" in svg
        assert "rect" in svg
        assert "Latency Distribution" in svg

    def test_histogram_svg_empty(self) -> None:
        assert _histogram_svg([]) == ""

    def test_histogram_svg_single_value(self) -> None:
        svg = _histogram_svg([100.0])
        assert "<svg" in svg

    def test_html_with_error_cases(self, tmp_path: Path) -> None:
        from tests.conftest import SharedErrorProvider

        provider = SharedErrorProvider(exception=RuntimeError("boom"))
        agent = Agent(provider=provider, config=AgentConfig(model="m"), tools=[get_weather])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="crash", name="error_test")])
        report = suite.run()
        path = tmp_path / "error_report.html"
        report.to_html(path)
        content = path.read_text()
        assert "error_test" in content
        assert "boom" in content


# ===========================================================================
# E2E: JUnit XML
# ===========================================================================


class TestJUnitXMLE2E:
    def test_junit_xml_structure(self, tmp_path: Path) -> None:
        agent = _make_agent(["Hello world"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="pass", name="passing_test", expect_contains="hello"),
                TestCase(input="fail", name="failing_test", expect_contains="goodbye"),
            ],
        )
        report = suite.run()
        path = tmp_path / "results.xml"
        report.to_junit_xml(path)

        tree = ET.parse(str(path))
        root = tree.getroot()
        assert root.tag == "testsuite"
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "1"

        testcases = root.findall("testcase")
        assert len(testcases) == 2

        # Check passing test has no failure element
        passing = [tc for tc in testcases if tc.attrib["name"] == "passing_test"][0]
        assert passing.find("failure") is None

        # Check failing test has failure element
        failing = [tc for tc in testcases if tc.attrib["name"] == "failing_test"][0]
        assert failing.find("failure") is not None

    def test_junit_with_errors(self, tmp_path: Path) -> None:
        from tests.conftest import SharedErrorProvider

        provider = SharedErrorProvider(exception=ValueError("broken"))
        agent = Agent(provider=provider, config=AgentConfig(model="m"), tools=[get_weather])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x", name="error_case")])
        report = suite.run()
        path = tmp_path / "errors.xml"
        report.to_junit_xml(path)

        tree = ET.parse(str(path))
        root = tree.getroot()
        assert root.attrib["errors"] == "1"
        error_el = root.find(".//error")
        assert error_el is not None
        assert "broken" in (error_el.attrib.get("message", ""))


# ===========================================================================
# E2E: Dataset loading → Suite → Report → Export
# ===========================================================================


class TestDatasetToReportPipeline:
    def test_json_to_html(self, tmp_path: Path) -> None:
        cases_data = [
            {"input": "Weather?", "expect_contains": "weather", "name": "weather"},
            {"input": "Hello", "expect_starts_with": "H", "name": "greeting", "tags": ["basic"]},
        ]
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases_data))

        cases = DatasetLoader.load(cases_path)
        assert len(cases) == 2
        assert cases[0].name == "weather"
        assert cases[1].tags == ["basic"]

        agent = _make_agent(["The weather is great", "Hi there"])
        suite = EvalSuite(agent=agent, cases=cases, name="pipeline-test")
        report = suite.run()

        # Export all formats
        report.to_html(tmp_path / "report.html")
        report.to_junit_xml(tmp_path / "results.xml")
        report.to_json(tmp_path / "results.json")

        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "results.xml").exists()
        assert (tmp_path / "results.json").exists()

        # Verify JSON report
        data = json.loads((tmp_path / "results.json").read_text())
        assert data["metadata"]["suite_name"] == "pipeline-test"
        assert len(data["cases"]) == 2


# ===========================================================================
# E2E: Regression detection
# ===========================================================================


class TestRegressionE2E:
    def test_baseline_save_compare_no_regression(self, tmp_path: Path) -> None:
        agent = _make_agent(["good response"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", name="stable", expect_contains="good")],
            name="reg-test",
        )
        report = suite.run()

        store = BaselineStore(tmp_path / "baselines")
        store.save(report)

        result = store.compare(report)
        assert not result.is_regression

    def test_regression_detected(self, tmp_path: Path) -> None:
        # Baseline: passing
        agent_good = _make_agent(["good response"])
        suite1 = EvalSuite(
            agent=agent_good,
            cases=[TestCase(input="x", name="check", expect_contains="good")],
            name="reg-test",
        )
        report1 = suite1.run()

        store = BaselineStore(tmp_path / "baselines")
        store.save(report1)

        # Current: failing
        agent_bad = _make_agent(["bad response"])
        suite2 = EvalSuite(
            agent=agent_bad,
            cases=[TestCase(input="x", name="check", expect_contains="good")],
            name="reg-test",
        )
        report2 = suite2.run()

        result = store.compare(report2)
        assert result.is_regression
        assert "check" in result.regressions


# ===========================================================================
# E2E: Pairwise A/B comparison
# ===========================================================================


class TestPairwiseE2E:
    def test_agent_a_wins(self) -> None:
        agent_a = _make_agent(["good answer with details"])
        agent_b = _make_agent(["wrong"])
        cases = [TestCase(input="Explain?", expect_contains="good")]

        pairwise = PairwiseEval(agent_a, agent_b, cases, agent_a_name="Good", agent_b_name="Bad")
        result = pairwise.run()
        assert result.a_wins == 1
        assert result.b_wins == 0
        assert result.winner == "A"
        assert result.report_a.accuracy == 1.0
        assert result.report_b.accuracy == 0.0

    def test_both_pass(self) -> None:
        """When both agents pass, result depends on latency — any outcome is valid."""
        agent_a = _make_agent(["same answer"])
        agent_b = _make_agent(["same answer"])
        cases = [TestCase(input="Test", expect_contains="same")]

        result = PairwiseEval(agent_a, agent_b, cases).run()
        # Both passed, so winner depends on latency difference
        assert result.a_wins + result.b_wins + result.ties == 1

    def test_pairwise_summary(self) -> None:
        agent_a = _make_agent(["win"])
        agent_b = _make_agent(["lose"])
        cases = [TestCase(input="x", expect_contains="win")]

        result = PairwiseEval(
            agent_a, agent_b, cases, agent_a_name="Fast", agent_b_name="Slow"
        ).run()
        s = result.summary()
        assert "Fast" in s
        assert "1 wins" in s


# ===========================================================================
# E2E: Snapshot testing
# ===========================================================================


class TestSnapshotE2E:
    def test_first_run_creates_snapshot(self, tmp_path: Path) -> None:
        agent = _make_agent(["Hello world"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="Hi", name="greeting")],
        )
        report = suite.run()

        store = SnapshotStore(tmp_path / "snapshots")
        result = store.compare(report, "test-suite")
        assert result.has_changes  # All new
        assert "greeting" in result.new_cases

        store.save(report, "test-suite")
        result2 = store.compare(report, "test-suite")
        assert not result2.has_changes
        assert "greeting" in result2.unchanged

    def test_detect_output_change(self, tmp_path: Path) -> None:
        agent1 = _make_agent(["Response A"])
        suite1 = EvalSuite(agent=agent1, cases=[TestCase(input="x", name="test")])
        report1 = suite1.run()

        store = SnapshotStore(tmp_path / "snapshots")
        store.save(report1, "changing")

        agent2 = _make_agent(["Response B"])
        suite2 = EvalSuite(agent=agent2, cases=[TestCase(input="x", name="test")])
        report2 = suite2.run()

        result = store.compare(report2, "changing")
        assert result.has_changes
        assert result.changed_count == 1
        content_diffs = [d for d in result.diffs if d.field == "content"]
        assert len(content_diffs) == 1
        assert content_diffs[0].expected == "Response A"
        assert content_diffs[0].actual == "Response B"


# ===========================================================================
# E2E: Badge generation
# ===========================================================================


class TestBadgeE2E:
    def test_badge_from_real_run(self, tmp_path: Path) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="a"), TestCase(input="b")],
        )
        report = suite.run()

        path = tmp_path / "badge.svg"
        generate_badge(report, path)

        content = path.read_text()
        assert "<svg" in content
        assert "100%" in content
        assert "#4ade80" in content  # green

    def test_detailed_badge(self, tmp_path: Path) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="a"), TestCase(input="b", expect_contains="nope")],
        )
        report = suite.run()

        path = tmp_path / "badge_detail.svg"
        generate_detailed_badge(report, path)
        content = path.read_text()
        assert "1/2 pass" in content

    def test_badge_colors(self, tmp_path: Path) -> None:
        """Test different accuracy levels produce different colors."""
        from selectools.evals.badge import _badge_color

        assert _badge_color(1.0) == "#4ade80"  # green
        assert _badge_color(0.92) == "#22d3ee"  # cyan
        assert _badge_color(0.85) == "#3b82f6"  # blue
        assert _badge_color(0.75) == "#fbbf24"  # yellow
        assert _badge_color(0.6) == "#f97316"  # orange
        assert _badge_color(0.3) == "#f87171"  # red


# ===========================================================================
# E2E: Report statistics
# ===========================================================================


class TestReportStatsE2E:
    def test_all_latency_percentiles(self) -> None:
        agent = _make_agent(["ok"])
        cases = [TestCase(input=f"q{i}") for i in range(20)]
        suite = EvalSuite(agent=agent, cases=cases)
        report = suite.run()

        assert report.latency_p50 > 0
        assert report.latency_p95 >= report.latency_p50
        assert report.latency_p99 >= report.latency_p95
        assert report.latency_mean > 0

    def test_cost_tracking(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        report = suite.run()
        assert report.total_cost >= 0
        assert report.cost_per_case >= 0
        assert report.total_tokens >= 0

    def test_report_summary_string(self) -> None:
        agent = _make_agent(["hello"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x", expect_contains="hello")],
            name="summary-test",
        )
        report = suite.run()
        s = report.summary()
        assert "summary-test" in s
        assert "100.0%" in s
        assert "1 pass" in s

    def test_report_repr(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x")])
        report = suite.run()
        r = repr(report)
        assert "EvalReport" in r
        assert "100.00%" in r

    def test_failures_by_evaluator(self) -> None:
        agent = _make_agent(["hello"])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="a", expect_contains="goodbye"),
                TestCase(input="b", expect_starts_with="Z"),
                TestCase(input="c", expect_contains="nope"),
            ],
        )
        report = suite.run()
        by_eval = report.failures_by_evaluator()
        assert "contains" in by_eval
        assert by_eval["contains"] == 2

    def test_to_dict_roundtrip(self) -> None:
        agent = _make_agent(["ok"])
        suite = EvalSuite(agent=agent, cases=[TestCase(input="x", name="test1")])
        report = suite.run()
        d = report.to_dict()
        assert d["metadata"]["suite_name"] == "eval"
        assert d["cases"][0]["name"] == "test1"
        assert d["summary"]["accuracy"] == 1.0


# ===========================================================================
# E2E: Synthetic test generation
# ===========================================================================


class TestGeneratorE2E:
    def test_generate_from_tools(self) -> None:
        provider = SharedFakeProvider(
            responses=[
                json.dumps(
                    [
                        {
                            "input": "What is the weather?",
                            "name": "weather_q",
                            "expect_tool": "get_weather",
                        },
                        {
                            "input": "Search for Python",
                            "name": "search_q",
                            "expect_tool": "search_docs",
                        },
                    ]
                )
            ]
        )
        cases = generate_cases(provider, "fake-model", [get_weather, search_docs], n=2)
        assert len(cases) == 2
        assert cases[0].name == "weather_q"
        assert cases[0].expect_tool == "get_weather"


# ===========================================================================
# E2E: LLM evaluators with real Agent + judge provider
# ===========================================================================


class TestLLMEvaluatorsE2E:
    def _run_with_llm_eval(self, evaluator: Any) -> EvalReport:
        agent = _make_agent(["Paris is the capital of France."])
        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(
                    input="What is the capital of France?",
                    reference="Paris is the capital of France.",
                    context="France is a country in Europe. Its capital is Paris.",
                    rubric="Check if the answer is factually correct.",
                )
            ],
            evaluators=[evaluator],
        )
        return suite.run()

    def _make_judge(self, score: int) -> SharedFakeProvider:
        return SharedFakeProvider(responses=[f"Analysis complete. Score: {score}"])

    def test_correctness_pass(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(CorrectnessEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_correctness_fail(self) -> None:
        judge = self._make_judge(3)
        report = self._run_with_llm_eval(CorrectnessEvaluator(judge, "fake"))
        assert report.fail_count == 1

    def test_relevance(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(RelevanceEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_faithfulness(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(FaithfulnessEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_hallucination(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(HallucinationEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_toxicity(self) -> None:
        judge = self._make_judge(10)
        report = self._run_with_llm_eval(ToxicityEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_coherence(self) -> None:
        judge = self._make_judge(8)
        report = self._run_with_llm_eval(CoherenceEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_completeness(self) -> None:
        judge = self._make_judge(8)
        report = self._run_with_llm_eval(CompletenessEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_bias(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(BiasEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_summary_eval(self) -> None:
        judge = self._make_judge(8)
        report = self._run_with_llm_eval(SummaryEvaluator(judge, "fake"))
        assert report.pass_count == 1

    def test_llm_judge_custom_rubric(self) -> None:
        judge = self._make_judge(9)
        report = self._run_with_llm_eval(
            LLMJudgeEvaluator(judge, "fake", default_rubric="Be accurate", threshold=7.0)
        )
        assert report.pass_count == 1

    def test_llm_evaluator_skip_without_fields(self) -> None:
        """LLM evaluators that require reference/context skip if not provided."""
        judge = self._make_judge(10)
        agent = _make_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            cases=[TestCase(input="x")],  # no reference or context
            evaluators=[
                CorrectnessEvaluator(judge, "fake"),
                FaithfulnessEvaluator(judge, "fake"),
                HallucinationEvaluator(judge, "fake"),
                SummaryEvaluator(judge, "fake"),
            ],
        )
        report = suite.run()
        assert report.pass_count == 1  # all skip → pass


# ===========================================================================
# E2E: CLI (__main__)
# ===========================================================================


class TestCLIE2E:
    def test_cli_help(self) -> None:
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "selectools.evals", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "compare" in result.stdout

    def test_cli_run_help(self) -> None:
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "selectools.evals", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--html" in result.stdout
        assert "--junit" in result.stdout
        assert "--provider" in result.stdout


# ===========================================================================
# E2E: Serve dashboard (unit-level — no HTTP server)
# ===========================================================================


class TestServeDashboard:
    def test_dashboard_html_has_all_elements(self) -> None:
        from selectools.evals.serve import _DASHBOARD_HTML

        assert "<!DOCTYPE html>" in _DASHBOARD_HTML
        assert "Live Dashboard" in _DASHBOARD_HTML
        assert "/api/state" in _DASHBOARD_HTML
        assert "NichevLabs" in _DASHBOARD_HTML
        assert "progress-bar" in _DASHBOARD_HTML
        assert "accuracy" in _DASHBOARD_HTML
        assert "poll()" in _DASHBOARD_HTML

    def test_dashboard_handler_state(self) -> None:
        from selectools.evals.serve import _DashboardHandler

        state = {"status": "running", "completed": 5, "total_cases": 10}
        _DashboardHandler.dashboard_state = state
        assert _DashboardHandler.dashboard_state["status"] == "running"
