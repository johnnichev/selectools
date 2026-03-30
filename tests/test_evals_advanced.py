"""Tests for advanced eval features: pairwise, generator, badge, snapshot, serve."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from selectools.evals import CaseResult, CaseVerdict, EvalMetadata, EvalReport, EvalSuite, TestCase
from selectools.evals.badge import generate_badge, generate_detailed_badge
from selectools.evals.generator import _parse_generated_cases, generate_cases
from selectools.evals.pairwise import PairwiseCaseResult, PairwiseEval, PairwiseReport
from selectools.evals.snapshot import SnapshotDiff, SnapshotResult, SnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(
    content: str = "hello",
    tool_calls: Optional[List[Any]] = None,
    iterations: int = 1,
    cost_usd: float = 0.001,
    total_tokens: int = 100,
) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.tool_calls = tool_calls or []
    result.iterations = iterations
    result.parsed = None
    result.reasoning = None
    usage = MagicMock()
    usage.total_cost_usd = cost_usd
    usage.total_tokens = total_tokens
    result.usage = usage
    return result


def _make_mock_agent(responses: List[Any]) -> MagicMock:
    agent = MagicMock()
    agent._model = "gpt-test"
    agent.provider = MagicMock()
    type(agent.provider).__name__ = "MockProvider"
    clone = MagicMock()
    call_count = 0

    def run_side_effect(prompt: str, response_format: Any = None) -> Any:
        nonlocal call_count
        idx = call_count % len(responses)
        call_count += 1
        return responses[idx]

    clone.run = MagicMock(side_effect=run_side_effect)
    agent._clone_for_isolation = MagicMock(return_value=clone)
    return agent


def _make_report(verdicts: List[CaseVerdict], suite_name: str = "test") -> EvalReport:
    cases = []
    for i, v in enumerate(verdicts):
        tc = TestCase(input=f"case_{i}", name=f"case_{i}")
        ar = _make_agent_result(content=f"response_{i}")
        cases.append(
            CaseResult(
                case=tc,
                verdict=v,
                agent_result=ar,
                latency_ms=100.0 + i * 50,
                cost_usd=0.001 * (i + 1),
                tokens=100 * (i + 1),
                tool_calls=["tool_a"] if v == CaseVerdict.PASS else [],
            )
        )
    meta = EvalMetadata(suite_name, "gpt-test", "Fake", 0, "abc", len(verdicts), 500, "0.17.0")
    return EvalReport(metadata=meta, case_results=cases)


# ===========================================================================
# Pairwise A/B Evaluation
# ===========================================================================


class TestPairwiseReport:
    def test_basic_counts(self) -> None:
        tc = TestCase(input="x", name="test")
        cr_a = CaseResult(case=tc, verdict=CaseVerdict.PASS, latency_ms=50)
        cr_b = CaseResult(case=tc, verdict=CaseVerdict.FAIL, latency_ms=100)
        pr = PairwiseCaseResult(case=tc, result_a=cr_a, result_b=cr_b, winner="A", reason="test")

        report = PairwiseReport(
            name="test", agent_a_name="Fast", agent_b_name="Slow", case_results=[pr]
        )
        assert report.a_wins == 1
        assert report.b_wins == 0
        assert report.ties == 0
        assert report.winner == "A"

    def test_tie(self) -> None:
        tc = TestCase(input="x")
        cr = CaseResult(case=tc, verdict=CaseVerdict.PASS, latency_ms=50)
        pr = PairwiseCaseResult(case=tc, result_a=cr, result_b=cr, winner="tie", reason="same")
        report = PairwiseReport(name="t", agent_a_name="A", agent_b_name="B", case_results=[pr])
        assert report.winner == "tie"

    def test_summary(self) -> None:
        tc = TestCase(input="x")
        cr = CaseResult(case=tc, verdict=CaseVerdict.PASS)
        pr = PairwiseCaseResult(case=tc, result_a=cr, result_b=cr, winner="A", reason="faster")
        report = PairwiseReport(name="test", agent_a_name="A", agent_b_name="B", case_results=[pr])
        s = report.summary()
        assert "A" in s
        assert "1 wins" in s

    def test_to_dict(self) -> None:
        tc = TestCase(input="x", name="q1")
        cr = CaseResult(case=tc, verdict=CaseVerdict.PASS)
        pr = PairwiseCaseResult(case=tc, result_a=cr, result_b=cr, winner="B", reason="cheaper")
        report = PairwiseReport(name="cmp", agent_a_name="A", agent_b_name="B", case_results=[pr])
        d = report.to_dict()
        assert d["winner"] == "B"
        assert len(d["cases"]) == 1


class TestPairwiseEval:
    def test_run(self) -> None:
        agent_a = _make_mock_agent([_make_agent_result(content="good")])
        agent_b = _make_mock_agent([_make_agent_result(content="bad")])
        cases = [TestCase(input="hello", expect_contains="good")]

        pairwise = PairwiseEval(agent_a, agent_b, cases, agent_a_name="Good", agent_b_name="Bad")
        result = pairwise.run()
        assert result.a_wins == 1
        assert result.b_wins == 0
        assert result.report_a is not None
        assert result.report_b is not None


# ===========================================================================
# Synthetic Test Case Generator
# ===========================================================================


class TestGeneratorParsing:
    def test_parse_json_array(self) -> None:
        text = json.dumps(
            [
                {"input": "hello", "name": "test1", "expect_tool": "greet"},
                {"input": "bye", "name": "test2", "tags": ["farewell"]},
            ]
        )
        cases = _parse_generated_cases(text)
        assert len(cases) == 2
        assert cases[0].expect_tool == "greet"
        assert cases[1].tags == ["farewell"]

    def test_parse_with_markdown_fences(self) -> None:
        text = '```json\n[{"input": "hello"}]\n```'
        cases = _parse_generated_cases(text)
        assert len(cases) == 1

    def test_parse_embedded_array(self) -> None:
        text = 'Here are the cases:\n[{"input": "test"}]\nDone.'
        cases = _parse_generated_cases(text)
        assert len(cases) == 1

    def test_parse_invalid_json(self) -> None:
        cases = _parse_generated_cases("not json at all")
        assert cases == []

    def test_parse_empty_array(self) -> None:
        cases = _parse_generated_cases("[]")
        assert cases == []

    def test_parse_missing_input(self) -> None:
        text = json.dumps([{"name": "no_input"}])
        cases = _parse_generated_cases(text)
        assert cases == []

    def test_parse_with_extra_fields(self) -> None:
        text = json.dumps(
            [{"input": "hi", "expect_contains": "hello", "expect_not_contains": "bye"}]
        )
        cases = _parse_generated_cases(text)
        assert len(cases) == 1
        assert cases[0].expect_contains == "hello"
        assert cases[0].expect_not_contains == "bye"


class TestGenerateCases:
    def test_generate_with_mock_provider(self) -> None:
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search the web"
        mock_tool.parameters = {"query": "str"}

        provider = MagicMock()
        response_msg = MagicMock()
        response_msg.content = json.dumps(
            [
                {"input": "Find Python tutorials", "name": "search_basic", "expect_tool": "search"},
                {"input": "", "name": "empty_query", "tags": ["edge_case"]},
            ]
        )
        provider.complete = MagicMock(return_value=(response_msg, MagicMock()))

        cases = generate_cases(provider, "gpt-test", [mock_tool], n=2)
        # The empty input case won't be generated since it has input=""
        assert len(cases) >= 1
        assert cases[0].expect_tool == "search"


# ===========================================================================
# Badge Generation
# ===========================================================================


class TestBadge:
    def test_generate_badge(self, tmp_path: Any) -> None:
        report = _make_report([CaseVerdict.PASS, CaseVerdict.PASS])
        path = tmp_path / "badge.svg"
        generate_badge(report, path)
        content = path.read_text()
        assert "<svg" in content
        assert "100%" in content
        assert "#4ade80" in content  # green for 100%

    def test_badge_yellow(self, tmp_path: Any) -> None:
        report = _make_report([CaseVerdict.PASS, CaseVerdict.FAIL, CaseVerdict.FAIL])
        path = tmp_path / "badge.svg"
        generate_badge(report, path, label="tests")
        content = path.read_text()
        assert "tests" in content

    def test_detailed_badge(self, tmp_path: Any) -> None:
        report = _make_report([CaseVerdict.PASS, CaseVerdict.FAIL])
        path = tmp_path / "badge_detail.svg"
        generate_detailed_badge(report, path)
        content = path.read_text()
        assert "1/2 pass" in content


# ===========================================================================
# Snapshot Testing
# ===========================================================================


class TestSnapshotStore:
    def test_save_and_load(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report = _make_report([CaseVerdict.PASS, CaseVerdict.FAIL])
        path = store.save(report, "my-agent")
        assert path.exists()

        loaded = store.load("my-agent")
        assert loaded is not None
        assert "case_0_0" in loaded
        assert loaded["case_0_0"]["content"] == "response_0"

    def test_load_missing(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        assert store.load("nonexistent") is None

    def test_compare_no_snapshot(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report = _make_report([CaseVerdict.PASS])
        result = store.compare(report, "new-suite")
        assert result.has_changes
        assert len(result.new_cases) == 1

    def test_compare_unchanged(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report = _make_report([CaseVerdict.PASS, CaseVerdict.FAIL])
        store.save(report, "stable")

        result = store.compare(report, "stable")
        assert not result.has_changes
        assert len(result.unchanged) == 2

    def test_compare_content_changed(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report1 = _make_report([CaseVerdict.PASS])
        store.save(report1, "changing")

        # Create report with different content
        report2 = _make_report([CaseVerdict.PASS])
        report2.case_results[0].agent_result.content = "different_response"

        result = store.compare(report2, "changing")
        assert result.has_changes
        assert result.changed_count == 1
        assert result.diffs[0].field == "content"

    def test_compare_new_case(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report1 = _make_report([CaseVerdict.PASS])
        store.save(report1, "growing")

        report2 = _make_report([CaseVerdict.PASS, CaseVerdict.PASS])
        result = store.compare(report2, "growing")
        assert "case_1_1" in result.new_cases

    def test_compare_removed_case(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report1 = _make_report([CaseVerdict.PASS, CaseVerdict.PASS])
        store.save(report1, "shrinking")

        report2 = _make_report([CaseVerdict.PASS])
        result = store.compare(report2, "shrinking")
        assert "case_1_1" in result.removed_cases

    def test_summary(self, tmp_path: Any) -> None:
        store = SnapshotStore(tmp_path / "snapshots")
        report1 = _make_report([CaseVerdict.PASS])
        store.save(report1, "summary")

        report2 = _make_report([CaseVerdict.PASS])
        report2.case_results[0].agent_result.content = "changed"

        result = store.compare(report2, "summary")
        s = result.summary()
        assert "Changed" in s
        assert "content" in s

    def test_duplicate_named_cases_no_collision(self, tmp_path: Any) -> None:
        """Regression: two cases with the same name must not overwrite each other in snapshot."""
        # Both cases share the same name — previously the second silently overwrote the first
        tc1 = TestCase(input="first input", name="same_name")
        tc2 = TestCase(input="second input", name="same_name")
        ar1 = _make_agent_result(content="response_1")
        ar2 = _make_agent_result(content="response_2")
        cr1 = CaseResult(case=tc1, verdict=CaseVerdict.PASS, agent_result=ar1)
        cr2 = CaseResult(case=tc2, verdict=CaseVerdict.PASS, agent_result=ar2)
        meta = EvalMetadata("dup", "m", "p", 0, "r", 2, 100.0, "0.1")
        report = EvalReport(metadata=meta, case_results=[cr1, cr2])

        store = SnapshotStore(tmp_path / "snapshots")
        store.save(report, "dup-test")
        loaded = store.load("dup-test")
        assert loaded is not None
        # Both entries must be present — keys differ by index suffix
        assert len(loaded) == 2
        contents = {v["content"] for v in loaded.values()}
        assert "response_1" in contents
        assert "response_2" in contents


class TestSnapshotResult:
    def test_no_changes(self) -> None:
        result = SnapshotResult(unchanged=["a", "b"])
        assert not result.has_changes
        assert result.changed_count == 0

    def test_has_changes(self) -> None:
        result = SnapshotResult(diffs=[SnapshotDiff("a", "content", "old", "new")])
        assert result.has_changes
        assert result.changed_count == 1

    def test_new_cases(self) -> None:
        result = SnapshotResult(new_cases=["a"])
        assert result.has_changes


# ===========================================================================
# Serve (unit tests only — no actual HTTP server)
# ===========================================================================


class TestServeState:
    """Test the dashboard state construction without starting a server."""

    def test_dashboard_html_content(self) -> None:
        from selectools.evals.serve import _DASHBOARD_HTML

        assert "<!DOCTYPE html>" in _DASHBOARD_HTML
        assert "Live Dashboard" in _DASHBOARD_HTML
        assert "NichevLabs" in _DASHBOARD_HTML
        assert "/api/state" in _DASHBOARD_HTML
