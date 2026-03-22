"""Pairwise A/B evaluation — compare two agents head-to-head."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import __version__
from ..agent import Agent
from .evaluators import DEFAULT_EVALUATORS
from .report import EvalReport
from .suite import EvalSuite
from .types import CaseResult, CaseVerdict, EvalMetadata, TestCase


@dataclass
class PairwiseCaseResult:
    """Result of comparing two agents on a single test case."""

    case: TestCase
    result_a: CaseResult
    result_b: CaseResult
    winner: str  # "A", "B", or "tie"
    reason: str


@dataclass
class PairwiseReport:
    """Aggregated A/B comparison results."""

    name: str
    agent_a_name: str
    agent_b_name: str
    case_results: List[PairwiseCaseResult] = field(default_factory=list)
    report_a: Optional[EvalReport] = None
    report_b: Optional[EvalReport] = None

    @property
    def a_wins(self) -> int:
        return sum(1 for cr in self.case_results if cr.winner == "A")

    @property
    def b_wins(self) -> int:
        return sum(1 for cr in self.case_results if cr.winner == "B")

    @property
    def ties(self) -> int:
        return sum(1 for cr in self.case_results if cr.winner == "tie")

    @property
    def winner(self) -> str:
        """Overall winner: 'A', 'B', or 'tie'."""
        if self.a_wins > self.b_wins:
            return "A"
        elif self.b_wins > self.a_wins:
            return "B"
        return "tie"

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Pairwise Comparison: {self.name}",
            f"  Agent A ({self.agent_a_name}): {self.a_wins} wins",
            f"  Agent B ({self.agent_b_name}): {self.b_wins} wins",
            f"  Ties: {self.ties}",
            f"  Winner: {self.winner}",
            "",
        ]
        if self.report_a and self.report_b:
            lines.extend(
                [
                    f"  Accuracy:  A={self.report_a.accuracy:.1%}  B={self.report_b.accuracy:.1%}",
                    f"  Latency:   A={self.report_a.latency_p50:.0f}ms  "
                    f"B={self.report_b.latency_p50:.0f}ms",
                    f"  Cost:      A=${self.report_a.total_cost:.6f}  "
                    f"B=${self.report_b.total_cost:.6f}",
                ]
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "name": self.name,
            "agent_a": self.agent_a_name,
            "agent_b": self.agent_b_name,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "winner": self.winner,
            "cases": [
                {
                    "name": cr.case.name or cr.case.input[:60],
                    "winner": cr.winner,
                    "reason": cr.reason,
                    "a_verdict": cr.result_a.verdict.value,
                    "b_verdict": cr.result_b.verdict.value,
                    "a_latency": cr.result_a.latency_ms,
                    "b_latency": cr.result_b.latency_ms,
                }
                for cr in self.case_results
            ],
            "summary_a": self.report_a.to_dict()["summary"] if self.report_a else None,
            "summary_b": self.report_b.to_dict()["summary"] if self.report_b else None,
        }


class PairwiseEval:
    """Compare two agents head-to-head on the same test cases.

    Example::

        comparison = PairwiseEval(
            agent_a=fast_agent,
            agent_b=accurate_agent,
            cases=cases,
        )
        result = comparison.run()
        print(result.summary())
    """

    def __init__(
        self,
        agent_a: Agent,
        agent_b: Agent,
        cases: List[TestCase],
        *,
        name: str = "pairwise",
        agent_a_name: str = "A",
        agent_b_name: str = "B",
        evaluators: Optional[List[Any]] = None,
        max_concurrency: int = 1,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.cases = cases
        self.name = name
        self.agent_a_name = agent_a_name
        self.agent_b_name = agent_b_name
        self.evaluators = evaluators
        self.max_concurrency = max_concurrency

    def run(self) -> PairwiseReport:
        """Run both agents and compare results."""
        suite_a = EvalSuite(
            agent=self.agent_a,
            cases=self.cases,
            name=f"{self.name}_A",
            evaluators=self.evaluators,
            max_concurrency=self.max_concurrency,
        )
        suite_b = EvalSuite(
            agent=self.agent_b,
            cases=self.cases,
            name=f"{self.name}_B",
            evaluators=self.evaluators,
            max_concurrency=self.max_concurrency,
        )

        report_a = suite_a.run()
        report_b = suite_b.run()

        pairwise_results: List[PairwiseCaseResult] = []
        for cr_a, cr_b in zip(report_a.case_results, report_b.case_results):
            winner, reason = self._compare_case(cr_a, cr_b)
            pairwise_results.append(
                PairwiseCaseResult(
                    case=cr_a.case,
                    result_a=cr_a,
                    result_b=cr_b,
                    winner=winner,
                    reason=reason,
                )
            )

        return PairwiseReport(
            name=self.name,
            agent_a_name=self.agent_a_name,
            agent_b_name=self.agent_b_name,
            case_results=pairwise_results,
            report_a=report_a,
            report_b=report_b,
        )

    def _compare_case(self, cr_a: CaseResult, cr_b: CaseResult) -> tuple[str, str]:
        """Determine the winner for a single case."""
        a_pass = cr_a.verdict == CaseVerdict.PASS
        b_pass = cr_b.verdict == CaseVerdict.PASS

        if a_pass and not b_pass:
            return "A", f"A passed, B {cr_b.verdict.value}"
        if b_pass and not a_pass:
            return "B", f"B passed, A {cr_a.verdict.value}"
        if not a_pass and not b_pass:
            a_fails = len(cr_a.failures)
            b_fails = len(cr_b.failures)
            if a_fails < b_fails:
                return "A", f"A had fewer failures ({a_fails} vs {b_fails})"
            if b_fails < a_fails:
                return "B", f"B had fewer failures ({b_fails} vs {a_fails})"
            return "tie", "both failed with same number of issues"

        # Both passed — compare on latency
        if cr_a.latency_ms < cr_b.latency_ms * 0.8:
            return "A", f"A was faster ({cr_a.latency_ms:.0f}ms vs {cr_b.latency_ms:.0f}ms)"
        if cr_b.latency_ms < cr_a.latency_ms * 0.8:
            return "B", f"B was faster ({cr_b.latency_ms:.0f}ms vs {cr_a.latency_ms:.0f}ms)"

        # Both passed, similar latency — compare cost
        if cr_a.cost_usd < cr_b.cost_usd * 0.8:
            return "A", f"A was cheaper (${cr_a.cost_usd:.6f} vs ${cr_b.cost_usd:.6f})"
        if cr_b.cost_usd < cr_a.cost_usd * 0.8:
            return "B", f"B was cheaper (${cr_b.cost_usd:.6f} vs ${cr_a.cost_usd:.6f})"

        return "tie", "both passed with similar performance"
