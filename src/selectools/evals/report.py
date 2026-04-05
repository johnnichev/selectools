"""EvalReport — aggregated evaluation results with statistics."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..stability import stable
from .types import CaseResult, CaseVerdict, EvalMetadata


@stable
@dataclass
class EvalReport:
    """Aggregated evaluation results with statistics."""

    metadata: EvalMetadata
    case_results: List[CaseResult] = field(default_factory=list)

    # --- Aggregate properties ---

    @property
    def accuracy(self) -> float:
        """Return weighted accuracy as a float between 0.0 and 1.0."""
        total_weight = sum(
            cr.case.weight for cr in self.case_results if cr.verdict != CaseVerdict.SKIP
        )
        if total_weight == 0:
            return 0.0
        pass_weight = sum(
            cr.case.weight for cr in self.case_results if cr.verdict == CaseVerdict.PASS
        )
        return pass_weight / total_weight

    @property
    def pass_count(self) -> int:
        return sum(1 for cr in self.case_results if cr.verdict == CaseVerdict.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for cr in self.case_results if cr.verdict == CaseVerdict.FAIL)

    @property
    def error_count(self) -> int:
        return sum(1 for cr in self.case_results if cr.verdict == CaseVerdict.ERROR)

    @property
    def skip_count(self) -> int:
        return sum(1 for cr in self.case_results if cr.verdict == CaseVerdict.SKIP)

    @property
    def total_cost(self) -> float:
        return sum(cr.cost_usd for cr in self.case_results)

    @property
    def total_tokens(self) -> int:
        return sum(cr.tokens for cr in self.case_results)

    @property
    def latency_p50(self) -> float:
        latencies = self._latencies()
        return statistics.median(latencies) if latencies else 0.0

    @property
    def latency_p95(self) -> float:
        latencies = sorted(self._latencies())
        if not latencies:
            return 0.0
        idx = min(math.ceil(len(latencies) * 0.95), len(latencies)) - 1
        return latencies[idx]

    @property
    def latency_p99(self) -> float:
        latencies = sorted(self._latencies())
        if not latencies:
            return 0.0
        idx = min(math.ceil(len(latencies) * 0.99), len(latencies)) - 1
        return latencies[idx]

    @property
    def latency_mean(self) -> float:
        latencies = self._latencies()
        return statistics.mean(latencies) if latencies else 0.0

    @property
    def cost_per_case(self) -> float:
        n = len(self.case_results)
        return self.total_cost / n if n else 0.0

    # --- Filtering ---

    def filter_by_tag(self, tag: str) -> List[CaseResult]:
        return [cr for cr in self.case_results if tag in cr.case.tags]

    def filter_by_verdict(self, verdict: CaseVerdict) -> List[CaseResult]:
        return [cr for cr in self.case_results if cr.verdict == verdict]

    def failures_by_evaluator(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cr in self.case_results:
            for f in cr.failures:
                counts[f.evaluator_name] = counts.get(f.evaluator_name, 0) + 1
        return counts

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "metadata": {
                "suite_name": self.metadata.suite_name,
                "model": self.metadata.model,
                "provider": self.metadata.provider,
                "timestamp": self.metadata.timestamp,
                "run_id": self.metadata.run_id,
                "total_cases": self.metadata.total_cases,
                "duration_ms": self.metadata.duration_ms,
                "selectools_version": self.metadata.selectools_version,
                "tags": self.metadata.tags,
            },
            "summary": {
                "accuracy": self.accuracy,
                "pass": self.pass_count,
                "fail": self.fail_count,
                "error": self.error_count,
                "skip": self.skip_count,
                "total_cost": self.total_cost,
                "total_tokens": self.total_tokens,
                "latency_p50": self.latency_p50,
                "latency_p95": self.latency_p95,
                "cost_per_case": self.cost_per_case,
            },
            "cases": [
                {
                    "name": cr.case.name or cr.case.input[:60],
                    "input": cr.case.input,
                    "verdict": cr.verdict.value,
                    "latency_ms": cr.latency_ms,
                    "cost_usd": cr.cost_usd,
                    "tokens": cr.tokens,
                    "tool_calls": cr.tool_calls,
                    "error": cr.error,
                    "failures": [
                        {
                            "evaluator": f.evaluator_name,
                            "expected": str(f.expected),
                            "actual": str(f.actual),
                            "message": f.message,
                        }
                        for f in cr.failures
                    ],
                }
                for cr in self.case_results
            ],
        }

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Write JSON report to file (atomic write via temp file)."""
        dest = Path(filepath)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(dest)

    def to_html(self, filepath: Union[str, Path], history: Optional[Any] = None) -> None:
        """Write self-contained HTML report to file.

        Args:
            filepath: Path to write the HTML file.
            history: Optional HistoryTrend for accuracy trend chart.
        """
        from .html import render_html_report

        render_html_report(self, filepath, history=history)

    def to_junit_xml(self, filepath: Union[str, Path]) -> None:
        """Write JUnit XML for CI integration."""
        from .junit import render_junit_xml

        render_junit_xml(self, filepath)

    def to_markdown(self) -> str:
        """Generate a markdown summary for GitHub issues, Slack, or PRs."""
        acc_emoji = "🟢" if self.accuracy >= 0.9 else "🟡" if self.accuracy >= 0.7 else "🔴"
        lines = [
            f"## {acc_emoji} Eval Report: `{self.metadata.suite_name}`",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Accuracy** | **{self.accuracy:.1%}** "
            f"({self.pass_count} pass, {self.fail_count} fail, {self.error_count} error) |",
            f"| **Latency** | p50: {self.latency_p50:.0f}ms, p95: {self.latency_p95:.0f}ms |",
            f"| **Cost** | ${self.total_cost:.6f} (${self.cost_per_case:.6f}/case) |",
            f"| **Tokens** | {self.total_tokens:,} |",
            f"| **Model** | {self.metadata.model or 'unknown'} |",
        ]

        failures = [
            cr for cr in self.case_results if cr.verdict not in (CaseVerdict.PASS, CaseVerdict.SKIP)
        ]
        if failures:
            lines.extend(
                [
                    "",
                    "<details>",
                    f"<summary>Failed cases ({len(failures)})</summary>",
                    "",
                    "| Case | Verdict | Issue |",
                    "|---|---|---|",
                ]
            )
            for cr in failures[:20]:
                name = cr.case.name or cr.case.input[:50]
                issues = "; ".join(f.message for f in cr.failures) or cr.error or ""
                lines.append(f"| {name} | `{cr.verdict.value}` | {issues[:100]} |")
            lines.extend(["", "</details>"])

        lines.extend(
            [
                "",
                f"<sub>Generated by Selectools v{self.metadata.selectools_version} "
                f"— an open-source project from <a href='https://nichevlabs.com'>NichevLabs</a></sub>",
            ]
        )
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"EvalReport: {self.metadata.suite_name}",
            f"  Model: {self.metadata.model or 'unknown'}",
            f"  Cases: {self.metadata.total_cases}",
            f"  Accuracy: {self.accuracy:.1%}  "
            f"({self.pass_count} pass, {self.fail_count} fail, "
            f"{self.error_count} error)",
            f"  Latency: p50={self.latency_p50:.0f}ms  p95={self.latency_p95:.0f}ms",
            f"  Cost: ${self.total_cost:.6f}  (${self.cost_per_case:.6f}/case)",
            f"  Tokens: {self.total_tokens}",
            f"  Duration: {self.metadata.duration_ms:.0f}ms",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EvalReport(accuracy={self.accuracy:.2%}, "
            f"pass={self.pass_count}, fail={self.fail_count}, "
            f"error={self.error_count}, "
            f"latency_p50={self.latency_p50:.1f}ms, "
            f"cost=${self.total_cost:.6f})"
        )

    # --- Internal ---

    def _latencies(self) -> List[float]:
        return [cr.latency_ms for cr in self.case_results if cr.verdict != CaseVerdict.SKIP]
