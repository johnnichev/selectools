"""Baseline storage and regression detection for eval runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .types import CaseVerdict


@dataclass
class RegressionResult:
    """Result of comparing current run against a baseline."""

    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    accuracy_delta: float = 0.0
    latency_p50_delta: float = 0.0
    cost_delta: float = 0.0

    @property
    def is_regression(self) -> bool:
        return len(self.regressions) > 0 or self.accuracy_delta < -0.01


class BaselineStore:
    """Persist and load eval baselines for regression detection.

    Stores baselines as JSON files in a directory, keyed by suite name.
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)

    def save(self, report: Any) -> Path:
        """Save an EvalReport as the new baseline for its suite name."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{report.metadata.suite_name}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))
        return path

    def load(self, suite_name: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved baseline by suite name."""
        path = self._dir / f"{suite_name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def compare(self, current: Any) -> RegressionResult:
        """Compare current report against stored baseline.

        Matches cases by name, falling back to input text.
        """
        baseline = self.load(current.metadata.suite_name)
        if baseline is None:
            return RegressionResult()

        baseline_verdicts: Dict[str, str] = {}
        for case_data in baseline.get("cases", []):
            key = case_data.get("name") or case_data.get("input", "")[:60]
            baseline_verdicts[key] = case_data.get("verdict", "")

        baseline_summary = baseline.get("summary", {})
        baseline_accuracy = baseline_summary.get("accuracy", 0.0)
        baseline_p50 = baseline_summary.get("latency_p50", 0.0)
        baseline_cost = baseline_summary.get("total_cost", 0.0)

        regressions: List[str] = []
        improvements: List[str] = []

        for cr in current.case_results:
            key = cr.case.name or cr.case.input[:60]
            old_verdict = baseline_verdicts.get(key)
            if old_verdict is None:
                continue
            if old_verdict == CaseVerdict.PASS.value and cr.verdict != CaseVerdict.PASS:
                regressions.append(key)
            elif old_verdict != CaseVerdict.PASS.value and cr.verdict == CaseVerdict.PASS:
                improvements.append(key)

        return RegressionResult(
            regressions=regressions,
            improvements=improvements,
            accuracy_delta=current.accuracy - baseline_accuracy,
            latency_p50_delta=current.latency_p50 - baseline_p50,
            cost_delta=current.total_cost - baseline_cost,
        )
