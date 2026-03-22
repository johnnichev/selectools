"""Eval history — track accuracy, cost, and latency over time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class HistoryEntry:
    """A single historical eval run."""

    run_id: str
    suite_name: str
    timestamp: float
    accuracy: float
    pass_count: int
    fail_count: int
    error_count: int
    total_cost: float
    total_tokens: int
    latency_p50: float
    latency_p95: float
    total_cases: int
    model: str
    duration_ms: float


@dataclass
class HistoryTrend:
    """Trend analysis across historical runs."""

    entries: List[HistoryEntry] = field(default_factory=list)

    @property
    def accuracy_trend(self) -> List[float]:
        """Accuracy values over time."""
        return [e.accuracy for e in self.entries]

    @property
    def cost_trend(self) -> List[float]:
        """Total cost values over time."""
        return [e.total_cost for e in self.entries]

    @property
    def latency_trend(self) -> List[float]:
        """P50 latency values over time."""
        return [e.latency_p50 for e in self.entries]

    @property
    def accuracy_delta(self) -> float:
        """Change in accuracy between last two runs."""
        if len(self.entries) < 2:
            return 0.0
        return self.entries[-1].accuracy - self.entries[-2].accuracy

    @property
    def cost_delta(self) -> float:
        """Change in cost between last two runs."""
        if len(self.entries) < 2:
            return 0.0
        return self.entries[-1].total_cost - self.entries[-2].total_cost

    @property
    def is_improving(self) -> bool:
        """True if accuracy is trending up over last 3 runs."""
        accs = self.accuracy_trend
        if len(accs) < 3:
            return self.accuracy_delta >= 0
        return accs[-1] >= accs[-2] >= accs[-3]

    @property
    def is_degrading(self) -> bool:
        """True if accuracy is trending down over last 3 runs."""
        accs = self.accuracy_trend
        if len(accs) < 3:
            return self.accuracy_delta < -0.01
        return accs[-1] <= accs[-2] <= accs[-3] and accs[-1] < accs[-3]

    def summary(self) -> str:
        """Human-readable trend summary."""
        if not self.entries:
            return "No history."
        n = len(self.entries)
        latest = self.entries[-1]
        lines = [
            f"Eval History: {latest.suite_name} ({n} runs)",
            f"  Latest: {latest.accuracy:.1%} accuracy, "
            f"${latest.total_cost:.4f}, {latest.latency_p50:.0f}ms p50",
        ]
        if n >= 2:
            lines.append(f"  Accuracy delta: {self.accuracy_delta:+.2%}")
            lines.append(f"  Cost delta: ${self.cost_delta:+.6f}")
            if self.is_improving:
                lines.append("  Trend: improving")
            elif self.is_degrading:
                lines.append("  Trend: degrading")
            else:
                lines.append("  Trend: stable")
        return "\n".join(lines)


class HistoryStore:
    """Track eval results over time.

    Stores each eval run as a JSON line in a history file, one file per
    suite name. Enables trend analysis across runs.

    Example::

        store = HistoryStore("./eval-history")
        report = suite.run()
        store.record(report)

        trend = store.trend("my-suite")
        print(trend.summary())
        print(f"Improving: {trend.is_improving}")
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)

    def record(self, report: Any) -> None:
        """Record an eval report to history."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{report.metadata.suite_name}.jsonl"

        entry = {
            "run_id": report.metadata.run_id,
            "suite_name": report.metadata.suite_name,
            "timestamp": report.metadata.timestamp,
            "accuracy": report.accuracy,
            "pass_count": report.pass_count,
            "fail_count": report.fail_count,
            "error_count": report.error_count,
            "total_cost": report.total_cost,
            "total_tokens": report.total_tokens,
            "latency_p50": report.latency_p50,
            "latency_p95": report.latency_p95,
            "total_cases": report.metadata.total_cases,
            "model": report.metadata.model,
            "duration_ms": report.metadata.duration_ms,
        }

        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def trend(self, suite_name: str, *, last_n: Optional[int] = None) -> HistoryTrend:
        """Get trend data for a suite.

        Args:
            suite_name: Name of the eval suite.
            last_n: Only return the last N entries. None = all.
        """
        path = self._dir / f"{suite_name}.jsonl"
        if not path.exists():
            return HistoryTrend()

        entries: List[HistoryEntry] = []
        for line in path.read_text().strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                HistoryEntry(
                    run_id=data["run_id"],
                    suite_name=data["suite_name"],
                    timestamp=data["timestamp"],
                    accuracy=data["accuracy"],
                    pass_count=data["pass_count"],
                    fail_count=data["fail_count"],
                    error_count=data["error_count"],
                    total_cost=data["total_cost"],
                    total_tokens=data["total_tokens"],
                    latency_p50=data["latency_p50"],
                    latency_p95=data["latency_p95"],
                    total_cases=data["total_cases"],
                    model=data["model"],
                    duration_ms=data["duration_ms"],
                )
            )

        if last_n is not None:
            entries = entries[-last_n:]

        return HistoryTrend(entries=entries)

    def list_suites(self) -> List[str]:
        """List all suite names with history."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.jsonl")]
