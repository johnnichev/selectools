"""
Tool usage analytics for tracking and analyzing agent performance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ToolMetrics:
    """
    Metrics for a single tool's usage.

    Tracks execution statistics including call counts, success rates,
    timing information, cost attribution, parameter usage patterns,
    and streaming-specific metrics.

    Attributes:
        name: Tool name.
        total_calls: Total number of times tool was called.
        successful_calls: Number of successful executions.
        failed_calls: Number of failed executions.
        total_duration: Total execution time in seconds.
        total_cost: Total cost attributed to this tool (USD).
        parameter_usage: Frequency count of parameter values used.
        total_chunks: Total number of chunks streamed (for streaming tools).
        streaming_calls: Number of calls where tool streamed results.
    """

    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    total_cost: float = 0.0
    parameter_usage: Dict[str, Dict[Any, int]] = field(default_factory=dict)
    total_chunks: int = 0
    streaming_calls: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage (0-100)."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage (0-100)."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100.0

    @property
    def avg_duration(self) -> float:
        """Calculate average execution time in seconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration / self.total_calls

    def to_dict(self) -> dict:
        """Convert metrics to dictionary format."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(self.success_rate, 2),
            "failure_rate": round(self.failure_rate, 2),
            "total_duration": round(self.total_duration, 4),
            "avg_duration": round(self.avg_duration, 4),
            "total_cost": round(self.total_cost, 6),
            "parameter_usage": self.parameter_usage,
            "total_chunks": self.total_chunks,
            "streaming_calls": self.streaming_calls,
        }


class AgentAnalytics:
    """
    Analytics tracker for agent tool usage.

    Collects and aggregates metrics about tool execution including
    call counts, success rates, execution times, and costs.

    Example:
        >>> config = AgentConfig(enable_analytics=True)
        >>> agent = Agent(tools=[...], provider=provider, config=config)
        >>> response = agent.run([...])
        >>>
        >>> # Get analytics
        >>> analytics = agent.get_analytics()
        >>> print(analytics.summary())
        >>>
        >>> # Export to file
        >>> analytics.to_json("analytics.json")
        >>> analytics.to_csv("analytics.csv")
    """

    def __init__(self):
        """Initialize analytics tracker."""
        self._metrics: Dict[str, ToolMetrics] = {}

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration: float,
        params: Dict[str, Any],
        cost: float = 0.0,
        chunk_count: int = 0,
    ) -> None:
        """
        Record a tool execution.

        Args:
            tool_name: Name of the tool that was called.
            success: Whether the execution succeeded.
            duration: Execution time in seconds.
            params: Parameters passed to the tool.
            cost: Cost attributed to this tool call (USD).
            chunk_count: Number of chunks streamed (0 for non-streaming tools).
        """
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(name=tool_name)

        metrics = self._metrics[tool_name]
        metrics.total_calls += 1
        metrics.total_duration += duration
        metrics.total_cost += cost

        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1

        # Track streaming metrics
        if chunk_count > 0:
            metrics.streaming_calls += 1
            metrics.total_chunks += chunk_count

        # Track parameter usage
        for param_name, param_value in params.items():
            if param_name not in metrics.parameter_usage:
                metrics.parameter_usage[param_name] = {}

            # Convert value to string for consistent tracking
            value_str = str(param_value)

            # Limit tracked values to prevent memory bloat
            if len(value_str) > 100:
                value_str = f"{value_str[:100]}..."

            if value_str not in metrics.parameter_usage[param_name]:
                metrics.parameter_usage[param_name][value_str] = 0

            metrics.parameter_usage[param_name][value_str] += 1

    def get_metrics(self, tool_name: str) -> ToolMetrics | None:
        """
        Get metrics for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            ToolMetrics for the tool, or None if tool hasn't been called.
        """
        return self._metrics.get(tool_name)

    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """
        Get metrics for all tools.

        Returns:
            Dictionary mapping tool names to their metrics.
        """
        return self._metrics.copy()

    def summary(self) -> str:
        """
        Generate human-readable analytics summary.

        Returns:
            Formatted string with metrics for all tools.
        """
        if not self._metrics:
            return "No tool usage data collected."

        lines = ["", "=" * 60, "Tool Usage Analytics", "=" * 60, ""]

        # Sort tools by call count (most used first)
        sorted_tools = sorted(self._metrics.values(), key=lambda m: m.total_calls, reverse=True)

        for metrics in sorted_tools:
            lines.append(f"\n{metrics.name}:")
            lines.append(f"  Calls: {metrics.total_calls}")
            lines.append(
                f"  Success rate: {metrics.success_rate:.1f}% "
                f"({metrics.successful_calls}/{metrics.total_calls})"
            )
            if metrics.failed_calls > 0:
                lines.append(f"  Failures: {metrics.failed_calls}")
            lines.append(f"  Avg duration: {metrics.avg_duration:.3f}s")
            lines.append(f"  Total duration: {metrics.total_duration:.3f}s")
            if metrics.total_cost > 0:
                lines.append(f"  Total cost: ${metrics.total_cost:.6f}")

            # Show streaming stats if applicable
            if metrics.streaming_calls > 0:
                avg_chunks = metrics.total_chunks / metrics.streaming_calls
                lines.append(f"  Streaming calls: {metrics.streaming_calls}/{metrics.total_calls}")
                lines.append(f"  Total chunks: {metrics.total_chunks} (avg: {avg_chunks:.1f}/call)")

            # Show most common parameter values
            if metrics.parameter_usage:
                lines.append("  Common parameters:")
                for param_name, value_counts in metrics.parameter_usage.items():
                    # Get most common value
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    value, count = most_common
                    if len(value_counts) == 1:
                        lines.append(f"    {param_name}: {value}")
                    else:
                        lines.append(
                            f"    {param_name}: {value} ({count}x, " f"{len(value_counts)} unique)"
                        )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export analytics as dictionary.

        Returns:
            Dictionary with metrics for all tools.
        """
        return {
            "tools": {name: metrics.to_dict() for name, metrics in self._metrics.items()},
            "summary": {
                "total_tools": len(self._metrics),
                "total_calls": sum(m.total_calls for m in self._metrics.values()),
                "total_duration": sum(m.total_duration for m in self._metrics.values()),
                "total_cost": sum(m.total_cost for m in self._metrics.values()),
            },
        }

    def to_json(self, filepath: str | Path) -> None:
        """
        Export analytics to JSON file.

        Args:
            filepath: Path to output JSON file.

        Example:
            >>> analytics.to_json("analytics.json")
        """
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, filepath: str | Path) -> None:
        """
        Export analytics to CSV file.

        Args:
            filepath: Path to output CSV file.

        Example:
            >>> analytics.to_csv("analytics.csv")
        """
        filepath = Path(filepath)

        if not self._metrics:
            # Write empty CSV with headers
            with filepath.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "tool_name",
                        "total_calls",
                        "successful_calls",
                        "failed_calls",
                        "success_rate",
                        "avg_duration",
                        "total_duration",
                        "total_cost",
                    ]
                )
            return

        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "tool_name",
                    "total_calls",
                    "successful_calls",
                    "failed_calls",
                    "success_rate",
                    "avg_duration",
                    "total_duration",
                    "total_cost",
                ]
            )

            # Write metrics
            for metrics in sorted(
                self._metrics.values(), key=lambda m: m.total_calls, reverse=True
            ):
                writer.writerow(
                    [
                        metrics.name,
                        metrics.total_calls,
                        metrics.successful_calls,
                        metrics.failed_calls,
                        f"{metrics.success_rate:.2f}",
                        f"{metrics.avg_duration:.4f}",
                        f"{metrics.total_duration:.4f}",
                        f"{metrics.total_cost:.6f}",
                    ]
                )

    def reset(self) -> None:
        """Clear all analytics data."""
        self._metrics.clear()


__all__ = ["ToolMetrics", "AgentAnalytics"]
