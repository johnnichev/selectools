"""
Token usage and cost tracking for LLM API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UsageStats:
    """
    Tracks token usage and costs for a single API call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        total_tokens: Total tokens used (prompt + completion).
        cost_usd: Estimated cost in USD for this call.
        model: Model name used for this call.
        provider: Provider name (openai, anthropic, gemini, etc.).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""

    def __post_init__(self):
        """Ensure total_tokens is consistent."""
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class AgentUsage:
    """
    Aggregates usage stats across multiple agent iterations.

    Tracks cumulative token usage, costs, and per-tool breakdowns for
    monitoring and optimization.

    Attributes:
        total_prompt_tokens: Cumulative prompt tokens across all calls.
        total_completion_tokens: Cumulative completion tokens across all calls.
        total_tokens: Cumulative total tokens across all calls.
        total_cost_usd: Cumulative cost in USD across all calls.
        tool_usage: Dictionary mapping tool names to call counts.
        tool_tokens: Dictionary mapping tool names to total tokens used.
        iterations: List of UsageStats for each iteration.
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Per-tool breakdown
    tool_usage: Dict[str, int] = field(default_factory=dict)
    tool_tokens: Dict[str, int] = field(default_factory=dict)

    # Per-iteration tracking
    iterations: List[UsageStats] = field(default_factory=list)

    def add_usage(self, stats: UsageStats, tool_name: Optional[str] = None) -> None:
        """
        Add usage stats from a single iteration.

        Args:
            stats: UsageStats object from an API call.
            tool_name: Optional tool name if this iteration involved a tool call.
        """
        self.total_prompt_tokens += stats.prompt_tokens
        self.total_completion_tokens += stats.completion_tokens
        self.total_tokens += stats.total_tokens
        self.total_cost_usd += stats.cost_usd
        self.iterations.append(stats)

        if tool_name:
            self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
            self.tool_tokens[tool_name] = self.tool_tokens.get(tool_name, 0) + stats.total_tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for logging/display.

        Returns:
            Dictionary containing all usage statistics.
        """
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "tool_usage": self.tool_usage,
            "tool_tokens": self.tool_tokens,
            "iterations": len(self.iterations),
        }

    def __str__(self) -> str:
        """
        Pretty print usage summary.

        Returns:
            Formatted string with usage statistics and emojis.
        """
        lines = [
            "\n" + "=" * 60,
            "📊 Usage Summary",
            "=" * 60,
            f"Total Tokens: {self.total_tokens:,}",
            f"  - Prompt: {self.total_prompt_tokens:,}",
            f"  - Completion: {self.total_completion_tokens:,}",
            f"Total Cost: ${self.total_cost_usd:.6f}",
            f"Iterations: {len(self.iterations)}",
        ]

        if self.tool_usage:
            lines.append("\nTool Usage:")
            for tool_name, count in sorted(self.tool_usage.items(), key=lambda x: -x[1]):
                tokens = self.tool_tokens.get(tool_name, 0)
                lines.append(f"  - {tool_name}: {count} calls, {tokens:,} tokens")

        lines.append("=" * 60 + "\n")
        return "\n".join(lines)


__all__ = ["UsageStats", "AgentUsage"]
