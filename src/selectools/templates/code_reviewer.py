"""Code reviewer agent template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import Provider

from ..agent.config import AgentConfig
from ..agent.core import Agent
from ..tools.decorators import tool


@tool(description="Read the contents of a source file")
def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    return f"Contents of {filepath}: [file contents, 150 lines]"


@tool(description="Search code for patterns or symbols")
def search_code(pattern: str, directory: str = ".") -> str:
    """Search codebase for a pattern. Returns matching files and lines."""
    return f"Found '{pattern}' in 5 files across {directory}"


@tool(description="Submit a code review comment")
def add_review_comment(file: str, line: int, comment: str, severity: str = "suggestion") -> str:
    """Add a review comment. Severity: suggestion, warning, error."""
    return f"Comment added to {file}:{line} [{severity}]: {comment[:80]}"


SYSTEM_PROMPT = """You are an experienced code reviewer.

Your responsibilities:
1. Read source files with read_file
2. Search for patterns and dependencies with search_code
3. Provide actionable feedback with add_review_comment

Review checklist:
- Correctness: Does the code do what it claims?
- Security: SQL injection, XSS, path traversal, hardcoded secrets?
- Performance: N+1 queries, unnecessary allocations, blocking in async?
- Readability: Clear naming, appropriate abstractions, minimal comments?
- Testing: Are edge cases covered?

Guidelines:
- Be specific — reference exact lines and suggest fixes
- Prioritize severity (errors > warnings > suggestions)
- Acknowledge good patterns, not just problems
- Keep feedback concise and actionable"""


def build(provider: "Provider", **overrides: Any) -> Agent:
    """Build a code reviewer agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o"),
        "max_iterations": overrides.pop("max_iterations", 6),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[read_file, search_code, add_review_comment],
        config=AgentConfig(**config_kwargs),
    )
