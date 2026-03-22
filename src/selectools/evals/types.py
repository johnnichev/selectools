"""Core data types for the eval framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CaseVerdict(str, Enum):
    """Verdict for a single evaluated test case."""

    PASS = "pass"  # nosec B105
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class TestCase:
    """A single test case for agent evaluation.

    Only ``input`` is required. All ``expect_*`` fields are optional —
    only the ones you set will be checked.
    """

    input: str
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Tool assertions
    expect_tool: Optional[str] = None
    expect_tools: Optional[List[str]] = None
    expect_tool_args: Optional[Dict[str, Dict[str, Any]]] = None

    # Content assertions
    expect_contains: Optional[str] = None
    expect_not_contains: Optional[str] = None
    expect_output: Optional[str] = None
    expect_output_regex: Optional[str] = None

    # Structured output assertions
    expect_parsed: Optional[Dict[str, Any]] = None

    # Performance assertions
    expect_iterations_lte: Optional[int] = None
    expect_latency_ms_lte: Optional[float] = None
    expect_cost_usd_lte: Optional[float] = None

    # Custom assertion
    custom_evaluator: Optional[Callable[..., bool]] = None
    custom_evaluator_name: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    response_format: Optional[Any] = None


@dataclass
class EvalFailure:
    """A single assertion failure within a test case."""

    evaluator_name: str
    expected: Any
    actual: Any
    message: str


@dataclass
class CaseResult:
    """Result of evaluating a single TestCase."""

    case: TestCase
    verdict: CaseVerdict
    agent_result: Optional[Any] = None
    failures: List[EvalFailure] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    tool_calls: List[str] = field(default_factory=list)


@dataclass
class EvalMetadata:
    """Metadata about an eval run."""

    suite_name: str
    model: str
    provider: str
    timestamp: float
    run_id: str
    total_cases: int
    duration_ms: float
    selectools_version: str
    tags: Dict[str, str] = field(default_factory=dict)
