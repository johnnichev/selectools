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

    # Content format assertions
    expect_json: Optional[bool] = None
    expect_starts_with: Optional[str] = None
    expect_ends_with: Optional[str] = None
    expect_min_length: Optional[int] = None
    expect_max_length: Optional[int] = None
    expect_min_words: Optional[int] = None
    expect_max_words: Optional[int] = None
    expect_valid_python: Optional[bool] = None
    expect_valid_sql: Optional[bool] = None
    expect_valid_urls: Optional[bool] = None
    expect_markdown: Optional[bool] = None

    # Tool order/uniqueness assertions
    expect_tool_order: Optional[List[str]] = None
    expect_unique_tools: Optional[bool] = None

    # Safety assertions
    expect_no_pii: Optional[bool] = None
    expect_no_injection: Optional[bool] = None
    expect_refusal: Optional[bool] = None

    # Sentiment assertion
    expect_sentiment: Optional[str] = None  # "positive", "negative", "neutral"

    # LLM-as-judge fields
    reference: Optional[str] = None
    context: Optional[str] = None
    rubric: Optional[str] = None
    expected_tone: Optional[str] = None  # "professional", "casual", "formal", etc.

    # Performance assertions
    expect_iterations_lte: Optional[int] = None
    expect_latency_ms_lte: Optional[float] = None
    expect_cost_usd_lte: Optional[float] = None

    # Custom assertion
    custom_evaluator: Optional[Callable[..., bool]] = None
    custom_evaluator_name: Optional[str] = None

    # Readability assertion (Flesch Reading Ease score, higher = easier)
    expect_readability_gte: Optional[float] = None

    # Agent trajectory: tool names called in this order (subsequence check)
    expect_trajectory: Optional[List[str]] = None

    # Tool efficiency: maximum number of tool calls allowed
    expect_max_tools: Optional[int] = None

    # Semantic similarity against reference (TF-IDF cosine, 0-1)
    expect_semantic_similarity_gte: Optional[float] = None

    # Multi-turn coherence heuristic check
    expect_coherent_turns: Optional[bool] = None

    # JSON schema validation (JSON Schema dict)
    expect_json_schema: Optional[Dict[str, Any]] = None

    # Keyword presence: all keywords must appear in response
    expect_keywords: Optional[List[str]] = None

    # Keyword density: min ratio of keyword occurrences to total words
    expect_keyword_density_min: Optional[float] = None

    # Forbidden words: none of these may appear in response
    expect_no_keywords: Optional[List[str]] = None

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
