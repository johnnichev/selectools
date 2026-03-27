"""
Nested configuration groups for AgentConfig.

These dataclasses group related config fields for cleaner YAML
serialization and IDE autocompletion. AgentConfig accepts both
flat kwargs (backward compat) and nested objects.

Usage::

    # Flat (existing, still works):
    config = AgentConfig(max_retries=3, coherence_check=True)

    # Nested (new, for YAML and clarity):
    config = AgentConfig(
        retry=RetryConfig(max_retries=3),
        coherence=CoherenceConfig(enabled=True),
    )

    # Both access patterns work:
    config.max_retries        # 3
    config.retry.max_retries  # 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..cache import Cache
    from ..cancellation import CancellationToken
    from ..entity_memory import EntityMemory
    from ..guardrails import GuardrailsPipeline
    from ..knowledge import KnowledgeMemory
    from ..knowledge_graph import KnowledgeGraphMemory
    from ..observer import AgentObserver
    from ..policy import ToolPolicy
    from ..providers.base import Provider
    from ..sessions import SessionStore
    from ..usage import AgentUsage


@dataclass
class RetryConfig:
    """Retry and timeout settings."""

    max_retries: int = 2
    backoff_seconds: float = 1.0
    request_timeout: Optional[float] = 30.0
    rate_limit_cooldown_seconds: float = 5.0


@dataclass
class ToolConfig:
    """Tool execution settings."""

    timeout_seconds: Optional[float] = None
    policy: Optional["ToolPolicy"] = None
    confirm_action: Optional[Any] = None  # ConfirmAction type
    approval_timeout: float = 60.0
    parallel_execution: bool = True


@dataclass
class CoherenceConfig:
    """Coherence checking settings."""

    enabled: bool = False
    provider: Optional["Provider"] = None
    model: Optional[str] = None
    fail_closed: bool = False


@dataclass
class GuardrailsConfig:
    """Input/output guardrail settings."""

    pipeline: Optional["GuardrailsPipeline"] = None
    screen_tool_output: bool = False
    output_screening_patterns: Optional[List[str]] = None


@dataclass
class SessionConfig:
    """Session persistence settings."""

    store: Optional["SessionStore"] = None
    session_id: Optional[str] = None


@dataclass
class SummarizeConfig:
    """Summarize-on-trim settings."""

    enabled: bool = False
    provider: Optional["Provider"] = None
    model: Optional[str] = None
    max_tokens: int = 150


@dataclass
class MemoryConfig:
    """Memory subsystem settings."""

    entity_memory: Optional["EntityMemory"] = None
    knowledge_graph: Optional["KnowledgeGraphMemory"] = None
    knowledge_memory: Optional["KnowledgeMemory"] = None


@dataclass
class BudgetConfig:
    """Token and cost budget settings."""

    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    cost_warning_threshold: Optional[float] = None
    cancellation_token: Optional["CancellationToken"] = None


@dataclass
class TraceConfig:
    """Tracing and observability settings."""

    tool_result_chars: Optional[int] = 200
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None


@dataclass
class CompressConfig:
    """Prompt compression settings."""

    enabled: bool = False
    threshold: float = 0.75
    keep_recent: int = 4


__all__ = [
    "RetryConfig",
    "ToolConfig",
    "CoherenceConfig",
    "GuardrailsConfig",
    "SessionConfig",
    "SummarizeConfig",
    "MemoryConfig",
    "BudgetConfig",
    "TraceConfig",
    "CompressConfig",
]
