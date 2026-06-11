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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

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
    """Tool execution settings.

    Attributes:
        timeout_seconds: Maximum execution time for each tool call. None = no timeout.
        policy: Optional ToolPolicy with allow/review/deny rules.
        confirm_action: Callback invoked for tools whose policy decision is ``review``.
        approval_timeout: Seconds to wait for a confirm_action response.
        parallel_execution: Execute multiple tool calls concurrently.
        compress_results: When True, tool results longer than ``compress_threshold``
            characters are summarized by a one-shot LLM call before being appended
            to the conversation. Compressed results are prefixed with
            ``[compressed from N chars]`` so the model knows. Terminal-tool results
            and results matching ``stop_condition`` are never compressed (they
            become the agent's final answer verbatim). On summarization failure the
            result is truncated to ``compress_threshold`` chars with a
            ``[truncated from N chars; compression failed]`` marker instead of
            crashing the tool loop. Default: False (zero overhead).
        compress_threshold: Character length above which a tool result is
            compressed. Default: 2000.
        compress_provider: Optional dedicated provider for compression calls.
            Falls back to the agent's own provider when None — note this means
            compression is billed at the agent's model rates unless
            ``compress_model`` selects a cheaper model. Default: None.
        compress_model: Model override for compression calls. Defaults to the
            agent's effective model. A fast/cheap model is recommended.
            Default: None.
        require_approval: Agent-level human-in-the-loop gate. A list of tool
            names (or the string ``"*"`` for all tools) that must be approved
            before executing — the centralized equivalent of marking each tool
            with ``requires_approval=True``. Gated calls are routed to
            ``approval_handler`` (or ``confirm_action`` as fallback). A denied
            call returns a standardized "denied by approval handler" tool
            result; the loop continues and the model sees the denial.
            Default: None (no agent-level gate).
        approval_handler: Sync or async callable receiving an
            ``selectools.policy.ApprovalRequest`` (tool name, args, reason,
            preview) and returning a truthy value to approve or falsy to deny.
            Used for every ``review`` policy decision when set (takes
            precedence over ``confirm_action``). Async handlers work from both
            ``run()`` and ``arun()``/``astream()``. Default: None.

    Raises:
        ValueError: If ``require_approval`` is set without an
            ``approval_handler`` or ``confirm_action`` — gated tools would be
            unconditionally denied, which is a misconfiguration (use
            ``ToolPolicy(deny=[...])`` to hard-block tools instead).
    """

    timeout_seconds: Optional[float] = None
    policy: Optional["ToolPolicy"] = None
    confirm_action: Optional[Any] = None  # ConfirmAction type
    approval_timeout: float = 60.0
    parallel_execution: bool = True
    compress_results: bool = False
    compress_threshold: int = 2000
    compress_provider: Optional["Provider"] = None
    compress_model: Optional[str] = None
    require_approval: Optional[Union[str, List[str]]] = None
    approval_handler: Optional[Any] = None  # ApprovalHandler type

    def __post_init__(self) -> None:
        if self.require_approval and self.approval_handler is None and self.confirm_action is None:
            raise ValueError(
                "ToolConfig.require_approval is set but neither approval_handler "
                "nor confirm_action is configured — gated tools would always be "
                "denied. Provide an approval_handler, or use "
                "ToolPolicy(deny=[...]) to hard-block tools instead."
            )


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
