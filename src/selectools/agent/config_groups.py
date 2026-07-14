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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..stability import beta, stable

if TYPE_CHECKING:
    from ..cancellation import CancellationToken
    from ..entity_memory import EntityMemory
    from ..guardrails import GuardrailsPipeline
    from ..knowledge import KnowledgeMemory
    from ..knowledge_graph import KnowledgeGraphMemory
    from ..policy import ToolPolicy
    from ..providers.base import Provider
    from ..sessions import SessionStore
    from ..unified_memory import UnifiedMemory


@stable
@dataclass
class RetryConfig:
    """Retry and timeout settings."""

    max_retries: int = 2
    backoff_seconds: float = 1.0
    request_timeout: Optional[float] = 30.0
    rate_limit_cooldown_seconds: float = 5.0


@stable
@dataclass
class ToolConfig:
    """Tool execution settings.

    Attributes:
        timeout_seconds: Maximum execution time for each tool call. None = no timeout.
        policy: Optional ToolPolicy with allow/review/deny rules.
        confirm_action: Callback invoked for tools whose policy decision is ``review``.
        approval_timeout: Seconds to wait for a confirm_action or
            approval_handler response. NOTE: while waiting, blocking sync
            handlers (and async handlers invoked from sync ``run()``) each
            occupy one slot of the shared 16-worker
            ``selectools_tool_timeout`` thread pool, which also serves
            tool-timeout enforcement across ALL agents in the process. Many
            concurrent long-blocking approvals can exhaust the pool — keep
            handlers prompt and this timeout deliberate. Timed-out approval
            futures are cancelled so still-queued handlers never fire after
            the call was denied.
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
            ``compress_model`` selects a cheaper model. When set,
            ``compress_model`` MUST also be set: the agent's model almost
            never exists on a different provider, and the resulting persistent
            404 would silently degrade every oversized result to the
            truncation fallback. Default: None.
        compress_model: Model override for compression calls. Defaults to the
            agent's effective model. A fast/cheap model is recommended.
            Required when ``compress_provider`` is set. Default: None.
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
            preview) and returning a bool: ``True`` to approve, ``False`` to
            deny. Any non-bool return value (coroutine, generator, mock, ...)
            fails CLOSED and denies the call. Used for every ``review``
            policy decision when set (takes precedence over
            ``confirm_action``). Async handlers — both coroutine functions
            and instances with ``async def __call__`` — work from both
            ``run()`` and ``arun()``/``astream()``. ``request.tool_args`` is
            a defensive copy: mutating it never changes what the tool
            executes with. Denials are memoized per (tool name, args) within
            a run so an identical retried call does not re-page the human;
            approvals are re-requested every time. Default: None.

    Raises:
        ValueError: If ``compress_provider`` is set without an explicit
            ``compress_model``; or if ``require_approval`` is set without an
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
        if self.compress_provider is not None and self.compress_model is None:
            raise ValueError(
                "ToolConfig: compress_provider is set but compress_model is not. "
                "The agent's own model is unlikely to exist on a dedicated "
                "compression provider, which would 404 on every call and "
                "silently degrade all oversized results to the truncation "
                "fallback. Set compress_model explicitly."
            )
        if self.require_approval and self.approval_handler is None and self.confirm_action is None:
            raise ValueError(
                "ToolConfig.require_approval is set but neither approval_handler "
                "nor confirm_action is configured — gated tools would always be "
                "denied. Provide an approval_handler, or use "
                "ToolPolicy(deny=[...]) to hard-block tools instead."
            )


@stable
@dataclass
class CoherenceConfig:
    """Coherence checking settings."""

    enabled: bool = False
    provider: Optional["Provider"] = None
    model: Optional[str] = None
    fail_closed: bool = False


@stable
@dataclass
class GuardrailsConfig:
    """Input/output guardrail settings."""

    pipeline: Optional["GuardrailsPipeline"] = None
    screen_tool_output: bool = False
    output_screening_patterns: Optional[List[str]] = None


@beta
@dataclass
class StructuredOutputConfig:
    """Structured-output (``response_format``) behavior settings (beta, v1.1).

    Attributes:
        native: Use the provider's native structured-output mode (OpenAI
            ``json_schema``, Gemini ``response_json_schema``) when the provider
            advertises support, instead of injecting a schema instruction into
            the system prompt. Providers without native support keep the
            prompt-injection + parse behavior unchanged. Default: True.
        final_turn_only: Keep the schema out of tool-loop turns entirely.
            The loop runs to convergence with no schema pressure (so tool
            calling and the text ``ToolCallParser`` work normally), then one
            extra synthesis call — with the schema applied and no tools —
            produces the validated structured final answer. In ``astream()``
            the synthesis JSON is delivered only via the terminal
            ``AgentResult`` (``.content`` / ``.parsed``), never leaked as
            content chunks. Costs at most one extra provider call per run
            (see ``reuse_loop_answer`` / ``should_finalize`` /
            ``single_pass`` for the paths that avoid it). Default: False.
        reuse_loop_answer: In ``final_turn_only`` mode, when the loop's
            converged answer already parses and validates against the schema,
            use it directly and skip the synthesis call entirely (v1.2,
            issue #164). Default: True.
        should_finalize: Optional predicate ``(messages, last_response_text)
            -> bool`` consulted in ``final_turn_only`` mode when the
            converged answer did NOT validate. Return ``False`` to skip the
            synthesis call for turns that need no structured output — the
            run finishes with ``parsed=None`` and
            ``AgentResult.structured_status == "skipped"``. A validating
            answer wins before the predicate runs. Default: None (always
            finalize, the v1.1 behavior).
        single_pass: In ``final_turn_only`` mode with a provider that
            advertises ``supports_native_structured_output_with_tools``
            (OpenAI/Azure), carry the schema natively on the loop calls so
            the converged answer IS the structured object — combined with
            ``reuse_loop_answer`` this eliminates the synthesis call (v1.2,
            issue #166). Providers without combined support fall back to the
            separate synthesis call. NOTE: in ``astream()`` the final answer
            is then JSON and streams as content chunks (like plain native
            mode). Default: False.
    """

    native: bool = True
    final_turn_only: bool = False
    reuse_loop_answer: bool = True
    should_finalize: Optional[Any] = None  # Callable[[List[Message], str], bool]
    single_pass: bool = False


@stable
@dataclass
class SessionConfig:
    """Session persistence settings."""

    store: Optional["SessionStore"] = None
    session_id: Optional[str] = None


@stable
@dataclass
class SummarizeConfig:
    """Summarize-on-trim settings."""

    enabled: bool = False
    provider: Optional["Provider"] = None
    model: Optional[str] = None
    max_tokens: int = 150


@stable
@dataclass
class MemoryConfig:
    """Memory subsystem settings.

    The ``unified*`` fields below are **beta** (v1.1): they wire
    :class:`~selectools.unified_memory.UnifiedMemory` into the agent so it
    builds context from all tiers before each call and persists the completed
    turn (with STM -> LTM auto-promotion) after each run.

    Attributes:
        entity_memory: Optional EntityMemory for per-turn entity extraction
            and context injection.
        knowledge_graph: Optional KnowledgeGraphMemory for relationship
            triple extraction and query-scoped context.
        knowledge_memory: Optional KnowledgeMemory for durable facts and the
            auto-injected ``remember`` tool.
        unified: Enable unified tiered memory (beta). The agent builds a
            :class:`~selectools.unified_memory.UnifiedMemory` from the tier
            parameters below and drives it for context assembly and post-turn
            persistence. Mutually exclusive with ``entity_memory``,
            ``knowledge_graph``, ``knowledge_memory``, the Agent ``memory=``
            parameter, and ``session_store``. Default: False.
        unified_memory: Optional pre-built UnifiedMemory instance (beta).
            Implies ``unified=True``. Use this to inject custom tiers
            (``UnifiedMemory(short_term=..., long_term=..., entity_memory=...)``,
            custom scorers, summarizers). When set, the scalar tier
            parameters below are ignored — the instance's own settings win.
            Default: None.
        importance_threshold: Minimum importance score (0.0-1.0) for STM ->
            LTM promotion. Default: 0.7.
        short_term_limit: Rolling short-term window size, in messages
            (one turn = two messages). Default: 100.
        long_term_limit: Maximum long-term entries before importance-based
            eviction. Default: 1000.
        episodic_retention_days: Episodes older than this are pruned.
            Default: 30.
        auto_promote: Score and promote short-term items as they age out of
            the rolling window. Default: True.
        context_max_tokens: Token budget passed to
            ``UnifiedMemory.assemble_context()`` when the agent injects
            unified context; compaction triggers at 70% of this budget.
            Default: 4000.

    Raises:
        ValueError: If unified memory is enabled together with
            ``entity_memory``, ``knowledge_graph``, or ``knowledge_memory``;
            or if any unified tier parameter is out of range while unified
            memory is enabled (the parameters are inert otherwise).
    """

    entity_memory: Optional["EntityMemory"] = None
    knowledge_graph: Optional["KnowledgeGraphMemory"] = None
    knowledge_memory: Optional["KnowledgeMemory"] = None
    # -- Unified tiered memory (beta, v1.1) --
    unified: bool = False
    unified_memory: Optional["UnifiedMemory"] = None
    importance_threshold: float = 0.7
    short_term_limit: int = 100
    long_term_limit: int = 1000
    episodic_retention_days: int = 30
    auto_promote: bool = True
    context_max_tokens: int = 4000

    @property
    def _unified_enabled(self) -> bool:
        """Whether unified memory is requested (flag or injected instance)."""
        return self.unified or self.unified_memory is not None

    def __post_init__(self) -> None:
        if not self._unified_enabled:
            return
        if (
            self.entity_memory is not None
            or self.knowledge_graph is not None
            or self.knowledge_memory is not None
        ):
            raise ValueError(
                "MemoryConfig: unified memory is mutually exclusive with "
                "entity_memory/knowledge_graph/knowledge_memory. Inject custom "
                "tiers via MemoryConfig(unified_memory=UnifiedMemory(...)) instead."
            )
        if not 0.0 <= self.importance_threshold <= 1.0:
            raise ValueError("MemoryConfig.importance_threshold must be between 0.0 and 1.0")
        if self.short_term_limit < 1:
            raise ValueError("MemoryConfig.short_term_limit must be at least 1")
        if self.long_term_limit < 1:
            raise ValueError("MemoryConfig.long_term_limit must be at least 1")
        if self.episodic_retention_days < 1:
            raise ValueError("MemoryConfig.episodic_retention_days must be at least 1")
        if self.context_max_tokens < 1:
            raise ValueError("MemoryConfig.context_max_tokens must be at least 1")


@stable
@dataclass
class BudgetConfig:
    """Token and cost budget settings."""

    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    cost_warning_threshold: Optional[float] = None
    cancellation_token: Optional["CancellationToken"] = None


@stable
@dataclass
class TraceConfig:
    """Tracing and observability settings."""

    tool_result_chars: Optional[int] = 200
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None


@stable
@dataclass
class CompressConfig:
    """Prompt compression settings."""

    enabled: bool = False
    threshold: float = 0.75
    keep_recent: int = 4


__stability__ = "stable"

__all__ = [
    "RetryConfig",
    "StructuredOutputConfig",
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
