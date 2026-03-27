"""
Configuration options for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional, Union

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

# Hook type definitions
HookCallable = Callable[..., None]
Hooks = Dict[str, HookCallable]

ConfirmAction = Union[
    Callable[[str, Dict[str, Any], str], bool],
    Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, bool]],
]


@dataclass
class AgentConfig:
    """
    Configuration options for customizing agent behavior.

    Controls model selection, retry logic, timeouts, and verbosity for debugging.
    Sensible defaults are provided for all options.

    Attributes:
        model: Model identifier to use (e.g., "gpt-5-mini", "claude-sonnet-4-6").
        temperature: LLM temperature (0.0 = deterministic, higher = more creative). Default: 0.0.
        max_tokens: Maximum tokens in LLM response. Default: 1000.
        max_iterations: Maximum tool-calling iterations before stopping. Default: 6.
        verbose: Print detailed logs for debugging. Default: False.
        stream: Enable streaming responses (if provider supports it). Default: False.
        request_timeout: Timeout for LLM API requests in seconds. Default: 30.0.
        max_retries: Maximum retry attempts for failed LLM requests. Default: 2.
        retry_backoff_seconds: Seconds to wait between retries (multiplied by attempt number). Default: 1.0.
        rate_limit_cooldown_seconds: Cooldown period after rate limit errors. Default: 5.0.
        tool_timeout_seconds: Maximum execution time for each tool call. None = no timeout. Default: None.
        cost_warning_threshold: Print warning when total cost exceeds this USD amount. None = no warnings. Default: None.
        enable_analytics: Enable detailed analytics tracking. Default: False.
        system_prompt: Custom system instructions to replace the default prompt. Default: None (uses built-in instructions).
        hooks: Optional dict of lifecycle hooks for observability. Default: None.
               Available hooks:
               - 'on_agent_start': Called at start of run/arun with (messages,)
               - 'on_agent_end': Called at end with (response, usage)
               - 'on_iteration_start': Called at start of each iteration with (iteration_num, messages)
               - 'on_iteration_end': Called at end of each iteration with (iteration_num, response)
               - 'on_tool_start': Called before tool execution with (tool_name, tool_args)
               - 'on_tool_chunk': Called for each chunk from streaming tools with (tool_name, chunk)
               - 'on_tool_end': Called after tool execution with (tool_name, result, duration)
               - 'on_tool_error': Called on tool error with (tool_name, error, tool_args)
               - 'on_llm_start': Called before LLM call with (messages, model)
               - 'on_llm_end': Called after LLM call with (response, usage)
               - 'on_error': Called on any error with (error, context)
        observers: List of AgentObserver instances for structured lifecycle events.
               Observers receive richer data than hooks (run_id, call_id, system_prompt)
               and are the recommended integration path for tracing systems like
               Langfuse, OpenTelemetry, and Datadog.  Existing hooks continue to
               work unchanged.  Default: [].
        routing_only: If True, returns tool selection without executing it. Default: False.
        parallel_tool_execution: Execute multiple tool calls concurrently when the LLM
               requests more than one tool in a single response. Uses asyncio.gather for
               async and ThreadPoolExecutor for sync execution. Default: True.
        cache: Optional response cache.  When set, the agent checks the cache
               before calling the LLM provider and stores successful responses.
               Any object satisfying the ``Cache`` protocol can be used (e.g.
               ``InMemoryCache``, ``RedisCache``).  Default: None (caching disabled).
        tool_policy: Optional ToolPolicy with allow/review/deny rules.
               Evaluated before every tool execution. Default: None (no policy).
        confirm_action: Callback invoked for tools whose policy decision is
               ``review``.  Signature: ``(tool_name, tool_args, reason) -> bool``.
               Async callables are also supported.  If ``None``, reviewed tools
               are denied by default.  Default: None.
        approval_timeout: Seconds to wait for a confirm_action response before
               denying the call.  Default: 60.
        trace_tool_result_chars: Maximum characters of tool result text stored in
               TraceStep entries.  Set to ``None`` for unlimited.  Default: 200.
        trace_metadata: Arbitrary key-value pairs attached to every ``AgentTrace``
               created by this agent (e.g. ``{"user_id": "u123", "env": "prod"}``).
               Default: {} (empty).
        parent_run_id: Optional run-ID of a parent agent.  When set, every
               ``AgentTrace`` created by this agent records a ``parent_run_id``
               so nested/chained agent calls can be linked in tracing systems.
               Default: None.
        guardrails: Optional GuardrailsPipeline with input and output guardrails.
               Input guardrails run on user messages before the LLM call.
               Output guardrails run on LLM responses after they return.
               Default: None (no guardrails).
        screen_tool_output: Enable prompt-injection screening on ALL tool outputs.
               Individual tools can also opt-in via ``@tool(screen_output=True)``.
               Default: False.
        output_screening_patterns: Extra regex patterns (strings) for tool output
               screening in addition to the built-in injection detectors.
               Default: None.
        coherence_check: Enable LLM-based coherence checking that verifies each
               proposed tool call matches the user's original intent.  Adds one
               extra LLM call per tool-call iteration.  Default: False.
        coherence_provider: Optional separate provider for coherence checks.
               Uses the agent's own provider if not set.  Default: None.
        coherence_model: Model to use for coherence checks.  Defaults to the
               agent's configured model.  Using a fast/cheap model is recommended.
               Default: None.
        reasoning_strategy: Optional reasoning pattern to inject into the system
               prompt.  Valid values: ``"react"`` (Thought → Action → Observation),
               ``"cot"`` (Chain-of-Thought step-by-step), ``"plan_then_act"``
               (plan first, then execute).  Default: None (no strategy).
        compress_context: Proactively summarize old messages when the estimated
               token count exceeds ``compress_threshold`` of the model's context
               window.  Operates on the per-call history view only — the permanent
               ``ConversationMemory`` is never modified.  Default: False.
        compress_threshold: Fraction of the model's context window at which
               compression is triggered.  Range: ``(0.0, 1.0]``.  Default: 0.75.
        compress_keep_recent: Number of recent conversation turns to always keep
               verbatim.  Each "turn" is one user + one assistant message pair.
               Default: 4.
    """

    name: str = "agent"
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    max_iterations: int = 6
    verbose: bool = False
    stream: bool = False
    request_timeout: Optional[float] = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    rate_limit_cooldown_seconds: float = 5.0
    tool_timeout_seconds: Optional[float] = None
    cost_warning_threshold: Optional[float] = None
    enable_analytics: bool = False
    system_prompt: Optional[str] = None
    hooks: Optional[Hooks] = None
    observers: List[AgentObserver] = field(default_factory=list)
    routing_only: bool = False
    parallel_tool_execution: bool = True
    cache: Optional[Cache] = None
    tool_policy: Optional[ToolPolicy] = None
    confirm_action: Optional[ConfirmAction] = None
    approval_timeout: float = 60.0
    trace_tool_result_chars: Optional[int] = 200
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    guardrails: Optional[GuardrailsPipeline] = None
    screen_tool_output: bool = False
    output_screening_patterns: Optional[List[str]] = None
    coherence_check: bool = False
    coherence_provider: Optional[Provider] = None
    coherence_model: Optional[str] = None
    coherence_fail_closed: bool = False
    session_store: Optional[SessionStore] = None
    session_id: Optional[str] = None
    summarize_on_trim: bool = False
    summarize_provider: Optional[Provider] = None
    summarize_model: Optional[str] = None
    summarize_max_tokens: int = 150
    entity_memory: Optional[EntityMemory] = None
    knowledge_graph: Optional[KnowledgeGraphMemory] = None
    knowledge_memory: Optional[KnowledgeMemory] = None
    stop_condition: Optional[Callable[[str, str], bool]] = None
    reasoning_strategy: Optional[str] = None
    model_selector: Optional[Callable[[int, List, "AgentUsage"], str]] = None
    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    cancellation_token: Optional[CancellationToken] = None
    compress_context: bool = False
    compress_threshold: float = 0.75
    compress_keep_recent: int = 4

    # -- Nested config groups (optional, for YAML / clean APIs) --
    # When provided, nested values take precedence over flat fields.
    retry: Optional[Any] = None  # RetryConfig
    tool: Optional[Any] = None  # ToolConfig
    coherence: Optional[Any] = None  # CoherenceConfig (name collision with coherence_check)
    guardrail: Optional[Any] = None  # GuardrailsConfig
    session: Optional[Any] = None  # SessionConfig
    summarize: Optional[Any] = None  # SummarizeConfig
    memory: Optional[Any] = None  # MemoryConfig
    budget: Optional[Any] = None  # BudgetConfig
    trace: Optional[Any] = None  # TraceConfig
    compress: Optional[Any] = None  # CompressConfig

    def __post_init__(self) -> None:  # noqa: D105
        from .config_groups import (
            BudgetConfig,
            CoherenceConfig,
            CompressConfig,
            GuardrailsConfig,
            MemoryConfig,
            RetryConfig,
            SessionConfig,
            SummarizeConfig,
            ToolConfig,
            TraceConfig,
        )

        # Auto-unpack dicts into config objects (for YAML / dict-based config)
        def _unpack(val, cls):
            if isinstance(val, dict):
                return cls(**val)
            return val

        self.retry = _unpack(self.retry, RetryConfig)
        self.tool = _unpack(self.tool, ToolConfig)
        self.coherence = _unpack(self.coherence, CoherenceConfig)
        self.guardrail = _unpack(self.guardrail, GuardrailsConfig)
        self.session = _unpack(self.session, SessionConfig)
        self.summarize = _unpack(self.summarize, SummarizeConfig)
        self.memory = _unpack(self.memory, MemoryConfig)
        self.budget = _unpack(self.budget, BudgetConfig)
        self.trace = _unpack(self.trace, TraceConfig)
        self.compress = _unpack(self.compress, CompressConfig)

        # Sync nested -> flat (nested takes precedence when provided)
        if isinstance(self.retry, RetryConfig):
            self.max_retries = self.retry.max_retries
            self.retry_backoff_seconds = self.retry.backoff_seconds
            self.request_timeout = self.retry.request_timeout
            self.rate_limit_cooldown_seconds = self.retry.rate_limit_cooldown_seconds
        else:
            self.retry = RetryConfig(
                max_retries=self.max_retries,
                backoff_seconds=self.retry_backoff_seconds,
                request_timeout=self.request_timeout,
                rate_limit_cooldown_seconds=self.rate_limit_cooldown_seconds,
            )

        if isinstance(self.tool, ToolConfig):
            self.tool_timeout_seconds = self.tool.timeout_seconds
            self.tool_policy = self.tool.policy
            self.confirm_action = self.tool.confirm_action
            self.approval_timeout = self.tool.approval_timeout
            self.parallel_tool_execution = self.tool.parallel_execution
        else:
            self.tool = ToolConfig(
                timeout_seconds=self.tool_timeout_seconds,
                policy=self.tool_policy,
                confirm_action=self.confirm_action,
                approval_timeout=self.approval_timeout,
                parallel_execution=self.parallel_tool_execution,
            )

        if isinstance(self.coherence, CoherenceConfig):
            self.coherence_check = self.coherence.enabled
            self.coherence_provider = self.coherence.provider
            self.coherence_model = self.coherence.model
            self.coherence_fail_closed = self.coherence.fail_closed
        else:
            self.coherence = CoherenceConfig(
                enabled=self.coherence_check,
                provider=self.coherence_provider,
                model=self.coherence_model,
                fail_closed=self.coherence_fail_closed,
            )

        if isinstance(self.guardrail, GuardrailsConfig):
            self.guardrails = self.guardrail.pipeline
            self.screen_tool_output = self.guardrail.screen_tool_output
            self.output_screening_patterns = self.guardrail.output_screening_patterns
        else:
            self.guardrail = GuardrailsConfig(
                pipeline=self.guardrails,
                screen_tool_output=self.screen_tool_output,
                output_screening_patterns=self.output_screening_patterns,
            )

        if isinstance(self.session, SessionConfig):
            self.session_store = self.session.store
            self.session_id = self.session.session_id
        else:
            self.session = SessionConfig(
                store=self.session_store,
                session_id=self.session_id,
            )

        if isinstance(self.summarize, SummarizeConfig):
            self.summarize_on_trim = self.summarize.enabled
            self.summarize_provider = self.summarize.provider
            self.summarize_model = self.summarize.model
            self.summarize_max_tokens = self.summarize.max_tokens
        else:
            self.summarize = SummarizeConfig(
                enabled=self.summarize_on_trim,
                provider=self.summarize_provider,
                model=self.summarize_model,
                max_tokens=self.summarize_max_tokens,
            )

        if isinstance(self.memory, MemoryConfig):
            self.entity_memory = self.memory.entity_memory
            self.knowledge_graph = self.memory.knowledge_graph
            self.knowledge_memory = self.memory.knowledge_memory
        else:
            self.memory = MemoryConfig(
                entity_memory=self.entity_memory,
                knowledge_graph=self.knowledge_graph,
                knowledge_memory=self.knowledge_memory,
            )

        if isinstance(self.budget, BudgetConfig):
            self.max_total_tokens = self.budget.max_total_tokens
            self.max_cost_usd = self.budget.max_cost_usd
            self.cost_warning_threshold = self.budget.cost_warning_threshold
            self.cancellation_token = self.budget.cancellation_token
        else:
            self.budget = BudgetConfig(
                max_total_tokens=self.max_total_tokens,
                max_cost_usd=self.max_cost_usd,
                cost_warning_threshold=self.cost_warning_threshold,
                cancellation_token=self.cancellation_token,
            )

        if isinstance(self.trace, TraceConfig):
            self.trace_tool_result_chars = self.trace.tool_result_chars
            self.trace_metadata = self.trace.metadata
            self.parent_run_id = self.trace.parent_run_id
        else:
            self.trace = TraceConfig(
                tool_result_chars=self.trace_tool_result_chars,
                metadata=self.trace_metadata,
                parent_run_id=self.parent_run_id,
            )

        if isinstance(self.compress, CompressConfig):
            self.compress_context = self.compress.enabled
            self.compress_threshold = self.compress.threshold
            self.compress_keep_recent = self.compress.keep_recent
        else:
            self.compress = CompressConfig(
                enabled=self.compress_context,
                threshold=self.compress_threshold,
                keep_recent=self.compress_keep_recent,
            )

        # Hooks deprecation (existing logic)
        if self.hooks is not None:
            import warnings

            warnings.warn(
                "AgentConfig.hooks is deprecated and will be removed in a future "
                "version. Use AgentConfig.observers with AgentObserver subclasses "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
