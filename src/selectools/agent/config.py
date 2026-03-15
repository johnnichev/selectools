"""
Configuration options for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ..cache import Cache
    from ..entity_memory import EntityMemory
    from ..guardrails import GuardrailsPipeline
    from ..knowledge import KnowledgeMemory
    from ..knowledge_graph import KnowledgeGraphMemory
    from ..observer import AgentObserver
    from ..policy import ToolPolicy
    from ..providers.base import Provider
    from ..sessions import SessionStore

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
        model: Model identifier to use (e.g., "gpt-4o", "claude-3-5-sonnet-20240620").
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
    """

    name: str = "agent"
    model: str = "gpt-4o"
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
    session_store: Optional[SessionStore] = None
    session_id: Optional[str] = None
    summarize_on_trim: bool = False
    summarize_provider: Optional[Provider] = None
    summarize_model: Optional[str] = None
    summarize_max_tokens: int = 150
    entity_memory: Optional[EntityMemory] = None
    knowledge_graph: Optional[KnowledgeGraphMemory] = None
    knowledge_memory: Optional[KnowledgeMemory] = None
