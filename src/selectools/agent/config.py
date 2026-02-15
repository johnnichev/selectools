"""
Configuration options for the agent.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

# Hook type definitions
HookCallable = Callable[..., None]
Hooks = Dict[str, HookCallable]


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
               - 'on_llm_end': Called after LLM call with (response, usage)
               - 'on_error': Called on any error with (error, context)
        routing_only: If True, returns tool selection without executing it. Default: False.
        parallel_tool_execution: Execute multiple tool calls concurrently when the LLM
               requests more than one tool in a single response. Uses asyncio.gather for
               async and ThreadPoolExecutor for sync execution. Default: True.
    """

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
    routing_only: bool = False
    parallel_tool_execution: bool = True
