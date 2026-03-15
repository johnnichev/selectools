"""
Provider-agnostic agent loop implementing the TOOL_CALL contract.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Union, cast

from ..analytics import AgentAnalytics
from ..cache import CacheKeyBuilder
from ..coherence import CoherenceResult, acheck_coherence, check_coherence
from ..guardrails.base import GuardrailError
from ..parser import ToolCallParser
from ..policy import PolicyDecision, ToolPolicy
from ..prompt import PromptBuilder
from ..providers.base import Provider, ProviderError
from ..providers.openai_provider import OpenAIProvider
from ..security import screen_output as screen_tool_output
from ..structured import (
    ResponseFormat,
    build_schema_instruction,
    parse_and_validate,
    schema_from_response_format,
    validation_retry_message,
)
from ..tools import Tool
from ..trace import AgentTrace, StepType, TraceStep
from ..types import AgentResult, Message, Role, StreamChunk, ToolCall
from ..usage import AgentUsage
from .config import AgentConfig

if TYPE_CHECKING:
    from ..memory import ConversationMemory


@dataclass
class _RunContext:
    """Private context object carrying state for a single run/arun/astream invocation."""

    trace: AgentTrace
    run_id: str
    original_system_prompt: str
    history_checkpoint: int
    response_format: Optional[ResponseFormat]
    user_text_for_coherence: str = ""
    iteration: int = 0
    all_tool_calls: List[ToolCall] = field(default_factory=list)
    last_tool_name: Optional[str] = None
    last_tool_args: Dict[str, Any] = field(default_factory=dict)
    reasoning_history: List[str] = field(default_factory=list)


class Agent:
    """
    Provider-agnostic AI agent that iteratively calls tools to accomplish tasks.

    The Agent manages the loop of:
    1. Sending conversation history to an LLM
    2. Parsing the response for tool calls
    3. Executing requested tools
    4. Repeating until the task is complete

    Supports both synchronous and asynchronous execution, conversation memory,
    streaming responses, and multiple LLM providers (OpenAI, Anthropic, Gemini, etc).

    Attributes:
        tools: List of available tools the agent can use.
        provider: LLM provider adapter (OpenAI, Anthropic, Gemini, or custom).
        prompt_builder: Generates system prompts with tool schemas.
        parser: Extracts tool calls from LLM responses.
        config: Configuration options for behavior customization.
        memory: Optional conversation memory for multi-turn dialogues.

    Example:
        >>> # Basic agent
        >>> tools = [search_tool, calculator_tool]
        >>> agent = Agent(tools=tools, provider=OpenAIProvider())
        >>>
        >>> response = agent.run([
        ...     Message(role=Role.USER, content="What's 2+2 and search for Python?")
        ... ])
        >>> print(response.content)

        >>> # Agent with memory for multi-turn conversations
        >>> memory = ConversationMemory(max_messages=20)
        >>> agent = Agent(tools=tools, provider=provider, memory=memory)
        >>>
        >>> # First turn
        >>> agent.run([Message(role=Role.USER, content="My name is Alice")])
        >>>
        >>> # Second turn - context is preserved
        >>> agent.run([Message(role=Role.USER, content="What's my name?")])

        >>> # Async execution for high-performance applications
        >>> response = await agent.arun([Message(...)])
    """

    def __init__(
        self,
        tools: List[Tool],
        provider: Optional[Provider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        parser: Optional[ToolCallParser] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        """
        Initialize a new Agent instance.

        Args:
            tools: List of Tool instances the agent can use (minimum 1 required).
            provider: LLM provider adapter. Defaults to OpenAIProvider().
            prompt_builder: Custom prompt builder. Defaults to PromptBuilder().
            parser: Custom tool call parser. Defaults to ToolCallParser().
            config: Agent configuration options. Defaults to AgentConfig().
            memory: Optional conversation memory for maintaining history across turns.

        Raises:
            ValueError: If tools list is empty.

        Example:
            >>> agent = Agent(
            ...     tools=[search, calculator],
            ...     provider=OpenAIProvider(api_key="sk-..."),
            ...     config=AgentConfig(model="gpt-4o", verbose=True),
            ...     memory=ConversationMemory(max_messages=20)
            ... )
        """
        if not tools:
            raise ValueError("Agent requires at least one tool.")

        self.tools = tools
        self._tools_by_name = {tool.name: tool for tool in tools}
        self.provider = provider or OpenAIProvider()
        if prompt_builder:
            self.prompt_builder = prompt_builder
        elif config and config.system_prompt:
            self.prompt_builder = PromptBuilder(base_instructions=config.system_prompt)
        else:
            self.prompt_builder = PromptBuilder()
        self.parser = parser or ToolCallParser()
        self.config = config or AgentConfig()
        self.memory = memory
        self.usage = AgentUsage()
        self.analytics = AgentAnalytics() if self.config.enable_analytics else None

        self._system_prompt = self.prompt_builder.build(self.tools)
        self._history: List[Message] = []

        # Auto-load session from store if configured (only if no memory was provided)
        if self.config.session_store and self.config.session_id and self.memory is None:
            loaded = self.config.session_store.load(self.config.session_id)
            if loaded is not None:
                self.memory = loaded

        # Auto-add remember tool if knowledge_memory is configured
        if self.config.knowledge_memory and "remember" not in self._tools_by_name:
            from ..toolbox.memory_tools import make_remember_tool

            remember_tool = make_remember_tool(self.config.knowledge_memory)
            self.tools.append(remember_tool)
            self._tools_by_name[remember_tool.name] = remember_tool
            self._system_prompt = self.prompt_builder.build(self.tools)

    @property
    def name(self) -> str:
        """Return the agent's name from config."""
        return self.config.name

    def __call__(
        self,
        messages: Union[str, List[Message]],
        **kwargs: Any,
    ) -> AgentResult:
        """Allow calling the agent directly as a shorthand for run()."""
        return self.run(messages, **kwargs)

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent at runtime.

        The system prompt is rebuilt to include the new tool's schema so that
        the LLM can immediately use it in subsequent iterations.

        Args:
            tool: Tool instance to add.

        Raises:
            ValueError: If a tool with the same name already exists.
        """
        if tool.name in self._tools_by_name:
            raise ValueError(
                f"Tool '{tool.name}' already exists. " f"Use replace_tool() to update it."
            )
        self.tools.append(tool)
        self._tools_by_name[tool.name] = tool
        self._system_prompt = self.prompt_builder.build(self.tools)

    def add_tools(self, tools: List[Tool]) -> None:
        """
        Add multiple tools to the agent at runtime.

        Args:
            tools: List of Tool instances to add.

        Raises:
            ValueError: If any tool name already exists.
        """
        for t in tools:
            if t.name in self._tools_by_name:
                raise ValueError(
                    f"Tool '{t.name}' already exists. " f"Use replace_tool() to update it."
                )
        for t in tools:
            self.tools.append(t)
            self._tools_by_name[t.name] = t
        self._system_prompt = self.prompt_builder.build(self.tools)

    def remove_tool(self, tool_name: str) -> Tool:
        """
        Remove a tool from the agent by name.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            The removed Tool instance.

        Raises:
            KeyError: If no tool with that name exists.
            ValueError: If removing would leave the agent with zero tools.
        """
        if tool_name not in self._tools_by_name:
            raise KeyError(
                f"Tool '{tool_name}' not found. "
                f"Available: {', '.join(self._tools_by_name.keys())}"
            )
        if len(self.tools) <= 1:
            raise ValueError("Agent requires at least one tool.")

        removed = self._tools_by_name.pop(tool_name)
        self.tools = [t for t in self.tools if t.name != tool_name]
        self._system_prompt = self.prompt_builder.build(self.tools)
        return removed

    def replace_tool(self, tool: Tool) -> Optional[Tool]:
        """
        Replace an existing tool with an updated version (same or new name).

        If a tool with the given name already exists it is swapped out;
        otherwise the tool is simply added.

        Args:
            tool: The new Tool instance.

        Returns:
            The old Tool instance that was replaced, or ``None`` if no tool
            with that name existed.
        """
        old = self._tools_by_name.get(tool.name)
        if old is not None:
            self.tools = [t if t.name != tool.name else tool for t in self.tools]
        else:
            self.tools.append(tool)
        self._tools_by_name[tool.name] = tool
        self._system_prompt = self.prompt_builder.build(self.tools)
        return old

    def reset(self) -> None:
        """Reset agent state for reuse. Clears history, usage stats, and memory."""
        self._history = []
        self.usage = AgentUsage()
        if self.analytics:
            self.analytics = AgentAnalytics()
        if self.memory:
            self.memory.clear()

    def _call_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Safely call a hook if it exists, swallowing any exceptions."""
        if not self.config.hooks or hook_name not in self.config.hooks:
            return
        try:
            self.config.hooks[hook_name](*args, **kwargs)
        except Exception:  # noqa: BLE001 # nosec B110
            pass

    def _notify_observers(self, method: str, *args: Any) -> None:
        """Call *method* on every registered observer, swallowing errors."""
        for obs in self.config.observers:
            try:
                getattr(obs, method)(*args)
            except Exception:  # noqa: BLE001 # nosec B110
                pass

    def _truncate_tool_result(self, result: Optional[str]) -> Optional[str]:
        """Truncate tool result text for trace storage."""
        if result is None:
            return None
        limit = self.config.trace_tool_result_chars
        if limit is None:
            return result
        return result[:limit]

    _fallback_run_id: threading.local = threading.local()
    _lock_type: type = type(threading.Lock())

    def _wire_fallback_observer(self, run_id: Optional[str]) -> None:
        """If the provider is a FallbackProvider, wire its on_fallback to observers.

        Thread-safe: uses a lock + refcount so multiple concurrent ``run()``
        calls (e.g. from ``batch()``) share a single callback on the provider
        while each thread's ``run_id`` is carried via a thread-local.
        The lock persists on the provider to avoid races from delete-then-recreate.
        """
        if not run_id or not self.config.observers:
            return
        provider = self.provider
        if not hasattr(provider, "on_fallback"):
            return

        Agent._fallback_run_id.value = run_id

        raw_lock = getattr(provider, "_fb_wire_lock", None)
        if not isinstance(raw_lock, Agent._lock_type):
            raw_lock = threading.Lock()
            provider._fb_wire_lock = raw_lock  # type: ignore[attr-defined]
        lock = cast(threading.Lock, raw_lock)

        agent_ref = self

        with lock:
            refcount: int = getattr(provider, "_fb_wire_refcount", 0)
            if refcount == 0:
                provider._fb_original_on_fallback = provider.on_fallback  # type: ignore[attr-defined]
                user_cb = provider.on_fallback

                def _observer_fallback(
                    failed: str,
                    next_p: str,
                    exc: Exception,
                ) -> None:
                    rid = getattr(Agent._fallback_run_id, "value", "")
                    agent_ref._notify_observers(
                        "on_provider_fallback",
                        rid,
                        failed,
                        next_p,
                        exc,
                    )
                    if user_cb:
                        try:
                            user_cb(failed, next_p, exc)
                        except Exception:  # nosec B110
                            pass

                provider.on_fallback = _observer_fallback  # type: ignore[attr-defined]

            provider._fb_wire_refcount = refcount + 1  # type: ignore[attr-defined]

    def _unwire_fallback_observer(self) -> None:
        """Restore FallbackProvider's original on_fallback callback (thread-safe).

        The lock is kept on the provider (never deleted) to prevent races when
        concurrent threads overlap wire / unwire calls.
        """
        provider = self.provider
        raw_lock = getattr(provider, "_fb_wire_lock", None)
        if not isinstance(raw_lock, Agent._lock_type):
            return
        lock = cast(threading.Lock, raw_lock)

        with lock:
            refcount: int = getattr(provider, "_fb_wire_refcount", 0) - 1
            if refcount < 0:
                refcount = 0
            provider._fb_wire_refcount = refcount  # type: ignore[attr-defined]
            if refcount == 0:
                original = getattr(provider, "_fb_original_on_fallback", None)
                provider.on_fallback = original  # type: ignore[attr-defined]
                if hasattr(provider, "_fb_original_on_fallback"):
                    try:
                        delattr(provider, "_fb_original_on_fallback")
                    except Exception:  # nosec B110
                        pass

    def _memory_add(self, msg: Message, run_id: str) -> None:
        """Add message to memory and notify observers if trimming occurred."""
        if not self.memory:
            return
        before = len(self.memory)
        self.memory.add(msg)
        after = len(self.memory)
        removed = (before + 1) - after
        if removed > 0:
            self._notify_observers(
                "on_memory_trim",
                run_id,
                removed,
                after,
                "enforce_limits",
            )
            self._maybe_summarize_trim(run_id)

    def _memory_add_many(self, msgs: List[Message], run_id: str) -> None:
        """Add multiple messages to memory and notify observers if trimming occurred."""
        if not self.memory or not msgs:
            return
        before = len(self.memory)
        self.memory.add_many(msgs)
        after = len(self.memory)
        removed = (before + len(msgs)) - after
        if removed > 0:
            self._notify_observers(
                "on_memory_trim",
                run_id,
                removed,
                after,
                "enforce_limits",
            )
            self._maybe_summarize_trim(run_id)

    def _maybe_summarize_trim(self, run_id: str) -> None:
        """Generate a summary of trimmed messages if summarize_on_trim is enabled."""
        if not self.config.summarize_on_trim or not self.memory:
            return
        trimmed = self.memory._last_trimmed
        if not trimmed:
            return
        try:
            provider = self.config.summarize_provider or self.provider
            model = self.config.summarize_model or self.config.model
            text_parts = []
            for m in trimmed:
                prefix = m.role.value.upper()
                text_parts.append(f"{prefix}: {m.content or ''}")
            trimmed_text = "\n".join(text_parts)

            prompt_msg = Message(
                role=Role.USER,
                content=(
                    "Summarize the following conversation excerpt in 2-3 sentences. "
                    "Focus on key facts, decisions, and context that would be useful "
                    "for continuing the conversation:\n\n" + trimmed_text
                ),
            )
            result = provider.complete(
                model=model,
                system_prompt="You are a concise summarizer.",
                messages=[prompt_msg],
                max_tokens=self.config.summarize_max_tokens,
            )
            # Provider returns (Message, UsageStats) tuple
            summary_msg = result[0] if isinstance(result, tuple) else result
            summary_text = summary_msg.content or ""
            if summary_text:
                existing = self.memory.summary
                if existing:
                    self.memory.summary = existing + " " + summary_text
                else:
                    self.memory.summary = summary_text
                self._notify_observers("on_memory_summarize", run_id, self.memory.summary)
        except Exception:  # nosec B110
            pass  # never crash the agent for a summarization failure

    def _session_save(self, run_id: str) -> None:
        """Auto-save memory to session store if configured."""
        store = self.config.session_store
        sid = self.config.session_id
        if not store or not sid or not self.memory:
            return
        try:
            store.save(sid, self.memory)
            self._notify_observers("on_session_save", run_id, sid, len(self.memory))
        except Exception:  # nosec B110
            pass  # never crash the agent for a persistence failure

    def _extract_entities(self, run_id: str) -> None:
        """Extract entities from recent messages if entity_memory is configured."""
        em = self.config.entity_memory
        if not em:
            return
        try:
            recent = self._history[-em._relevance_window :]
            entities = em.extract_entities(recent, model=self.config.model)
            if entities:
                em.update(entities)
                self._notify_observers(
                    "on_entity_extraction",
                    run_id,
                    len(entities),
                )
        except Exception:  # nosec B110
            pass  # never crash the agent for entity extraction failure

    def _extract_kg_triples(self, run_id: str) -> None:
        """Extract relationship triples from recent messages if knowledge_graph is configured."""
        kg = self.config.knowledge_graph
        if not kg:
            return
        try:
            recent = self._history[-kg._relevance_window :]
            triples = kg.extract_triples(recent, model=self.config.model)
            if triples:
                kg.store.add_many(triples)
                self._notify_observers(
                    "on_kg_extraction",
                    run_id,
                    len(triples),
                )
        except Exception:  # nosec B110
            pass  # never crash the agent for KG extraction failure

    def _prepare_run(
        self,
        messages: List[Message],
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
    ) -> _RunContext:
        """Shared setup for run(), arun(), and astream().

        Saves original system prompt, applies response_format, creates trace,
        wires observers, runs input guardrails, loads memory/session, injects
        knowledge context, and returns a _RunContext carrying all state.
        """
        original_system_prompt = self._system_prompt
        if response_format is not None:
            schema = schema_from_response_format(response_format)
            self._system_prompt = self._system_prompt + build_schema_instruction(schema)

        trace = self._new_trace()
        if parent_run_id is not None:
            trace.parent_run_id = parent_run_id
        run_id = trace.run_id
        history_checkpoint = len(self._history)

        self._wire_fallback_observer(run_id)
        self._notify_observers("on_run_start", run_id, messages, self._system_prompt)

        # Input guardrails
        for msg in messages:
            if msg.role == Role.USER and msg.content:
                msg.content = self._run_input_guardrails(msg.content, trace)

        # Memory / session loading
        if self.memory:
            if self.config.session_store and self.config.session_id:
                self._notify_observers(
                    "on_session_load", run_id, self.config.session_id, len(self.memory)
                )
            self._history = self.memory.get_history() + list(messages)
            self._memory_add_many(list(messages), run_id)
            if self.memory.summary:
                self._history.insert(
                    0,
                    Message(
                        role=Role.SYSTEM,
                        content=f"[Conversation Summary] {self.memory.summary}",
                    ),
                )
        else:
            self._history.extend(messages)

        # Knowledge memory context
        if self.config.knowledge_memory:
            km_ctx = self.config.knowledge_memory.build_context()
            if km_ctx:
                self._history.insert(
                    0,
                    Message(role=Role.SYSTEM, content=km_ctx),
                )

        # Entity memory context
        if self.config.entity_memory:
            entity_ctx = self.config.entity_memory.build_context()
            if entity_ctx:
                self._history.insert(
                    0,
                    Message(role=Role.SYSTEM, content=entity_ctx),
                )

        # Extract user text for coherence checks
        user_text_for_coherence = ""
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                user_text_for_coherence = msg.content
                break

        # Knowledge graph context
        if self.config.knowledge_graph:
            kg_ctx = self.config.knowledge_graph.build_context(query=user_text_for_coherence)
            if kg_ctx:
                self._history.insert(
                    0,
                    Message(role=Role.SYSTEM, content=kg_ctx),
                )

        return _RunContext(
            trace=trace,
            run_id=run_id,
            original_system_prompt=original_system_prompt,
            history_checkpoint=history_checkpoint,
            response_format=response_format,
            user_text_for_coherence=user_text_for_coherence,
        )

    def _finalize_run(
        self,
        ctx: _RunContext,
        final_response: Message,
        parsed: Any = None,
    ) -> AgentResult:
        """Shared teardown for run(), arun(), and astream().

        Appends final response to history, saves session, extracts entities/KG,
        and builds the AgentResult.
        """
        self._history.append(final_response)
        self._memory_add(final_response, ctx.run_id)
        self._extract_entities(ctx.run_id)
        self._extract_kg_triples(ctx.run_id)
        self._session_save(ctx.run_id)
        self._call_hook("on_agent_end", final_response, self.usage)
        result = AgentResult(
            message=final_response,
            tool_name=ctx.last_tool_name,
            tool_args=ctx.last_tool_args,
            iterations=ctx.iteration,
            tool_calls=ctx.all_tool_calls,
            parsed=parsed,
            reasoning=ctx.reasoning_history[-1] if ctx.reasoning_history else None,
            reasoning_history=ctx.reasoning_history,
            trace=ctx.trace,
            provider_used=getattr(self.provider, "provider_used", None),
            usage=copy.copy(self.usage),
        )
        self._notify_observers("on_run_end", ctx.run_id, result)
        return result

    def _build_max_iterations_result(self, ctx: _RunContext) -> AgentResult:
        """Build the AgentResult for when max iterations are reached."""
        final_response = Message(
            role=Role.ASSISTANT,
            content=f"Maximum iterations ({self.config.max_iterations}) reached without resolution.",
        )
        self._history.append(final_response)
        self._memory_add(final_response, ctx.run_id)
        self._call_hook("on_agent_end", final_response, self.usage)
        result = AgentResult(
            message=final_response,
            tool_name=ctx.last_tool_name,
            tool_args=ctx.last_tool_args,
            iterations=ctx.iteration,
            tool_calls=ctx.all_tool_calls,
            reasoning=ctx.reasoning_history[-1] if ctx.reasoning_history else None,
            reasoning_history=ctx.reasoning_history,
            trace=ctx.trace,
            provider_used=getattr(self.provider, "provider_used", None),
            usage=copy.copy(self.usage),
        )
        self._notify_observers("on_run_end", ctx.run_id, result)
        return result

    def _process_response(
        self,
        ctx: _RunContext,
        response_msg: Message,
    ) -> tuple:
        """Shared post-provider response processing for run(), arun(), and astream().

        Applies output guardrails, extracts tool calls (with response_format guard),
        extracts reasoning, adds tool_selection trace steps.

        Returns (response_text, tool_calls_to_execute, reasoning_text).
        """
        response_text = response_msg.content or ""
        response_text = self._run_output_guardrails(response_text, ctx.trace)
        response_msg.content = response_text

        tool_calls_to_execute: List[ToolCall] = []
        if response_msg.tool_calls:
            tool_calls_to_execute = response_msg.tool_calls
        elif ctx.response_format is None:
            parse_result = self.parser.parse(response_text)
            if parse_result.tool_call:
                tool_calls_to_execute.append(parse_result.tool_call)

        reasoning_text = self._extract_reasoning(response_msg, tool_calls_to_execute)
        if reasoning_text:
            ctx.reasoning_history.append(reasoning_text)

        if tool_calls_to_execute:
            for tc in tool_calls_to_execute:
                ctx.trace.add(
                    TraceStep(
                        type="tool_selection",
                        tool_name=tc.tool_name,
                        tool_args=tc.parameters,
                        reasoning=reasoning_text,
                        summary=f"Selected {tc.tool_name}",
                    )
                )

        return response_text, tool_calls_to_execute, reasoning_text

    def _run_input_guardrails(self, content: str, trace: Optional[AgentTrace] = None) -> str:
        """Run input guardrails on user content.  Returns (possibly rewritten) content."""
        if not self.config.guardrails or not self.config.guardrails.input:
            return content
        result = self.config.guardrails.check_input(content)
        if trace and not result.passed:
            trace.add(
                TraceStep(
                    type="guardrail",
                    summary=f"Input guardrail: {result.reason}",
                )
            )
        return result.content

    def _run_output_guardrails(self, content: str, trace: Optional[AgentTrace] = None) -> str:
        """Run output guardrails on LLM response.  Returns (possibly rewritten) content."""
        if not self.config.guardrails or not self.config.guardrails.output:
            return content
        result = self.config.guardrails.check_output(content)
        if trace and not result.passed:
            trace.add(
                TraceStep(
                    type="guardrail",
                    summary=f"Output guardrail: {result.reason}",
                )
            )
        return result.content

    def _screen_tool_result(self, tool_name: str, result: str) -> str:
        """Screen a tool result for prompt injection if the tool or config requires it."""
        tool = self._tools_by_name.get(tool_name)
        should_screen = self.config.screen_tool_output or (
            tool is not None and getattr(tool, "screen_output", False)
        )
        if not should_screen:
            return result
        screening = screen_tool_output(
            result,
            extra_patterns=self.config.output_screening_patterns,
        )
        return screening.content

    def _check_coherence(
        self,
        user_message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Sync coherence check.  Returns error string or None."""
        if not self.config.coherence_check:
            return None
        provider = self.config.coherence_provider or self.provider
        model = self.config.coherence_model or self.config.model
        result = check_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
        )
        if not result.coherent:
            return (
                f"Coherence check failed for tool '{tool_name}': "
                f"{result.explanation or 'Tool call does not match user intent'}"
            )
        return None

    async def _acheck_coherence(
        self,
        user_message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Async coherence check.  Returns error string or None."""
        if not self.config.coherence_check:
            return None
        provider = self.config.coherence_provider or self.provider
        model = self.config.coherence_model or self.config.model
        result = await acheck_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
        )
        if not result.coherent:
            return (
                f"Coherence check failed for tool '{tool_name}': "
                f"{result.explanation or 'Tool call does not match user intent'}"
            )
        return None

    def _new_trace(self) -> AgentTrace:
        """Create an ``AgentTrace`` pre-configured from ``AgentConfig``."""
        return AgentTrace(
            parent_run_id=self.config.parent_run_id,
            metadata=dict(self.config.trace_metadata),
        )

    @staticmethod
    def _normalize_messages(messages: Union[str, List[Message]]) -> List[Message]:
        """Convert a string prompt to a message list, or pass through as-is."""
        if isinstance(messages, str):
            return [Message(role=Role.USER, content=messages)]
        return messages

    @staticmethod
    def _extract_reasoning(response_msg: Message, tool_calls: List[ToolCall]) -> Optional[str]:
        """Return explanatory text the LLM sent alongside tool calls.

        Providers typically return both text content and tool_use blocks in
        the same response.  The text portion explains *why* the tool was
        chosen.  If no tool calls were made or the text is empty, returns
        ``None``.
        """
        if not tool_calls:
            return None
        text = (response_msg.content or "").strip()
        return text if text else None

    def _check_policy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        run_id: str = "",
    ) -> Optional[str]:
        """Evaluate tool policy and confirm_action. Returns error string or None."""
        if not self.config.tool_policy:
            return None

        result = self.config.tool_policy.evaluate(tool_name, tool_args)
        decision_str = result.decision.value

        if run_id:
            self._notify_observers(
                "on_policy_decision",
                run_id,
                tool_name,
                decision_str,
                result.reason,
                tool_args,
            )

        if result.decision == PolicyDecision.ALLOW:
            return None

        if result.decision == PolicyDecision.DENY:
            return f"Tool '{tool_name}' denied by policy: {result.reason}"

        if result.decision == PolicyDecision.REVIEW:
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.config.confirm_action, tool_name, tool_args, result.reason
                    )
                    try:
                        approved = future.result(timeout=self.config.approval_timeout)
                    except FuturesTimeoutError:
                        return (
                            f"Tool '{tool_name}' approval timed out "
                            f"after {self.config.approval_timeout}s"
                        )
                if not approved:
                    return f"Tool '{tool_name}' rejected by reviewer: {result.reason}"
            except FuturesTimeoutError:
                return (
                    f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
                )
            except Exception as exc:
                return f"Tool '{tool_name}' approval failed: {exc}"

        return None

    async def _acheck_policy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        run_id: str = "",
    ) -> Optional[str]:
        """Async version of _check_policy."""
        if not self.config.tool_policy:
            return None

        result = self.config.tool_policy.evaluate(tool_name, tool_args)
        decision_str = result.decision.value

        if run_id:
            self._notify_observers(
                "on_policy_decision",
                run_id,
                tool_name,
                decision_str,
                result.reason,
                tool_args,
            )

        if result.decision == PolicyDecision.ALLOW:
            return None

        if result.decision == PolicyDecision.DENY:
            return f"Tool '{tool_name}' denied by policy: {result.reason}"

        if result.decision == PolicyDecision.REVIEW:
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            try:
                import asyncio
                import inspect

                if inspect.iscoroutinefunction(self.config.confirm_action):
                    approved = await asyncio.wait_for(
                        self.config.confirm_action(tool_name, tool_args, result.reason),
                        timeout=self.config.approval_timeout,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    approved = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            self.config.confirm_action,
                            tool_name,
                            tool_args,
                            result.reason,
                        ),
                        timeout=self.config.approval_timeout,
                    )
                if not approved:
                    return f"Tool '{tool_name}' rejected by reviewer: {result.reason}"
            except asyncio.TimeoutError:
                return (
                    f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
                )
            except Exception as exc:
                return f"Tool '{tool_name}' approval failed: {exc}"

        return None

    def ask(
        self,
        prompt: str,
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> AgentResult:
        """
        Send a single user prompt and return the agent's response.

        Convenience wrapper around :meth:`run` that removes the need to
        construct ``Message`` and ``Role`` objects manually.

        Args:
            prompt: Plain-text question or instruction.
            stream_handler: Optional callback for streaming responses.
            response_format: Optional Pydantic model class or JSON Schema dict.
                When provided the LLM is instructed to return valid JSON and
                the result is validated. Access via ``result.parsed``.

        Returns:
            AgentResult with the response and metadata.

        Example:
            >>> result = agent.ask("What is the capital of France?")
            >>> print(result.content)
            Paris
        """
        return self.run(
            [Message(role=Role.USER, content=prompt)],
            stream_handler=stream_handler,
            response_format=response_format,
        )

    async def aask(
        self,
        prompt: str,
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> AgentResult:
        """
        Async version of :meth:`ask`.

        Args:
            prompt: Plain-text question or instruction.
            stream_handler: Optional callback for streaming responses.
            response_format: Optional Pydantic model class or JSON Schema dict.

        Returns:
            AgentResult with the response and metadata.

        Example:
            >>> result = await agent.aask("What is the capital of France?")
            >>> print(result.content)
        """
        return await self.arun(
            [Message(role=Role.USER, content=prompt)],
            stream_handler=stream_handler,
            response_format=response_format,
        )

    def batch(
        self,
        prompts: List[str],
        max_concurrency: int = 5,
        response_format: Optional[ResponseFormat] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[AgentResult]:
        """
        Process multiple prompts concurrently using a thread pool.

        Each prompt is processed in full isolation — a lightweight clone of the
        agent is used per item so that concurrent threads never share ``_history``
        or ``usage`` state.  Results are returned in the same order as the input
        prompts.  Individual failures are captured per-result without cancelling
        the rest of the batch.

        Token usage from all items is aggregated back to ``self.usage`` after
        the batch completes.

        Args:
            prompts: List of plain-text prompts.
            max_concurrency: Maximum parallel requests. Default: 5.
            response_format: Optional Pydantic model or JSON Schema applied to each prompt.
            on_progress: Optional ``(completed, total)`` callback.

        Returns:
            List of AgentResult in the same order as *prompts*.
        """
        batch_id = uuid.uuid4().hex
        batch_start = time.time()
        self._notify_observers("on_batch_start", batch_id, len(prompts))

        usage_lock = threading.Lock()
        progress_lock = threading.Lock()
        completed = 0

        def _run_one(prompt: str) -> AgentResult:
            nonlocal completed
            clone = self._clone_for_isolation()
            try:
                result = clone.run(prompt, response_format=response_format)
            except Exception as exc:
                result = AgentResult(
                    message=Message(role=Role.ASSISTANT, content=f"Batch error: {exc}"),
                )
            with usage_lock:
                self.usage.merge(clone.usage)
            with progress_lock:
                completed += 1
                count = completed
            if on_progress:
                try:
                    on_progress(count, len(prompts))
                except Exception:  # nosec B110
                    pass
            return result

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(_run_one, p) for p in prompts]
            results = [cast(AgentResult, f.result()) for f in futures]

        errors = sum(1 for r in results if r.content and r.content.startswith("Batch error:"))
        self._notify_observers(
            "on_batch_end",
            batch_id,
            len(results),
            errors,
            (time.time() - batch_start) * 1000,
        )
        return results

    async def abatch(
        self,
        prompts: List[str],
        max_concurrency: int = 10,
        response_format: Optional[ResponseFormat] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[AgentResult]:
        """
        Async version of :meth:`batch`.

        Each prompt runs in an isolated clone of the agent (fresh ``_history``
        and ``usage``).  Usage is aggregated back to ``self.usage`` after the
        batch completes.

        Args:
            prompts: List of plain-text prompts.
            max_concurrency: Maximum parallel requests. Default: 10.
            response_format: Optional Pydantic model or JSON Schema applied to each prompt.
            on_progress: Optional ``(completed, total)`` callback.

        Returns:
            List of AgentResult in the same order as *prompts*.
        """
        batch_id = uuid.uuid4().hex
        batch_start = time.time()
        self._notify_observers("on_batch_start", batch_id, len(prompts))

        semaphore = asyncio.Semaphore(max_concurrency)
        completed = 0

        async def _run_one(prompt: str) -> AgentResult:
            nonlocal completed
            clone = self._clone_for_isolation()
            async with semaphore:
                try:
                    result = await clone.arun(prompt, response_format=response_format)
                except Exception as exc:
                    result = AgentResult(
                        message=Message(role=Role.ASSISTANT, content=f"Batch error: {exc}"),
                    )
            self.usage.merge(clone.usage)
            completed += 1
            if on_progress:
                try:
                    on_progress(completed, len(prompts))
                except Exception:  # nosec B110
                    pass
            return result

        results = list(await asyncio.gather(*[_run_one(p) for p in prompts]))
        errors = sum(1 for r in results if r.content and r.content.startswith("Batch error:"))
        self._notify_observers(
            "on_batch_end",
            batch_id,
            len(results),
            errors,
            (time.time() - batch_start) * 1000,
        )
        return results

    def run(
        self,
        messages: Union[str, List[Message]],
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute the agent loop with the provided conversation history.

        If the agent was initialized with a ConversationMemory, the new messages
        will be appended to the existing memory history, and the final response
        will be automatically saved to memory.

        Args:
            messages: A plain-text prompt (str) or list of Message objects.
                A string is automatically wrapped as a single user message.
            stream_handler: Optional callback for streaming responses.
            response_format: Optional Pydantic model class or JSON Schema dict.
                When provided the LLM is instructed to return valid JSON and
                the result is validated and available via ``result.parsed``.

        Returns:
            AgentResult with the final response and tool call metadata.

        Examples:
            >>> result = agent.run("What is Python?")
            >>> result = agent.run([Message(role=Role.USER, content="Hello")])
        """
        messages = self._normalize_messages(messages)
        self._call_hook("on_agent_start", messages)
        ctx = self._prepare_run(
            messages, response_format=response_format, parent_run_id=parent_run_id
        )

        try:
            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1
                self._call_hook("on_iteration_start", ctx.iteration, self._history)
                self._notify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )

                response_msg = self._call_provider(
                    stream_handler=stream_handler, trace=ctx.trace, run_id=ctx.run_id
                )
                response_text, tool_calls_to_execute, reasoning_text = self._process_response(
                    ctx, response_msg
                )

                if not tool_calls_to_execute:
                    parsed = None
                    if ctx.response_format is not None:
                        try:
                            parsed = parse_and_validate(response_text, ctx.response_format)
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                True,
                                ctx.iteration,
                            )
                        except (ValueError, TypeError) as exc:
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                False,
                                ctx.iteration,
                                str(exc),
                            )
                            if ctx.iteration < self.config.max_iterations:
                                ctx.trace.add(
                                    TraceStep(
                                        type="structured_retry",
                                        error=str(exc),
                                        summary=f"Validation failed: {exc}",
                                    )
                                )
                                retry_msg = Message(
                                    role=Role.USER,
                                    content=validation_retry_message(exc),
                                )
                                self._history.append(
                                    Message(role=Role.ASSISTANT, content=response_text)
                                )
                                self._history.append(retry_msg)
                                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._call_hook("on_iteration_end", ctx.iteration, response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    return self._finalize_run(ctx, final_response, parsed=parsed)

                if self.config.routing_only and tool_calls_to_execute:
                    self._call_hook("on_iteration_end", ctx.iteration, response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=ctx.iteration,
                        tool_calls=tool_calls_to_execute,
                        reasoning=reasoning_text,
                        reasoning_history=ctx.reasoning_history,
                        trace=ctx.trace,
                        provider_used=getattr(self.provider, "provider_used", None),
                        usage=copy.copy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = self._execute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        response_text,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                else:
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        call_id = tool_call.id or ""
                        ctx.all_tool_calls.append(tool_call)
                        ctx.last_tool_name = tool_name
                        ctx.last_tool_args = parameters

                        if self.config.verbose:
                            print(
                                f"[agent] Iteration {ctx.iteration}: tool={tool_name} params={parameters}"
                            )

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=error_message,
                                    summary=f"Unknown tool {tool_name}",
                                )
                            )
                            continue

                        policy_error = self._check_policy(tool_name, parameters, ctx.run_id)
                        if policy_error:
                            self._append_tool_result(
                                policy_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=policy_error,
                                    summary=f"Policy denied {tool_name}",
                                )
                            )
                            continue

                        coherence_error = self._check_coherence(
                            ctx.user_text_for_coherence, tool_name, parameters
                        )
                        if coherence_error:
                            self._append_tool_result(
                                coherence_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=coherence_error,
                                    summary=f"Coherence check failed for {tool_name}",
                                )
                            )
                            continue

                        try:
                            start_time = time.time()
                            self._call_hook("on_tool_start", tool_name, parameters)
                            self._notify_observers(
                                "on_tool_start", ctx.run_id, call_id, tool_name, parameters
                            )

                            chunk_counter = {"count": 0}

                            def chunk_callback(chunk: str) -> None:
                                chunk_counter["count"] += 1
                                self._call_hook("on_tool_chunk", tool_name, chunk)
                                self._notify_observers(
                                    "on_tool_chunk",
                                    ctx.run_id,
                                    call_id,
                                    tool_name,
                                    chunk,
                                )

                            result = self._execute_tool_with_timeout(
                                tool, parameters, chunk_callback
                            )
                            result = self._screen_tool_result(tool_name, result)
                            duration = time.time() - start_time
                            self._call_hook("on_tool_end", tool_name, result, duration)
                            self._notify_observers(
                                "on_tool_end",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                result,
                                duration * 1000,
                            )

                            ctx.trace.add(
                                TraceStep(
                                    type="tool_execution",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    tool_args=parameters,
                                    tool_result=self._truncate_tool_result(result),
                                    summary=f"{tool_name} → {len(result)} chars",
                                )
                            )

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=True,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter["count"],
                                )
                            if self.usage.iterations:
                                self.usage.tool_usage[tool.name] = (
                                    self.usage.tool_usage.get(tool.name, 0) + 1
                                )
                                self.usage.tool_tokens[tool.name] = (
                                    self.usage.tool_tokens.get(tool.name, 0)
                                    + self.usage.iterations[-1].total_tokens
                                )

                            self._append_tool_result(
                                result,
                                tool_name,
                                tool_call.id,
                                tool_result=result,
                                run_id=ctx.run_id,
                            )

                        except Exception as exc:
                            duration = time.time() - start_time
                            self._call_hook("on_tool_error", tool_name, exc, parameters)
                            self._notify_observers(
                                "on_tool_error",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                exc,
                                parameters,
                                duration * 1000,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    error=str(exc),
                                    summary=f"{tool_name} failed: {exc}",
                                )
                            )
                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=False,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=0,
                                )

                            error_message = f"Error executing tool '{tool_name}': {exc}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )

                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )
                continue

            return self._build_max_iterations_result(ctx)
        except Exception as exc:
            if not self.memory:
                self._history = self._history[: ctx.history_checkpoint]
            self._call_hook("on_error", exc, {"messages": messages, "iteration": ctx.iteration})
            self._notify_observers(
                "on_error",
                ctx.run_id,
                exc,
                {"messages": messages, "iteration": ctx.iteration},
            )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = ctx.original_system_prompt

    def _call_provider(
        self,
        stream_handler: Optional[Callable[[str], None]] = None,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> Message:
        call_start = time.time()

        cache_key: Optional[str] = None
        if self.config.cache and not (
            self.config.stream and getattr(self.provider, "supports_streaming", False)
        ):
            cache_key = CacheKeyBuilder.build(
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
            )
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                cached_msg = cast(Message, cached[0])
                cached_usage = cached[1]
                self.usage.add_usage(cached_usage, tool_name=None)
                self._call_hook("on_llm_start", self._history, self.config.model)
                self._call_hook("on_llm_end", cached_msg.content, cached_usage)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self.config.model,
                        self._system_prompt,
                    )
                    self._notify_observers(
                        "on_llm_end",
                        run_id,
                        cached_msg.content,
                        cached_usage,
                    )
                    self._notify_observers(
                        "on_cache_hit",
                        run_id,
                        self.config.model,
                        cached_msg.content or "",
                    )
                    self._notify_observers("on_usage", run_id, cached_usage)
                if self.config.verbose:
                    print("[agent] cache hit -- skipping provider call")
                if trace is not None:
                    trace.add(
                        TraceStep(
                            type="cache_hit",
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self.config.model,
                            summary=f"Cache hit: {self.config.model}",
                        )
                    )
                return cached_msg

        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                self._call_hook("on_llm_start", self._history, self.config.model)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self.config.model,
                        self._system_prompt,
                    )

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = self._streaming_call(stream_handler=stream_handler)
                    self._call_hook("on_llm_end", response_text, None)
                    if run_id:
                        self._notify_observers("on_llm_end", run_id, response_text, None)
                    return Message(role=Role.ASSISTANT, content=response_text)

                response_msg, usage_stats = self.provider.complete(
                    model=self.config.model,
                    system_prompt=self._system_prompt,
                    messages=self._history,
                    tools=self.tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout,
                )
                response_text = response_msg.content or ""

                self.usage.add_usage(usage_stats, tool_name=None)

                if cache_key is not None and self.config.cache:
                    self.config.cache.set(cache_key, (response_msg, usage_stats))

                self._call_hook("on_llm_end", response_text, usage_stats)
                if run_id:
                    self._notify_observers("on_llm_end", run_id, response_text, usage_stats)
                    self._notify_observers("on_usage", run_id, usage_stats)

                if (
                    self.config.cost_warning_threshold
                    and self.usage.total_cost_usd > self.config.cost_warning_threshold
                ):
                    print(
                        f"\n⚠️  Cost Warning: Total cost ${self.usage.total_cost_usd:.6f} "
                        f"exceeds threshold ${self.config.cost_warning_threshold:.6f}\n"
                    )

                if self.config.verbose:
                    print(
                        f"[agent] tokens: {usage_stats.total_tokens:,} "
                        f"(prompt: {usage_stats.prompt_tokens:,}, completion: {usage_stats.completion_tokens:,}), "
                        f"cost: ${usage_stats.cost_usd:.6f}"
                    )

                if trace is not None:
                    trace.add(
                        TraceStep(
                            type="llm_call",
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self.config.model,
                            prompt_tokens=usage_stats.prompt_tokens,
                            completion_tokens=usage_stats.completion_tokens,
                            summary=f"{self.config.model} → {len(response_text)} chars",
                        )
                    )
                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                backoff = 0.0
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    backoff += self.config.rate_limit_cooldown_seconds * attempts
                if self.config.retry_backoff_seconds:
                    backoff += self.config.retry_backoff_seconds * attempts
                if run_id:
                    self._notify_observers(
                        "on_llm_retry",
                        run_id,
                        attempts,
                        self.config.max_retries,
                        exc,
                        backoff,
                    )
                if backoff > 0:
                    time.sleep(backoff)

        if trace is not None:
            trace.add(
                TraceStep(
                    type="llm_call",
                    duration_ms=(time.time() - call_start) * 1000,
                    model=self.config.model,
                    error=last_error,
                    summary=f"Provider error: {last_error}",
                )
            )
        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    def _streaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []
        for chunk in self.provider.stream(
            model=self.config.model,
            system_prompt=self._system_prompt,
            messages=self._history,
            tools=self.tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
        ):
            if chunk:
                aggregated.append(str(chunk))
                if stream_handler:
                    stream_handler(str(chunk))

        return "".join(aggregated)

    def _append_tool_result(
        self,
        tool_content: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tool_result: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Update history with tool output."""
        tool_msg = Message(
            role=Role.TOOL,
            content=tool_content,
            tool_name=tool_name,
            tool_result=tool_result,
            tool_call_id=tool_call_id,
        )
        self._history.append(tool_msg)
        self._memory_add(tool_msg, run_id or "")

    # ------------------------------------------------------------------
    # Parallel tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> tuple:
        """Execute multiple tool calls concurrently using ThreadPoolExecutor.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int

        def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            policy_error = self._check_policy(tool_name, parameters, run_id or "")
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            call_id = tc.id or ""
            start = time.time()
            self._call_hook("on_tool_start", tool_name, parameters)
            if run_id:
                self._notify_observers("on_tool_start", run_id, call_id, tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._call_hook("on_tool_chunk", tool_name, chunk)
                if run_id:
                    self._notify_observers(
                        "on_tool_chunk",
                        run_id,
                        call_id,
                        tool_name,
                        chunk,
                    )

            try:
                result = self._execute_tool_with_timeout(tool, parameters, chunk_cb)
                dur = time.time() - start
                self._call_hook("on_tool_end", tool_name, result, dur)
                if run_id:
                    self._notify_observers(
                        "on_tool_end",
                        run_id,
                        call_id,
                        tool_name,
                        result,
                        dur * 1000,
                    )
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                self._call_hook("on_tool_error", tool_name, exc, parameters)
                if run_id:
                    self._notify_observers(
                        "on_tool_error",
                        run_id,
                        call_id,
                        tool_name,
                        exc,
                        parameters,
                        dur * 1000,
                    )
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        # Submit all tool calls to the thread pool
        with ThreadPoolExecutor(max_workers=len(tool_calls_to_execute)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls_to_execute]
            results = [f.result() for f in futures]  # preserves order

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}

        for r in results:
            all_tool_calls.append(r.tool_call)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )
            if not r.is_error and self.usage.iterations:
                self.usage.tool_usage[r.tool_call.tool_name] = (
                    self.usage.tool_usage.get(r.tool_call.tool_name, 0) + 1
                )
                self.usage.tool_tokens[r.tool_call.tool_name] = (
                    self.usage.tool_tokens.get(r.tool_call.tool_name, 0)
                    + self.usage.iterations[-1].total_tokens
                )

            if trace is not None:
                step_type: StepType = "error" if r.is_error else "tool_execution"
                trace.add(
                    TraceStep(
                        type=step_type,
                        duration_ms=r.duration * 1000,
                        tool_name=r.tool_call.tool_name,
                        tool_args=r.tool_call.parameters,
                        tool_result=(
                            self._truncate_tool_result(r.result) if not r.is_error else None
                        ),
                        error=r.result if r.is_error else None,
                        summary=f"{r.tool_call.tool_name} → {'error' if r.is_error else f'{len(r.result)} chars'}",
                    )
                )

            if r.is_error:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    run_id=run_id,
                )
            else:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=r.result,
                    run_id=run_id,
                )

        return last_tool_name, last_tool_args

    async def _aexecute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> tuple:
        """Execute multiple tool calls concurrently using asyncio.gather.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int

        async def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            call_id = tc.id or ""
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            policy_error = await self._acheck_policy(tool_name, parameters, run_id or "")
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            start = time.time()
            self._call_hook("on_tool_start", tool_name, parameters)
            if run_id:
                self._notify_observers("on_tool_start", run_id, call_id, tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._call_hook("on_tool_chunk", tool_name, chunk)
                if run_id:
                    self._notify_observers(
                        "on_tool_chunk",
                        run_id,
                        call_id,
                        tool_name,
                        chunk,
                    )

            try:
                result = await self._aexecute_tool_with_timeout(tool, parameters, chunk_cb)
                dur = time.time() - start
                self._call_hook("on_tool_end", tool_name, result, dur)
                if run_id:
                    self._notify_observers(
                        "on_tool_end",
                        run_id,
                        call_id,
                        tool_name,
                        result,
                        dur * 1000,
                    )
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                self._call_hook("on_tool_error", tool_name, exc, parameters)
                if run_id:
                    self._notify_observers(
                        "on_tool_error",
                        run_id,
                        call_id,
                        tool_name,
                        exc,
                        parameters,
                        dur * 1000,
                    )
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls_to_execute])

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}

        for r in results:
            all_tool_calls.append(r.tool_call)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )

            if trace is not None:
                step_type: StepType = "error" if r.is_error else "tool_execution"
                trace.add(
                    TraceStep(
                        type=step_type,
                        duration_ms=r.duration * 1000,
                        tool_name=r.tool_call.tool_name,
                        tool_args=r.tool_call.parameters,
                        tool_result=(
                            self._truncate_tool_result(r.result) if not r.is_error else None
                        ),
                        error=r.result if r.is_error else None,
                        summary=f"{r.tool_call.tool_name} → {'error' if r.is_error else f'{len(r.result)} chars'}",
                    )
                )

            if r.is_error:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    run_id=run_id,
                )
            else:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=r.result,
                    run_id=run_id,
                )

        return last_tool_name, last_tool_args

    def _execute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run tool.execute with optional timeout and chunk callback."""
        if not self.config.tool_timeout_seconds:
            return tool.execute(parameters, chunk_callback=chunk_callback)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.execute, parameters, chunk_callback)
        try:
            return future.result(timeout=self.config.tool_timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )
        finally:
            executor.shutdown(wait=False)

    def _clone_for_isolation(self) -> "Agent":
        """Create a lightweight clone for batch processing with isolated state."""
        clone = copy.copy(self)
        clone._history = []
        clone.usage = AgentUsage()
        clone.memory = None
        clone.analytics = None
        return clone

    def _is_rate_limit_error(self, message: str) -> bool:
        lowered = message.lower()
        return "rate limit" in lowered or "429" in lowered

    # Async methods
    async def astream(
        self,
        messages: Union[str, List[Message]],
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
    ) -> AsyncGenerator[Union[StreamChunk, AgentResult], None]:
        """
        Stream the agent's response token-by-token.

        Args:
            messages: A plain-text prompt (str) or list of Message objects.
            response_format: Optional Pydantic model class or JSON Schema dict.
            parent_run_id: Optional run-ID of a parent agent for trace linking.

        Yields:
            StreamChunk: Intermediate content chunks.
            AgentResult: The final result object (yielded at the very end).
        """
        messages = self._normalize_messages(messages)
        self._call_hook("on_agent_start", messages)
        ctx = self._prepare_run(
            messages, response_format=response_format, parent_run_id=parent_run_id
        )

        try:
            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1
                self._call_hook("on_iteration_start", ctx.iteration, self._history)
                self._notify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )

                full_content = ""
                current_tool_calls: List[ToolCall] = []

                has_astream = (
                    hasattr(self.provider, "astream")
                    and self.provider.astream is not None
                    and self.provider.supports_streaming
                )

                self._call_hook("on_llm_start", self._history, self.config.model)
                self._notify_observers(
                    "on_llm_start",
                    ctx.run_id,
                    self._history,
                    self.config.model,
                    self._system_prompt,
                )
                llm_start = time.time()

                if not has_astream:
                    if hasattr(self.provider, "acomplete") and getattr(
                        self.provider, "supports_async", False
                    ):
                        response_msg, _usage = await self.provider.acomplete(
                            model=self.config.model,
                            system_prompt=self._system_prompt,
                            messages=self._history,
                            tools=self.tools,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            timeout=self.config.request_timeout,
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        response_msg, _usage = await loop.run_in_executor(
                            None,
                            lambda: self.provider.complete(
                                model=self.config.model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                tools=self.tools,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )
                    if _usage:
                        self.usage.add_usage(_usage, tool_name=None)
                    self._call_hook("on_llm_end", response_msg.content, _usage)
                    self._notify_observers("on_llm_end", ctx.run_id, response_msg.content, _usage)
                    if _usage:
                        self._notify_observers("on_usage", ctx.run_id, _usage)
                    yield StreamChunk(content=response_msg.content)
                    full_content = response_msg.content
                    if response_msg.tool_calls:
                        current_tool_calls = response_msg.tool_calls
                else:
                    gen = self.provider.astream(
                        model=self.config.model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )

                    async for item in gen:
                        if isinstance(item, str):
                            yield StreamChunk(content=item)
                            full_content += item
                        elif isinstance(item, ToolCall):
                            current_tool_calls.append(item)
                            yield StreamChunk(tool_calls=[item])

                    self._call_hook("on_llm_end", full_content, None)
                    self._notify_observers("on_llm_end", ctx.run_id, full_content, None)

                ctx.trace.add(
                    TraceStep(
                        type="llm_call",
                        duration_ms=(time.time() - llm_start) * 1000,
                        model=self.config.model,
                        summary=f"{self.config.model} → {len(full_content)} chars (stream)",
                    )
                )

                response_msg = Message(
                    role=Role.ASSISTANT,
                    content=full_content,
                    tool_calls=current_tool_calls or None,
                )

                # Use _process_response for output guardrails, parsing, reasoning
                response_text, tool_calls_to_execute, reasoning_text = self._process_response(
                    ctx, response_msg
                )

                if not tool_calls_to_execute:
                    parsed = None
                    if ctx.response_format is not None:
                        try:
                            parsed = parse_and_validate(response_text, ctx.response_format)
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                True,
                                ctx.iteration,
                            )
                        except (ValueError, TypeError) as exc:
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                False,
                                ctx.iteration,
                                str(exc),
                            )
                            if ctx.iteration < self.config.max_iterations:
                                ctx.trace.add(
                                    TraceStep(
                                        type="structured_retry",
                                        error=str(exc),
                                        summary=f"Validation failed: {exc}",
                                    )
                                )
                                retry_msg = Message(
                                    role=Role.USER,
                                    content=validation_retry_message(exc),
                                )
                                self._history.append(
                                    Message(role=Role.ASSISTANT, content=response_text)
                                )
                                self._history.append(retry_msg)
                                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._call_hook("on_iteration_end", ctx.iteration, response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = self._finalize_run(ctx, final_response, parsed=parsed)
                    yield _result
                    return

                if self.config.routing_only:
                    self._call_hook("on_iteration_end", ctx.iteration, full_content)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, full_content or ""
                    )
                    _result = AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=ctx.iteration,
                        tool_calls=tool_calls_to_execute,
                        reasoning=reasoning_text,
                        reasoning_history=ctx.reasoning_history,
                        trace=ctx.trace,
                        provider_used=getattr(self.provider, "provider_used", None),
                        usage=copy.copy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                # Append response AFTER checking for tool calls (matches run/arun)
                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = await self._aexecute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        full_content,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                else:
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        call_id = tool_call.id or ""
                        ctx.all_tool_calls.append(tool_call)
                        ctx.last_tool_name = tool_name
                        ctx.last_tool_args = parameters

                        if self.config.verbose:
                            print(
                                f"[agent] Iteration {ctx.iteration}: tool={tool_name} params={parameters}"
                            )

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=error_message,
                                    summary=f"Unknown tool {tool_name}",
                                )
                            )
                            continue

                        policy_error = await self._acheck_policy(tool_name, parameters, ctx.run_id)
                        if policy_error:
                            self._append_tool_result(
                                policy_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=policy_error,
                                    summary=f"Policy denied {tool_name}",
                                )
                            )
                            continue

                        coherence_error = await self._acheck_coherence(
                            ctx.user_text_for_coherence, tool_name, parameters
                        )
                        if coherence_error:
                            self._append_tool_result(
                                coherence_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=coherence_error,
                                    summary=f"Coherence check failed for {tool_name}",
                                )
                            )
                            continue

                        try:
                            start_time = time.time()
                            self._call_hook("on_tool_start", tool_name, parameters)
                            self._notify_observers(
                                "on_tool_start", ctx.run_id, call_id, tool_name, parameters
                            )

                            chunk_counter = {"count": 0}

                            def chunk_callback(chunk: str) -> None:
                                chunk_counter["count"] += 1
                                self._call_hook("on_tool_chunk", tool_name, chunk)
                                self._notify_observers(
                                    "on_tool_chunk",
                                    ctx.run_id,
                                    call_id,
                                    tool_name,
                                    chunk,
                                )

                            result = await self._aexecute_tool_with_timeout(
                                tool, parameters, chunk_callback
                            )
                            result = self._screen_tool_result(tool_name, result)
                            duration = time.time() - start_time
                            self._call_hook("on_tool_end", tool_name, result, duration)
                            self._notify_observers(
                                "on_tool_end",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                result,
                                duration * 1000,
                            )

                            ctx.trace.add(
                                TraceStep(
                                    type="tool_execution",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    tool_args=parameters,
                                    tool_result=self._truncate_tool_result(result),
                                    summary=f"{tool_name} → {len(result)} chars",
                                )
                            )

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=True,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter["count"],
                                )
                            if self.usage.iterations:
                                self.usage.tool_usage[tool.name] = (
                                    self.usage.tool_usage.get(tool.name, 0) + 1
                                )
                                self.usage.tool_tokens[tool.name] = (
                                    self.usage.tool_tokens.get(tool.name, 0)
                                    + self.usage.iterations[-1].total_tokens
                                )

                            self._append_tool_result(
                                result,
                                tool_name,
                                tool_call.id,
                                tool_result=result,
                                run_id=ctx.run_id,
                            )

                        except Exception as exc:
                            duration = time.time() - start_time
                            self._call_hook("on_tool_error", tool_name, exc, parameters)
                            self._notify_observers(
                                "on_tool_error",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                exc,
                                parameters,
                                duration * 1000,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    error=str(exc),
                                    summary=f"{tool_name} failed: {exc}",
                                )
                            )
                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=False,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=0,
                                )

                            error_message = f"Error executing tool '{tool_name}': {exc}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )

                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )

            _result = self._build_max_iterations_result(ctx)
            yield _result
            return
        except Exception as exc:
            if not self.memory:
                self._history = self._history[: ctx.history_checkpoint]
            self._call_hook("on_error", exc, {"messages": messages, "iteration": ctx.iteration})
            self._notify_observers(
                "on_error",
                ctx.run_id,
                exc,
                {"messages": messages, "iteration": ctx.iteration},
            )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = ctx.original_system_prompt

    async def arun(
        self,
        messages: Union[str, List[Message]],
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Async version of run().

        Execute the agent loop asynchronously with the provided conversation history.
        Uses provider async methods if available, falls back to sync in executor.

        Args:
            messages: A plain-text prompt (str) or list of Message objects.
            stream_handler: Optional callback for streaming responses.
            response_format: Optional Pydantic model class or JSON Schema dict.
            parent_run_id: Optional run-ID of a parent agent for trace linking.

        Returns:
            AgentResult with the final response and tool call metadata.
        """
        messages = self._normalize_messages(messages)
        self._call_hook("on_agent_start", messages)
        ctx = self._prepare_run(
            messages, response_format=response_format, parent_run_id=parent_run_id
        )

        try:
            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1
                self._call_hook("on_iteration_start", ctx.iteration, self._history)
                self._notify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )

                response_msg = await self._acall_provider(
                    stream_handler=stream_handler, trace=ctx.trace, run_id=ctx.run_id
                )
                response_text, tool_calls_to_execute, reasoning_text = self._process_response(
                    ctx, response_msg
                )

                if not tool_calls_to_execute:
                    parsed = None
                    if ctx.response_format is not None:
                        try:
                            parsed = parse_and_validate(response_text, ctx.response_format)
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                True,
                                ctx.iteration,
                            )
                        except (ValueError, TypeError) as exc:
                            self._notify_observers(
                                "on_structured_validate",
                                ctx.run_id,
                                False,
                                ctx.iteration,
                                str(exc),
                            )
                            if ctx.iteration < self.config.max_iterations:
                                ctx.trace.add(
                                    TraceStep(
                                        type="structured_retry",
                                        error=str(exc),
                                        summary=f"Validation failed: {exc}",
                                    )
                                )
                                retry_msg = Message(
                                    role=Role.USER,
                                    content=validation_retry_message(exc),
                                )
                                self._history.append(
                                    Message(role=Role.ASSISTANT, content=response_text)
                                )
                                self._history.append(retry_msg)
                                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._call_hook("on_iteration_end", ctx.iteration, response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    return self._finalize_run(ctx, final_response, parsed=parsed)

                if self.config.routing_only and tool_calls_to_execute:
                    self._call_hook("on_iteration_end", ctx.iteration, response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = AgentResult(
                        message=response_msg,
                        tool_name=tool_calls_to_execute[-1].tool_name,
                        tool_args=tool_calls_to_execute[-1].parameters,
                        iterations=ctx.iteration,
                        tool_calls=tool_calls_to_execute,
                        reasoning=reasoning_text,
                        reasoning_history=ctx.reasoning_history,
                        trace=ctx.trace,
                        provider_used=getattr(self.provider, "provider_used", None),
                        usage=copy.copy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args = await self._aexecute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        response_text,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                else:
                    for tool_call in tool_calls_to_execute:
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters
                        call_id = tool_call.id or ""
                        ctx.all_tool_calls.append(tool_call)
                        ctx.last_tool_name = tool_name
                        ctx.last_tool_args = parameters

                        if self.config.verbose:
                            print(
                                f"[agent] Iteration {ctx.iteration}: tool={tool_name} params={parameters}"
                            )

                        tool = self._tools_by_name.get(tool_name)
                        if not tool:
                            error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self._tools_by_name.keys())}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=error_message,
                                    summary=f"Unknown tool {tool_name}",
                                )
                            )
                            continue

                        policy_error = await self._acheck_policy(tool_name, parameters, ctx.run_id)
                        if policy_error:
                            self._append_tool_result(
                                policy_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=policy_error,
                                    summary=f"Policy denied {tool_name}",
                                )
                            )
                            continue

                        coherence_error = await self._acheck_coherence(
                            ctx.user_text_for_coherence, tool_name, parameters
                        )
                        if coherence_error:
                            self._append_tool_result(
                                coherence_error,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    tool_name=tool_name,
                                    error=coherence_error,
                                    summary=f"Coherence check failed for {tool_name}",
                                )
                            )
                            continue

                        try:
                            start_time = time.time()
                            self._call_hook("on_tool_start", tool_name, parameters)
                            self._notify_observers(
                                "on_tool_start", ctx.run_id, call_id, tool_name, parameters
                            )

                            chunk_counter = {"count": 0}

                            def chunk_callback(chunk: str) -> None:
                                chunk_counter["count"] += 1
                                self._call_hook("on_tool_chunk", tool_name, chunk)
                                self._notify_observers(
                                    "on_tool_chunk",
                                    ctx.run_id,
                                    call_id,
                                    tool_name,
                                    chunk,
                                )

                            result = await self._aexecute_tool_with_timeout(
                                tool, parameters, chunk_callback
                            )
                            result = self._screen_tool_result(tool_name, result)
                            duration = time.time() - start_time
                            self._call_hook("on_tool_end", tool_name, result, duration)
                            self._notify_observers(
                                "on_tool_end",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                result,
                                duration * 1000,
                            )

                            ctx.trace.add(
                                TraceStep(
                                    type="tool_execution",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    tool_args=parameters,
                                    tool_result=self._truncate_tool_result(result),
                                    summary=f"{tool_name} → {len(result)} chars",
                                )
                            )

                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=True,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=chunk_counter["count"],
                                )
                            if self.usage.iterations:
                                self.usage.tool_usage[tool.name] = (
                                    self.usage.tool_usage.get(tool.name, 0) + 1
                                )
                                self.usage.tool_tokens[tool.name] = (
                                    self.usage.tool_tokens.get(tool.name, 0)
                                    + self.usage.iterations[-1].total_tokens
                                )

                            self._append_tool_result(
                                result,
                                tool_name,
                                tool_call.id,
                                tool_result=result,
                                run_id=ctx.run_id,
                            )

                        except Exception as exc:
                            duration = time.time() - start_time
                            self._call_hook("on_tool_error", tool_name, exc, parameters)
                            self._notify_observers(
                                "on_tool_error",
                                ctx.run_id,
                                call_id,
                                tool_name,
                                exc,
                                parameters,
                                duration * 1000,
                            )
                            ctx.trace.add(
                                TraceStep(
                                    type="error",
                                    duration_ms=duration * 1000,
                                    tool_name=tool_name,
                                    error=str(exc),
                                    summary=f"{tool_name} failed: {exc}",
                                )
                            )
                            if self.analytics:
                                self.analytics.record_tool_call(
                                    tool_name=tool.name,
                                    success=False,
                                    duration=duration,
                                    params=parameters,
                                    cost=0.0,
                                    chunk_count=0,
                                )

                            error_message = f"Error executing tool '{tool_name}': {exc}"
                            self._append_tool_result(
                                error_message,
                                tool_name,
                                tool_call.id,
                                run_id=ctx.run_id,
                            )

                self._call_hook("on_iteration_end", ctx.iteration, response_text)
                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )

            return self._build_max_iterations_result(ctx)
        except Exception as exc:
            if not self.memory:
                self._history = self._history[: ctx.history_checkpoint]
            self._call_hook("on_error", exc, {"messages": messages, "iteration": ctx.iteration})
            self._notify_observers(
                "on_error",
                ctx.run_id,
                exc,
                {"messages": messages, "iteration": ctx.iteration},
            )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = ctx.original_system_prompt

    async def _acall_provider(
        self,
        stream_handler: Optional[Callable[[str], None]] = None,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
    ) -> Message:
        """Async version of _call_provider with retry logic."""
        call_start = time.time()

        cache_key: Optional[str] = None
        if self.config.cache and not (
            self.config.stream and getattr(self.provider, "supports_streaming", False)
        ):
            cache_key = CacheKeyBuilder.build(
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
            )
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                cached_msg = cast(Message, cached[0])
                cached_usage = cached[1]
                self.usage.add_usage(cached_usage, tool_name=None)
                self._call_hook("on_llm_start", self._history, self.config.model)
                self._call_hook("on_llm_end", cached_msg.content, cached_usage)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self.config.model,
                        self._system_prompt,
                    )
                    self._notify_observers(
                        "on_llm_end",
                        run_id,
                        cached_msg.content,
                        cached_usage,
                    )
                    self._notify_observers(
                        "on_cache_hit",
                        run_id,
                        self.config.model,
                        cached_msg.content or "",
                    )
                    self._notify_observers("on_usage", run_id, cached_usage)
                if self.config.verbose:
                    print("[agent] cache hit -- skipping provider call")
                if trace is not None:
                    trace.add(
                        TraceStep(
                            type="cache_hit",
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self.config.model,
                            summary=f"Cache hit: {self.config.model}",
                        )
                    )
                return cached_msg

        attempts = 0
        last_error: Optional[str] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                self._call_hook("on_llm_start", self._history, self.config.model)
                if run_id:
                    self._notify_observers(
                        "on_llm_start",
                        run_id,
                        self._history,
                        self.config.model,
                        self._system_prompt,
                    )

                if self.config.stream and getattr(self.provider, "supports_streaming", False):
                    response_text = await self._astreaming_call(stream_handler=stream_handler)
                    self._call_hook("on_llm_end", response_text, None)
                    if run_id:
                        self._notify_observers("on_llm_end", run_id, response_text, None)
                    return Message(role=Role.ASSISTANT, content=response_text)

                # Check if provider has async support
                if hasattr(self.provider, "acomplete") and getattr(
                    self.provider, "supports_async", False
                ):
                    response_msg, usage_stats = await self.provider.acomplete(
                        model=self.config.model,
                        system_prompt=self._system_prompt,
                        messages=self._history,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.request_timeout,
                    )
                    response_text = response_msg.content or ""
                else:
                    # Fallback to sync in executor
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        response_msg, usage_stats = await loop.run_in_executor(
                            executor,
                            lambda: self.provider.complete(
                                model=self.config.model,
                                system_prompt=self._system_prompt,
                                messages=self._history,
                                tools=self.tools,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                timeout=self.config.request_timeout,
                            ),
                        )
                    response_text = response_msg.content or ""

                self.usage.add_usage(usage_stats, tool_name=None)

                if cache_key is not None and self.config.cache:
                    self.config.cache.set(cache_key, (response_msg, usage_stats))

                self._call_hook("on_llm_end", response_text, usage_stats)
                if run_id:
                    self._notify_observers("on_llm_end", run_id, response_text, usage_stats)
                    self._notify_observers("on_usage", run_id, usage_stats)

                if (
                    self.config.cost_warning_threshold
                    and self.usage.total_cost_usd > self.config.cost_warning_threshold
                ):
                    print(
                        f"\n⚠️  Cost Warning: Total cost ${self.usage.total_cost_usd:.6f} "
                        f"exceeds threshold ${self.config.cost_warning_threshold:.6f}\n"
                    )

                if self.config.verbose:
                    print(
                        f"[agent] tokens: {usage_stats.total_tokens:,} "
                        f"(prompt: {usage_stats.prompt_tokens:,}, completion: {usage_stats.completion_tokens:,}), "
                        f"cost: ${usage_stats.cost_usd:.6f}"
                    )

                if trace is not None:
                    trace.add(
                        TraceStep(
                            type="llm_call",
                            duration_ms=(time.time() - call_start) * 1000,
                            model=self.config.model,
                            prompt_tokens=usage_stats.prompt_tokens,
                            completion_tokens=usage_stats.completion_tokens,
                            summary=f"{self.config.model} → {len(response_text)} chars",
                        )
                    )
                return response_msg
            except ProviderError as exc:
                last_error = str(exc)
                if self.config.verbose:
                    print(
                        f"[agent] provider error attempt {attempts}/{self.config.max_retries + 1}: {exc}"
                    )
                if attempts > self.config.max_retries:
                    break
                backoff = 0.0
                if (
                    self._is_rate_limit_error(last_error)
                    and self.config.rate_limit_cooldown_seconds
                ):
                    backoff += self.config.rate_limit_cooldown_seconds * attempts
                if self.config.retry_backoff_seconds:
                    backoff += self.config.retry_backoff_seconds * attempts
                if run_id:
                    self._notify_observers(
                        "on_llm_retry",
                        run_id,
                        attempts,
                        self.config.max_retries,
                        exc,
                        backoff,
                    )
                if backoff > 0:
                    await asyncio.sleep(backoff)

        if trace is not None:
            trace.add(
                TraceStep(
                    type="llm_call",
                    duration_ms=(time.time() - call_start) * 1000,
                    model=self.config.model,
                    error=last_error,
                    summary=f"Provider error: {last_error}",
                )
            )
        return Message(
            role=Role.ASSISTANT, content=f"Provider error: {last_error or 'unknown error'}"
        )

    async def _astreaming_call(self, stream_handler: Optional[Callable[[str], None]] = None) -> str:
        """Async version of _streaming_call."""
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []

        if hasattr(self.provider, "astream") and getattr(self.provider, "supports_async", False):
            stream = self.provider.astream(  # type: ignore[attr-defined]
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )
            async for chunk in stream:
                if isinstance(chunk, str) and chunk:
                    aggregated.append(chunk)
                    if stream_handler:
                        stream_handler(chunk)
        else:
            for chunk in self.provider.stream(
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            ):
                if chunk:
                    aggregated.append(str(chunk))
                    if stream_handler:
                        stream_handler(str(chunk))

        return "".join(aggregated)

    async def _aexecute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Async version of _execute_tool_with_timeout."""
        if not self.config.tool_timeout_seconds:
            return await tool.aexecute(parameters, chunk_callback=chunk_callback)

        try:
            return await asyncio.wait_for(
                tool.aexecute(parameters, chunk_callback=chunk_callback),
                timeout=self.config.tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )

    # Usage tracking convenience methods
    @property
    def total_cost(self) -> float:
        """Get the total cost in USD for all API calls."""
        return self.usage.total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens used across all API calls."""
        return self.usage.total_tokens

    def get_usage_summary(self) -> str:
        """
        Get a formatted summary of token usage and costs.

        Returns:
            Formatted string with usage statistics including token counts,
            costs, and per-tool breakdown.
        """
        return str(self.usage)

    def get_analytics(self) -> Optional[AgentAnalytics]:
        """Get the analytics tracker if enabled."""
        return self.analytics

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self.usage = AgentUsage()
