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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Union, cast

from ..analytics import AgentAnalytics
from ..parser import ToolCallParser
from ..prompt import PromptBuilder
from ..providers.base import Provider, ProviderError
from ..providers.openai_provider import OpenAIProvider
from ..stability import stable
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
from ._lifecycle import _LifecycleMixin
from ._memory_manager import _MemoryManagerMixin
from ._provider_caller import _ProviderCallerMixin
from ._tool_executor import _ToolExecutorMixin
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
    terminal_tool_result: Optional[str] = None


@stable
class Agent(_ToolExecutorMixin, _ProviderCallerMixin, _LifecycleMixin, _MemoryManagerMixin):
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
        strategy = config.reasoning_strategy if config else None
        if prompt_builder:
            self.prompt_builder = prompt_builder
        elif config and config.system_prompt:
            self.prompt_builder = PromptBuilder(
                base_instructions=config.system_prompt,
                reasoning_strategy=strategy,
            )
        else:
            self.prompt_builder = PromptBuilder(reasoning_strategy=strategy)
        self.parser = parser or ToolCallParser()
        self.config = config or AgentConfig()
        self.memory = memory
        self.usage = AgentUsage()
        self.analytics = AgentAnalytics() if self.config.enable_analytics else None

        self._system_prompt = self.prompt_builder.build(self.tools)
        self._history: List[Message] = []

        # Hooks deprecation: wrap hooks dict as an observer adapter
        if self.config.hooks:
            import warnings

            warnings.warn(
                "AgentConfig.hooks is deprecated and will be removed in a future version. "
                "Use AgentConfig.observers with AgentObserver subclasses instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            from ..observer import _HooksAdapter

            adapter = _HooksAdapter(self.config.hooks)
            self.config.observers = [adapter] + list(self.config.observers)

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

    @property
    def _effective_model(self) -> str:
        """The model to use for the current iteration.

        Returns the model set by ``model_selector`` if active, otherwise
        falls back to ``self.config.model``.
        """
        return getattr(self, "_current_model", None) or self.config.model

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

    def _prepare_run(
        self,
        messages: List[Message],
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
        skip_guardrails: bool = False,
    ) -> _RunContext:
        """Shared setup for run(), arun(), and astream().

        Saves original system prompt, applies response_format, creates trace,
        wires observers, runs input guardrails, loads memory/session, injects
        knowledge context, and returns a _RunContext carrying all state.
        """
        self._current_model: Optional[str] = None
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

        # Extract user text for coherence checks BEFORE guardrails may redact it
        user_text_for_coherence = ""
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                user_text_for_coherence = msg.content
                break

        # Input guardrails (operate on copies to avoid mutating caller's objects)
        # In async mode, guardrails are applied separately via _arun_input_guardrails
        if self.config.guardrails and self.config.guardrails.input and not skip_guardrails:
            messages = [copy.copy(msg) for msg in messages]
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
            if len(self._history) > 200:
                import warnings

                warnings.warn(
                    f"Agent history has {len(self._history)} messages without memory configured. "
                    f"Consider using ConversationMemory to prevent unbounded growth.",
                    stacklevel=3,
                )

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

        # Knowledge graph context
        if self.config.knowledge_graph:
            kg_ctx = self.config.knowledge_graph.build_context(query=user_text_for_coherence)
            if kg_ctx:
                self._history.insert(
                    0,
                    Message(role=Role.SYSTEM, content=kg_ctx),
                )

        # Prompt compression (modifies self._history view only, not self.memory)
        self._maybe_compress_context(run_id, trace)

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
            usage=copy.deepcopy(self.usage),
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
        self._extract_entities(ctx.run_id)
        self._extract_kg_triples(ctx.run_id)
        self._session_save(ctx.run_id)
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
            usage=copy.deepcopy(self.usage),
        )
        self._notify_observers("on_run_end", ctx.run_id, result)
        return result

    def _check_budget(self, ctx: _RunContext) -> Optional[str]:
        """Check whether the token or cost budget has been exceeded.

        Returns a reason string if the budget is exceeded, or ``None`` if OK.
        """
        if (
            self.config.max_total_tokens is not None
            and self.usage.total_tokens >= self.config.max_total_tokens
        ):
            return (
                f"Token budget exceeded: {self.usage.total_tokens}"
                f"/{self.config.max_total_tokens} tokens"
            )
        if (
            self.config.max_cost_usd is not None
            and self.usage.total_cost_usd >= self.config.max_cost_usd
        ):
            return (
                f"Cost budget exceeded: ${self.usage.total_cost_usd:.6f}"
                f"/${self.config.max_cost_usd:.6f}"
            )
        return None

    def _build_budget_exceeded_result(self, ctx: _RunContext, reason: str) -> AgentResult:
        """Build the AgentResult when a budget limit is hit."""
        ctx.trace.add(TraceStep(type=StepType.BUDGET_EXCEEDED, summary=reason))
        self._notify_observers(
            "on_budget_exceeded",
            ctx.run_id,
            reason,
            self.usage.total_tokens,
            self.usage.total_cost_usd,
        )
        final_response = Message(role=Role.ASSISTANT, content=reason)
        self._history.append(final_response)
        self._memory_add(final_response, ctx.run_id)
        self._extract_entities(ctx.run_id)
        self._extract_kg_triples(ctx.run_id)
        self._session_save(ctx.run_id)
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
            usage=copy.deepcopy(self.usage),
        )
        self._notify_observers("on_run_end", ctx.run_id, result)
        return result

    def _build_cancelled_result(self, ctx: _RunContext) -> AgentResult:
        """Build the AgentResult when a cancellation token fires."""
        reason = "Agent run was cancelled"
        ctx.trace.add(TraceStep(type=StepType.CANCELLED, summary=reason))
        self._notify_observers("on_cancelled", ctx.run_id, ctx.iteration, reason)
        final_response = Message(role=Role.ASSISTANT, content=reason)
        self._history.append(final_response)
        self._memory_add(final_response, ctx.run_id)
        self._session_save(ctx.run_id)
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
            usage=copy.deepcopy(self.usage),
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
                        type=StepType.TOOL_SELECTION,
                        tool_name=tc.tool_name,
                        tool_args=tc.parameters,
                        reasoning=reasoning_text,
                        summary=f"Selected {tc.tool_name}",
                    )
                )

        return response_text, tool_calls_to_execute, reasoning_text

    async def _aprocess_response(
        self,
        ctx: "_RunContext",
        response_msg: Message,
    ) -> tuple:
        """Async version of _process_response — uses async output guardrails."""
        response_text = response_msg.content or ""
        response_text = await self._arun_output_guardrails(response_text, ctx.trace)
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
                        type=StepType.TOOL_SELECTION,
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
        if trace is not None and (not result.passed or result.guardrail_name):
            trace.add(
                TraceStep(
                    type=StepType.GUARDRAIL,
                    summary=f"Input guardrail: {result.guardrail_name or result.reason}",
                )
            )
        return result.content

    def _run_output_guardrails(self, content: str, trace: Optional[AgentTrace] = None) -> str:
        """Run output guardrails on LLM response.  Returns (possibly rewritten) content."""
        if not self.config.guardrails or not self.config.guardrails.output:
            return content
        result = self.config.guardrails.check_output(content)
        if trace is not None and (not result.passed or result.guardrail_name):
            trace.add(
                TraceStep(
                    type=StepType.GUARDRAIL,
                    summary=f"Output guardrail: {result.guardrail_name or result.reason}",
                )
            )
        return result.content

    async def _arun_input_guardrails(self, content: str, trace: Optional[AgentTrace] = None) -> str:
        """Async input guardrails — calls ``acheck()`` to avoid blocking the event loop."""
        if not self.config.guardrails or not self.config.guardrails.input:
            return content
        result = await self.config.guardrails.acheck_input(content)
        if trace is not None and (not result.passed or result.guardrail_name):
            trace.add(
                TraceStep(
                    type=StepType.GUARDRAIL,
                    summary=f"Input guardrail: {result.guardrail_name or result.reason}",
                )
            )
        return result.content

    async def _arun_output_guardrails(
        self, content: str, trace: Optional[AgentTrace] = None
    ) -> str:
        """Async output guardrails — calls ``acheck()`` to avoid blocking the event loop."""
        if not self.config.guardrails or not self.config.guardrails.output:
            return content
        result = await self.config.guardrails.acheck_output(content)
        if trace is not None and (not result.passed or result.guardrail_name):
            trace.add(
                TraceStep(
                    type=StepType.GUARDRAIL,
                    summary=f"Output guardrail: {result.guardrail_name or result.reason}",
                )
            )
        return result.content

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

    def ask(
        self,
        prompt: str,
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
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
            parent_run_id: Optional run-ID of a parent agent for trace linking.

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
            parent_run_id=parent_run_id,
        )

    async def aask(
        self,
        prompt: str,
        stream_handler: Optional[Callable[[str], None]] = None,
        response_format: Optional[ResponseFormat] = None,
        parent_run_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Async version of :meth:`ask`.

        Args:
            prompt: Plain-text question or instruction.
            stream_handler: Optional callback for streaming responses.
            response_format: Optional Pydantic model class or JSON Schema dict.
            parent_run_id: Optional run-ID of a parent agent for trace linking.

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
            parent_run_id=parent_run_id,
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
        saved_system_prompt = self._system_prompt
        ctx: Optional[_RunContext] = None
        try:
            ctx = self._prepare_run(
                messages, response_format=response_format, parent_run_id=parent_run_id
            )

            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1

                # Cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    return self._build_cancelled_result(ctx)

                # Budget check (R1)
                budget_msg = self._check_budget(ctx)
                if budget_msg:
                    return self._build_budget_exceeded_result(ctx, budget_msg)

                # Model selection (R10)
                if self.config.model_selector:
                    selected = self.config.model_selector(
                        ctx.iteration, ctx.all_tool_calls, self.usage
                    )
                    if selected != self._effective_model:
                        old_model = self._effective_model
                        self._current_model = selected
                        self._notify_observers(
                            "on_model_switch", ctx.run_id, ctx.iteration, old_model, selected
                        )

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
                                        type=StepType.STRUCTURED_RETRY,
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
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    return self._finalize_run(ctx, final_response, parsed=parsed)

                if self.config.routing_only and tool_calls_to_execute:
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
                        usage=copy.deepcopy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args, _terminal = self._execute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        response_text,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                        user_text_for_coherence=ctx.user_text_for_coherence,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                    if _terminal is not None:
                        ctx.terminal_tool_result = _terminal
                else:
                    for tool_call in tool_calls_to_execute:
                        terminal = self._execute_single_tool(ctx, tool_call)
                        if terminal:
                            break

                if ctx.terminal_tool_result is not None:
                    final_response = Message(role=Role.ASSISTANT, content=ctx.terminal_tool_result)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    return self._finalize_run(ctx, final_response)

                # Post-tool cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    return self._build_cancelled_result(ctx)

                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )
                continue

            return self._build_max_iterations_result(ctx)
        except Exception as exc:
            if ctx is not None:
                if not self.memory:
                    self._history = self._history[: ctx.history_checkpoint]
                self._notify_observers(
                    "on_error",
                    ctx.run_id,
                    exc,
                    {"messages": messages, "iteration": ctx.iteration},
                )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = saved_system_prompt

    def _clone_for_isolation(self) -> "Agent":
        """Create a lightweight clone for batch processing with isolated state."""
        clone = copy.copy(self)
        clone._history = []
        clone.usage = AgentUsage()
        clone.memory = None
        clone.analytics = None
        return clone

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
        saved_system_prompt = self._system_prompt
        ctx: Optional[_RunContext] = None
        try:
            ctx = self._prepare_run(
                messages,
                response_format=response_format,
                parent_run_id=parent_run_id,
                skip_guardrails=True,
            )

            # Async input guardrails (non-blocking).
            # Only process the newly added messages — those are always at the tail of
            # self._history (appended last by _prepare_run). Applying guardrails to
            # the entire history would re-validate previously processed turns from
            # memory, which wastes work and can corrupt already-validated content.
            if self.config.guardrails and self.config.guardrails.input:
                new_msg_start = len(self._history) - len(messages)
                for i in range(new_msg_start, len(self._history)):
                    msg = self._history[i]
                    if msg.role == Role.USER and msg.content:
                        self._history[i] = copy.copy(msg)
                        self._history[i].content = await self._arun_input_guardrails(
                            msg.content, ctx.trace
                        )

            await self._anotify_observers("on_run_start", ctx.run_id, messages, self._system_prompt)

            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1

                # Cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    _result = self._build_cancelled_result(ctx)
                    await self._anotify_observers(
                        "on_cancelled", ctx.run_id, ctx.iteration, "Agent run was cancelled"
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                # Budget check (R1)
                budget_msg = self._check_budget(ctx)
                if budget_msg:
                    _result = self._build_budget_exceeded_result(ctx, budget_msg)
                    await self._anotify_observers(
                        "on_budget_exceeded",
                        ctx.run_id,
                        budget_msg,
                        self.usage.total_tokens,
                        self.usage.total_cost_usd,
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                # Model selection (R10)
                if self.config.model_selector:
                    selected = self.config.model_selector(
                        ctx.iteration, ctx.all_tool_calls, self.usage
                    )
                    if selected != self._effective_model:
                        old_model = self._effective_model
                        self._current_model = selected
                        self._notify_observers(
                            "on_model_switch", ctx.run_id, ctx.iteration, old_model, selected
                        )
                        await self._anotify_observers(
                            "on_model_switch", ctx.run_id, ctx.iteration, old_model, selected
                        )

                self._notify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )
                await self._anotify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )

                full_content = ""
                current_tool_calls: List[ToolCall] = []

                has_astream = (
                    hasattr(self.provider, "astream")
                    and self.provider.astream is not None
                    and self.provider.supports_streaming
                )

                self._notify_observers(
                    "on_llm_start",
                    ctx.run_id,
                    self._history,
                    self._effective_model,
                    self._system_prompt,
                )
                await self._anotify_observers(
                    "on_llm_start",
                    ctx.run_id,
                    self._history,
                    self._effective_model,
                    self._system_prompt,
                )
                llm_start = time.time()

                if not has_astream:
                    if hasattr(self.provider, "acomplete") and getattr(
                        self.provider, "supports_async", False
                    ):
                        response_msg, _usage = await self.provider.acomplete(
                            model=self._effective_model,
                            system_prompt=self._system_prompt,
                            messages=self._history,
                            tools=self.tools,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            timeout=self.config.request_timeout,
                        )
                    else:
                        loop = asyncio.get_running_loop()
                        response_msg, _usage = await loop.run_in_executor(
                            None,
                            lambda: self.provider.complete(
                                model=self._effective_model,
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
                    _response_content = response_msg.content or ""
                    self._notify_observers("on_llm_end", ctx.run_id, _response_content, _usage)
                    await self._anotify_observers(
                        "on_llm_end", ctx.run_id, _response_content, _usage
                    )
                    if _usage:
                        self._notify_observers("on_usage", ctx.run_id, _usage)
                        await self._anotify_observers("on_usage", ctx.run_id, _usage)
                    yield StreamChunk(content=_response_content)
                    full_content = _response_content
                    if response_msg.tool_calls:
                        current_tool_calls = response_msg.tool_calls
                else:
                    gen = self.provider.astream(
                        model=self._effective_model,
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

                    self._notify_observers("on_llm_end", ctx.run_id, full_content, None)
                    await self._anotify_observers("on_llm_end", ctx.run_id, full_content, None)

                ctx.trace.add(
                    TraceStep(
                        type=StepType.LLM_CALL,
                        duration_ms=(time.time() - llm_start) * 1000,
                        model=self._effective_model,
                        summary=f"{self._effective_model} → {len(full_content)} chars (stream)",
                    )
                )

                response_msg = Message(
                    role=Role.ASSISTANT,
                    content=full_content,
                    tool_calls=current_tool_calls or None,
                )

                # Use async _process_response for non-blocking output guardrails
                response_text, tool_calls_to_execute, reasoning_text = (
                    await self._aprocess_response(ctx, response_msg)
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
                                        type=StepType.STRUCTURED_RETRY,
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
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                await self._anotify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    await self._anotify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = self._finalize_run(ctx, final_response, parsed=parsed)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                if self.config.routing_only:
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, full_content or ""
                    )
                    await self._anotify_observers(
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
                        usage=copy.deepcopy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                # Append response AFTER checking for tool calls (matches run/arun)
                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args, _terminal = await self._aexecute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        full_content,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                        user_text_for_coherence=ctx.user_text_for_coherence,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                    if _terminal is not None:
                        ctx.terminal_tool_result = _terminal
                else:
                    for tool_call in tool_calls_to_execute:
                        terminal = await self._aexecute_single_tool(ctx, tool_call)
                        if terminal:
                            break

                if ctx.terminal_tool_result is not None:
                    final_response = Message(role=Role.ASSISTANT, content=ctx.terminal_tool_result)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    await self._anotify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = self._finalize_run(ctx, final_response)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                # Post-tool cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    _result = self._build_cancelled_result(ctx)
                    await self._anotify_observers(
                        "on_cancelled", ctx.run_id, ctx.iteration, "Agent run was cancelled"
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    yield _result
                    return

                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )
                await self._anotify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )

            _result = self._build_max_iterations_result(ctx)
            await self._anotify_observers("on_run_end", ctx.run_id, _result)
            yield _result
            return
        except Exception as exc:
            if ctx is not None:
                if not self.memory:
                    self._history = self._history[: ctx.history_checkpoint]
                self._notify_observers(
                    "on_error",
                    ctx.run_id,
                    exc,
                    {"messages": messages, "iteration": ctx.iteration},
                )
                await self._anotify_observers(
                    "on_error",
                    ctx.run_id,
                    exc,
                    {"messages": messages, "iteration": ctx.iteration},
                )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = saved_system_prompt

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
        saved_system_prompt = self._system_prompt
        ctx: Optional[_RunContext] = None
        try:
            ctx = self._prepare_run(
                messages,
                response_format=response_format,
                parent_run_id=parent_run_id,
                skip_guardrails=True,
            )

            # Async input guardrails (non-blocking).
            # Only process the newly added messages — those are always at the tail of
            # self._history (appended last by _prepare_run). Applying guardrails to
            # the entire history would re-validate previously processed turns from
            # memory, which wastes work and can corrupt already-validated content.
            if self.config.guardrails and self.config.guardrails.input:
                new_msg_start = len(self._history) - len(messages)
                for i in range(new_msg_start, len(self._history)):
                    msg = self._history[i]
                    if msg.role == Role.USER and msg.content:
                        self._history[i] = copy.copy(msg)
                        self._history[i].content = await self._arun_input_guardrails(
                            msg.content, ctx.trace
                        )

            await self._anotify_observers("on_run_start", ctx.run_id, messages, self._system_prompt)

            while ctx.iteration < self.config.max_iterations:
                ctx.iteration += 1

                # Cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    result = self._build_cancelled_result(ctx)
                    await self._anotify_observers(
                        "on_cancelled", ctx.run_id, ctx.iteration, "cancelled"
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, result)
                    return result

                # Budget check (R1)
                budget_msg = self._check_budget(ctx)
                if budget_msg:
                    result = self._build_budget_exceeded_result(ctx, budget_msg)
                    await self._anotify_observers(
                        "on_budget_exceeded",
                        ctx.run_id,
                        budget_msg,
                        self.usage.total_tokens,
                        self.usage.total_cost_usd,
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, result)
                    return result

                # Model selection (R10)
                if self.config.model_selector:
                    selected = self.config.model_selector(
                        ctx.iteration, ctx.all_tool_calls, self.usage
                    )
                    if selected != self._effective_model:
                        old_model = self._effective_model
                        self._current_model = selected
                        self._notify_observers(
                            "on_model_switch", ctx.run_id, ctx.iteration, old_model, selected
                        )
                        await self._anotify_observers(
                            "on_model_switch", ctx.run_id, ctx.iteration, old_model, selected
                        )

                self._notify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )
                await self._anotify_observers(
                    "on_iteration_start", ctx.run_id, ctx.iteration, self._history
                )

                response_msg = await self._acall_provider(
                    stream_handler=stream_handler, trace=ctx.trace, run_id=ctx.run_id
                )
                response_text, tool_calls_to_execute, reasoning_text = (
                    await self._aprocess_response(ctx, response_msg)
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
                                        type=StepType.STRUCTURED_RETRY,
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
                                self._notify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                await self._anotify_observers(
                                    "on_iteration_end",
                                    ctx.run_id,
                                    ctx.iteration,
                                    response_text or "",
                                )
                                continue

                    final_response = Message(role=Role.ASSISTANT, content=response_text)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    await self._anotify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = self._finalize_run(ctx, final_response, parsed=parsed)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                if self.config.routing_only and tool_calls_to_execute:
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    await self._anotify_observers(
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
                        usage=copy.deepcopy(self.usage),
                    )
                    self._notify_observers("on_run_end", ctx.run_id, _result)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                self._history.append(response_msg)
                self._memory_add(response_msg, ctx.run_id)

                use_parallel = (
                    self.config.parallel_tool_execution and len(tool_calls_to_execute) > 1
                )

                if use_parallel:
                    _last_name, _last_args, _terminal = await self._aexecute_tools_parallel(
                        tool_calls_to_execute,
                        ctx.all_tool_calls,
                        ctx.iteration,
                        response_text,
                        trace=ctx.trace,
                        run_id=ctx.run_id,
                        user_text_for_coherence=ctx.user_text_for_coherence,
                    )
                    ctx.last_tool_name = _last_name
                    ctx.last_tool_args = _last_args
                    if _terminal is not None:
                        ctx.terminal_tool_result = _terminal
                else:
                    for tool_call in tool_calls_to_execute:
                        terminal = await self._aexecute_single_tool(ctx, tool_call)
                        if terminal:
                            break

                if ctx.terminal_tool_result is not None:
                    final_response = Message(role=Role.ASSISTANT, content=ctx.terminal_tool_result)
                    self._notify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    await self._anotify_observers(
                        "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                    )
                    _result = self._finalize_run(ctx, final_response)
                    await self._anotify_observers("on_run_end", ctx.run_id, _result)
                    return _result

                # Post-tool cancellation check (R2)
                if self.config.cancellation_token and self.config.cancellation_token.is_cancelled:
                    result = self._build_cancelled_result(ctx)
                    await self._anotify_observers(
                        "on_cancelled", ctx.run_id, ctx.iteration, "cancelled"
                    )
                    await self._anotify_observers("on_run_end", ctx.run_id, result)
                    return result

                self._notify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )
                await self._anotify_observers(
                    "on_iteration_end", ctx.run_id, ctx.iteration, response_text or ""
                )

            result = self._build_max_iterations_result(ctx)
            await self._anotify_observers("on_run_end", ctx.run_id, result)
            return result
        except Exception as exc:
            if ctx is not None:
                if not self.memory:
                    self._history = self._history[: ctx.history_checkpoint]
                self._notify_observers(
                    "on_error",
                    ctx.run_id,
                    exc,
                    {"messages": messages, "iteration": ctx.iteration},
                )
                await self._anotify_observers(
                    "on_error",
                    ctx.run_id,
                    exc,
                    {"messages": messages, "iteration": ctx.iteration},
                )
            raise
        finally:
            self._unwire_fallback_observer()
            self._system_prompt = saved_system_prompt

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
