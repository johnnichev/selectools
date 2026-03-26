"""Mixin providing memory management methods for the Agent class."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ..token_estimation import estimate_run_tokens
from ..trace import AgentTrace, StepType, TraceStep
from ..types import Message, Role

if TYPE_CHECKING:
    from ..memory import ConversationMemory


def _format_messages_as_text(messages: List[Message]) -> str:
    """Render a list of messages as 'ROLE: content' lines joined by newlines."""
    return "\n".join(f"{m.role.value.upper()}: {m.content or ''}" for m in messages)


class _MemoryManagerMixin:
    """Mixin that provides memory management methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, memory,
    _history, _notify_observers, etc.) which are expected to be provided
    by the Agent class that inherits from this mixin.
    """

    # Declared here so mypy resolves the type when _maybe_compress_context
    # both reads and assigns self._history (assignment triggers has-type errors
    # without this annotation).
    _history: List[Message]

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
            trimmed_text = _format_messages_as_text(trimmed)
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
            entities = em.extract_entities(recent, model=self._effective_model)
            if entities:
                em.update(entities)
                self._notify_observers(
                    "on_entity_extraction",
                    run_id,
                    len(entities),
                )
        except Exception:  # nosec B110
            pass  # never crash the agent for entity extraction failure

    def _maybe_compress_context(self, run_id: str, trace: AgentTrace) -> None:
        """Proactively summarize old history messages if context is getting full.

        Only fires when ``config.compress_context`` is True and the estimated
        token fill-rate exceeds ``config.compress_threshold``.  Modifies
        ``self._history`` (the per-call view) only — ``self.memory`` is untouched.
        """
        if not self.config.compress_context:
            return

        estimate = estimate_run_tokens(
            messages=self._history,
            tools=self.tools,
            system_prompt=self._system_prompt,
            model=self._effective_model,
        )
        context_window = estimate.context_window or 128_000
        if context_window == 0:
            return

        fill_rate = estimate.total_tokens / context_window
        if fill_rate < self.config.compress_threshold:
            return

        # Each "turn" is one user + one assistant message, so keep_recent * 2 messages.
        keep_recent = self.config.compress_keep_recent * 2
        system_msgs: List[Message] = []
        non_system: List[Message] = []
        for m in self._history:
            (system_msgs if m.role == Role.SYSTEM else non_system).append(m)

        if len(non_system) <= keep_recent:
            return  # nothing old enough to compress

        to_compress = non_system[:-keep_recent] if keep_recent else non_system
        to_keep_recent = non_system[-keep_recent:] if keep_recent else []

        if len(to_compress) < 2:
            return

        try:
            compressed_text = _format_messages_as_text(to_compress)

            prompt_msg = Message(
                role=Role.USER,
                content=(
                    "Summarize the following conversation excerpt in 3-5 sentences. "
                    "Preserve key facts, decisions, and context needed to continue "
                    "the conversation:\n\n" + compressed_text
                ),
            )
            result = self.provider.complete(
                model=self._effective_model,
                system_prompt="You are a concise summarizer.",
                messages=[prompt_msg],
                max_tokens=300,
            )
            summary_msg = result[0] if isinstance(result, tuple) else result
            summary_text = (summary_msg.content or "").strip()
            if not summary_text:
                return

            summary_message = Message(
                role=Role.SYSTEM,
                content=f"[Compressed context] {summary_text}",
            )
            self._history = system_msgs + [summary_message] + to_keep_recent

            after_estimate = estimate_run_tokens(
                messages=self._history,
                tools=self.tools,
                system_prompt=self._system_prompt,
                model=self._effective_model,
            )
            trace.steps.append(
                TraceStep(
                    type=StepType.PROMPT_COMPRESSED,
                    summary=(
                        f"Compressed {len(to_compress)} messages: "
                        f"{estimate.total_tokens}→{after_estimate.total_tokens} tokens"
                    ),
                    prompt_tokens=estimate.total_tokens,
                    completion_tokens=after_estimate.total_tokens,
                )
            )
            self._notify_observers(
                "on_prompt_compressed",
                run_id,
                estimate.total_tokens,
                after_estimate.total_tokens,
                len(to_compress),
            )
        except Exception:  # nosec B110
            pass  # never crash the agent for a compression failure

    def _extract_kg_triples(self, run_id: str) -> None:
        """Extract relationship triples from recent messages if knowledge_graph is configured."""
        kg = self.config.knowledge_graph
        if not kg:
            return
        try:
            recent = self._history[-kg._relevance_window :]
            triples = kg.extract_triples(recent, model=self._effective_model)
            if triples:
                kg.store.add_many(triples)
                self._notify_observers(
                    "on_kg_extraction",
                    run_id,
                    len(triples),
                )
        except Exception:  # nosec B110
            pass  # never crash the agent for KG extraction failure
