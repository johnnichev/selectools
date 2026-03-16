"""Mixin providing memory management methods for the Agent class."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ..types import Message, Role

if TYPE_CHECKING:
    from ..memory import ConversationMemory


class _MemoryManagerMixin:
    """Mixin that provides memory management methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, memory,
    _history, _notify_observers, etc.) which are expected to be provided
    by the Agent class that inherits from this mixin.
    """

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
