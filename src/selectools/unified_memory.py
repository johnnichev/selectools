"""
Unified memory — tiered memory lifecycle orchestrating the existing memory systems.

``UnifiedMemory`` composes the four existing memory systems into one lifecycle:

- **Short-term** (:class:`~selectools.memory.ConversationMemory`): rolling
  message window; items age out as the window slides.
- **Long-term** (:class:`~selectools.knowledge.KnowledgeMemory`): durable
  facts.  Items whose importance is at or above ``importance_threshold`` are
  auto-promoted from short-term when they age out (``auto_promote=True``) or
  on explicit :meth:`UnifiedMemory.consolidate`.
- **Entity** (:class:`~selectools.entity_memory.EntityMemory`): structured
  entity tracking, fed per turn when an entity memory is provided.
- **Episodic** (:class:`EpisodicMemory`): date-keyed interaction history with
  retention-based pruning.

Importance scoring is rule-based by default (see
:data:`DEFAULT_IMPORTANCE_RULES`); an LLM-based scorer can be plugged in via
the ``scorer=`` callable without requiring a provider.

Context compaction: when the assembled context exceeds ``compaction_threshold``
(default 70%) of the requested token limit, older short-term content is
compacted.  With a ``summarizer=`` callable the old segment is summarized;
without one it is replaced by a truncation marker.

Example::

    memory = UnifiedMemory(
        importance_threshold=0.7,
        short_term_limit=100,
        long_term_limit=1000,
        episodic_retention_days=30,
        auto_promote=True,
    )
    memory.add_turn("My name is John", "Nice to meet you, John!")
    context = memory.assemble_context(max_tokens=4000)
    results = memory.recall("user's name")
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._time import ensure_aware, parse_iso
from .entity_memory import EntityMemory
from .knowledge import KnowledgeEntry, KnowledgeMemory
from .memory import ConversationMemory
from .stability import beta, register_stability
from .token_estimation import estimate_tokens
from .types import Message, Role

logger = logging.getLogger(__name__)

#: Importance assigned when no rule matches.  Deliberately below the default
#: promotion threshold (0.7) so unremarkable turns stay short-term only.
DEFAULT_BASE_IMPORTANCE = 0.3
register_stability("DEFAULT_BASE_IMPORTANCE", "beta")


# ======================================================================
# Importance scoring — rule table
# ======================================================================


@beta
@dataclass
class ImportanceRule:
    """A single rule mapping a regex pattern to an importance score.

    Attributes:
        name: Rule identifier; used as the long-term entry ``category``
            when this rule triggers a promotion.
        pattern: Case-insensitive regular expression matched against the
            message text.
        score: Importance score (0.0-1.0) assigned when the pattern matches.
        description: Human-readable rationale for the rule.
    """

    name: str
    pattern: str
    score: float
    description: str = ""
    _regex: "re.Pattern[str]" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        self._regex = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """Whether this rule's pattern matches *text*."""
        return bool(self._regex.search(text))


#: Default rule table.  When several rules match, the highest score wins.
#: Scores follow the roadmap sketch (names ~0.9, preferences ~0.75,
#: locations ~0.6) generalized into a configurable table.
DEFAULT_IMPORTANCE_RULES: Tuple[ImportanceRule, ...] = (
    ImportanceRule(
        name="identity",
        pattern=r"\b(my name is|call me|i'?m called|i am called)\b",
        score=0.9,
        description="User self-identification: names are near-permanent facts.",
    ),
    ImportanceRule(
        name="relationship",
        pattern=(
            r"\bmy (wife|husband|spouse|partner|daughter|son|kids?|children|"
            r"mother|father|mom|dad|brother|sister|boss|manager|friend)\b"
        ),
        score=0.85,
        description="People in the user's life: stable, high-recall-value facts.",
    ),
    ImportanceRule(
        name="preference",
        pattern=r"\b(i (prefer|like|love|hate|dislike|enjoy)|my favou?rite|i always|i never)\b",
        score=0.75,
        description="Stated preferences and habits: durable but may evolve.",
    ),
    ImportanceRule(
        name="goal",
        pattern=r"\b(my goal|i (decided|plan|want) to|we (decided|agreed))\b",
        score=0.7,
        description="Goals and decisions: important while active, may expire.",
    ),
    ImportanceRule(
        name="location",
        pattern=r"\b(i live in|i'?m from|i am from|based in|i work (at|in|for)|moving to)\b",
        score=0.6,
        description="Locations and affiliations: useful context, changes over time.",
    ),
    ImportanceRule(
        name="date_fact",
        pattern=r"\b(birthday|anniversary|deadline)\b",
        score=0.6,
        description="Date-anchored facts: useful for follow-ups and reminders.",
    ),
)


@beta
def score_importance(
    text: str,
    rules: Optional[Sequence[ImportanceRule]] = None,
) -> float:
    """Score *text* against a rule table.

    Returns the highest score among matching rules, or
    :data:`DEFAULT_BASE_IMPORTANCE` when no rule matches.

    Args:
        text: The text to score.
        rules: Rule table to use.  Defaults to :data:`DEFAULT_IMPORTANCE_RULES`.
            Passing a custom table fully replaces the defaults.
    """
    rule = _best_rule(text, DEFAULT_IMPORTANCE_RULES if rules is None else rules)
    return rule.score if rule is not None else DEFAULT_BASE_IMPORTANCE


def _best_rule(text: str, rules: Sequence[ImportanceRule]) -> Optional[ImportanceRule]:
    best: Optional[ImportanceRule] = None
    for rule in rules:
        if rule.matches(text) and (best is None or rule.score > best.score):
            best = rule
    return best


# ======================================================================
# Episodic memory — date-keyed interaction history
# ======================================================================


@beta
@dataclass
class Episode:
    """A single user/assistant interaction recorded in episodic memory."""

    user: str
    assistant: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, str]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        ts = ensure_aware(parse_iso(data["timestamp"]))
        return cls(user=data.get("user", ""), assistant=data.get("assistant", ""), timestamp=ts)


@beta
class EpisodicMemory:
    """Date-keyed interaction history with retention-based pruning.

    Episodes are bucketed by UTC calendar date.  The whole structure is
    JSON-serializable via :meth:`to_dict` / :meth:`from_dict`.

    Args:
        retention_days: Default retention window used by :meth:`prune`
            when no override is given.
    """

    def __init__(self, retention_days: int = 30) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self.retention_days = retention_days
        self._episodes: Dict[str, List[Episode]] = {}
        self._lock = threading.RLock()

    def add(
        self,
        user: str,
        assistant: str,
        when: Optional[datetime] = None,
    ) -> Episode:
        """Record one interaction.  ``when`` defaults to now (UTC)."""
        ts = when if when is not None else datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        episode = Episode(user=user, assistant=assistant, timestamp=ts)
        key = ts.date().isoformat()
        with self._lock:
            self._episodes.setdefault(key, []).append(episode)
        return episode

    def recent(self, days: int) -> List[Episode]:
        """Episodes from the last *days* days, in chronological order."""
        if days < 1:
            raise ValueError("days must be at least 1")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            episodes = [e for day in self._episodes.values() for e in day]
        return sorted(
            (e for e in episodes if e.timestamp >= cutoff),
            key=lambda e: e.timestamp,
        )

    def search(self, terms: Sequence[str], days: Optional[int] = None) -> List[Episode]:
        """Episodes containing any of *terms* (case-insensitive substring).

        Args:
            terms: Lowercased search terms.
            days: Optional date filter; only episodes from the last *days*
                days are considered.  Defaults to the retention window.
        """
        window = self.retention_days if days is None else days
        matches = []
        for episode in self.recent(window):
            haystack = f"{episode.user} {episode.assistant}".lower()
            if any(term in haystack for term in terms):
                matches.append(episode)
        return matches

    def prune(self, retention_days: Optional[int] = None) -> int:
        """Drop whole days older than the retention window.

        Returns:
            Number of episodes removed.
        """
        window = self.retention_days if retention_days is None else retention_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window)).date()
        removed = 0
        with self._lock:
            for key in list(self._episodes):
                try:
                    day = datetime.strptime(key, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if day < cutoff:
                    removed += len(self._episodes.pop(key))
        return removed

    def clear(self) -> None:
        """Remove all episodes."""
        with self._lock:
            self._episodes.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "retention_days": self.retention_days,
                "episodes": {
                    day: [e.to_dict() for e in items] for day, items in self._episodes.items()
                },
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        memory = cls(retention_days=data.get("retention_days", 30))
        for day, items in data.get("episodes", {}).items():
            memory._episodes[day] = [Episode.from_dict(item) for item in items]
        return memory

    def __len__(self) -> int:
        with self._lock:
            return sum(len(day) for day in self._episodes.values())


# ======================================================================
# InMemoryKnowledgeStore — process-local KnowledgeStore implementation
# ======================================================================


@beta
class InMemoryKnowledgeStore:
    """Dict-backed :class:`~selectools.knowledge.KnowledgeStore` implementation.

    Used as the default long-term store for :class:`UnifiedMemory` so the
    zero-arg constructor needs no filesystem or database.  Semantics mirror
    ``FileKnowledgeStore`` (importance-ordered queries, TTL filtering,
    persistent entries survive pruning).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._lock = threading.RLock()

    def save(self, entry: KnowledgeEntry) -> str:
        with self._lock:
            self._entries[entry.id] = entry
        return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        since_aware: Optional[datetime] = None
        if since is not None:
            since_aware = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        with self._lock:
            entries = list(self._entries.values())
        result = []
        for entry in entries:
            if entry.is_expired:
                continue
            if category and entry.category != category:
                continue
            if entry.importance < min_importance:
                continue
            if since_aware and entry.created_at < since_aware:
                continue
            result.append(entry)
        result.sort(key=lambda e: e.importance, reverse=True)
        return result[:limit]

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
            if max_age_days is not None
            else None
        )
        removed = 0
        with self._lock:
            for entry_id, entry in list(self._entries.items()):
                if entry.persistent:
                    continue
                if (
                    entry.is_expired
                    or (cutoff is not None and entry.created_at < cutoff)
                    or entry.importance < min_importance
                ):
                    del self._entries[entry_id]
                    removed += 1
        return removed


# ======================================================================
# Recall results
# ======================================================================


@beta
@dataclass
class RecallResult:
    """A single federated recall hit.

    Attributes:
        content: The recalled text.
        source: Which tier produced it: ``"long_term"``, ``"entity"``,
            or ``"episodic"``.
        score: Merge score used for ordering (see :meth:`UnifiedMemory.recall`).
        timestamp: When the underlying item was created/last seen, if known.
        metadata: Tier-specific extras (category, entity type, etc.).
    """

    content: str
    source: str
    score: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# UnifiedMemory
# ======================================================================


@beta
class UnifiedMemory:
    """Tiered memory with auto-promotion, orchestrating the existing systems.

    Lifecycle::

        add_turn() ──> Short-term (ConversationMemory, rolling window)
            │                │ ages out (auto_promote) / consolidate()
            │                ▼  importance >= threshold
            │          Long-term (KnowledgeMemory)
            ├────────> Episodic (date-keyed, retention-pruned)
            └────────> Entity (EntityMemory, when provided)

    Sub-memories are dependency-injected so callers pick backends; the
    zero-arg default builds in-memory versions (the long-term tier uses an
    :class:`InMemoryKnowledgeStore` with a temp scratch directory for the
    legacy daily-log files).

    Args:
        importance_threshold: Minimum importance score for promotion to
            long-term memory.
        short_term_limit: Rolling window size for short-term memory,
            in messages (one turn = two messages).  Ignored when a
            ``short_term`` instance is injected.
        long_term_limit: Maximum long-term entries before importance-based
            eviction.  Ignored when a ``long_term`` instance is injected.
        episodic_retention_days: Episodes older than this are pruned.
        auto_promote: Score and promote short-term items as they age out
            of the rolling window.  When ``False``, promotion only happens
            via explicit :meth:`consolidate`.
        short_term: Optional pre-built :class:`ConversationMemory`.
        long_term: Optional pre-built :class:`KnowledgeMemory`.
        entity_memory: Optional :class:`EntityMemory`.  When provided, each
            turn is fed through entity extraction (requires its provider).
            When ``None`` the entity tier is disabled — no LLM required.
        episodic: Optional pre-built :class:`EpisodicMemory`.
        importance_rules: Custom rule table replacing
            :data:`DEFAULT_IMPORTANCE_RULES`.
        scorer: Optional importance scorer (e.g. LLM-based) called with the
            message text, returning a 0.0-1.0 score.  Overrides the rule
            score; rules still supply the promotion category.  Failures
            fall back to the rule table.
        summarizer: Optional callable summarizing old short-term text during
            compaction.  Without it, compaction falls back to a truncation
            marker (``[... N earlier messages compacted ...]``).
        token_counter: Token estimator used for compaction decisions.
            Defaults to :func:`selectools.token_estimation.estimate_tokens`.
        compaction_threshold: Fraction of ``max_tokens`` that triggers
            compaction in :meth:`assemble_context`.  Default 0.7.
    """

    def __init__(
        self,
        importance_threshold: float = 0.7,
        short_term_limit: int = 100,
        long_term_limit: int = 1000,
        episodic_retention_days: int = 30,
        auto_promote: bool = True,
        *,
        short_term: Optional[ConversationMemory] = None,
        long_term: Optional[KnowledgeMemory] = None,
        entity_memory: Optional[EntityMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        importance_rules: Optional[Sequence[ImportanceRule]] = None,
        scorer: Optional[Callable[[str], float]] = None,
        summarizer: Optional[Callable[[str], str]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        compaction_threshold: float = 0.7,
    ) -> None:
        if not 0.0 <= importance_threshold <= 1.0:
            raise ValueError("importance_threshold must be between 0.0 and 1.0")
        if short_term_limit < 1:
            raise ValueError("short_term_limit must be at least 1")
        if long_term_limit < 1:
            raise ValueError("long_term_limit must be at least 1")
        if episodic_retention_days < 1:
            raise ValueError("episodic_retention_days must be at least 1")
        if not 0.0 < compaction_threshold <= 1.0:
            raise ValueError("compaction_threshold must be in (0.0, 1.0]")

        self.importance_threshold = importance_threshold
        self.episodic_retention_days = episodic_retention_days
        self.auto_promote = auto_promote
        self._rules: Tuple[ImportanceRule, ...] = (
            DEFAULT_IMPORTANCE_RULES if importance_rules is None else tuple(importance_rules)
        )
        self._scorer = scorer
        self._summarizer = summarizer
        self._count: Callable[[str], int] = (
            token_counter if token_counter is not None else (lambda text: estimate_tokens(text))
        )
        self._compaction_threshold = compaction_threshold

        self._short_term = (
            short_term
            if short_term is not None
            else ConversationMemory(max_messages=short_term_limit)
        )
        self._long_term = (
            long_term if long_term is not None else self._default_long_term(long_term_limit)
        )
        self._entity_memory = entity_memory
        self._episodic = (
            episodic if episodic is not None else EpisodicMemory(episodic_retention_days)
        )

        self._lock = threading.RLock()
        self._stm_mirror: List[Message] = []
        self._promoted_hashes: set[str] = set()

    @staticmethod
    def _default_long_term(long_term_limit: int) -> KnowledgeMemory:
        scratch = tempfile.mkdtemp(prefix="selectools-unified-memory-")
        return KnowledgeMemory(
            directory=scratch,
            store=InMemoryKnowledgeStore(),
            max_entries=long_term_limit,
        )

    # ------------------------------------------------------------------
    # Tier accessors
    # ------------------------------------------------------------------

    @property
    def short_term(self) -> ConversationMemory:
        """The short-term tier."""
        return self._short_term

    @property
    def long_term(self) -> KnowledgeMemory:
        """The long-term tier."""
        return self._long_term

    @property
    def entity_memory(self) -> Optional[EntityMemory]:
        """The entity tier, or ``None`` when disabled."""
        return self._entity_memory

    @property
    def episodic(self) -> EpisodicMemory:
        """The episodic tier."""
        return self._episodic

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_message: "str | Message",
        assistant_message: "str | Message",
        when: Optional[datetime] = None,
    ) -> None:
        """Record one conversation turn across all tiers.

        The turn enters short-term memory; messages pushed out of the rolling
        window are scored and promoted to long-term when ``auto_promote`` is
        on.  The turn is also recorded as an episode (pruning anything past
        the retention window) and, when an entity memory is configured, fed
        through entity extraction.

        Args:
            user_message: User text or a :class:`Message`.
            assistant_message: Assistant text or a :class:`Message`.
            when: Episode timestamp override (defaults to now, UTC).
        """
        user_msg = (
            user_message
            if isinstance(user_message, Message)
            else Message(role=Role.USER, content=user_message)
        )
        assistant_msg = (
            assistant_message
            if isinstance(assistant_message, Message)
            else Message(role=Role.ASSISTANT, content=assistant_message)
        )

        with self._lock:
            self._short_term.add_many([user_msg, assistant_msg])
            self._stm_mirror.extend([user_msg, assistant_msg])
            aged_out = self._collect_aged_out()
            if self.auto_promote:
                for message in aged_out:
                    self._promote_if_important(message)

            self._episodic.add(user_msg.content or "", assistant_msg.content or "", when=when)
            self._episodic.prune(self.episodic_retention_days)

        if self._entity_memory is not None:
            try:
                extracted = self._entity_memory.extract_entities([user_msg, assistant_msg])
                if extracted:
                    self._entity_memory.update(extracted)
            except Exception:
                logger.warning("Entity tier update failed", exc_info=True)

    def _collect_aged_out(self) -> List[Message]:
        """Messages that left the short-term window since the last check."""
        current_ids = {id(m) for m in self._short_term.get_history()}
        aged_out: List[Message] = []
        while self._stm_mirror and id(self._stm_mirror[0]) not in current_ids:
            aged_out.append(self._stm_mirror.pop(0))
        return aged_out

    def consolidate(self) -> int:
        """Score everything currently in short-term memory and promote.

        Promotion is idempotent: each distinct message content is promoted
        at most once (content-hash dedup), so calling :meth:`consolidate`
        repeatedly — or letting an already-consolidated item age out later —
        never duplicates long-term entries.  Also prunes episodic memory.

        Returns:
            Number of items promoted to long-term memory.
        """
        with self._lock:
            promoted = sum(
                1
                for message in self._short_term.get_history()
                if self._promote_if_important(message)
            )
            self._episodic.prune(self.episodic_retention_days)
            return promoted

    def _promote_if_important(self, message: Message) -> bool:
        content = (message.content or "").strip()
        if not content or message.role not in (Role.USER, Role.ASSISTANT):
            return False
        score, rule = self._score(content)
        if score < self.importance_threshold:
            return False
        digest = hashlib.sha256(content.lower().encode("utf-8")).hexdigest()
        if digest in self._promoted_hashes:
            return False
        self._long_term.remember(
            content,
            category=rule.name if rule is not None else "general",
            importance=score,
            metadata={"source": "unified_memory", "role": message.role.value},
        )
        self._promoted_hashes.add(digest)
        return True

    def _score(self, text: str) -> Tuple[float, Optional[ImportanceRule]]:
        rule = _best_rule(text, self._rules)
        rule_score = rule.score if rule is not None else DEFAULT_BASE_IMPORTANCE
        if self._scorer is not None:
            try:
                return max(0.0, min(1.0, float(self._scorer(text)))), rule
            except Exception:
                logger.warning("Custom scorer failed; falling back to rules", exc_info=True)
        return rule_score, rule

    # ------------------------------------------------------------------
    # Read path — context assembly with compaction
    # ------------------------------------------------------------------

    def assemble_context(
        self,
        max_tokens: int = 4000,
        *,
        include_conversation: bool = True,
    ) -> str:
        """Build a single context string from all tiers.

        Sections, in order: long-term knowledge, known entities, recent
        episodes, and the short-term conversation.

        Compaction: when the assembled context exceeds
        ``compaction_threshold * max_tokens`` (default 70%), older short-term
        messages are compacted — progressively halving the number of recent
        messages kept verbatim.  With a ``summarizer`` the old segment is
        summarized; otherwise it is replaced by a
        ``[... N earlier messages compacted ...]`` marker.  If the budget is
        still exceeded with only the latest turn kept, the episodic and then
        the entity sections are dropped, and as a last resort the result is
        hard-truncated with a ``[... context truncated ...]`` marker.

        Args:
            max_tokens: Token budget the caller plans to spend on context.
            include_conversation: Include the short-term conversation section.
                Pass ``False`` when the conversation is delivered separately
                (e.g. the agent integration sends short-term history as
                structured messages and only needs the other tiers here).
                Default: True.

        Returns:
            The assembled (possibly compacted) context string.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        with self._lock:
            budget = max(1, int(max_tokens * self._compaction_threshold))
            long_term_section = self._long_term.build_context()
            entity_section = (
                self._entity_memory.build_context() if self._entity_memory is not None else ""
            )
            episodic_section = self._episodic_section()
            messages = self._short_term.get_history() if include_conversation else []

        def render(conversation: str, drop_level: int = 0) -> str:
            parts = [long_term_section]
            if drop_level < 2:
                parts.append(entity_section)
            if drop_level < 1:
                parts.append(episodic_section)
            parts.append(conversation)
            return "\n\n".join(p for p in parts if p)

        conversation = self._render_conversation(messages)
        context = render(conversation)
        if self._count(context) <= budget:
            return context

        recent_n = len(messages)
        while recent_n > 2:
            recent_n = max(2, recent_n // 2)
            old, recent = messages[:-recent_n], messages[-recent_n:]
            conversation = self._render_conversation(recent, compact_block=self._compact(old))
            context = render(conversation)
            if self._count(context) <= budget:
                return context

        for drop_level in (1, 2):
            context = render(conversation, drop_level=drop_level)
            if self._count(context) <= budget:
                return context

        return self._hard_truncate(context, budget)

    def _render_conversation(self, messages: Sequence[Message], compact_block: str = "") -> str:
        if not messages and not compact_block:
            return ""
        lines = ["[Conversation]"]
        if compact_block:
            lines.append(compact_block)
        for message in messages:
            lines.append(f"{message.role.value.upper()}: {message.content or ''}")
        return "\n".join(lines)

    def _compact(self, old_messages: Sequence[Message]) -> str:
        if not old_messages:
            return ""
        if self._summarizer is not None:
            old_text = "\n".join(f"{m.role.value.upper()}: {m.content or ''}" for m in old_messages)
            try:
                summary = self._summarizer(old_text)
                return f"[Earlier conversation summary]\n{summary}"
            except Exception:
                logger.warning(
                    "Summarizer failed; falling back to truncation marker", exc_info=True
                )
        return f"[... {len(old_messages)} earlier messages compacted ...]"

    def _hard_truncate(self, context: str, budget: int) -> str:
        marker = "\n[... context truncated ...]"
        text = context
        while text and self._count(text + marker) > budget:
            text = text[: max(0, int(len(text) * 0.9) - 1)]
        return (text + marker) if text else marker.strip()

    def _episodic_section(self, days: int = 3, max_episodes: int = 10) -> str:
        episodes = self._episodic.recent(days)[-max_episodes:]
        if not episodes:
            return ""
        lines = ["[Recent Episodes]"]
        for episode in episodes:
            stamp = episode.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"- {stamp} user: {_clip(episode.user)} | assistant: {_clip(episode.assistant)}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read path — federated recall
    # ------------------------------------------------------------------

    def recall(self, query: str, limit: int = 10, days: Optional[int] = None) -> List[RecallResult]:
        """Federated recall across long-term, entity, and episodic tiers.

        Merge rule (documented so ordering is predictable):

        - **Long-term**: ``importance * (0.5 + 0.5 * overlap)`` where
          ``overlap`` is the fraction of query terms found in the entry.
          A high-importance exact match scores up to 1.0.
        - **Entity**: ``0.55 + 0.3 * overlap`` against the entity name,
          type, and attributes (max 0.85 — below a strong long-term hit).
        - **Episodic**: ``0.35 + 0.2 * overlap`` (max 0.55 — raw history
          ranks below distilled knowledge).

        Results are sorted by score descending; ties break newest-first.
        Items with zero term overlap are excluded.

        Args:
            query: Free-text query; split into lowercase terms (>2 chars).
            limit: Maximum results returned.
            days: Episodic date filter — only episodes from the last *days*
                days are searched.  Defaults to the retention window.

        Returns:
            Ordered list of :class:`RecallResult`.
        """
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        results: List[RecallResult] = []

        for entry in self._long_term.store.query(limit=1000):
            overlap = _overlap(terms, entry.content.lower())
            if overlap == 0.0:
                continue
            results.append(
                RecallResult(
                    content=entry.content,
                    source="long_term",
                    score=entry.importance * (0.5 + 0.5 * overlap),
                    timestamp=entry.updated_at,
                    metadata={"category": entry.category, "id": entry.id},
                )
            )

        if self._entity_memory is not None:
            for entity in self._entity_memory.entities:
                attrs = " ".join(f"{k} {v}" for k, v in entity.attributes.items())
                haystack = f"{entity.name} {entity.entity_type} {attrs}".lower()
                overlap = _overlap(terms, haystack)
                if overlap == 0.0:
                    continue
                attr_str = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                content = f"{entity.name} [{entity.entity_type}]"
                if attr_str:
                    content += f" ({attr_str})"
                results.append(
                    RecallResult(
                        content=content,
                        source="entity",
                        score=0.55 + 0.3 * overlap,
                        timestamp=datetime.fromtimestamp(entity.last_mentioned, tz=timezone.utc),
                        metadata={"entity_type": entity.entity_type},
                    )
                )

        window = self.episodic_retention_days if days is None else days
        for episode in self._episodic.search(terms, days=window):
            haystack = f"{episode.user} {episode.assistant}".lower()
            results.append(
                RecallResult(
                    content=f"user: {episode.user} | assistant: {episode.assistant}",
                    source="episodic",
                    score=0.35 + 0.2 * _overlap(terms, haystack),
                    timestamp=episode.timestamp,
                    metadata={"date": episode.timestamp.date().isoformat()},
                )
            )

        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        results.sort(key=lambda r: (-r.score, -(r.timestamp or epoch).timestamp()))
        return results[:limit]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self, include_long_term: bool = False) -> None:
        """Reset short-term, episodic, and promotion-dedup state.

        Long-term memory is preserved unless ``include_long_term=True``
        (which deletes every stored entry).  The entity tier is left intact.
        """
        with self._lock:
            self._short_term.clear()
            self._stm_mirror.clear()
            self._promoted_hashes.clear()
            self._episodic.clear()
            if include_long_term:
                for entry in self._long_term.store.query(limit=1_000_000):
                    self._long_term.store.delete(entry.id)

    def __repr__(self) -> str:
        return (
            f"UnifiedMemory(short_term={len(self._short_term)} msgs, "
            f"long_term={self._long_term.store.count()} entries, "
            f"episodic={len(self._episodic)} episodes, "
            f"threshold={self.importance_threshold}, auto_promote={self.auto_promote})"
        )


def _overlap(terms: Sequence[str], haystack: str) -> float:
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in haystack) / len(terms)


def _clip(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


register_stability("DEFAULT_IMPORTANCE_RULES", "beta")

__stability__ = "beta"

__all__ = [
    "DEFAULT_BASE_IMPORTANCE",
    "DEFAULT_IMPORTANCE_RULES",
    "Episode",
    "EpisodicMemory",
    "ImportanceRule",
    "InMemoryKnowledgeStore",
    "RecallResult",
    "UnifiedMemory",
    "score_importance",
]
