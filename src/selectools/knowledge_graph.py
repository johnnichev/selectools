"""
Knowledge graph memory — extract and track relationship triples across conversations.

Maintains a store of subject-relation-object triples extracted from conversation,
with LLM-powered extraction, keyword-based querying, and context building for
prompt injection.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import Message, Role


@dataclass
class Triple:
    """A relationship triple extracted from conversation.

    Attributes:
        subject: The entity performing or having the relationship.
        relation: The type of relationship (e.g. "works_at", "knows", "uses").
        object: The entity being related to.
        confidence: Confidence score from 0.0 to 1.0.
        source_turn: The conversation turn index where this was extracted.
        created_at: Unix timestamp of creation.
    """

    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source_turn: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source_turn": self.source_turn,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triple":
        return cls(
            subject=data["subject"],
            relation=data["relation"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source_turn=data.get("source_turn", 0),
            created_at=data.get("created_at", 0),
        )


# ======================================================================
# TripleStore protocol and backends
# ======================================================================


@runtime_checkable
class TripleStore(Protocol):
    """Protocol for triple storage backends."""

    def add(self, triple: Triple) -> None: ...
    def add_many(self, triples: List[Triple]) -> None: ...
    def query(self, keywords: List[str]) -> List[Triple]: ...
    def all(self) -> List[Triple]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
    def to_list(self) -> List[Dict[str, Any]]: ...


class InMemoryTripleStore:
    """In-memory triple store backed by a list."""

    def __init__(self, max_triples: int = 200) -> None:
        self._triples: List[Triple] = []
        self._max_triples = max_triples

    def add(self, triple: Triple) -> None:
        self._triples.append(triple)
        self._prune()

    def add_many(self, triples: List[Triple]) -> None:
        self._triples.extend(triples)
        self._prune()

    def query(self, keywords: List[str]) -> List[Triple]:
        if not keywords:
            return []
        results = []
        lower_keywords = [k.lower() for k in keywords]
        for t in self._triples:
            text = f"{t.subject} {t.relation} {t.object}".lower()
            if any(kw in text for kw in lower_keywords):
                results.append(t)
        return results

    def all(self) -> List[Triple]:
        return list(self._triples)

    def count(self) -> int:
        return len(self._triples)

    def clear(self) -> None:
        self._triples.clear()

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._triples]

    def _prune(self) -> None:
        if len(self._triples) > self._max_triples:
            excess = len(self._triples) - self._max_triples
            self._triples = self._triples[excess:]


class SQLiteTripleStore:
    """SQLite-backed triple store for persistent storage.

    Creates a fresh connection for each operation (matches SQLiteVectorStore pattern).
    """

    def __init__(self, db_path: str, max_triples: int = 200) -> None:
        self._db_path = db_path
        self._max_triples = max_triples
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_turn INTEGER DEFAULT 0,
                    created_at REAL DEFAULT 0
                )
                """
            )
            # Note: LIKE '%keyword%' queries cannot use B-tree indexes efficiently.
            # Consider FTS5 for full-text search if query performance becomes an issue.
            conn.commit()
        finally:
            conn.close()

    def add(self, triple: Triple) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO triples (subject, relation, object, confidence, source_turn, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    triple.subject,
                    triple.relation,
                    triple.object,
                    triple.confidence,
                    triple.source_turn,
                    triple.created_at,
                ),
            )
            conn.commit()
            self._prune(conn)
        finally:
            conn.close()

    def add_many(self, triples: List[Triple]) -> None:
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO triples (subject, relation, object, confidence, source_turn, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (t.subject, t.relation, t.object, t.confidence, t.source_turn, t.created_at)
                    for t in triples
                ],
            )
            conn.commit()
            self._prune(conn)
        finally:
            conn.close()

    def query(self, keywords: List[str]) -> List[Triple]:
        if not keywords:
            return []
        conn = self._connect()
        try:
            conditions = []
            params: List[str] = []

            def _escape_like(kw: str) -> str:
                return kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

            for kw in keywords:
                like = f"%{_escape_like(kw)}%"
                conditions.append(
                    "(LOWER(subject) LIKE LOWER(?) ESCAPE '\\'"
                    " OR LOWER(relation) LIKE LOWER(?) ESCAPE '\\'"
                    " OR LOWER(object) LIKE LOWER(?) ESCAPE '\\')"
                )
                params.extend([like, like, like])
            where = " OR ".join(conditions)
            query = (
                f"SELECT subject, relation, object, confidence, source_turn, created_at"  # nosec B608
                f" FROM triples WHERE {where} ORDER BY created_at DESC"
            )
            rows = conn.execute(query, params).fetchall()
            return [
                Triple(
                    subject=r[0],
                    relation=r[1],
                    object=r[2],
                    confidence=r[3],
                    source_turn=r[4],
                    created_at=r[5],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def all(self) -> List[Triple]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT subject, relation, object, confidence, source_turn, created_at"
                " FROM triples ORDER BY created_at ASC"
            ).fetchall()
            return [
                Triple(
                    subject=r[0],
                    relation=r[1],
                    object=r[2],
                    confidence=r[3],
                    source_turn=r[4],
                    created_at=r[5],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0])
        finally:
            conn.close()

    def clear(self) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM triples")
            conn.commit()
        finally:
            conn.close()

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.all()]

    def _prune(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        if count > self._max_triples:
            excess = count - self._max_triples
            conn.execute(
                "DELETE FROM triples WHERE id IN ("
                "  SELECT id FROM triples ORDER BY created_at ASC LIMIT ?"
                ")",
                (excess,),
            )
            conn.commit()


# ======================================================================
# KnowledgeGraphMemory
# ======================================================================


_EXTRACTION_PROMPT = (
    "Extract relationship triples from the following conversation messages. "
    "Return a JSON array of objects with keys: subject, relation, object, confidence. "
    "confidence should be a float from 0.0 to 1.0. "
    "Relations should be concise verb phrases (e.g. 'works_at', 'knows', 'prefers', 'is_a'). "
    "Only extract clearly stated relationships. "
    "Return ONLY the JSON array, no other text.\n\n"
)


class KnowledgeGraphMemory:
    """Maintains a knowledge graph of relationship triples from conversation.

    Uses an LLM to extract triples from recent messages, stores them in a
    configurable backend, and builds context strings for prompt injection.

    Args:
        provider: LLM provider for triple extraction.
        model: Model to use for extraction.  Defaults to the agent's model.
        storage: Triple store backend.  Pass ``"memory"`` for in-memory (default)
            or an existing ``TripleStore`` instance.
        max_triples: Maximum triples to store (for new in-memory stores).
        max_context_triples: Maximum triples to include in context injection.
        relevance_window: Number of recent messages to extract triples from.
    """

    def __init__(
        self,
        provider: Any,
        model: Optional[str] = None,
        storage: Any = "memory",
        max_triples: int = 200,
        max_context_triples: int = 15,
        relevance_window: int = 10,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_context_triples = max_context_triples
        self._relevance_window = relevance_window

        if storage == "memory":
            self._store: Any = InMemoryTripleStore(max_triples=max_triples)
        elif isinstance(storage, TripleStore):
            self._store = storage
        else:
            self._store = storage

    @property
    def store(self) -> Any:
        """The underlying triple store."""
        return self._store

    def extract_triples(
        self,
        messages: List[Message],
        model: Optional[str] = None,
    ) -> List[Triple]:
        """Extract relationship triples from messages using the LLM.

        Args:
            messages: Messages to extract triples from.
            model: Optional model override.

        Returns:
            List of newly extracted Triple objects.
        """
        text_parts = []
        recent = messages[-self._relevance_window :]
        for m in recent:
            if m.content:
                text_parts.append(f"{m.role.value.upper()}: {m.content}")

        if not text_parts:
            return []

        conversation_text = "\n".join(text_parts)
        prompt = Message(
            role=Role.USER,
            content=_EXTRACTION_PROMPT + conversation_text,
        )

        try:
            result = self._provider.complete(
                model=model or self._model or "gpt-4o-mini",
                system_prompt="You extract relationship triples from text. Always return valid JSON.",
                messages=[prompt],
                max_tokens=500,
            )
            response_msg = result[0] if isinstance(result, tuple) else result
            raw_text = (response_msg.content or "").strip()
            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(line for line in lines if not line.strip().startswith("```"))
            triples_data = json.loads(raw_text)
            if not isinstance(triples_data, list):
                return []

            now = time.time()
            extracted = []
            for item in triples_data:
                if not isinstance(item, dict):
                    continue
                if "subject" not in item or "relation" not in item or "object" not in item:
                    continue
                extracted.append(
                    Triple(
                        subject=item["subject"],
                        relation=item["relation"],
                        object=item["object"],
                        confidence=float(item.get("confidence", 1.0)),
                        source_turn=len(messages),
                        created_at=now,
                    )
                )
            return extracted
        except Exception:
            return []

    def query_relevant(self, query: str) -> List[Triple]:
        """Query the store for triples relevant to a text query.

        Splits the query into keywords and returns matching triples,
        limited by ``max_context_triples``.

        Args:
            query: Free-text query to match against triples.

        Returns:
            List of matching Triple objects.
        """
        keywords = [w for w in query.lower().split() if len(w) > 2]
        if not keywords:
            return []
        results: List[Triple] = self._store.query(keywords)
        return results[: self._max_context_triples]

    def build_context(self, query: str = "") -> str:
        """Build a context string for prompt injection.

        If a query is provided, returns only query-relevant triples.
        Otherwise returns the most recent triples up to ``max_context_triples``.

        Args:
            query: Optional free-text query to filter triples.

        Returns:
            A formatted ``[Known Relationships]`` block.
        """
        if query:
            triples = self.query_relevant(query)
        else:
            all_triples = self._store.all()
            triples = all_triples[-self._max_context_triples :]

        if not triples:
            return ""

        lines = ["[Known Relationships]"]
        for t in triples:
            conf = f" (confidence: {t.confidence:.1f})" if t.confidence < 1.0 else ""
            lines.append(f"- {t.subject} --[{t.relation}]--> {t.object}{conf}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_triples": self._max_context_triples,
            "relevance_window": self._relevance_window,
            "triples": self._store.to_list(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], provider: Any) -> "KnowledgeGraphMemory":
        mem = cls(
            provider=provider,
            max_context_triples=data.get("max_context_triples", 15),
            relevance_window=data.get("relevance_window", 10),
        )
        triples = [Triple.from_dict(t) for t in data.get("triples", [])]
        if triples:
            mem._store.add_many(triples)
        return mem


__all__ = [
    "Triple",
    "TripleStore",
    "InMemoryTripleStore",
    "SQLiteTripleStore",
    "KnowledgeGraphMemory",
]
