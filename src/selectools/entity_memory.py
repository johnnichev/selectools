"""
Entity memory — auto-extract and track named entities across conversations.

Maintains a registry of entities mentioned in conversation, with LLM-powered
extraction, deduplication, and LRU pruning.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Message, Role


@dataclass
class Entity:
    """A named entity extracted from conversation.

    Attributes:
        name: Canonical name of the entity (e.g. "Alice", "Python 3.12").
        entity_type: Category (e.g. "person", "technology", "company").
        attributes: Key-value pairs of known facts about the entity.
        first_mentioned: Unix timestamp of first mention.
        last_mentioned: Unix timestamp of most recent mention.
        mention_count: Number of times the entity has been mentioned.
    """

    name: str
    entity_type: str
    attributes: Dict[str, str] = field(default_factory=dict)
    first_mentioned: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    mention_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "first_mentioned": self.first_mentioned,
            "last_mentioned": self.last_mentioned,
            "mention_count": self.mention_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            name=data["name"],
            entity_type=data.get("entity_type", "unknown"),
            attributes=data.get("attributes", {}),
            first_mentioned=data.get("first_mentioned", 0),
            last_mentioned=data.get("last_mentioned", 0),
            mention_count=data.get("mention_count", 1),
        )


_EXTRACTION_PROMPT = (
    "Extract named entities from the following conversation messages. "
    "Return a JSON array of objects with keys: name, entity_type, attributes. "
    "entity_type should be one of: person, organization, technology, location, concept, other. "
    "attributes should be a dict of known facts. "
    "Only extract entities that are clearly identifiable. "
    "Return ONLY the JSON array, no other text.\n\n"
)


class EntityMemory:
    """Maintains a registry of entities mentioned in conversation.

    Uses an LLM to extract entities from recent messages, merges them into
    a persistent registry, and builds context strings for prompt injection.

    Args:
        provider: LLM provider for entity extraction.
        model: Model to use for extraction.  Defaults to the agent's model.
        max_entities: Maximum entities to track.  Oldest-mentioned are pruned.
        relevance_window: Number of recent messages to extract entities from.
    """

    def __init__(
        self,
        provider: Any,
        model: Optional[str] = None,
        max_entities: int = 50,
        relevance_window: int = 10,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_entities = max_entities
        self._relevance_window = relevance_window
        self._entities: Dict[str, Entity] = {}
        self._lock = threading.Lock()

    @property
    def entities(self) -> List[Entity]:
        """All tracked entities, sorted by most recently mentioned."""
        with self._lock:
            return sorted(
                self._entities.values(),
                key=lambda e: e.last_mentioned,
                reverse=True,
            )

    def extract_entities(
        self,
        messages: List[Message],
        model: Optional[str] = None,
    ) -> List[Entity]:
        """Extract entities from messages using the LLM.

        Args:
            messages: Messages to extract entities from.
            model: Optional model override.

        Returns:
            List of newly extracted Entity objects.
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
                system_prompt="You extract named entities from text. Always return valid JSON.",
                messages=[prompt],
                max_tokens=500,
            )
            response_msg = result[0] if isinstance(result, tuple) else result
            raw_text = (response_msg.content or "").strip()
            # Strip markdown code fences if present
            raw_text = re.sub(r"^```\w*\n?", "", raw_text, count=1)
            raw_text = re.sub(r"\n?```\s*$", "", raw_text, count=1)
            entities_data = json.loads(raw_text)
            if not isinstance(entities_data, list):
                return []

            now = time.time()
            extracted = []
            for item in entities_data:
                if not isinstance(item, dict) or "name" not in item:
                    continue
                raw_attrs = item.get("attributes", {})
                attrs = raw_attrs if isinstance(raw_attrs, dict) else {}
                extracted.append(
                    Entity(
                        name=item["name"],
                        entity_type=item.get("entity_type", "other"),
                        attributes=attrs,
                        first_mentioned=now,
                        last_mentioned=now,
                        mention_count=1,
                    )
                )
            return extracted
        except Exception:
            return []

    def update(self, entities: List[Entity]) -> None:
        """Merge extracted entities into the registry.

        Deduplicates by name (case-insensitive), updates mention counts
        and attributes, and prunes if over ``max_entities``.

        Thread-safe: protected by an internal lock.
        """
        now = time.time()
        with self._lock:
            for entity in entities:
                key = entity.name.lower()
                if key in self._entities:
                    existing = self._entities[key]
                    existing.mention_count += 1
                    existing.last_mentioned = now
                    existing.attributes.update(entity.attributes)
                else:
                    entity.last_mentioned = now
                    self._entities[key] = entity

            # LRU prune: remove least recently mentioned
            if len(self._entities) > self._max_entities:
                sorted_keys = sorted(
                    self._entities.keys(),
                    key=lambda k: self._entities[k].last_mentioned,
                )
                excess = len(self._entities) - self._max_entities
                for k in sorted_keys[:excess]:
                    del self._entities[k]

    def build_context(self) -> str:
        """Build a context string for prompt injection.

        Returns:
            A formatted ``[Known Entities]`` block listing all tracked entities.
        """
        if not self.entities:  # Uses the locked property
            return ""

        lines = ["[Known Entities]"]
        for entity in self.entities:
            raw_attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
            attrs = ", ".join(f"{k}: {v}" for k, v in raw_attrs.items())
            attr_str = f" ({attrs})" if attrs else ""
            lines.append(f"- {entity.name} [{entity.entity_type}]{attr_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_entities": self._max_entities,
            "relevance_window": self._relevance_window,
            "entities": [e.to_dict() for e in self._entities.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], provider: Any) -> "EntityMemory":
        mem = cls(
            provider=provider,
            max_entities=data.get("max_entities", 50),
            relevance_window=data.get("relevance_window", 10),
        )
        for e_data in data.get("entities", []):
            entity = Entity.from_dict(e_data)
            mem._entities[entity.name.lower()] = entity
        return mem


__all__ = ["Entity", "EntityMemory"]
