"""
Memory tools — ``remember`` and ``recall`` tools for cross-session knowledge memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..stability import beta, stable
from ..tools import Tool, ToolParameter

if TYPE_CHECKING:
    from ..knowledge import KnowledgeEntry, KnowledgeMemory


@stable
def make_remember_tool(knowledge: "KnowledgeMemory") -> Tool:
    """Create a ``remember`` tool bound to a KnowledgeMemory instance.

    The tool allows the agent to store facts and preferences that persist
    across sessions.

    Args:
        knowledge: The KnowledgeMemory instance to store facts in.

    Returns:
        A Tool that the agent can call to remember information.
    """

    def _remember(content: str, category: str = "general", persistent: str = "false") -> str:
        is_persistent = persistent.lower() in ("true", "yes", "1")
        return knowledge.remember(content=content, category=category, persistent=is_persistent)

    return Tool(
        name="remember",
        description=(
            "Store a piece of information for future reference. "
            "Use this to remember user preferences, important facts, "
            "or context that should persist across conversations. "
            "Set persistent=true for long-term facts that should never expire."
        ),
        parameters=[
            ToolParameter(
                name="content",
                param_type=str,
                description="The information to remember.",
                required=True,
            ),
            ToolParameter(
                name="category",
                param_type=str,
                description="Category tag (e.g. 'preference', 'fact', 'context'). Default: 'general'.",
                required=False,
            ),
            ToolParameter(
                name="persistent",
                param_type=str,
                description="Set to 'true' for long-term facts. Default: 'false'.",
                required=False,
            ),
        ],
        function=_remember,
    )


_DEFAULT_RECALL_LIMIT = 5
# Entries fetched (importance-ordered) before in-process keyword ranking.
# KnowledgeMemory exposes no text search, so recall ranks a window of the
# highest-importance entries. With the default max_entries (50) this covers the
# whole store. NOTE: a custom store holding more than this many entries can hide
# a low-importance keyword match below the window — raise this (or add a
# store-level search) if you run recall against a large unbounded store.
_RECALL_FETCH_WINDOW = 500


@beta
def make_recall_tool(knowledge: "KnowledgeMemory") -> Tool:
    """Create a ``recall`` tool bound to a KnowledgeMemory instance.

    The tool allows the agent to search previously stored facts and
    preferences.  Entries are fetched from the underlying store and
    keyword-matched against the query (case-insensitive, any token
    matches content or category), ranked by match count then importance.

    Args:
        knowledge: The KnowledgeMemory instance to search.

    Returns:
        A Tool that the agent can call to recall stored information.
    """

    def _recall(query: str, limit: str = "5") -> str:
        try:
            max_results = int(limit)
        except ValueError:
            max_results = _DEFAULT_RECALL_LIMIT
        if max_results < 1:
            max_results = _DEFAULT_RECALL_LIMIT

        entries: List["KnowledgeEntry"] = knowledge.store.query(limit=_RECALL_FETCH_WINDOW)
        tokens = [t for t in query.lower().split() if t]
        if tokens:
            scored = []
            for entry in entries:
                haystack = f"{entry.content} {entry.category}".lower()
                score = sum(1 for t in tokens if t in haystack)
                if score:
                    scored.append((score, entry))
            scored.sort(key=lambda pair: (-pair[0], -pair[1].importance))
            matches = [entry for _, entry in scored]
        else:
            # Empty query: fall back to the most important entries.
            matches = entries

        matches = matches[:max_results]
        if not matches:
            return f"No memories found matching '{query}'."

        lines = [f"Found {len(matches)} memor{'y' if len(matches) == 1 else 'ies'}:"]
        for i, entry in enumerate(matches, 1):
            tag = f"[{entry.category}] " if entry.category != "general" else ""
            lines.append(f"{i}. {tag}{entry.content}")
        return "\n".join(lines)

    return Tool(
        name="recall",
        description=(
            "Search previously stored memories for relevant information. "
            "Use this to retrieve user preferences, important facts, "
            "or context saved in earlier conversations before asking the user. "
            "Returns the closest matches ranked by relevance and importance."
        ),
        parameters=[
            ToolParameter(
                name="query",
                param_type=str,
                description="Keywords describing the information to look up.",
                required=True,
            ),
            ToolParameter(
                name="limit",
                param_type=str,
                description="Maximum number of memories to return. Default: '5'.",
                required=False,
            ),
        ],
        function=_recall,
    )


__stability__ = "stable"

__all__ = ["make_recall_tool", "make_remember_tool"]
