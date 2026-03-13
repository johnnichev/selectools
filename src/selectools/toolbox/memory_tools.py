"""
Memory tools — provides a ``remember`` tool for cross-session knowledge memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tools import Tool, ToolParameter

if TYPE_CHECKING:
    from ..knowledge import KnowledgeMemory


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


__all__ = ["make_remember_tool"]
