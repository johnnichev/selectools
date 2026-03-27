"""Research assistant agent template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import Provider

from ..agent.config import AgentConfig
from ..agent.core import Agent
from ..tools.decorators import tool


@tool(description="Search the web for information on a topic")
def web_search(query: str) -> str:
    """Search the web and return top results."""
    return f"Search results for '{query}': Multiple relevant sources found."


@tool(description="Read and extract content from a URL")
def read_url(url: str) -> str:
    """Fetch and extract main content from a web page."""
    return f"Content from {url}: [Article content extracted]"


@tool(description="Save research notes for later reference")
def save_notes(topic: str, notes: str) -> str:
    """Save research notes on a topic."""
    return f"Notes saved for '{topic}': {len(notes)} characters stored."


SYSTEM_PROMPT = """You are a thorough research assistant.

Your responsibilities:
1. Search for information using web_search
2. Read detailed content from sources using read_url
3. Organize findings with save_notes

Guidelines:
- Search broadly first, then dive deep into relevant sources
- Always cite your sources
- Distinguish between facts and opinions
- Summarize findings clearly
- Flag conflicting information across sources"""


def build(provider: "Provider", **overrides: Any) -> Agent:
    """Build a research assistant agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o-mini"),
        "max_iterations": overrides.pop("max_iterations", 8),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[web_search, read_url, save_notes],
        config=AgentConfig(**config_kwargs),
    )
