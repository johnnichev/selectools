"""Customer support agent template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import Provider

from ..agent.config import AgentConfig
from ..agent.core import Agent
from ..tools.decorators import tool


@tool(description="Look up customer account details by customer ID or email")
def lookup_customer(identifier: str) -> str:
    """Look up a customer's account. Returns account status, plan, and recent activity."""
    return f"Customer {identifier}: Active account, Pro plan, last login 2 days ago."


@tool(description="Search the knowledge base for help articles")
def search_help_articles(query: str) -> str:
    """Search help center articles matching the query."""
    return f"Help articles for '{query}': Found 3 relevant articles. See help.example.com"


@tool(description="Create a support ticket for the customer")
def create_ticket(subject: str, priority: str = "medium") -> str:
    """Create a support ticket. Priority: low, medium, high, urgent."""
    return f"Ticket created: '{subject}' (priority: {priority}). Ticket ID: TK-12345"


SYSTEM_PROMPT = """You are a friendly and professional customer support agent.

Your responsibilities:
1. Help customers with account questions using lookup_customer
2. Find answers in the knowledge base using search_help_articles
3. Escalate complex issues by creating tickets with create_ticket

Guidelines:
- Always greet the customer warmly
- Look up their account before asking them to repeat information
- Search help articles before escalating
- Only create tickets for issues you cannot resolve directly
- Be concise but thorough"""


def build(provider: "Provider", **overrides: Any) -> Agent:
    """Build a customer support agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o-mini"),
        "max_iterations": overrides.pop("max_iterations", 5),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[lookup_customer, search_help_articles, create_ticket],
        config=AgentConfig(**config_kwargs),
    )
