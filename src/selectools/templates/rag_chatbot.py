"""RAG chatbot agent template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import Provider

from ..agent.config import AgentConfig
from ..agent.core import Agent
from ..tools.decorators import tool


@tool(description="Search the knowledge base for relevant documents")
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the vector store for documents matching the query."""
    return f"Found {top_k} relevant documents for '{query}'"


@tool(description="Get the full content of a specific document by ID")
def get_document(doc_id: str) -> str:
    """Retrieve the full text of a document."""
    return f"Document {doc_id}: [full document content]"


SYSTEM_PROMPT = """You are a helpful chatbot that answers questions using a knowledge base.

Your responsibilities:
1. Search for relevant documents using search_knowledge_base
2. Read full documents when needed with get_document
3. Answer questions based on the retrieved information

Guidelines:
- Always search the knowledge base before answering
- Base your answers on the retrieved documents
- If the knowledge base doesn't contain the answer, say so clearly
- Quote relevant passages when appropriate
- Be conversational but accurate"""


def build(provider: "Provider", **overrides: Any) -> Agent:
    """Build a RAG chatbot agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o-mini"),
        "max_iterations": overrides.pop("max_iterations", 4),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[search_knowledge_base, get_document],
        config=AgentConfig(**config_kwargs),
    )
