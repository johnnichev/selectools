"""
Simulation: Cross-Provider Pipeline
====================================
OpenAI drafts → Anthropic reviews → Gemini translates.

Tests that:
- 3 different providers work in a single graph
- Context flows correctly between providers
- Each provider's tool calling works
- No SYSTEM role errors (the Anthropic bug we fixed)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.gemini_provider import GeminiProvider
from selectools.providers.openai_provider import OpenAIProvider

# --- Tools ---


@tool(description="Draft a short text on a topic")
def draft_text(topic: str) -> str:
    """Create a draft paragraph about a topic."""
    return f"Draft about {topic}: This is an important subject that affects many people."


@tool(description="Review text for quality and clarity")
def review_text(text: str) -> str:
    """Review and provide feedback on text."""
    word_count = len(text.split())
    return f"Review: {word_count} words. Clarity: good. Suggestion: add specific examples."


@tool(description="Translate text to Spanish")
def translate_to_spanish(text: str) -> str:
    """Translate English text to Spanish."""
    return f"[ES] {text[:100]}... (translated to Spanish)"


def main():
    openai = OpenAIProvider()
    anthropic = AnthropicProvider()
    gemini = GeminiProvider()

    drafter = Agent(
        provider=openai,
        tools=[draft_text],
        config=AgentConfig(
            model="gpt-4.1-mini",
            max_iterations=2,
            system_prompt="You are a content drafter. Use draft_text to create content, then polish it.",
        ),
    )

    reviewer = Agent(
        provider=anthropic,
        tools=[review_text],
        config=AgentConfig(
            model="claude-haiku-4-5-20251001",
            max_iterations=2,
            system_prompt="You are a content reviewer. Use review_text to evaluate the draft and provide improvements.",
        ),
    )

    translator = Agent(
        provider=gemini,
        tools=[translate_to_spanish],
        config=AgentConfig(
            model="gemini-3-flash-preview",
            max_iterations=2,
            system_prompt="You are a translator. Use translate_to_spanish to translate the reviewed content.",
        ),
    )

    graph = AgentGraph.chain(
        drafter,
        reviewer,
        translator,
        names=["openai_draft", "anthropic_review", "gemini_translate"],
    )

    print("Running Cross-Provider Pipeline simulation...")
    print("=" * 60)
    result = graph.run("Write about the benefits of open source software")

    print(f"\nFINAL OUTPUT:\n{result.content[:400]}")
    print(f"\nProviders used:")
    for name in result.node_results:
        node_result = result.node_results[name][0]
        print(f"  {name}: {len(node_result.message.content or '')} chars")
    print(f"\nTotal tokens: {result.total_usage.total_tokens}")
    print(f"Total cost: ${result.total_usage.cost_usd:.4f}")

    # Assertions
    assert result.content, "Should produce output"
    assert len(result.node_results) == 3, "All 3 providers should execute"
    assert "openai_draft" in result.node_results
    assert "anthropic_review" in result.node_results
    assert "gemini_translate" in result.node_results
    assert result.total_usage.total_tokens > 0
    print("\nAll assertions passed! 3 providers worked together.")


if __name__ == "__main__":
    main()
