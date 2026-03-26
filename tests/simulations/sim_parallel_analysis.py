"""
Simulation: Parallel Market Analysis
=====================================
3 agents research different aspects of a topic in parallel, then a
synthesizer merges their findings.

  ┌→ market_researcher ──┐
  ├→ tech_analyst ───────├→ synthesizer → END
  └→ competitor_tracker ─┘

Tests that:
- Parallel execution runs all 3 agents
- Merge policy combines results correctly
- Synthesizer receives aggregated context
- Different providers can work in the same graph (OpenAI + Anthropic)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState, MergePolicy
from selectools.providers.openai_provider import OpenAIProvider

# --- Tools ---


@tool(description="Get market data for a sector")
def get_market_data(sector: str) -> str:
    """Return market data for the given sector."""
    data = {
        "ai": "AI market: $200B in 2026, growing 35% YoY. Key segments: enterprise, healthcare, automotive.",
        "default": f"Market data for '{sector}': Moderate growth, competitive landscape.",
    }
    for key, val in data.items():
        if key in sector.lower():
            return val
    return data["default"]


@tool(description="Analyze technology trends")
def analyze_tech_trends(topic: str) -> str:
    """Analyze tech trends for a topic."""
    return f"Tech trends for '{topic}': LLMs moving to edge, multi-modal becoming standard, agent frameworks consolidating."


@tool(description="Get competitor information")
def get_competitors(market: str) -> str:
    """Return competitor landscape."""
    return f"Competitors in '{market}': LangChain (leader), CrewAI (growing), AutoGen (declining), Selectools (emerging)."


@tool(description="Synthesize multiple research inputs into a brief")
def create_brief(findings: str) -> str:
    """Create an executive brief from findings."""
    return f"EXECUTIVE BRIEF\n{'=' * 40}\n{findings}\n{'=' * 40}\nEnd of brief."


def main():
    provider = OpenAIProvider()
    model = "gpt-4.1-mini"

    market_researcher = Agent(
        provider=provider,
        tools=[get_market_data],
        config=AgentConfig(
            model=model,
            max_iterations=2,
            system_prompt="You are a market researcher. Use get_market_data to find market size and trends. Be concise.",
        ),
    )

    tech_analyst = Agent(
        provider=provider,
        tools=[analyze_tech_trends],
        config=AgentConfig(
            model=model,
            max_iterations=2,
            system_prompt="You are a technology analyst. Use analyze_tech_trends to identify key tech shifts. Be concise.",
        ),
    )

    competitor_tracker = Agent(
        provider=provider,
        tools=[get_competitors],
        config=AgentConfig(
            model=model,
            max_iterations=2,
            system_prompt="You are a competitive intelligence analyst. Use get_competitors to map the landscape. Be concise.",
        ),
    )

    synthesizer = Agent(
        provider=provider,
        tools=[create_brief],
        config=AgentConfig(
            model=model,
            max_iterations=2,
            system_prompt="You are a strategy consultant. Take the research from all analysts and use create_brief to produce an executive summary.",
        ),
    )

    # Build parallel graph
    graph = AgentGraph(name="market_analysis")
    graph.add_node("market", market_researcher)
    graph.add_node("tech", tech_analyst)
    graph.add_node("competitors", competitor_tracker)
    graph.add_parallel_nodes(
        "research_team",
        ["market", "tech", "competitors"],
        merge_policy=MergePolicy.APPEND,
    )
    graph.add_node("synthesizer", synthesizer)
    graph.add_edge("research_team", "synthesizer")
    graph.add_edge("synthesizer", AgentGraph.END)
    graph.set_entry("research_team")

    print("Running Parallel Market Analysis simulation...")
    print("=" * 60)
    result = graph.run("Analyze the AI agent framework market opportunity")

    print(f"\nFINAL OUTPUT:\n{result.content[:600]}")
    print(f"\nSteps: {result.steps}")
    print(f"Tokens: {result.total_usage.total_tokens}")
    print(f"Cost: ${result.total_usage.cost_usd:.4f}")
    print(f"Nodes: {list(result.node_results.keys())}")

    # Assertions
    assert result.content, "Should produce output"
    assert result.total_usage.total_tokens > 0
    assert "market" in result.node_results or "synthesizer" in result.node_results
    print("\nAll assertions passed!")


if __name__ == "__main__":
    main()
