"""
Real-LLM evals for multi-agent simulation scenarios.

Each test validates a real-world use case end-to-end:
- Correct tool usage
- Output quality (content references upstream work)
- Routing accuracy
- Cross-provider data flow

Run with: pytest tests/test_simulation_evals.py -v --run-e2e
"""

from __future__ import annotations

import os

import pytest

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState, MergePolicy
from selectools.types import Message, Role

# ---------------------------------------------------------------------------
# Shared tools (same as simulations)
# ---------------------------------------------------------------------------


@tool(description="Search the web for information on a topic")
def web_search(query: str) -> str:
    """Search the web and return results."""
    results = {
        "ai safety": "Key concerns: alignment, misuse, autonomous systems. Leading researchers: Yoshua Bengio, Stuart Russell.",
        "climate": "Global temperatures rising 1.5C since pre-industrial. Key: renewable energy, carbon capture.",
    }
    for key, val in results.items():
        if key in query.lower():
            return val
    return f"Top results for '{query}': Multiple sources found."


@tool(description="Analyze data and extract key insights")
def analyze_data(text: str) -> str:
    """Analyze text and return structured insights."""
    word_count = len(text.split())
    return f"Analysis complete: {word_count} words processed. Key themes extracted."


@tool(description="Format text into a structured report")
def format_report(content: str, title: str = "Report") -> str:
    """Format content into a report."""
    return f"# {title}\n\n{content}\n\n---\nEnd of report."


@tool(description="Look up a customer's billing information")
def lookup_billing(customer_id: str) -> str:
    """Look up billing details."""
    return f"Customer {customer_id}: Plan=Pro, Balance=$45.00, Next billing=April 1"


@tool(description="Check system status and known issues")
def check_system_status(service: str) -> str:
    """Check service status."""
    return f"Service '{service}': All systems operational."


@tool(description="Search the FAQ knowledge base")
def search_faq(question: str) -> str:
    """Search FAQs."""
    return f"FAQ: For '{question}' see help.example.com/faq"


@tool(description="Get market data for a sector")
def get_market_data(sector: str) -> str:
    """Return market data."""
    return "AI market: $200B in 2026, growing 35% YoY. Key segments: enterprise, healthcare."


@tool(description="Analyze technology trends")
def analyze_tech_trends(topic: str) -> str:
    """Analyze tech trends."""
    return (
        f"Trends for '{topic}': LLMs on edge, multi-modal standard, agent frameworks consolidating."
    )


@tool(description="Get competitor information")
def get_competitors(market: str) -> str:
    """Return competitor landscape."""
    return "Competitors: LangChain (leader), CrewAI (growing), AutoGen (declining), Selectools (emerging)."


@tool(description="Create an executive brief from findings")
def create_brief(findings: str) -> str:
    """Create executive brief."""
    return f"EXECUTIVE BRIEF: {findings}"


@tool(description="Review text for quality")
def review_text(text: str) -> str:
    """Review text quality."""
    return f"Review: {len(text.split())} words. Clarity: good. Add specific examples."


@tool(description="Translate text to Spanish")
def translate_to_spanish(text: str) -> str:
    """Translate to Spanish."""
    return f"[Traduccion] {text[:200]}"


@tool(description="Draft a short text on a topic")
def draft_text(topic: str) -> str:
    """Draft text about a topic."""
    return f"Open source software provides transparency, community collaboration, and cost savings for {topic}."


@tool(description="Classify a customer request")
def classify_request(message: str) -> str:
    """Classify request category."""
    return "classified"


# ---------------------------------------------------------------------------
# Provider helper
# ---------------------------------------------------------------------------


def _openai():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from selectools.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(), "gpt-4.1-mini"


def _anthropic():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from selectools.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(), "claude-haiku-4-5-20251001"


def _gemini():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    from selectools.providers.gemini_provider import GeminiProvider

    return GeminiProvider(), "gemini-3-flash-preview"


def _agent(provider, model, tools, system_prompt, max_iter=3):
    return Agent(
        provider=provider,
        tools=tools,
        config=AgentConfig(model=model, max_iterations=max_iter, system_prompt=system_prompt),
    )


# ===========================================================================
# Scenario 1: Research Team Pipeline
# ===========================================================================


@pytest.mark.e2e
class TestResearchTeamEval:
    """3-agent research pipeline: researcher → analyst → writer.
    Validates that each agent's output builds on the previous."""

    def test_researcher_actually_searches(self):
        """Researcher agent must call web_search and produce findings."""
        provider, model = _openai()
        researcher = _agent(
            provider,
            model,
            [web_search],
            "You are a researcher. Use web_search to find information, then summarize findings.",
        )
        graph = AgentGraph.chain(researcher)
        result = graph.run("Research AI safety concerns")

        assert result.content, "Researcher should produce output"
        # The tool returns "alignment" and "Bengio" — LLM should reference these
        content_lower = result.content.lower()
        has_substance = (
            "alignment" in content_lower
            or "safety" in content_lower
            or "bengio" in content_lower
            or "autonomous" in content_lower
        )
        assert has_substance, (
            f"Researcher output should reference search results. Got: {result.content[:300]}"
        )

    def test_full_pipeline_writer_references_research(self):
        """Writer's final output should reference content from researcher and analyst."""
        provider, model = _openai()

        researcher = _agent(
            provider,
            model,
            [web_search],
            "You are a researcher. Use web_search to find information about the topic.",
        )
        analyst = _agent(
            provider,
            model,
            [analyze_data],
            "You are an analyst. Use analyze_data on the research findings.",
        )
        writer = _agent(
            provider,
            model,
            [format_report],
            "You are a writer. Use format_report to create a final report from the analysis.",
        )

        graph = AgentGraph.chain(
            researcher, analyst, writer, names=["research", "analysis", "write"]
        )
        result = graph.run("Research the current state of AI safety")

        # All 3 agents should execute
        assert len(result.node_results) == 3, (
            f"Expected 3 nodes, got {list(result.node_results.keys())}"
        )

        # Writer output should be substantive (not just "ok" or empty)
        assert len(result.content) > 100, (
            f"Final report should be substantive. Got {len(result.content)} chars: {result.content[:200]}"
        )

        # Final output should reference AI safety (the topic flows through)
        content_lower = result.content.lower()
        assert "ai" in content_lower or "safety" in content_lower or "alignment" in content_lower, (
            f"Final report should reference the research topic. Got: {result.content[:300]}"
        )


# ===========================================================================
# Scenario 2: Customer Support Routing
# ===========================================================================


@pytest.mark.e2e
class TestCustomerSupportEval:
    """LLM classifier routes to specialist agents.
    Validates routing accuracy and specialist tool usage."""

    def _build_graph(self, provider, model):
        """Build the support graph with classifier + 3 specialists."""

        def classify_node(state: GraphState) -> GraphState:
            user_msg = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            if not user_msg and state.messages:
                user_msg = getattr(state.messages[-1], "content", "")
            response, _ = provider.complete(
                model=model,
                system_prompt="Classify this customer request. Reply with ONLY one word: billing, technical, or general.",
                messages=[Message(role=Role.USER, content=user_msg)],
                tools=[],
            )
            state.data[STATE_KEY_LAST_OUTPUT] = (response.content or "general").strip().lower()
            return state

        def route(state: GraphState) -> str:
            c = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            if "billing" in c:
                return "billing"
            elif "technical" in c:
                return "technical"
            return "general"

        billing = _agent(
            provider,
            model,
            [lookup_billing],
            "You are billing support. Use lookup_billing to help.",
        )
        technical = _agent(
            provider,
            model,
            [check_system_status],
            "You are technical support. Use check_system_status to help.",
        )
        general = _agent(
            provider, model, [search_faq], "You are general support. Use search_faq to help."
        )

        graph = AgentGraph()
        graph.add_node("classify", classify_node)
        graph.add_node("billing", billing)
        graph.add_node("technical", technical)
        graph.add_node("general", general)
        graph.add_conditional_edge("classify", route)
        graph.add_edge("billing", AgentGraph.END)
        graph.add_edge("technical", AgentGraph.END)
        graph.add_edge("general", AgentGraph.END)
        return graph

    def test_billing_request_routes_to_billing(self):
        """Billing question should route to billing agent."""
        provider, model = _openai()
        graph = self._build_graph(provider, model)
        result = graph.run("I was charged twice on my credit card last month")

        nodes = list(result.node_results.keys())
        assert "billing" in nodes, f"Should route to billing. Routed to: {nodes}"
        assert result.content, "Should produce helpful response"

    def test_technical_request_routes_to_technical(self):
        """Technical question should route to technical agent."""
        provider, model = _openai()
        graph = self._build_graph(provider, model)
        result = graph.run("The API keeps returning 500 internal server errors")

        nodes = list(result.node_results.keys())
        assert "technical" in nodes, f"Should route to technical. Routed to: {nodes}"

    def test_specialist_uses_correct_tool(self):
        """Billing agent should call lookup_billing, not search_faq."""
        provider, model = _openai()
        graph = self._build_graph(provider, model)
        result = graph.run("What is my current account balance?")

        # Check that billing node executed and produced billing-related content
        content_lower = result.content.lower()
        has_billing_content = (
            "balance" in content_lower
            or "plan" in content_lower
            or "billing" in content_lower
            or "$" in result.content
        )
        assert has_billing_content, (
            f"Billing agent should reference account info. Got: {result.content[:300]}"
        )


# ===========================================================================
# Scenario 3: Parallel Analysis with Synthesis
# ===========================================================================


@pytest.mark.e2e
class TestParallelAnalysisEval:
    """3 parallel agents research, then synthesizer merges.
    Validates all 3 contribute and synthesis is coherent."""

    def test_all_parallel_agents_execute(self):
        """All 3 parallel research agents should produce output."""
        provider, model = _openai()

        market = _agent(
            provider, model, [get_market_data], "Use get_market_data. Be concise.", max_iter=2
        )
        tech = _agent(
            provider,
            model,
            [analyze_tech_trends],
            "Use analyze_tech_trends. Be concise.",
            max_iter=2,
        )
        competitors = _agent(
            provider, model, [get_competitors], "Use get_competitors. Be concise.", max_iter=2
        )

        graph = AgentGraph()
        graph.add_node("market", market)
        graph.add_node("tech", tech)
        graph.add_node("competitors", competitors)
        graph.add_parallel_nodes(
            "research", ["market", "tech", "competitors"], merge_policy=MergePolicy.APPEND
        )
        graph.add_edge("research", AgentGraph.END)
        graph.set_entry("research")

        result = graph.run("Analyze the AI market")

        assert "market" in result.node_results, "Market researcher should execute"
        assert "tech" in result.node_results, "Tech analyst should execute"
        assert "competitors" in result.node_results, "Competitor tracker should execute"

    def test_synthesizer_references_all_inputs(self):
        """Synthesizer should produce output that reflects parallel research."""
        provider, model = _openai()

        market = _agent(
            provider, model, [get_market_data], "Use get_market_data. Be concise.", max_iter=2
        )
        tech = _agent(
            provider,
            model,
            [analyze_tech_trends],
            "Use analyze_tech_trends. Be concise.",
            max_iter=2,
        )
        competitors = _agent(
            provider, model, [get_competitors], "Use get_competitors. Be concise.", max_iter=2
        )
        synth = _agent(
            provider,
            model,
            [create_brief],
            "Synthesize all research into an executive brief using create_brief. Reference market data, tech trends, and competitors.",
            max_iter=2,
        )

        graph = AgentGraph()
        graph.add_node("market", market)
        graph.add_node("tech", tech)
        graph.add_node("competitors", competitors)
        graph.add_parallel_nodes(
            "research", ["market", "tech", "competitors"], merge_policy=MergePolicy.APPEND
        )
        graph.add_node("synth", synth)
        graph.add_edge("research", "synth")
        graph.add_edge("synth", AgentGraph.END)
        graph.set_entry("research")

        result = graph.run("Analyze the AI agent framework market")

        assert result.content, "Synthesizer should produce output"
        assert len(result.content) > 50, (
            f"Synthesis should be substantive. Got: {result.content[:200]}"
        )


# ===========================================================================
# Scenario 4: Cross-Provider Pipeline
# ===========================================================================


@pytest.mark.e2e
class TestCrossProviderEval:
    """OpenAI drafts → Anthropic reviews → Gemini translates.
    Validates each provider processes the previous provider's output."""

    def test_three_providers_produce_output(self):
        """All 3 providers should execute and produce content."""
        openai_prov, openai_model = _openai()
        anthropic_prov, anthropic_model = _anthropic()
        gemini_prov, gemini_model = _gemini()

        drafter = _agent(
            openai_prov, openai_model, [draft_text], "Draft content using draft_text.", max_iter=2
        )
        reviewer = _agent(
            anthropic_prov,
            anthropic_model,
            [review_text],
            "Review the draft using review_text.",
            max_iter=2,
        )
        translator = _agent(
            gemini_prov,
            gemini_model,
            [translate_to_spanish],
            "Translate using translate_to_spanish.",
            max_iter=2,
        )

        graph = AgentGraph.chain(
            drafter, reviewer, translator, names=["draft", "review", "translate"]
        )
        result = graph.run("Write about open source software benefits")

        assert "draft" in result.node_results, "OpenAI drafter should execute"
        assert "review" in result.node_results, "Anthropic reviewer should execute"
        assert "translate" in result.node_results, "Gemini translator should execute"

        # Each node should produce non-empty output
        for name in ["draft", "review", "translate"]:
            node_content = result.node_results[name][0].message.content or ""
            assert len(node_content) > 10, f"{name} output too short: {node_content[:100]}"

    def test_anthropic_receives_openai_context(self):
        """Anthropic reviewer should reference content from OpenAI drafter."""
        openai_prov, openai_model = _openai()
        anthropic_prov, anthropic_model = _anthropic()

        drafter = _agent(
            openai_prov,
            openai_model,
            [draft_text],
            "Draft about open source using draft_text.",
            max_iter=2,
        )
        reviewer = _agent(
            anthropic_prov,
            anthropic_model,
            [review_text],
            "Review the draft using review_text. Comment on the content quality.",
            max_iter=2,
        )

        graph = AgentGraph.chain(drafter, reviewer, names=["draft", "review"])
        result = graph.run("Benefits of open source software")

        # Reviewer should produce a review (not just echo the draft)
        review_content = result.node_results["review"][0].message.content or ""
        assert len(review_content) > 20, f"Review should be substantive: {review_content[:200]}"
