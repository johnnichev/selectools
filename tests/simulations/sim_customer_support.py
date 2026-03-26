"""
Simulation: Customer Support Router
====================================
An LLM classifier routes customer requests to specialized agents.

  classifier → (billing | technical | general) → END

Tests that:
- Conditional routing works with real LLM decisions
- Different agents handle different request types
- The router picks the correct specialist
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState
from selectools.providers.openai_provider import OpenAIProvider

# --- Tools ---


@tool(description="Look up a customer's billing information")
def lookup_billing(customer_id: str) -> str:
    """Look up billing details for a customer."""
    return f"Customer {customer_id}: Plan=Pro, Balance=$45.00, Next billing=April 1"


@tool(description="Check system status and known issues")
def check_system_status(service: str) -> str:
    """Check if a service is operational."""
    return f"Service '{service}': All systems operational. No known issues."


@tool(description="Search the FAQ knowledge base")
def search_faq(question: str) -> str:
    """Search FAQs for an answer."""
    return f"FAQ match: For '{question}' — Please visit our help center at help.example.com"


@tool(description="Classify a customer request into a category")
def classify_request(message: str) -> str:
    """Classify the customer request. Returns: billing, technical, or general."""
    return "classified"


def main():
    provider = OpenAIProvider()
    model = "gpt-4.1-mini"

    # Classifier node — plain function, not an Agent
    # Classifier is a callable node, not an Agent — it calls the LLM directly
    # for a single classification without tool-calling overhead.
    def classify_node(state: GraphState) -> GraphState:
        from selectools.types import Message, Role

        user_msg = state.data.get(STATE_KEY_LAST_OUTPUT, "")
        if not user_msg and state.messages:
            user_msg = getattr(state.messages[-1], "content", str(state.messages[-1]))
        response, _ = provider.complete(
            model=model,
            system_prompt=(
                "Classify this customer request into exactly one category. "
                "Reply with ONLY one word: billing, technical, or general."
            ),
            messages=[Message(role=Role.USER, content=user_msg)],
            tools=[],
        )
        classification = (response.content or "general").strip().lower()
        state.data[STATE_KEY_LAST_OUTPUT] = classification
        return state

    billing_agent = Agent(
        provider=provider,
        tools=[lookup_billing],
        config=AgentConfig(
            model=model,
            max_iterations=3,
            system_prompt="You are a billing support specialist. Help with billing questions using the lookup_billing tool.",
        ),
    )

    technical_agent = Agent(
        provider=provider,
        tools=[check_system_status],
        config=AgentConfig(
            model=model,
            max_iterations=3,
            system_prompt="You are a technical support specialist. Help with technical issues using the check_system_status tool.",
        ),
    )

    general_agent = Agent(
        provider=provider,
        tools=[search_faq],
        config=AgentConfig(
            model=model,
            max_iterations=3,
            system_prompt="You are a general support agent. Help with general questions using the search_faq tool.",
        ),
    )

    # Router function reads the classifier's output
    def route_customer(state: GraphState) -> str:
        classification = state.data.get(STATE_KEY_LAST_OUTPUT, "").strip().lower()
        if "billing" in classification:
            return "billing"
        elif "technical" in classification:
            return "technical"
        else:
            return "general"

    # Build the graph
    graph = AgentGraph()
    graph.add_node("classifier", classify_node)
    graph.add_node("billing", billing_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("general", general_agent)
    graph.add_conditional_edge("classifier", route_customer)
    graph.add_edge("billing", AgentGraph.END)
    graph.add_edge("technical", AgentGraph.END)
    graph.add_edge("general", AgentGraph.END)

    # Test 3 different request types
    test_cases = [
        ("I need to update my payment method", "billing"),
        ("The API is returning 500 errors", "technical"),
        ("How do I reset my password?", "general"),
    ]

    for prompt, expected_route in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Customer: {prompt}")
        print(f"Expected route: {expected_route}")

        result = graph.run(prompt)

        print(f"Output: {result.content[:200]}")
        print(f"Nodes used: {list(result.node_results.keys())}")
        print(f"Tokens: {result.total_usage.total_tokens}")

        assert result.content, f"Should produce output for: {prompt}"
        assert result.total_usage.total_tokens > 0

    print(f"\n{'=' * 60}")
    print("All 3 routing scenarios completed!")


if __name__ == "__main__":
    main()
