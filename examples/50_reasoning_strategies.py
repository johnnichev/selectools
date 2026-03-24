#!/usr/bin/env python3
"""
Reasoning Strategies — ReAct, Chain-of-Thought, and Plan-Then-Act.

Demonstrates:
- reasoning_strategy="react"  — Thought → Action → Observation cycle
- reasoning_strategy="cot"    — step-by-step Chain of Thought
- reasoning_strategy="plan_then_act" — plan first, then execute

The strategy injects instructions into the system prompt so the LLM
follows a structured reasoning pattern. Combined with result.reasoning,
you get full visibility into the agent's thought process.

Prerequisites:
    pip install selectools
"""

import os

from selectools import REASONING_STRATEGIES, Agent, AgentConfig, tool
from selectools.providers.openai_provider import OpenAIProvider

api_key = os.environ.get("OPENAI_API_KEY", "")


@tool(description="Calculate a math expression")
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})  # nosec B307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(description="Look up a fact")
def lookup(topic: str) -> str:
    """Look up a fact about a topic."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991.",
        "earth": "Earth is the third planet from the Sun.",
        "selectools": "Selectools is a production-ready Python library for AI agents.",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'.")


def main():
    if not api_key:
        print("Set OPENAI_API_KEY to run this example with a real provider.")
        print()

    # Show available strategies
    print("Available reasoning strategies:")
    for name in sorted(REASONING_STRATEGIES):
        print(f"  - {name}")
    print()

    # Example 1: ReAct strategy
    print("=" * 60)
    print("Strategy: react")
    print("=" * 60)
    config = AgentConfig(
        model="gpt-4o",
        reasoning_strategy="react",
        max_iterations=4,
    )
    if api_key:
        agent = Agent(
            tools=[calculate, lookup],
            provider=OpenAIProvider(api_key=api_key),
            config=config,
        )
        result = agent.run("What year was Python created, and what is 2026 minus that year?")
        print(f"Answer: {result.content}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning[:200]}")
    else:
        agent = Agent(
            tools=[calculate, lookup],
            config=config,
        )
        print(f"System prompt includes: {'ReAct' in agent._system_prompt}")
    print()

    # Example 2: Chain of Thought
    print("=" * 60)
    print("Strategy: cot")
    print("=" * 60)
    config = AgentConfig(
        model="gpt-4o",
        reasoning_strategy="cot",
        max_iterations=4,
    )
    agent = Agent(
        tools=[calculate, lookup],
        provider=OpenAIProvider(api_key=api_key) if api_key else None,
        config=config,
    )
    if api_key:
        result = agent.run("Is Earth closer to the Sun than Mars?")
        print(f"Answer: {result.content}")
    else:
        print(f"System prompt includes: {'Chain of Thought' in agent._system_prompt}")
    print()

    # Example 3: Plan Then Act
    print("=" * 60)
    print("Strategy: plan_then_act")
    print("=" * 60)
    config = AgentConfig(
        model="gpt-4o",
        reasoning_strategy="plan_then_act",
        max_iterations=6,
    )
    agent = Agent(
        tools=[calculate, lookup],
        provider=OpenAIProvider(api_key=api_key) if api_key else None,
        config=config,
    )
    if api_key:
        result = agent.run(
            "Look up when Python was created, calculate how old it is, "
            "and tell me what selectools is."
        )
        print(f"Answer: {result.content}")
    else:
        print(f"System prompt includes: {'Plan Then Act' in agent._system_prompt}")


if __name__ == "__main__":
    main()
