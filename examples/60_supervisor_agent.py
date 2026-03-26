"""
Example 60: SupervisorAgent with multiple coordination strategies

Demonstrates all four supervisor strategies:
- plan_and_execute: LLM generates a plan, then executes each step
- round_robin: Each agent takes a turn each round
- dynamic: LLM router selects the best agent per step
- magentic: Magentic-One pattern with Task/Progress Ledgers

Uses mock agents with LocalProvider so no API keys are needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from selectools.orchestration import ModelSplit, SupervisorAgent, SupervisorStrategy
from selectools.orchestration.graph import GraphResult
from selectools.types import AgentResult, Message, Role, UsageStats


def make_mock_agent(name: str, response: str):
    """Create a mock agent that returns a predictable response."""
    agent = MagicMock()
    result = AgentResult(
        message=Message(role=Role.ASSISTANT, content=f"[{name}]: {response}"),
        iterations=1,
        usage=UsageStats(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    agent.arun = AsyncMock(return_value=result)
    return agent


def make_mock_provider(response: str):
    """Create a mock provider for the supervisor's planning/routing calls."""
    from selectools.usage import UsageStats

    provider = MagicMock()
    msg = Message(role=Role.ASSISTANT, content=response)
    provider.acomplete = AsyncMock(return_value=(msg, UsageStats()))
    return provider


async def demo_plan_and_execute():
    print("=== Strategy: plan_and_execute ===")
    plan = '[{"agent": "researcher", "task": "research AI safety"}, {"agent": "writer", "task": "write summary"}]'

    supervisor = SupervisorAgent(
        agents={
            "researcher": make_mock_agent("researcher", "AI safety research done"),
            "writer": make_mock_agent("writer", "Article written"),
        },
        provider=make_mock_provider(plan),
        strategy=SupervisorStrategy.PLAN_AND_EXECUTE,
        model_split=ModelSplit(planner_model="gpt-4o", executor_model="gpt-4o-mini"),
    )

    result = await supervisor.arun("Write a blog post about AI safety")
    print(f"Result: {result.content}")
    print(f"Steps: {result.steps}")


async def demo_round_robin():
    print("\n=== Strategy: round_robin ===")
    supervisor = SupervisorAgent(
        agents={
            "agent_a": make_mock_agent("A", "Contribution from A"),
            "agent_b": make_mock_agent("B", "Contribution from B"),
        },
        provider=make_mock_provider(""),
        strategy=SupervisorStrategy.ROUND_ROBIN,
        max_rounds=2,
    )

    result = await supervisor.arun("Collaborative task")
    print(f"Result: {result.content}")
    print(f"Stalls: {result.stalls}")


async def demo_dynamic():
    print("\n=== Strategy: dynamic ===")
    supervisor = SupervisorAgent(
        agents={
            "researcher": make_mock_agent("researcher", "research complete"),
            "analyst": make_mock_agent("analyst", "analysis done"),
        },
        provider=make_mock_provider("researcher"),  # always routes to researcher
        strategy=SupervisorStrategy.DYNAMIC,
        max_rounds=2,
    )

    result = await supervisor.arun("Analyze some data")
    print(f"Result: {result.content}")


async def demo_magentic():
    print("\n=== Strategy: magentic (Magentic-One) ===")
    done_ledger = '{"task_ledger": {"facts": ["task done"], "plan": []}, "progress_ledger": {"is_complete": true, "is_progressing": true, "next_agent": "DONE", "reason": "complete"}}'

    supervisor = SupervisorAgent(
        agents={
            "worker": make_mock_agent("worker", "work complete"),
        },
        provider=make_mock_provider(done_ledger),
        strategy=SupervisorStrategy.MAGENTIC,
        max_rounds=5,
        max_stalls=2,
    )

    result = await supervisor.arun("Complex autonomous task")
    print(f"Result: {result.content}")
    print(f"Stalls detected: {result.stalls}")


async def main():
    await demo_plan_and_execute()
    await demo_round_robin()
    await demo_dynamic()
    await demo_magentic()
    print("\nAll strategies completed!")


if __name__ == "__main__":
    asyncio.run(main())
