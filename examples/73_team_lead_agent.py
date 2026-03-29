"""
Example 73: TeamLeadAgent

The lead Agent generates a subtask plan and delegates to team members. Three
delegation strategies are demonstrated:

- sequential: tasks run one-by-one; each member sees prior work as context
- parallel:   all tasks run simultaneously via AgentGraph fan-out
- dynamic:    lead reviews each result and decides whether to reassign or finish

Pattern: lead → [member_a, member_b, ...] → lead synthesis → TeamLeadResult

Run: python examples/73_team_lead_agent.py
"""

import asyncio
from typing import List

from selectools import tool
from selectools.agent import Agent, AgentConfig
from selectools.patterns import TeamLeadAgent
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.usage import UsageStats


@tool(description="Placeholder tool — not called in this example")
def _noop(x: str) -> str:
    return x


class _ScriptedProvider(LocalProvider):
    """Returns scripted responses in order, cycling if exhausted."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._index = 0

    def complete(self, *, model, system_prompt, messages, **kwargs):  # type: ignore[override]
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return Message(role=Role.ASSISTANT, content=text), UsageStats()


def _make_agent(*responses: str) -> Agent:
    """Create an agent that returns scripted responses in order."""
    return Agent(tools=[_noop], provider=_ScriptedProvider(list(responses)), config=AgentConfig())


# ── Sequential strategy ───────────────────────────────────────────────────────


def demo_sequential():
    print("\n--- Sequential strategy ---")

    lead = _make_agent(
        # Planning call: assigns tasks to each member
        '[{"assignee": "researcher", "task": "research market size for AI dev tools"}, '
        '{"assignee": "analyst", "task": "identify top 3 competitors"}, '
        '{"assignee": "writer", "task": "draft the executive summary"}]',
        # Synthesis call: combines all results
        "Executive Summary: The AI developer tools market is valued at $4.2B with 35% YoY growth. "
        "Key competitors are LangChain, LlamaIndex, and AutoGen. "
        "selectools differentiates via its production-ready test suite and zero-dependency design.",
    )
    researcher = _make_agent(
        "Market size: $4.2B in 2025, projected $12B by 2028. 35% YoY growth driven by LLM adoption.",
    )
    analyst = _make_agent(
        "Top competitors: LangChain (most popular, complex), LlamaIndex (RAG-focused), "
        "AutoGen (Microsoft, multi-agent). Gap: none offer built-in eval frameworks.",
    )
    writer = _make_agent(
        "Draft: The AI tools market is booming. selectools targets teams that need reliability "
        "over flexibility — 2529 tests, 50 evaluators, and a self-hosted observability stack.",
    )

    agent = TeamLeadAgent(
        lead=lead,
        team={"researcher": researcher, "analyst": analyst, "writer": writer},
        delegation_strategy="sequential",
    )
    result = agent.run("Prepare a competitive analysis report for selectools")

    print(f"Content: {result.content[:200]}")
    print(f"Subtasks completed: {len(result.subtasks)}")
    for st in result.subtasks:
        print(f"  [{st.assignee}] {st.task[:60]} → {(st.result or '')[:60]}")


# ── Parallel strategy ─────────────────────────────────────────────────────────


def demo_parallel():
    print("\n--- Parallel strategy ---")

    lead = _make_agent(
        # Planning call
        '[{"assignee": "frontend", "task": "review the landing page copy"}, '
        '{"assignee": "backend", "task": "review the API documentation"}]',
        # Synthesis call
        "Both reviews are complete. Landing page copy is clear and benefit-focused. "
        "API docs need one correction: the /agents endpoint example is missing auth headers.",
    )
    frontend = _make_agent(
        "Landing page: strong headline, clear CTAs, good social proof. Minor: add a code snippet "
        "above the fold to appeal to developers.",
    )
    backend = _make_agent(
        "API docs are well-structured. Found one issue: the curl example for POST /agents "
        "is missing the Authorization: Bearer <token> header.",
    )

    agent = TeamLeadAgent(
        lead=lead,
        team={"frontend": frontend, "backend": backend},
        delegation_strategy="parallel",
    )
    result = agent.run("Review the selectools documentation for launch")

    print(f"Content: {result.content[:200]}")
    print(f"Subtasks: {len(result.subtasks)}")


# ── Dynamic strategy ──────────────────────────────────────────────────────────


def demo_dynamic():
    print("\n--- Dynamic strategy ---")

    lead = _make_agent(
        # Planning call
        '[{"assignee": "debugger", "task": "reproduce the reported import error on Python 3.9"}]',
        # Review call: declares complete after debugger finishes
        '{"complete": true, "reassignments": [], '
        '"synthesis": "Root cause identified: Python 3.9 does not support X|Y union syntax. '
        'Fix: replace all X|Y type hints with Optional[Union[X,Y]]. PR ready for review."}',
        # Synthesis call (reached if review JSON parse fails — safety net)
        "Fallback synthesis: issue resolved, PR ready.",
    )
    debugger = _make_agent(
        "Reproduced on Python 3.9.7. Stack trace points to `Union[str | None]` syntax in "
        "agent/config.py line 42. Python 3.9 requires `Optional[str]` instead.",
    )

    agent = TeamLeadAgent(
        lead=lead,
        team={"debugger": debugger},
        delegation_strategy="dynamic",
        max_reassignments=1,
    )
    result = agent.run("Investigate and fix the Python 3.9 compatibility bug")

    print(f"Content: {result.content[:200]}")
    print(f"Total assignment attempts: {result.total_assignments}")


# ── Async execution ───────────────────────────────────────────────────────────


async def demo_async():
    print("\n--- Async (sequential) ---")

    lead = _make_agent(
        '[{"assignee": "tester", "task": "write unit tests for the new feature"}]',
        "Tests written and passing. Coverage: 94%. Ready to merge.",
    )
    tester = _make_agent(
        "Written 12 unit tests covering happy path, edge cases, and error handling. All pass.",
    )

    agent = TeamLeadAgent(
        lead=lead,
        team={"tester": tester},
        delegation_strategy="sequential",
    )
    result = await agent.arun("Add tests for the new SemanticCache feature")
    print(f"[async] Content: {result.content[:150]}")
    print(f"[async] Subtasks: {len(result.subtasks)}")


def main():
    print("=" * 60)
    print("TeamLeadAgent — Example")
    print("=" * 60)

    demo_sequential()
    demo_parallel()
    demo_dynamic()

    print("\n[async] Running async demo...")
    asyncio.run(demo_async())

    print("\n✓ TeamLeadAgent example complete")


if __name__ == "__main__":
    main()
