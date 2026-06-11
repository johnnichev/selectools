"""
Example 109: Planning-as-Config

Add planning to ANY Agent with a config flag — no PlanAndExecuteAgent
wiring required. When `PlanningConfig.enabled=True` and the input looks
multi-step, the agent internally runs: plan -> (approve) -> execute steps
-> synthesize, reusing the PlanAndExecuteAgent pattern under the hood.

Simple single-step inputs skip planning automatically (cheap local
heuristic), so you only pay the planning overhead when it helps.

Run: python examples/109_planning_as_config.py  (offline, no API keys)
"""

from typing import List

from selectools import Agent, AgentConfig, PlanningConfig, tool
from selectools.patterns import PlanStep
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.usage import UsageStats

# ── Mock setup (no API keys needed) ──────────────────────────────────────────
# In production replace _ScriptedProvider with OpenAIProvider / AnthropicProvider


@tool(description="Placeholder tool — not called in this example")
def _noop(x: str) -> str:
    return x


class _ScriptedProvider(LocalProvider):
    """Returns scripted responses in order."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)

    def complete(self, *, model, system_prompt, messages, **kwargs):  # type: ignore[override]
        text = self._responses.pop(0) if self._responses else "done"
        return Message(role=Role.ASSISTANT, content=text), UsageStats()


_PLAN = (
    '[{"executor": "executor", "task": "research vector databases"},'
    ' {"executor": "executor", "task": "write the blog post"}]'
)
_COMPLEX_TASK = "Research vector databases, then write a short blog post about them."


def main() -> None:
    print("=" * 60)
    print("Planning-as-Config — Example")
    print("=" * 60)

    # ── 1. Auto-planning on a complex input ───────────────────────────────
    provider = _ScriptedProvider(
        [
            _PLAN,  # planner call -> JSON plan
            "Research notes: Pinecone, Qdrant, Weaviate, FAISS.",  # step 1
            "Draft: Vector databases power RAG retrieval...",  # step 2
            "Final post: Vector databases, explained simply...",  # synthesis
        ]
    )
    agent = Agent(
        [_noop],
        provider=provider,
        config=AgentConfig(planning=PlanningConfig(enabled=True)),
    )
    result = agent.run(_COMPLEX_TASK)
    print(f"\n[planned] Final: {result.content[:60]}...")
    print(f"[planned] Reasoning (the plan):\n{result.reasoning}")
    print(f"[planned] Trace metadata: {result.trace.metadata['planning']['steps_executed']} steps")

    # ── 2. Simple inputs skip planning (complexity gate) ──────────────────
    simple_provider = _ScriptedProvider(["4"])
    simple_agent = Agent(
        [_noop],
        provider=simple_provider,
        config=AgentConfig(planning=PlanningConfig(enabled=True)),
    )
    answer = simple_agent.run("What is 2+2?")
    print(f"\n[simple] '{answer.content}' — planning skipped, one provider call")

    # ── 3. Human-in-the-loop plan approval ─────────────────────────────────
    def review_plan(plan: List[PlanStep]):
        print("\n[approval] Proposed plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.task}")
        # Return True to approve, False to reject (falls back to a normal
        # run), or an edited List[PlanStep] to execute instead.
        return [PlanStep(executor_name="executor", task="research AND write in one pass")]

    hitl_provider = _ScriptedProvider(
        [
            _PLAN,  # planner
            "Combined research + draft in one pass.",  # edited single step
            "Final synthesized answer.",  # synthesis
        ]
    )
    hitl_agent = Agent(
        [_noop],
        provider=hitl_provider,
        config=AgentConfig(
            planning=PlanningConfig(
                enabled=True,
                auto_approve=False,
                plan_approval_handler=review_plan,
            )
        ),
    )
    hitl_result = hitl_agent.run(_COMPLEX_TASK)
    print(f"[approval] Result after edited plan: {hitl_result.content}")

    print("\n✓ Planning-as-config example complete")


if __name__ == "__main__":
    main()
