"""
Example 38 — Terminal Tools and Stop Conditions

Demonstrates how to stop the agent loop after a specific tool fires,
without making another LLM call. Two mechanisms:

1. `@tool(terminal=True)` — static, declarative
2. `AgentConfig(stop_condition=...)` — dynamic, result-dependent

Use cases: human-in-the-loop, multi-turn forms, payment flows, escalation.
"""

import json

from selectools import Agent, AgentConfig, AgentResult, Message, Role, tool
from selectools.providers.stubs import LocalProvider

# ---------------------------------------------------------------------------
# 1. Static terminal tool — @tool(terminal=True)
# ---------------------------------------------------------------------------


@tool(terminal=True, description="Present a question card to the student")
def present_question(question_id: int) -> str:
    """Terminal tool: the agent loop stops after this fires."""
    return json.dumps(
        {
            "action": "present_question",
            "question_id": question_id,
            "text": f"What is the capital of France? (Q{question_id})",
        }
    )


@tool(description="Record the student's answer")
def record_answer(question_id: int, answer: str) -> str:
    return json.dumps(
        {
            "action": "answer_recorded",
            "question_id": question_id,
            "answer": answer,
            "correct": answer.lower() == "paris",
        }
    )


def demo_terminal_tool():
    """The agent calls present_question, and the loop stops immediately."""
    agent = Agent(
        tools=[present_question, record_answer],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=5),
    )

    result: AgentResult = agent.ask("Start the quiz")

    print("=== Terminal Tool Demo ===")
    print(f"Content: {result.content[:80]}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {[tc.tool_name for tc in result.tool_calls]}")
    print(f"terminal attribute on tool: {present_question.terminal}")
    print()


# ---------------------------------------------------------------------------
# 2. Dynamic stop condition — AgentConfig(stop_condition=...)
# ---------------------------------------------------------------------------


@tool(description="Fetch the next step in the workflow")
def workflow_step(step_name: str) -> str:
    steps = {
        "gather_info": json.dumps({"action": "continue", "next": "review"}),
        "review": json.dumps({"action": "continue", "next": "approve"}),
        "approve": json.dumps({"action": "needs_human_approval", "form_url": "/approve/123"}),
    }
    return steps.get(step_name, json.dumps({"action": "unknown"}))


def demo_stop_condition():
    """The agent continues until a tool returns needs_human_approval."""
    agent = Agent(
        tools=[workflow_step],
        provider=LocalProvider(),
        config=AgentConfig(
            max_iterations=10,
            stop_condition=lambda tool_name, result: "needs_human_approval" in result,
        ),
    )

    result = agent.ask("Start the approval workflow")

    print("=== Stop Condition Demo ===")
    print(f"Content: {result.content[:80]}")
    print(f"Iterations: {result.iterations}")
    print(f"stop_condition configured: {agent.config.stop_condition is not None}")
    print()


# ---------------------------------------------------------------------------
# 3. Combined: terminal tool + async observer (realistic pattern)
# ---------------------------------------------------------------------------


def demo_combined():
    """Shows how terminal tools work alongside observers."""
    from selectools.observer import AgentObserver

    events = []

    class TrackingObserver(AgentObserver):
        def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
            events.append(f"{tool_name} ({duration_ms:.0f}ms)")

        def on_run_end(self, run_id, result):
            events.append(f"run_end ({result.iterations} iterations)")

    agent = Agent(
        tools=[present_question, record_answer],
        provider=LocalProvider(),
        config=AgentConfig(
            max_iterations=5,
            observers=[TrackingObserver()],
        ),
    )

    result = agent.ask("Present question 42")

    print("=== Combined Demo (terminal + observer) ===")
    print(f"Content: {result.content[:80]}")
    print(f"Observer events: {events}")
    print()


if __name__ == "__main__":
    demo_terminal_tool()
    demo_stop_condition()
    demo_combined()
