"""Pre-built agent patterns for selectools.

Higher-level agent architectures built on the v0.18.0 orchestration primitives.
Each pattern wires up an AgentGraph topology (or a direct async loop) for you,
so you get a battle-tested multi-agent workflow in one class.

Available patterns:

- ``PlanAndExecuteAgent``: planner generates a typed plan → executors handle each step.
- ``ReflectiveAgent``: actor produces a draft → critic evaluates → actor revises.
- ``DebateAgent``: multiple agents argue positions → judge synthesizes conclusion.
- ``TeamLeadAgent``: lead delegates subtasks to a team, reviews results, coordinates.

Usage::

    from selectools.patterns import PlanAndExecuteAgent, ReflectiveAgent

    agent = ReflectiveAgent(actor=writer, critic=reviewer, max_reflections=3)
    result = agent.run("Write a press release")
    print(result.final_draft)
"""

from .debate import DebateAgent, DebateResult, DebateRound
from .plan_and_execute import PlanAndExecuteAgent, PlanStep
from .reflective import ReflectionRound, ReflectiveAgent, ReflectiveResult
from .team_lead import Subtask, TeamLeadAgent, TeamLeadResult

__all__ = [
    # PlanAndExecute
    "PlanAndExecuteAgent",
    "PlanStep",
    # Reflective
    "ReflectiveAgent",
    "ReflectionRound",
    "ReflectiveResult",
    # Debate
    "DebateAgent",
    "DebateRound",
    "DebateResult",
    # TeamLead
    "TeamLeadAgent",
    "Subtask",
    "TeamLeadResult",
]
