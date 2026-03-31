"""TeamLeadAgent — lead delegates to a team, reviews results, coordinates handoffs.

Three delegation strategies:
- ``sequential``: Lead assigns subtasks one-by-one in order.
- ``parallel``:   Lead assigns all subtasks simultaneously (AgentGraph fan-out).
- ``dynamic``:    Lead uses LLM to decide the next assignment after each result.

The lead Agent is called to generate the initial subtask list as JSON.
In ``dynamic`` mode, the lead is called again after each team result to review
progress and optionally reassign work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..observer import AgentObserver

from ..orchestration.graph import AgentGraph
from ..orchestration.state import ContextMode, GraphState
from ..orchestration.supervisor import _safe_json_parse
from ..stability import beta
from ..types import Message, Role


@dataclass
class Subtask:
    """A unit of work assigned to a team member."""

    assignee: str
    task: str
    result: Optional[str] = None
    status: str = "pending"  # "pending" | "done" | "reassigned"
    attempt: int = 0


@dataclass
class TeamLeadResult:
    """Result of a TeamLeadAgent run."""

    content: str
    subtasks: List[Subtask]

    @property
    def total_assignments(self) -> int:
        return sum(s.attempt for s in self.subtasks)


_DELEGATION_SYSTEM = """You are a team lead. Break the task into subtasks and assign them to team members.

Respond with ONLY a JSON array:
[
  {{"assignee": "<member_name>", "task": "<specific task description>"}},
  ...
]

Team members: {members}
"""

_REVIEW_SYSTEM = """You are a team lead reviewing work in progress.

Original task: {task}
Work completed so far:
{work_log}

Based on this progress, decide what to do next.
Respond with ONLY this JSON:
{{
  "complete": <true|false>,
  "reassignments": [{{"assignee": "<name>", "task": "<task>"}}],
  "synthesis": "<final answer if complete, else empty string>"
}}

Team members: {members}
"""


@beta
class TeamLeadAgent:
    """Team lead delegates subtasks to agents and coordinates results.

    Args:
        lead: Agent that generates subtask plans and reviews progress.
        team: Dict mapping team member name → Agent.
        delegation_strategy: "sequential", "parallel", or "dynamic" (default).
        max_reassignments: Maximum reassignment cycles in dynamic mode (default: 2).
        observers: Optional AgentObserver instances.
        cancellation_token: Optional cancellation token.
        max_cost_usd: Optional cost budget.
    """

    def __init__(
        self,
        lead: "Agent",
        team: Dict[str, "Agent"],
        *,
        delegation_strategy: str = "dynamic",
        max_reassignments: int = 2,
        observers: Optional[List["AgentObserver"]] = None,
        cancellation_token: Optional["CancellationToken"] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        if delegation_strategy not in ("sequential", "parallel", "dynamic"):
            raise ValueError(
                f"delegation_strategy must be 'sequential', 'parallel', or 'dynamic', "
                f"got {delegation_strategy!r}"
            )
        if not team:
            raise ValueError("TeamLeadAgent requires at least one team member")
        self.lead = lead
        self.team = team
        self.delegation_strategy = delegation_strategy
        self.max_reassignments = max_reassignments
        self._observers = observers or []
        self._cancellation_token = cancellation_token
        self._max_cost_usd = max_cost_usd
        self._member_names = ", ".join(team.keys())

    def run(self, prompt: str) -> TeamLeadResult:
        """Execute synchronously."""
        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> TeamLeadResult:
        """Execute asynchronously using the configured delegation strategy."""
        if self.delegation_strategy == "sequential":
            return await self._run_sequential(prompt)
        elif self.delegation_strategy == "parallel":
            return await self._run_parallel(prompt)
        else:
            return await self._run_dynamic(prompt)

    async def _get_initial_subtasks(self, prompt: str) -> List[Subtask]:
        """Ask the lead to generate the initial subtask list."""
        system = _DELEGATION_SYSTEM.format(members=self._member_names)
        planning_prompt = f"{system}\n\nTask to delegate:\n{prompt}"
        result = await self.lead.arun([Message(role=Role.USER, content=planning_prompt)])
        raw = _safe_json_parse(result.content or "", default=[])

        if not isinstance(raw, list) or not raw:
            # Fallback: one task per team member
            return [Subtask(assignee=name, task=prompt) for name in self.team]

        subtasks: List[Subtask] = []
        for item in raw:
            if isinstance(item, dict):
                assignee = item.get("assignee", "")
                task = item.get("task", prompt)
                if assignee in self.team:
                    subtasks.append(Subtask(assignee=assignee, task=task))

        if not subtasks:
            return [Subtask(assignee=name, task=prompt) for name in self.team]
        return subtasks

    async def _run_sequential(self, prompt: str) -> TeamLeadResult:
        """Execute subtasks one-by-one in order."""
        subtasks = await self._get_initial_subtasks(prompt)
        work_log = ""

        for subtask in subtasks:
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break
            agent = self.team.get(subtask.assignee)
            if agent is None:
                continue
            subtask.attempt += 1
            msg = subtask.task
            if work_log:
                msg = f"Context from prior work:\n{work_log}\n\nYour task: {subtask.task}"
            result = await agent.arun([Message(role=Role.USER, content=msg)])
            subtask.result = result.content or ""
            subtask.status = "done"
            work_log += f"\n[{subtask.assignee}]: {subtask.result[:300]}"

        # Lead synthesizes
        synthesis_prompt = (
            f"Task: {prompt}\n\nTeam results:\n{work_log}\n\n"
            f"Please synthesize these results into a final answer."
        )
        synthesis = await self.lead.arun([Message(role=Role.USER, content=synthesis_prompt)])
        return TeamLeadResult(content=synthesis.content or work_log, subtasks=subtasks)

    async def _run_parallel(self, prompt: str) -> TeamLeadResult:
        """Execute all subtasks simultaneously using AgentGraph fan-out."""
        subtasks = await self._get_initial_subtasks(prompt)

        # Build parallel graph with all assigned team members
        graph = AgentGraph(
            name="team_lead_parallel",
            observers=self._observers,
            cancellation_token=self._cancellation_token,
            max_cost_usd=self._max_cost_usd,
        )

        node_names: List[str] = []
        for i, subtask in enumerate(subtasks):
            agent = self.team.get(subtask.assignee)
            if agent is None:
                continue
            node_name = f"member_{i}_{subtask.assignee}"
            graph.add_node(node_name, agent, context_mode=ContextMode.LAST_MESSAGE)
            node_names.append(node_name)

        if not node_names:
            return TeamLeadResult(content="", subtasks=subtasks)

        # Fan-out to parallel group, converge to END
        graph.add_parallel_nodes("team_work", node_names)
        graph.set_entry("team_work")
        graph.add_edge("team_work", AgentGraph.END)

        state = GraphState.from_prompt(prompt)
        result = await graph.arun(state)

        # Collect results from node_results
        work_log = ""
        for i, subtask in enumerate(subtasks):
            node_name = f"member_{i}_{subtask.assignee}"
            node_result_list = result.node_results.get(node_name, [])
            if node_result_list:
                subtask.result = getattr(node_result_list[-1], "content", "") or ""
                subtask.status = "done"
                subtask.attempt = 1
                work_log += f"\n[{subtask.assignee}]: {subtask.result[:300]}"

        # Lead synthesizes
        synthesis_prompt = (
            f"Task: {prompt}\n\nTeam results:\n{work_log}\n\n"
            f"Please synthesize these results into a final answer."
        )
        synthesis = await self.lead.arun([Message(role=Role.USER, content=synthesis_prompt)])
        return TeamLeadResult(content=synthesis.content or work_log, subtasks=subtasks)

    async def _run_dynamic(self, prompt: str) -> TeamLeadResult:
        """Lead reviews progress after each result and decides next assignment."""
        subtasks = await self._get_initial_subtasks(prompt)
        work_log = ""
        all_subtasks: List[Subtask] = list(subtasks)
        pending = list(subtasks)

        for _ in range(self.max_reassignments + len(subtasks)):
            if not pending:
                break
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            subtask = pending.pop(0)
            agent = self.team.get(subtask.assignee)
            if agent is None:
                continue

            subtask.attempt += 1
            context_msg = subtask.task
            if work_log:
                context_msg = f"Context:\n{work_log}\n\nYour task: {subtask.task}"

            result = await agent.arun([Message(role=Role.USER, content=context_msg)])
            subtask.result = result.content or ""
            subtask.status = "done"
            work_log += f"\n[{subtask.assignee}]: {subtask.result[:300]}"

            if not pending:
                # Ask lead to review and decide if done
                review_prompt = _REVIEW_SYSTEM.format(
                    task=prompt,
                    work_log=work_log,
                    members=self._member_names,
                )
                review_result = await self.lead.arun(
                    [Message(role=Role.USER, content=review_prompt)]
                )
                decision = _safe_json_parse(review_result.content or "", default={})

                if not isinstance(decision, dict):
                    decision = {}

                if decision.get("complete", True):
                    synthesis = decision.get("synthesis", "") or work_log
                    return TeamLeadResult(content=synthesis, subtasks=all_subtasks)

                # Add reassigned tasks back to the queue
                for reassignment in decision.get("reassignments", []):
                    if isinstance(reassignment, dict):
                        assignee = reassignment.get("assignee", "")
                        task = reassignment.get("task", prompt)
                        if assignee in self.team:
                            new_subtask = Subtask(assignee=assignee, task=task)
                            pending.append(new_subtask)
                            all_subtasks.append(new_subtask)

        # Final synthesis
        synthesis_prompt = (
            f"Task: {prompt}\n\nTeam results:\n{work_log}\n\n"
            f"Please synthesize these results into a final answer."
        )
        synthesis = await self.lead.arun([Message(role=Role.USER, content=synthesis_prompt)])
        return TeamLeadResult(content=synthesis.content or work_log, subtasks=all_subtasks)


__all__ = ["TeamLeadAgent", "Subtask", "TeamLeadResult"]
