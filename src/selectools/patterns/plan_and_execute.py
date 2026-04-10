"""PlanAndExecuteAgent — planner generates a typed plan, executors handle each step.

The planner Agent is called once to produce a JSON plan (list of steps).
Each step names an executor and describes its task. The executor agents run
in sequence; results are aggregated into a final output.

Replanning is supported: if a step fails and ``replanner=True``, the planner
is re-called with the failure context to revise the remaining steps.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..observer import AgentObserver

from .._async_utils import run_sync
from ..orchestration.graph import GraphResult
from ..orchestration.state import GraphState
from ..orchestration.supervisor import _safe_json_parse
from ..stability import beta
from ..trace import AgentTrace
from ..types import Message, Role
from ..usage import UsageStats


@dataclass
class PlanStep:
    """A single step in a PlanAndExecuteAgent execution plan."""

    executor_name: str
    task: str
    status: str = "pending"  # "pending" | "done" | "failed"
    result: Optional[str] = None


_PLANNER_SYSTEM = """You are a planning agent. Given a task, create a JSON execution plan.

Respond with ONLY a JSON array of steps:
[
  {{"executor": "<executor_name>", "task": "<specific task description>"}},
  ...
]

Available executors: {executors}
"""

_REPLANNER_SYSTEM = """A step in the execution plan has failed. Revise the remaining plan.

Original task: {task}
Completed steps: {completed}
Failed step: {failed_step} — Error: {error}
Available executors: {executors}

Respond with ONLY a JSON array of revised remaining steps:
[
  {{"executor": "<executor_name>", "task": "<revised task>"}},
  ...
]
"""


@beta
class PlanAndExecuteAgent:
    """Planner generates a structured plan; executor agents handle each step.

    The planner Agent is called to produce a JSON execution plan. Each step
    specifies an executor name and a task description. Steps run sequentially,
    with each executor called directly via arun(). The final executor's output
    becomes the result content.

    Args:
        planner: Agent responsible for generating the execution plan.
        executors: Dict mapping executor name → Agent.
        replanner: If True, re-calls the planner on step failure.
        max_replan_attempts: Maximum number of replanning cycles.
        observers: Optional AgentObserver instances.
        cancellation_token: Optional cancellation token.
        max_cost_usd: Optional cost budget (informational only).
    """

    def __init__(
        self,
        planner: "Agent",
        executors: Dict[str, "Agent"],
        *,
        replanner: bool = False,
        max_replan_attempts: int = 2,
        observers: Optional[List["AgentObserver"]] = None,
        cancellation_token: Optional["CancellationToken"] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        if not executors:
            raise ValueError("PlanAndExecuteAgent requires at least one executor")
        self.planner = planner
        self.executors = executors
        self.replanner = replanner
        self.max_replan_attempts = max_replan_attempts
        self._observers = observers or []
        self._cancellation_token = cancellation_token
        self._max_cost_usd = max_cost_usd
        self._executor_names = ", ".join(executors.keys())

    def run(self, prompt: str) -> GraphResult:
        """Execute synchronously."""
        return run_sync(self.arun(prompt))

    async def arun(self, prompt: str) -> GraphResult:
        """Execute asynchronously: plan → execute → aggregate."""
        plan = await self._call_planner(prompt)
        state = GraphState.from_prompt(prompt)
        state.data["__plan__"] = [{"executor": s.executor_name, "task": s.task} for s in plan]

        node_results: Dict[str, List[Any]] = {}
        last_content = ""
        step_index = 0
        replan_attempts = 0
        completed_names: List[str] = []

        remaining = list(plan)
        while remaining:
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            step = remaining.pop(0)
            executor = self.executors.get(step.executor_name)
            if executor is None:
                # Unknown executor — skip
                continue

            msg = step.task
            if last_content:
                msg = f"Context from prior steps:\n{last_content}\n\nYour task: {step.task}"

            try:
                result = await executor.arun([Message(role=Role.USER, content=msg)])
                step.result = result.content or ""
                step.status = "done"
                last_content += f"\n[{step.executor_name}]: {step.result}"

                node_name = f"step_{step_index}_{step.executor_name}"
                node_results[node_name] = [result]
                completed_names.append(step.executor_name)
                step_index += 1

            except Exception as exc:
                step.status = "failed"
                if self.replanner and replan_attempts < self.max_replan_attempts:
                    replan_attempts += 1
                    revised = await self._replan(
                        prompt, completed_names, step.executor_name, str(exc)
                    )
                    remaining = revised + remaining
                # Continue to next step regardless

        return GraphResult(
            content=last_content.strip(),
            state=state,
            node_results=node_results,
            trace=AgentTrace(),
            total_usage=UsageStats(),
        )

    async def _call_planner(self, prompt: str) -> List[PlanStep]:
        """Call the planner agent and parse the resulting JSON plan."""
        system = _PLANNER_SYSTEM.format(executors=self._executor_names)
        planning_prompt = f"{system}\n\nTask to plan:\n{prompt}"
        result = await self.planner.arun([Message(role=Role.USER, content=planning_prompt)])
        raw = _safe_json_parse(result.content or "", default=[])

        if not isinstance(raw, list) or not raw:
            # Fallback: one step per executor
            return [PlanStep(executor_name=name, task=prompt) for name in self.executors]

        steps: List[PlanStep] = []
        for item in raw:
            if isinstance(item, dict):
                executor = item.get("executor", "")
                task = item.get("task", prompt)
                if executor in self.executors:
                    steps.append(PlanStep(executor_name=executor, task=task))

        if not steps:
            return [PlanStep(executor_name=name, task=prompt) for name in self.executors]

        return steps

    async def _replan(
        self,
        original_prompt: str,
        completed: List[str],
        failed_step: str,
        error: str,
    ) -> List[PlanStep]:
        """Replan remaining steps after a failure."""
        system = _REPLANNER_SYSTEM.format(
            task=original_prompt,
            completed=", ".join(completed) if completed else "none",
            failed_step=failed_step,
            error=error,
            executors=self._executor_names,
        )
        result = await self.planner.arun([Message(role=Role.USER, content=system)])
        raw = _safe_json_parse(result.content or "", default=[])

        if not isinstance(raw, list):
            return []

        steps: List[PlanStep] = []
        for item in raw:
            if isinstance(item, dict):
                executor = item.get("executor", "")
                task = item.get("task", original_prompt)
                if executor in self.executors:
                    steps.append(PlanStep(executor_name=executor, task=task))
        return steps


__all__ = ["PlanAndExecuteAgent", "PlanStep"]
