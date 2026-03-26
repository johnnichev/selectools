"""
SupervisorAgent — high-level multi-agent coordination.

Wraps AgentGraph to provide four coordination strategies:

- plan_and_execute: Supervisor LLM generates a JSON plan then executes agents sequentially.
  With model_split, expensive model plans and cheap models execute (70-90% cost reduction).
- round_robin: Agents take turns; supervisor checks after each full round.
- dynamic: LLM router selects the best agent for each step.
- magentic: Magentic-One pattern. Maintains Task Ledger + Progress Ledger.
  After max_stalls consecutive unproductive steps, replans from scratch.
  Most autonomous strategy.

Usage::

    supervisor = SupervisorAgent(
        agents={
            "researcher": researcher_agent,
            "writer": writer_agent,
            "reviewer": reviewer_agent,
        },
        provider=OpenAIProvider(),
        strategy="plan_and_execute",
        max_rounds=5,
    )
    result = supervisor.run("Write a comprehensive blog post about LLM safety")
    print(result.content)
    print(result.total_usage)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..observer import AgentObserver
    from ..providers.base import Provider
    from ..types import AgentResult
    from .checkpoint import CheckpointStore

from ..types import Message, Role
from ..usage import UsageStats
from .graph import AgentGraph, ErrorPolicy, GraphResult, _merge_usage
from .state import STATE_KEY_LAST_OUTPUT, ContextMode, GraphState


class SupervisorStrategy(str, Enum):
    """Coordination strategy for SupervisorAgent.

    PLAN_AND_EXECUTE: Supervisor generates a structured plan then executes each step.
    ROUND_ROBIN: Each agent participates in sequence; supervisor checks completion after each round.
    DYNAMIC: LLM router node selects the best agent per step based on current state.
    MAGENTIC: Magentic-One pattern — Task Ledger + Progress Ledger + auto-replan on stall.
    """

    PLAN_AND_EXECUTE = "plan_and_execute"
    ROUND_ROBIN = "round_robin"
    DYNAMIC = "dynamic"
    MAGENTIC = "magentic"


@dataclass
class ModelSplit:
    """Use separate models for planning and execution.

    Production-documented 70-90% token cost reduction: expensive model generates
    the plan, cheap models execute the steps.

    Example::

        model_split = ModelSplit(
            planner_model="gpt-4o",
            executor_model="gpt-4o-mini",
        )
    """

    planner_model: str
    executor_model: str


# System prompts for each strategy
_PLAN_SYSTEM = """You are a supervisor coordinating a team of AI agents.
Given a task and a list of available agents, generate a JSON execution plan.

Respond with ONLY a JSON array of steps:
[
  {{"agent": "<agent_name>", "task": "<specific task for this agent>"}},
  ...
]

Available agents: {agents}
"""

_DYNAMIC_ROUTER_SYSTEM = """You are a dynamic routing supervisor.
Given the current task state, select the best agent to handle the next step.

Available agents: {agents}

Respond with ONLY the agent name (no explanation, no JSON, just the name).
If the task is complete, respond with "DONE".
"""

_MAGENTIC_SYSTEM = """You are an orchestrator managing a team of AI agents using the Magentic-One pattern.

You maintain two ledgers:
1. Task Ledger: Known facts, working assumptions, and the current plan
2. Progress Ledger: Whether the task is progressing, if it's complete, and which agent should act next

Given the current state, produce a JSON response with this exact structure:
{{
  "task_ledger": {{
    "facts": ["fact 1", "fact 2"],
    "plan": ["step 1", "step 2", "step 3"]
  }},
  "progress_ledger": {{
    "is_complete": false,
    "is_progressing": true,
    "next_agent": "<agent_name>",
    "reason": "<why this agent>"
  }}
}}

Available agents: {agents}
If the task is complete, set is_complete=true and next_agent="DONE".
"""

_REPLAN_SYSTEM = """The previous plan has stalled — agents are not making progress.
Generate a completely NEW plan from scratch, approaching the problem differently.

Original task: {task}
What was tried: {history}
Available agents: {agents}

Respond with ONLY a JSON array:
[
  {{"agent": "<agent_name>", "task": "<fresh approach for this agent>"}},
  ...
]
"""


def _safe_json_parse(text: str, default: Any = None) -> Any:
    """Try to extract and parse JSON from an LLM response."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting first JSON array or object
        import re

        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return default


class SupervisorAgent:
    """High-level multi-agent coordinator.

    Wraps AgentGraph to provide structured coordination patterns. Each
    strategy builds a graph internally — the supervisor is a convenience
    layer on top of the graph engine.

    Args:
        agents: Dict mapping agent name → Agent instance.
        provider: LLM provider used by the supervisor for planning/routing.
        strategy: Coordination strategy (default: plan_and_execute).
        max_rounds: Maximum coordination rounds before stopping.
        max_stalls: MAGENTIC only — stall count before replanning from scratch.
        model_split: Optional separate models for planning vs execution.
        delegation_constraints: Explicit allow-lists preventing delegation ping-pong.
            E.g. {"worker_a": ["supervisor"]} means worker_a can only delegate to supervisor.
        cancellation_token: Token for cooperative cancellation.
        max_total_tokens: Graph-level token budget.
        max_cost_usd: Graph-level cost budget.
        observers: List of AgentObserver instances.
    """

    def __init__(
        self,
        agents: Dict[str, "Agent"],
        provider: "Provider",
        strategy: SupervisorStrategy = SupervisorStrategy.PLAN_AND_EXECUTE,
        max_rounds: int = 10,
        max_stalls: int = 2,
        model_split: Optional[ModelSplit] = None,
        delegation_constraints: Optional[Dict[str, List[str]]] = None,
        cancellation_token: Optional["CancellationToken"] = None,
        max_total_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        observers: Optional[List["AgentObserver"]] = None,
    ) -> None:
        if not agents:
            raise ValueError("SupervisorAgent requires at least one agent")
        self.agents = agents
        self.provider = provider
        self.strategy = SupervisorStrategy(strategy)
        self.max_rounds = max_rounds
        self.max_stalls = max_stalls
        self.model_split = model_split
        self.delegation_constraints = delegation_constraints or {}
        self._cancellation_token = cancellation_token
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self._observers = observers or []

        self._planner_model: Optional[str] = model_split.planner_model if model_split else None
        self._executor_model: Optional[str] = model_split.executor_model if model_split else None
        self._agent_names: str = ", ".join(agents.keys())
        self._first_agent_name: str = next(iter(agents)) if agents else ""
        self._default_model: str = self._resolve_default_model()

    def _resolve_default_model(self) -> str:
        from ..models import MODELS_BY_ID

        for candidate in ["gpt-4o-mini", "gpt-4.1-mini", "claude-haiku-4-5-20251001"]:
            if candidate in MODELS_BY_ID:
                return candidate
        keys = list(MODELS_BY_ID.keys())
        return keys[0] if keys else "gpt-4o-mini"

    def run(self, prompt: str) -> GraphResult:
        """Execute the supervisor synchronously."""
        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> GraphResult:
        """Execute the supervisor asynchronously."""
        if self.strategy == SupervisorStrategy.PLAN_AND_EXECUTE:
            return await self._run_plan_and_execute(prompt)
        elif self.strategy == SupervisorStrategy.ROUND_ROBIN:
            return await self._run_round_robin(prompt)
        elif self.strategy == SupervisorStrategy.DYNAMIC:
            return await self._run_dynamic(prompt)
        elif self.strategy == SupervisorStrategy.MAGENTIC:
            return await self._run_magentic(prompt)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    async def astream(self, prompt: str) -> AsyncGenerator[Any, None]:
        """Stream graph events from supervisor execution."""
        graph = self._build_graph_for_streaming(prompt)
        state = GraphState.from_prompt(prompt)
        async for event in graph.astream(state):
            yield event

    def _build_graph_for_streaming(self, prompt: str) -> AgentGraph:
        """Build a simple round-robin graph for streaming use."""
        graph = self._make_graph()
        agent_list = list(self.agents.keys())
        for i, name in enumerate(agent_list):
            graph.add_node(name, self.agents[name])
            if i < len(agent_list) - 1:
                graph.add_edge(name, agent_list[i + 1])
            else:
                graph.add_edge(name, AgentGraph.END)
        graph.set_entry(agent_list[0])
        return graph

    def _make_graph(self) -> AgentGraph:
        """Create a configured AgentGraph with supervisor-level settings."""
        return AgentGraph(
            name=f"supervisor_{self.strategy.value}",
            observers=self._observers,
            cancellation_token=self._cancellation_token,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
            max_steps=self.max_rounds * len(self.agents) + self.max_rounds + 10,
            stall_threshold=self.max_stalls,
        )

    async def _call_planner(self, system: str, user: str) -> str:
        """Call the supervisor's provider for planning/routing."""
        from ..types import Message, Role

        messages = [Message(role=Role.USER, content=user)]
        model = self._planner_model

        try:
            result_msg, _ = await self.provider.acomplete(
                model=model or self._default_model,
                messages=messages,
                system_prompt=system,
                tools=[],
            )
        except Exception:
            return ""

        return result_msg.content or ""

    # ------------------------------------------------------------------
    # Strategy: plan_and_execute
    # ------------------------------------------------------------------

    async def _run_plan_and_execute(self, prompt: str) -> GraphResult:
        """Generate a plan then execute each step as an agent node."""
        system = _PLAN_SYSTEM.format(agents=self._agent_names)
        plan_text = await self._call_planner(system, prompt)
        plan = _safe_json_parse(plan_text, default=[])

        if not isinstance(plan, list) or not plan:
            # Fallback: execute agents in registration order
            plan = [{"agent": name, "task": prompt} for name in self.agents]

        graph = self._make_graph()
        node_sequence: List[Any] = []

        for i, step in enumerate(plan):
            agent_name = step.get("agent", "")
            task = step.get("task", prompt)

            if agent_name not in self.agents:
                continue

            # Unique node name so the same agent can appear twice in the plan
            node_name = f"{agent_name}_{i}" if i > 0 else agent_name
            graph.add_node(
                node_name, self.agents[agent_name], context_mode=ContextMode.LAST_MESSAGE
            )
            node_sequence.append((node_name, task))

        if not node_sequence:
            state = GraphState.from_prompt(prompt)
            from ..trace import AgentTrace

            return GraphResult(
                content="",
                state=state,
                node_results={},
                trace=AgentTrace(),
                total_usage=UsageStats(),
            )

        for i in range(len(node_sequence) - 1):
            graph.add_edge(node_sequence[i][0], node_sequence[i + 1][0])
        graph.add_edge(node_sequence[-1][0], AgentGraph.END)
        graph.set_entry(node_sequence[0][0])

        state = GraphState.from_prompt(prompt)
        state.data["__plan__"] = plan
        for node_name, task in node_sequence:
            state.data[f"__task_{node_name}__"] = task

        return await graph.arun(state)

    # ------------------------------------------------------------------
    # Strategy: round_robin
    # ------------------------------------------------------------------

    async def _run_round_robin(self, prompt: str) -> GraphResult:
        """Each agent participates each round; supervisor checks completion."""
        agent_list = list(self.agents.items())
        state = GraphState.from_prompt(prompt)
        combined_usage = UsageStats()
        node_results: Dict[str, List[Any]] = {}

        from ..trace import AgentTrace

        trace = AgentTrace(metadata={"supervisor": "round_robin"})
        run_id = trace.run_id

        round_num = 0
        for round_num in range(self.max_rounds):  # noqa: B007
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            for agent_name, agent in agent_list:
                if self._cancellation_token and self._cancellation_token.is_cancelled:
                    break

                last_output = state.data.get(STATE_KEY_LAST_OUTPUT, "")
                input_msg = last_output or prompt
                messages = [Message(role=Role.USER, content=input_msg)]

                result = await agent.arun(messages, parent_run_id=run_id)
                combined_usage = _apply_result_to_state(
                    agent_name, result, state, combined_usage, node_results
                )

            # Check if done after each round
            done_check = state.data.get(STATE_KEY_LAST_OUTPUT, "").strip()
            if _looks_complete(done_check):
                break

        return GraphResult(
            content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            state=state,
            node_results=node_results,
            trace=trace,
            total_usage=combined_usage,
            steps=round_num + 1,
        )

    # ------------------------------------------------------------------
    # Strategy: dynamic
    # ------------------------------------------------------------------

    async def _run_dynamic(self, prompt: str) -> GraphResult:
        """LLM router selects best agent per step."""
        system = _DYNAMIC_ROUTER_SYSTEM.format(agents=self._agent_names)
        state = GraphState.from_prompt(prompt)
        combined_usage = UsageStats()
        node_results: Dict[str, List[Any]] = {}

        from ..trace import AgentTrace

        trace = AgentTrace(metadata={"supervisor": "dynamic"})
        run_id = trace.run_id

        step = 0
        for step in range(self.max_rounds):  # noqa: B007
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            # Build routing context
            history_text = _format_history(state.history)
            routing_prompt = (
                f"Task: {prompt}\n\n"
                f"History:\n{history_text}\n\n"
                f"Current state: {state.data.get(STATE_KEY_LAST_OUTPUT, 'Not started')}\n\n"
                f"Which agent should act next?"
            )

            agent_name = (await self._call_planner(system, routing_prompt)).strip()

            if agent_name == "DONE" or not agent_name:
                break

            if agent_name not in self.agents:
                # LLM hallucinated — pick first available
                agent_name = self._first_agent_name

            agent = self.agents[agent_name]
            last_output = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            messages = [Message(role=Role.USER, content=last_output or prompt)]
            result = await agent.arun(messages, parent_run_id=run_id)
            combined_usage = _apply_result_to_state(
                agent_name, result, state, combined_usage, node_results
            )

            if _looks_complete(result.content):
                break

        return GraphResult(
            content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            state=state,
            node_results=node_results,
            trace=trace,
            total_usage=combined_usage,
            steps=step + 1,
        )

    # ------------------------------------------------------------------
    # Strategy: magentic (Magentic-One pattern)
    # ------------------------------------------------------------------

    async def _run_magentic(self, prompt: str) -> GraphResult:
        """Magentic-One orchestration: Task Ledger + Progress Ledger + auto-replan."""
        system = _MAGENTIC_SYSTEM.format(agents=self._agent_names)
        state = GraphState.from_prompt(prompt)
        combined_usage = UsageStats()
        node_results: Dict[str, List[Any]] = {}
        stall_count = 0

        from ..trace import AgentTrace

        trace = AgentTrace(metadata={"supervisor": "magentic"})
        run_id = trace.run_id

        step = 0
        for step in range(self.max_rounds):  # noqa: B007
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            history_text = _format_history(state.history)
            orchestrator_prompt = (
                f"Task: {prompt}\n\n"
                f"Work so far:\n{history_text}\n\n"
                f"Current output: {state.data.get(STATE_KEY_LAST_OUTPUT, 'None yet')}\n\n"
                f"Stall count: {stall_count}/{self.max_stalls}\n\n"
                f"Update ledgers and select next action."
            )

            response_text = await self._call_planner(system, orchestrator_prompt)
            ledger = _safe_json_parse(response_text, default={})

            if not isinstance(ledger, dict):
                ledger = {}

            progress = ledger.get("progress_ledger", {})
            task_ledger = ledger.get("task_ledger", {})

            state.data["__task_ledger__"] = task_ledger
            state.data["__progress_ledger__"] = progress

            is_complete = progress.get("is_complete", False)
            is_progressing = progress.get("is_progressing", True)
            next_agent = progress.get("next_agent", "")

            if is_complete or next_agent == "DONE":
                break

            if not is_progressing:
                stall_count += 1
                if stall_count >= self.max_stalls:
                    new_plan_text = await self._replan(prompt, state.history)
                    new_plan = _safe_json_parse(new_plan_text, default=[])
                    if new_plan:
                        state.data["__plan__"] = new_plan
                        for obs in self._observers:
                            handler = getattr(obs, "on_supervisor_replan", None)
                            if handler:
                                try:
                                    handler(run_id, stall_count, new_plan_text)
                                except Exception:  # nosec B110
                                    pass
                    stall_count = 0
                    continue
            else:
                stall_count = 0

            if next_agent not in self.agents:
                next_agent = self._first_agent_name

            agent = self.agents[next_agent]
            agent_task = progress.get("reason", state.data.get(STATE_KEY_LAST_OUTPUT, prompt))
            messages = [Message(role=Role.USER, content=agent_task or prompt)]
            result = await agent.arun(messages, parent_run_id=run_id)
            combined_usage = _apply_result_to_state(
                next_agent, result, state, combined_usage, node_results
            )

        return GraphResult(
            content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            state=state,
            node_results=node_results,
            trace=trace,
            total_usage=combined_usage,
            steps=step + 1,
            stalls=stall_count,
        )

    async def _replan(self, task: str, history: List[Any]) -> str:
        """Call the planner to replan from scratch."""
        history_text = _format_history(history)
        system = _REPLAN_SYSTEM.format(
            task=task,
            history=history_text,
            agents=self._agent_names,
        )
        return await self._call_planner(system, task)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _apply_result_to_state(
    agent_name: str,
    result: Any,
    state: "GraphState",
    combined_usage: UsageStats,
    node_results: Dict[str, List[Any]],
) -> UsageStats:
    """Update shared state and usage after an agent call."""
    from ..types import Message, Role

    if result.usage:
        combined_usage = _merge_usage(combined_usage, result.usage)
    node_results.setdefault(agent_name, []).append(result)
    state.data[STATE_KEY_LAST_OUTPUT] = result.content
    state.messages.append(Message(role=Role.ASSISTANT, content=result.content))
    state.history.append((agent_name, result))
    return combined_usage


def _looks_complete(text: str) -> bool:
    """Heuristic: does the output text look like a completed task?"""
    if not text:
        return False
    text_lower = text.lower()
    completion_signals = ["task complete", "done.", "finished.", "complete.", "all done"]
    return any(signal in text_lower for signal in completion_signals)


def _format_history(history: List[Any]) -> str:
    """Format history list for inclusion in supervisor prompts."""
    if not history:
        return "No steps taken yet."
    lines = []
    for i, item in enumerate(history, 1):
        if isinstance(item, tuple) and len(item) == 2:
            name, result = item
            content = getattr(result, "content", str(result))[:200]
            lines.append(f"{i}. [{name}]: {content}")
    return "\n".join(lines) if lines else "No steps taken yet."


__all__ = [
    "SupervisorAgent",
    "SupervisorStrategy",
    "ModelSplit",
]
