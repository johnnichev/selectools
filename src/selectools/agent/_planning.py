"""Planning-as-config adapter (ROADMAP P2).

Lets any ``Agent`` opt into a plan -> (approve) -> execute -> synthesize
flow via ``AgentConfig(planning=PlanningConfig(enabled=True))`` WITHOUT
reimplementing planning: the adapter wraps the existing
:class:`~selectools.patterns.plan_and_execute.PlanAndExecuteAgent` pattern.

Design notes
------------
- The ``agent/core.py`` touch is intentionally tiny: ``run()``/``arun()``
  call :func:`run_with_planning`/:func:`arun_with_planning` at the top.
  A ``None`` return means "planning did not engage" (gate not met, empty
  prompt, or plan rejected) and the normal agent loop proceeds unchanged.
  ``PlanningConfig`` defaulting to ``None`` on ``AgentConfig`` keeps the
  disabled path byte-identical and zero-overhead.
- The agent is cloned twice via ``Agent.clone_for_isolation()``: once as
  the planner (with optional ``PlanningConfig.provider``/``model``
  overrides) and once as the single executor named ``"executor"``. Both
  clones get ``config.planning = None`` so planned sub-runs can never
  recurse into planning.
- Streaming runs skip planning with a one-time ``UserWarning`` per agent
  instance — the pattern runs whole sub-agent calls and cannot stream
  tokens. Structured output IS supported: ``response_format`` is applied
  to the final synthesis call only.
- Aggregation: planner + executor usage is merged into ``agent.usage``
  on EVERY exit path (success, plan rejection, or a mid-flow exception)
  via a ``try/finally`` so no sub-run tokens are ever lost, and is
  reflected in ``AgentResult.usage``; tool calls from every step are
  collected into ``AgentResult.tool_calls``. Known gap: the returned
  ``AgentResult.trace`` is the synthesis run's trace (with the plan
  attached under ``trace.metadata["planning"]``); per-step traces live on
  the per-step ``AgentResult`` objects inside the pattern and are not
  merged, because ``PlanAndExecuteAgent`` returns an empty
  ``GraphResult.trace``.
- Budget continuity: ``clone_for_isolation()`` gives each clone a fresh
  ``AgentUsage``, which would silently reset ``max_total_tokens`` /
  ``max_cost_usd`` for the planned sub-runs. Each clone's usage is
  therefore seeded with the parent's current scalar totals so the
  lifetime caps keep binding across the whole planned flow; the
  ``finally`` merge folds back only the DELTA (clone usage minus the
  seeded baseline) so the baseline is never double-counted.
"""

from __future__ import annotations

import copy
import inspect
import re
import warnings
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from ..tool_parser import ResponseFormat
    from .core import Agent

from .._async_utils import run_sync
from ..patterns.plan_and_execute import PlanAndExecuteAgent, PlanStep
from ..token_estimation import estimate_tokens
from ..types import AgentResult, Message, Role, ToolCall
from ..usage import AgentUsage

_EXECUTOR_NAME = "executor"

_SYNTHESIS_PROMPT = (
    "You created and executed a multi-step plan for the task below.\n\n"
    "Original task:\n{task}\n\n"
    "Step results:\n{results}\n\n"
    "Using the step results above, write the final answer to the original task."
)

# ---------------------------------------------------------------------------
# Complexity gate — cheap local heuristic (deliberately NOT an LLM call and
# NOT RouterProvider's classifier).
# ---------------------------------------------------------------------------

_SEQUENCE_MARKERS = (
    " then ",
    " after that",
    " and then",
    " next,",
    " followed by",
    " finally",
    " afterwards",
    " once that",
    "step by step",
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S", re.MULTILINE)
_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")


def _complexity_score(text: str) -> int:
    """Score how multi-step a prompt looks. Deterministic and offline.

    Score = 1 (base) plus one point for each signal present:

    - a sequence connective ("then", "after that", "finally", ...)
    - an enumerated or bulleted list
    - three or more sentences
    - estimated length > 120 tokens (``selectools.token_estimation``,
      which falls back to ``len(text) // 4`` without tiktoken)

    A plain single-clause question scores 1, so the default
    ``PlanningConfig.min_complexity = 2`` skips planning for it.

    Review note: a bare-semicolon signal was deliberately REMOVED.
    Semicolons appear in pasted code far more often than as natural
    language clause separators (e.g. "Refactor x = 1; y = 2 in my code"
    used to plan at the default threshold), and genuinely multi-step
    prompts are already caught by the remaining signals.
    """
    lowered = f" {text.lower()} "
    score = 1
    if any(marker in lowered for marker in _SEQUENCE_MARKERS):
        score += 1
    if _LIST_ITEM_RE.search(text):
        score += 1
    if len(_SENTENCE_RE.findall(text)) >= 3:
        score += 1
    if estimate_tokens(text) > 120:
        score += 1
    return score


# ---------------------------------------------------------------------------
# Approval support
# ---------------------------------------------------------------------------


class _PlanRejected(Exception):
    """Raised internally when the approval handler rejects the plan."""


class _ApprovablePlanAndExecute(PlanAndExecuteAgent):
    """PlanAndExecuteAgent with an approval gate between plan and execution.

    The handler receives the parsed plan and returns ``True`` (approve,
    in-place edits kept), ``False`` (reject -> ``_PlanRejected``), or an
    edited ``List[PlanStep]``. Edited steps are re-normalized to the single
    internal executor so the handler cannot break dispatch.
    """

    def __init__(self, *args: object, approval_handler: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._approval_handler = approval_handler

    async def _call_planner(self, prompt: str) -> List[PlanStep]:
        plan = await super()._call_planner(prompt)
        handler = self._approval_handler
        if handler is None:
            return plan
        verdict: Union[bool, List[PlanStep]] = handler(plan)  # type: ignore[operator]
        if inspect.isawaitable(verdict):
            # Review finding: an async handler used to be treated as a silent
            # rejection (a coroutine is neither True nor a list). Fail loudly
            # instead so the misconfiguration is obvious.
            close = getattr(verdict, "close", None)
            if callable(close):
                close()  # avoid "coroutine was never awaited" RuntimeWarning
            raise TypeError(
                "plan_approval_handler must be a sync callable returning "
                "True, False, or List[PlanStep]; got an awaitable (async "
                "handlers are not supported — do any async work before "
                "returning the verdict)."
            )
        if verdict is True:
            return plan
        if not isinstance(verdict, list):
            raise _PlanRejected()
        edited = [
            PlanStep(executor_name=_EXECUTOR_NAME, task=step.task)
            for step in verdict
            if isinstance(step, PlanStep) and step.task
        ]
        if not edited:
            raise _PlanRejected()
        return edited


# ---------------------------------------------------------------------------
# Usage continuity helpers (review findings: budget bypass + lost tokens)
# ---------------------------------------------------------------------------

_USAGE_SCALAR_FIELDS = (
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_cost_usd",
    "total_embedding_tokens",
    "total_embedding_cost_usd",
)


def _seed_usage_from_parent(clone_usage: AgentUsage, parent_usage: AgentUsage) -> AgentUsage:
    """Seed a fresh clone ``AgentUsage`` with the parent's scalar totals.

    ``Agent.clone_for_isolation()`` resets ``usage`` to a fresh
    ``AgentUsage``, so the clone's ``_check_budget`` would compare against
    zero and ``max_total_tokens`` / ``max_cost_usd`` would effectively be
    granted again to every sub-run. Copying the parent's running totals
    into the clone makes the lifetime caps keep binding across the planned
    flow. Only the scalar totals are seeded; ``iterations`` and the
    per-tool dicts stay empty so they remain a pure delta.

    Returns a baseline snapshot of the seeded scalars for
    :func:`_merge_usage_delta`.
    """
    baseline = AgentUsage()
    for field_name in _USAGE_SCALAR_FIELDS:
        value = getattr(parent_usage, field_name)
        setattr(clone_usage, field_name, value)
        setattr(baseline, field_name, value)
    return baseline


def _merge_usage_delta(
    parent_usage: AgentUsage, clone_usage: AgentUsage, baseline: AgentUsage
) -> None:
    """Fold a clone's OWN spend into the parent, excluding the seeded baseline.

    Scalar totals are merged as ``clone - baseline`` so the parent totals
    seeded by :func:`_seed_usage_from_parent` are never double-counted.
    ``iterations`` and the per-tool dicts start empty on the clone, so they
    are merged wholesale.
    """
    for field_name in _USAGE_SCALAR_FIELDS:
        delta = getattr(clone_usage, field_name) - getattr(baseline, field_name)
        setattr(parent_usage, field_name, getattr(parent_usage, field_name) + delta)
    parent_usage.iterations.extend(clone_usage.iterations)
    for tool_name, count in clone_usage.tool_usage.items():
        parent_usage.tool_usage[tool_name] = parent_usage.tool_usage.get(tool_name, 0) + count
    for tool_name, tokens in clone_usage.tool_tokens.items():
        parent_usage.tool_tokens[tool_name] = parent_usage.tool_tokens.get(tool_name, 0) + tokens


# ---------------------------------------------------------------------------
# Entry points called from agent/core.py
# ---------------------------------------------------------------------------


def warn_streaming_planning_skip(agent: "Agent") -> None:
    """Warn (once per agent instance) that planning is skipped for streaming."""
    if getattr(agent, "_planning_stream_warned", False):
        return
    agent._planning_stream_warned = True  # type: ignore[attr-defined]
    warnings.warn(
        "PlanningConfig.enabled is set but a streaming run was requested; "
        "planning is skipped for streaming runs (PlanAndExecute runs whole "
        "sub-agent calls and cannot stream tokens).",
        UserWarning,
        stacklevel=4,
    )


def run_with_planning(
    agent: "Agent",
    messages: List[Message],
    *,
    response_format: Optional["ResponseFormat"] = None,
    parent_run_id: Optional[str] = None,
) -> Optional[AgentResult]:
    """Sync wrapper around :func:`arun_with_planning`."""
    return run_sync(
        arun_with_planning(
            agent,
            messages,
            response_format=response_format,
            parent_run_id=parent_run_id,
        )
    )


async def arun_with_planning(
    agent: "Agent",
    messages: List[Message],
    *,
    response_format: Optional["ResponseFormat"] = None,
    parent_run_id: Optional[str] = None,
) -> Optional[AgentResult]:
    """Run the planned flow for *agent*, or return ``None`` to fall back.

    Returns ``None`` when the complexity gate is not met, no user text is
    present, or the approval handler rejects the plan. The caller
    (``Agent.run``/``Agent.arun``) then proceeds with the normal loop.
    """
    cfg = agent.config.planning
    if cfg is None or not cfg.enabled:
        return None

    prompt = _latest_user_text(messages)
    if not prompt:
        return None
    if not cfg.always and _complexity_score(prompt) < cfg.min_complexity:
        return None

    planner = agent.clone_for_isolation()
    planner.config.planning = None
    if cfg.provider is not None:
        planner.provider = cfg.provider
    if cfg.model is not None:
        planner.config.model = cfg.model
        planner._current_model = cfg.model

    executor = agent.clone_for_isolation()
    executor.config.planning = None

    # Budget continuity (review finding): seed both clones with the parent's
    # running totals so max_total_tokens / max_cost_usd bind across the whole
    # planned flow instead of resetting per clone. The baselines let the
    # finally-merge below fold back only each clone's own spend.
    planner_baseline = _seed_usage_from_parent(planner.usage, agent.usage)
    executor_baseline = _seed_usage_from_parent(executor.usage, agent.usage)

    pattern = _ApprovablePlanAndExecute(
        planner=planner,
        executors={_EXECUTOR_NAME: executor},
        approval_handler=None if cfg.auto_approve else cfg.plan_approval_handler,
        cancellation_token=agent.config.cancellation_token,
    )
    try:
        try:
            graph_result = await pattern.arun(prompt)
        except _PlanRejected:
            warnings.warn(
                "Plan rejected by plan_approval_handler; falling back to standard "
                "(non-planned) execution.",
                UserWarning,
                stacklevel=2,
            )
            return None

        plan_dicts = list(graph_result.state.data.get("__plan__", []))
        steps_executed = len(graph_result.node_results)

        synthesis_msg = Message(
            role=Role.USER,
            content=_SYNTHESIS_PROMPT.format(
                task=prompt, results=graph_result.content or "(no step output)"
            ),
        )
        final = await executor.arun(
            [synthesis_msg], response_format=response_format, parent_run_id=parent_run_id
        )
    finally:
        # Usage continuity (review finding): merge sub-run usage into the
        # parent on EVERY exit path — success, plan rejection, or an
        # exception from pattern.arun / the synthesis call — exactly once.
        # Delta-merge so the seeded baselines are not double-counted.
        _merge_usage_delta(agent.usage, planner.usage, planner_baseline)
        _merge_usage_delta(agent.usage, executor.usage, executor_baseline)

    all_tool_calls: List[ToolCall] = []
    for results in graph_result.node_results.values():
        for step_result in results:
            all_tool_calls.extend(getattr(step_result, "tool_calls", None) or [])
    all_tool_calls.extend(final.tool_calls)

    reasoning = final.reasoning
    if cfg.reasoning and plan_dicts:
        plan_text = "\n".join(
            f"{i + 1}. {step.get('task', '')}" for i, step in enumerate(plan_dicts)
        )
        reasoning = f"Execution plan:\n{plan_text}"

    trace = final.trace
    if trace is not None:
        trace.metadata["planning"] = {
            "plan": plan_dicts,
            "steps_executed": steps_executed,
        }

    # Persist the turn on the parent agent's memory (clones run memory-less).
    if agent.memory is not None:
        run_id = trace.run_id if trace is not None else ""
        agent._memory_add_many(list(messages) + [final.message], run_id)
        agent._session_save(run_id)

    return AgentResult(
        message=final.message,
        tool_name=final.tool_name,
        tool_args=final.tool_args,
        iterations=steps_executed + 1,
        tool_calls=all_tool_calls,
        parsed=final.parsed,
        reasoning=reasoning,
        reasoning_history=final.reasoning_history,
        trace=trace,
        provider_used=final.provider_used,
        usage=copy.deepcopy(agent.usage),
        artifacts=final.artifacts,
    )


def _latest_user_text(messages: List[Message]) -> str:
    """Return the most recent user message content, or empty string."""
    for msg in reversed(messages):
        if msg.role == Role.USER and msg.content:
            return msg.content
    return ""


__all__ = [
    "arun_with_planning",
    "run_with_planning",
    "warn_streaming_planning_skip",
]
