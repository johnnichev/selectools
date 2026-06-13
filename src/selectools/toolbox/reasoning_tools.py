"""
Reasoning tools — turn reasoning into explicit, bounded, inspectable tool calls.

Today reasoning is a *prompt strategy* (``PromptBuilder(reasoning_strategy=...)``):
it nudges the model to think a certain way, but the thinking stays hidden inside
the model's output. These tools make reasoning a **composable step** instead: the
agent calls ``think`` / ``analyze`` to externalize each reasoning step, so the
chain shows up as structured tool calls in the trace — and is bounded by explicit
``min_steps`` / ``max_steps``.

Usage::

    from selectools import Agent
    from selectools.toolbox.reasoning_tools import make_reasoning_tools

    agent = Agent(
        tools=[*my_tools, *make_reasoning_tools(min_steps=1, max_steps=8)],
        provider=provider,
    )

Or keep a handle to inspect the recorded chain::

    from selectools.toolbox.reasoning_tools import ReasoningTools

    reasoning = ReasoningTools(min_steps=2, max_steps=6)
    agent = Agent(tools=[*my_tools, *reasoning.tools], provider=provider)
    agent.run("...")
    for step in reasoning.steps:
        print(step.index, step.kind, step.content)

Bounds:
    - ``max_steps`` is **enforced**: once reached, further ``think`` / ``analyze``
      calls are not recorded and return a message telling the agent to stop
      reasoning and answer (a real guard against reasoning loops).
    - ``min_steps`` is **guidance**: it is advertised in the tool descriptions
      and each call reports how many more are expected. A model cannot be forced
      to call a tool, so the floor is a nudge, not a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..stability import beta
from ..tools import Tool, ToolParameter

_DEFAULT_MIN_STEPS = 1
_DEFAULT_MAX_STEPS = 10


@beta
@dataclass
class ReasoningStep:
    """One recorded reasoning step."""

    index: int  # 1-based position in the chain
    kind: str  # "think" or "analyze"
    content: str


@beta
class ReasoningTools:
    """Produces ``think`` / ``analyze`` tools backed by a shared, bounded log.

    Both tools count against the same step budget. Inspect :attr:`steps` after a
    run to see the externalized reasoning chain; call :meth:`reset` to reuse the
    instance for another run.
    """

    def __init__(
        self,
        min_steps: int = _DEFAULT_MIN_STEPS,
        max_steps: Optional[int] = _DEFAULT_MAX_STEPS,
    ) -> None:
        if min_steps < 0:
            raise ValueError("min_steps must be >= 0")
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be >= 1 when set")
            if max_steps < min_steps:
                raise ValueError("max_steps must be >= min_steps")
        self.min_steps = min_steps
        self.max_steps = max_steps
        self._steps: List[ReasoningStep] = []

    @property
    def steps(self) -> List[ReasoningStep]:
        """The recorded reasoning chain (read-only view)."""
        return list(self._steps)

    @property
    def count(self) -> int:
        return len(self._steps)

    def reset(self) -> None:
        """Clear the recorded chain so the instance can drive another run."""
        self._steps.clear()

    def _record(self, kind: str, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return "Nothing recorded: provide a non-empty reasoning step."
        if self.max_steps is not None and self.count >= self.max_steps:
            return (
                f"Reasoning budget reached ({self.max_steps} steps). "
                f"Stop reasoning and give your final answer now."
            )
        step = ReasoningStep(index=self.count + 1, kind=kind, content=content)
        self._steps.append(step)
        remaining_min = max(0, self.min_steps - self.count)
        if remaining_min > 0:
            return (
                f"Recorded {kind} step {step.index}. "
                f"Reason at least {remaining_min} more time(s) before answering."
            )
        return f"Recorded {kind} step {step.index}. Continue reasoning or give your final answer."

    def think_tool(self) -> Tool:
        """A ``think`` tool: a scratchpad for one reasoning step."""
        floor = (
            f" Use it at least {self.min_steps} time(s) before answering."
            if self.min_steps > 0
            else ""
        )

        def _think(thought: str) -> str:
            return self._record("think", thought)

        return Tool(
            name="think",
            description=(
                "Record one step of your reasoning before acting. Use this as a "
                "scratchpad to work through the problem: what you know, what to do "
                "next, and why. It does not call any external system; it just makes "
                "your reasoning explicit." + floor
            ),
            parameters=[
                ToolParameter(
                    name="thought",
                    param_type=str,
                    description="Your reasoning for this step.",
                    required=True,
                ),
            ],
            function=_think,
        )

    def analyze_tool(self) -> Tool:
        """An ``analyze`` tool: evaluate results and decide the next step."""

        def _analyze(analysis: str) -> str:
            return self._record("analyze", analysis)

        return Tool(
            name="analyze",
            description=(
                "Record an analysis step: evaluate the result of a previous action "
                "or tool call, judge whether it moves you toward the goal, and "
                "decide what to do next. Counts against the same reasoning budget "
                "as 'think'."
            ),
            parameters=[
                ToolParameter(
                    name="analysis",
                    param_type=str,
                    description="Your evaluation and decision about what to do next.",
                    required=True,
                ),
            ],
            function=_analyze,
        )

    @property
    def tools(self) -> List[Tool]:
        """Both reasoning tools, sharing this instance's bounded log."""
        return [self.think_tool(), self.analyze_tool()]


@beta
def make_reasoning_tools(
    min_steps: int = _DEFAULT_MIN_STEPS,
    max_steps: Optional[int] = _DEFAULT_MAX_STEPS,
) -> List[Tool]:
    """Create ``think`` + ``analyze`` tools backed by a fresh bounded log.

    Convenience for the common case where you don't need to inspect the chain.
    Hold a :class:`ReasoningTools` instead if you want ``.steps`` afterward.
    """
    return ReasoningTools(min_steps=min_steps, max_steps=max_steps).tools


__stability__ = "beta"

__all__ = [
    "ReasoningStep",
    "ReasoningTools",
    "make_reasoning_tools",
]
