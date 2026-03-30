"""ReflectiveAgent — actor produces a draft, critic evaluates, actor revises.

The actor Agent produces output. The critic Agent evaluates it.
The actor revises based on the critique. This cycle repeats until either
the critic's response contains the stop_condition string (e.g. "approved")
or max_reflections is reached.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..observer import AgentObserver

from ..types import Message, Role


@dataclass
class ReflectionRound:
    """The output of one actor→critic cycle."""

    round_number: int
    draft: str
    critique: str
    approved: bool


@dataclass
class ReflectiveResult:
    """Result of a ReflectiveAgent run."""

    final_draft: str
    rounds: List[ReflectionRound]
    approved: bool

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)


class ReflectiveAgent:
    """Actor-Critic loop: actor drafts, critic evaluates, actor revises.

    Stops when the critic's response contains ``stop_condition`` (case-insensitive)
    or when ``max_reflections`` rounds have completed.

    Args:
        actor: Agent that produces the initial draft and revisions.
        critic: Agent that evaluates drafts and provides feedback.
        max_reflections: Maximum actor-critic rounds (default: 3).
        stop_condition: String in critic output that signals approval (default: "approved").
        observers: Optional AgentObserver instances (forwarded to both agents via their config).
        cancellation_token: Optional cancellation token.
    """

    def __init__(
        self,
        actor: "Agent",
        critic: "Agent",
        *,
        max_reflections: int = 3,
        stop_condition: str = "approved",
        observers: Optional[List["AgentObserver"]] = None,
        cancellation_token: Optional["CancellationToken"] = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.max_reflections = max_reflections
        self.stop_condition = stop_condition
        self._observers = observers or []
        self._cancellation_token = cancellation_token

    def run(self, prompt: str) -> ReflectiveResult:
        """Execute synchronously."""
        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> ReflectiveResult:
        """Execute asynchronously: actor → critic → actor → ..."""
        rounds: List[ReflectionRound] = []
        draft = ""
        critique = ""
        approved = False

        for round_num in range(self.max_reflections):
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            # Actor phase
            if round_num == 0:
                actor_input = prompt
            else:
                actor_input = (
                    f"Original task: {prompt}\n\n"
                    f"Your previous draft:\n{draft}\n\n"
                    f"Critic feedback:\n{critique}\n\n"
                    f"Please revise your draft based on the feedback above."
                )

            actor_result = await self.actor.arun([Message(role=Role.USER, content=actor_input)])
            draft = actor_result.content or ""

            # Critic phase
            critic_input = (
                f"Please evaluate the following response to this task: {prompt}\n\n"
                f"Response to evaluate:\n{draft}\n\n"
                f"Provide detailed feedback. If the response is satisfactory, "
                f"include the word '{self.stop_condition}' in your response."
            )
            critic_result = await self.critic.arun([Message(role=Role.USER, content=critic_input)])
            critique = critic_result.content or ""
            approved = self.stop_condition.lower() in critique.lower()

            rounds.append(
                ReflectionRound(
                    round_number=round_num,
                    draft=draft,
                    critique=critique,
                    approved=approved,
                )
            )

            if approved:
                break

        return ReflectiveResult(final_draft=draft, rounds=rounds, approved=approved)


__all__ = ["ReflectiveAgent", "ReflectionRound", "ReflectiveResult"]
