"""DebateAgent — multiple agents argue positions, a judge synthesizes the conclusion.

Each agent is assigned a position (e.g. "optimist", "skeptic"). Over max_rounds
rounds, each agent argues its position while seeing the previous round's
transcript. After all rounds, a judge Agent synthesizes a final answer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..observer import AgentObserver

from .._async_utils import run_sync
from ..stability import beta
from ..types import Message, Role


@dataclass
class DebateRound:
    """Arguments from all agents in one debate round."""

    round_number: int
    arguments: Dict[str, str]  # agent_name → argument text


@dataclass
class DebateResult:
    """Result of a DebateAgent run."""

    conclusion: str
    rounds: List[DebateRound]

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)


@beta
class DebateAgent:
    """Multi-agent debate: agents argue → judge synthesizes conclusion.

    Each agent argues its assigned position each round. Later rounds include
    the prior round transcript so agents can respond to each other.
    The judge is called once after all rounds to synthesize a conclusion.

    Args:
        agents: Dict mapping position name → Agent (minimum 2 agents required).
        judge: Agent that synthesizes the final conclusion.
        max_rounds: Number of debate rounds (default: 3).
        observers: Optional AgentObserver instances.
        cancellation_token: Optional cancellation token.
    """

    def __init__(
        self,
        agents: Dict[str, "Agent"],
        judge: "Agent",
        *,
        max_rounds: int = 3,
        observers: Optional[List["AgentObserver"]] = None,
        cancellation_token: Optional["CancellationToken"] = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError(f"DebateAgent requires at least 2 agents, got {len(agents)}")
        if max_rounds < 1:
            raise ValueError(f"DebateAgent requires max_rounds >= 1, got {max_rounds}")
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
        self._observers = observers or []
        self._cancellation_token = cancellation_token

    def run(self, prompt: str) -> DebateResult:
        """Execute synchronously."""
        return run_sync(self.arun(prompt))

    async def arun(self, prompt: str) -> DebateResult:
        """Execute asynchronously: agents debate → judge concludes."""
        rounds: List[DebateRound] = []
        prior_context = ""

        for round_num in range(self.max_rounds):
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break

            round_arguments: Dict[str, str] = {}

            for agent_name, agent in self.agents.items():
                if self._cancellation_token and self._cancellation_token.is_cancelled:
                    break

                if round_num == 0:
                    agent_input = (
                        f"Topic: {prompt}\n\n"
                        f"You are arguing as: {agent_name}\n"
                        f"Present your opening argument for this position."
                    )
                else:
                    agent_input = (
                        f"Topic: {prompt}\n\n"
                        f"You are arguing as: {agent_name}\n\n"
                        f"Previous discussion:\n{prior_context}\n\n"
                        f"Continue arguing your position, responding to the points made above."
                    )

                result = await agent.arun([Message(role=Role.USER, content=agent_input)])
                round_arguments[agent_name] = result.content or ""

            rounds.append(DebateRound(round_number=round_num, arguments=round_arguments))

            # Build context for next round (truncated to keep prompts manageable)
            prior_context += f"\n--- Round {round_num + 1} ---\n"
            for name, arg in round_arguments.items():
                prior_context += f"{name}: {arg[:400]}\n"

        # Judge synthesizes conclusion
        judge_input = (
            f"Topic: {prompt}\n\n"
            f"Full debate transcript:\n{prior_context}\n\n"
            f"Based on all arguments presented, provide a balanced synthesis and conclusion."
        )
        judge_result = await self.judge.arun([Message(role=Role.USER, content=judge_input)])
        conclusion = judge_result.content or ""

        return DebateResult(conclusion=conclusion, rounds=rounds)


__all__ = ["DebateAgent", "DebateRound", "DebateResult"]
