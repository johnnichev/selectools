"""
Loop Detection — catch pathological tool-call patterns before max_iterations.

Since v0.22.0, `AgentConfig.loop_detector` enables three composable detectors:

  - RepeatDetector:   same (tool, args) N times in a row
  - StallDetector:    same (tool, result) N times in a row — polling with no progress
  - PingPongDetector: a cycle of length K repeats M times

On detection, the agent either raises LoopDetectedError (default) or injects
a corrective system message and continues (LoopPolicy.INJECT_MESSAGE).

Prerequisites: No API key needed (uses a scripted provider)
Run: python examples/95_loop_detection.py
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from selectools import (
    Agent,
    AgentConfig,
    LoopDetectedError,
    LoopDetector,
    LoopPolicy,
    RepeatDetector,
)
from selectools.tools import tool
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats


@tool(description="Search the web")
def search(query: str) -> str:
    return "no results found"


@dataclass
class _RepeatingProvider:
    """Minimal provider that always returns the same tool call — simulates a stuck LLM."""

    name: str = "repeating"
    supports_streaming: bool = False
    supports_async: bool = True

    def _next(self) -> Tuple[Message, UsageStats]:
        call = ToolCall(tool_name="search", parameters={"query": "cats"}, id="tc_1")
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[call])
        return msg, UsageStats(model="fake", provider=self.name)

    def complete(
        self,
        *,
        model: str = "fake",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        return self._next()

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self._next()


def demo_raise() -> None:
    """Default policy: raise LoopDetectedError."""
    agent = Agent(
        tools=[search],
        provider=_RepeatingProvider(),
        config=AgentConfig(
            max_iterations=20,
            loop_detector=LoopDetector.default(),
        ),
    )
    try:
        agent.run("Find something")
    except LoopDetectedError as exc:
        print(f"[RAISE]  detector={exc.detector}  details={exc.details}")


def demo_inject_message() -> None:
    """INJECT_MESSAGE policy: add a corrective system message, keep looping until max_iterations."""
    detector = LoopDetector(
        detectors=[RepeatDetector(threshold=3)],
        policy=LoopPolicy.INJECT_MESSAGE,
        inject_message="You are repeating yourself. Try a different approach.",
    )
    agent = Agent(
        tools=[search],
        provider=_RepeatingProvider(),
        config=AgentConfig(max_iterations=5, loop_detector=detector),
    )
    result = agent.run("Find something")
    print(f"[INJECT] iterations={result.iterations}  no exception raised")


def main() -> None:
    print("Loop detection — three patterns, two policies.\n")
    demo_raise()
    demo_inject_message()


if __name__ == "__main__":
    main()
