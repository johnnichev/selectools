"""
StreamEnd — report usage and provider state from a streaming provider (offline).

Shows a custom provider that yields a StreamEnd as the last item of its stream,
so streamed turns count toward agent.usage, observers get real usage, and the
assistant message carries provider_state across the tool loop.

Prerequisites: none (offline)
Run: python examples/115_stream_end_provider.py
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple, Union

from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.observer import AgentObserver
from selectools.providers.base import Provider
from selectools.types import StreamEnd, ToolCall
from selectools.usage import UsageStats


@tool()
def lookup_answer() -> str:
    """Return the answer to everything."""
    return "42"


class ReplayingProvider(Provider):
    """A fake stateless API: turn 1 calls a tool, turn 2 answers.

    The reasoning items from turn 1 travel in provider_state and come back on
    turn 2's messages, exactly as a Responses-style API needs them replayed.
    """

    name = "replaying"
    supports_streaming = True
    supports_async = False

    def __init__(self) -> None:
        self.turn = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise NotImplementedError("this example streams")

    def stream(
        self, *, model: str, messages: List[Message], **kwargs: Any
    ) -> Iterator[Union[str, ToolCall, StreamEnd]]:
        self.turn += 1
        usage = UsageStats(prompt_tokens=20 * self.turn, completion_tokens=5, model=model)
        if self.turn == 1:
            yield ToolCall(tool_name="lookup_answer", parameters={}, id="call_1")
            yield StreamEnd(
                usage=usage,
                provider_state={"items": [{"type": "reasoning", "encrypted_content": "..."}]},
            )
            return
        replayed = [m.provider_state for m in messages if m.provider_state]
        print(f"turn 2 received provider_state from turn 1: {replayed}")
        for word in ("The", " answer", " is", " 42."):
            yield word
        yield StreamEnd(usage=usage)


class UsagePrinter(AgentObserver):
    def on_llm_end(self, run_id: str, content: Any, usage: Optional[UsageStats]) -> None:
        tokens = usage.total_tokens if usage else None
        print(f"on_llm_end usage: {tokens} tokens")


def main() -> None:
    agent = Agent(
        [lookup_answer],
        provider=ReplayingProvider(),
        config=AgentConfig(model="fake-model", stream=True, observers=[UsagePrinter()]),
    )
    result = agent.run("What is the answer?", stream_handler=lambda chunk: print(chunk, end=""))
    print()
    print(f"final: {result.message.content!r}")
    print(f"total tokens across both streamed turns: {result.usage.total_tokens}")
    turn_one = [m for m in agent._history if m.role == Role.ASSISTANT and m.tool_calls][0]
    print(f"turn 1 message kept provider_state: {turn_one.provider_state is not None}")


if __name__ == "__main__":
    main()
