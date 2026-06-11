#!/usr/bin/env python3
"""
Typed ToolResult returns + artifact side-channel (issue #59).

Two conventions every tool-using agent otherwise re-invents:

1. ToolResult — frozen dataclass base for typed tool returns. Subclasses set
   a ``kind`` discriminator as a ClassVar; the serializer re-injects it into
   the JSON the model sees (ClassVar fields are dropped by asdict()).
   Built-ins: Ambiguous, NotFound. Note the epistemics: ``not_found`` means
   "this tool observed no match from this source at this time", not "the
   entity does not exist".

2. emit_artifact() — tools that produce files (charts, PDFs, exports) attach
   them out-of-band instead of stuffing URLs into the reply string. The agent
   drains the per-run collector into AgentResult.artifacts, where channel
   layers (chat, email, Slack) deliver them. The shape is richer than a URL
   on purpose: URLs rot and signed links expire — sha256 + size let consumers
   identify an artifact without storing its body.

No API key needed. Runs offline with a scripted provider.

Run: python examples/99_tool_results_artifacts.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar, List, Tuple

from selectools import Agent, AgentConfig, Message, NotFound, Role, ToolCall, emit_artifact, tool
from selectools.providers.base import Provider
from selectools.results import ToolResult
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_CUSTOMERS = {"acme": {"id": 1, "name": "Acme Corp", "plan": "enterprise"}}


@dataclass(frozen=True)
class CustomerFound(ToolResult):
    """A user-defined typed result — kind survives serialization."""

    kind: ClassVar[str] = "customer_found"

    customer_id: int
    name: str
    plan: str


@tool()
def find_customer(query: str) -> ToolResult:
    """Look up a customer by name."""
    row = _CUSTOMERS.get(query.lower())
    if row is None:
        # Observation, not a truth claim: no match from THIS source NOW.
        return NotFound(entity="customer", query=query)
    return CustomerFound(customer_id=row["id"], name=row["name"], plan=row["plan"])


@tool()
def render_chart(title: str) -> str:
    """Render a revenue chart as PNG."""
    body = f"<fake png bytes for {title}>".encode()
    emit_artifact(
        f"https://files.example.com/charts/{title}.png",
        mime_type="image/png",
        filename=f"{title}.png",
        sha256=hashlib.sha256(body).hexdigest(),
        size=len(body),
        role="primary",
        retention="30d",
    )
    # The reply string stays LLM-friendly; the file travels on the side.
    return f"Rendered chart '{title}'."


# ---------------------------------------------------------------------------
# Scripted offline provider
# ---------------------------------------------------------------------------


class ScriptedProvider(Provider):
    """Issues a fixed sequence of tool calls, then a final answer."""

    name = "scripted"
    supports_streaming = False
    supports_async = False

    def __init__(self, tool_calls: List[ToolCall]) -> None:
        self.default_model = "scripted"
        self._tool_calls = tool_calls
        self._turn = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        usage = UsageStats(0, 0, 0, 0.0, model="scripted", provider="scripted")
        self._turn += 1
        if self._turn == 1:
            return Message(role=Role.ASSISTANT, content="", tool_calls=self._tool_calls), usage
        return Message(role=Role.ASSISTANT, content="Here is the Q2 report."), usage


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Typed results: kind survives into the JSON the LLM sees.
    print("== ToolResult serialization ==")
    print("hit: ", find_customer.execute({"query": "acme"}))
    print("miss:", find_customer.execute({"query": "globex"}))

    # 2. Artifact side-channel through a full agent run.
    print("\n== Artifact side-channel ==")
    provider = ScriptedProvider(
        [ToolCall(tool_name="render_chart", parameters={"title": "q2-revenue"}, id="c1")]
    )
    agent = Agent(
        tools=[find_customer, render_chart],
        provider=provider,
        config=AgentConfig(max_iterations=3),
    )
    result = agent.run("Chart Q2 revenue for Acme")
    print("reply:    ", result.content)
    for artifact in result.artifacts:
        print("artifact: ", artifact.url)
        print("           mime:", artifact.mime_type, "| size:", artifact.size, "bytes")
        print("           sha256:", (artifact.sha256 or "")[:16], "... | role:", artifact.role)


if __name__ == "__main__":
    main()
