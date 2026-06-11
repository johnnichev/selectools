#!/usr/bin/env python3
"""
Tool result compression (ROADMAP P2).

Verbose tool outputs (a web scrape returning 10KB of HTML, a SQL query
dumping 500 rows) waste context window on every subsequent iteration. With
``ToolConfig(compress_results=True)``, any tool result longer than
``compress_threshold`` characters is summarized by a one-shot LLM call
BEFORE it is appended to the conversation. The model sees a marked summary:

    [compressed from 9482 chars] 500 rows; ids 1-500; total=$12,403.50 ...

Guarantees:
- Zero overhead when disabled (the default).
- Terminal-tool results and stop_condition matches are NEVER compressed —
  they become the agent's final answer verbatim.
- If the compression call fails, the result is truncated with a
  ``[truncated from N chars; compression failed]`` marker. The loop never
  crashes and never loses progress.
- ``compress_provider`` / ``compress_model`` let you route compression to a
  cheap, fast model. Without them the agent's own provider+model is used —
  meaning compression is billed at your main model's rates.

No API key needed. Runs offline with scripted providers.

Run: python examples/107_tool_result_compression.py
"""

from __future__ import annotations

from typing import Any, List, Tuple

from selectools import Agent, AgentConfig, Message, Role, ToolCall, tool
from selectools.agent.config_groups import ToolConfig
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# A tool that returns a verbose result
# ---------------------------------------------------------------------------


@tool()
def query_orders() -> str:
    """Return all orders from the database (verbose!)."""
    rows = [f"order_id={i} customer=cust-{i % 7} total=${i * 3}.50" for i in range(1, 201)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Scripted offline providers
# ---------------------------------------------------------------------------


def _usage(model: str) -> UsageStats:
    return UsageStats(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0, model=model
    )


class ScriptedAgentProvider:
    """Main provider: first asks for the tool, then answers."""

    name = "scripted"
    supports_streaming = False
    supports_async = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, *, model: str, messages: List[Message], **kw: Any
    ) -> Tuple[Message, UsageStats]:
        self.calls += 1
        if self.calls == 1:
            msg = Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(tool_name="query_orders", parameters={})],
            )
        else:
            tool_msg = next(m for m in messages if m.role == Role.TOOL)
            msg = Message(
                role=Role.ASSISTANT,
                content=f"Answered from a {len(tool_msg.content)}-char tool message.",
            )
        return msg, _usage(model)


class CheapSummarizer:
    """Dedicated compression provider (stands in for a fast/cheap model)."""

    name = "cheap-summarizer"
    supports_streaming = False
    supports_async = False

    def complete(
        self, *, model: str, messages: List[Message], **kw: Any
    ) -> Tuple[Message, UsageStats]:
        raw = messages[-1].content or ""
        n_rows = raw.count("order_id=")
        summary = f"{n_rows} orders, ids 1-{n_rows}, customers cust-0..cust-6."
        return Message(role=Role.ASSISTANT, content=summary), _usage(model)


def main() -> None:
    config = AgentConfig(
        max_iterations=3,
        tool=ToolConfig(
            compress_results=True,
            compress_threshold=2000,
            compress_provider=CheapSummarizer(),
            compress_model="cheap-model",
        ),
    )
    agent = Agent(tools=[query_orders], provider=ScriptedAgentProvider(), config=config)
    result = agent.run("How many orders do we have?")

    tool_msg = next(m for m in agent._history if m.role == Role.TOOL)
    print(f"Raw tool output:    {len(query_orders.execute({}))} chars")
    print(f"Message the model saw ({len(tool_msg.content)} chars):")
    print(f"  {tool_msg.content}")
    print(f"Final answer: {result.content}")

    assert tool_msg.content.startswith("[compressed from")


if __name__ == "__main__":
    main()
