#!/usr/bin/env python3
"""
Agent-level human-in-the-loop (ROADMAP P2).

Standalone agents (no AgentGraph needed) can gate specific tools behind an
approval handler:

    config = AgentConfig(tool=ToolConfig(
        require_approval=["execute_shell", "send_email"],  # or "*" for all
        approval_handler=my_callback,                      # sync OR async
    ))

The handler receives a structured ``ApprovalRequest`` (tool name, args,
reason, one-line preview) and returns truthy to approve, falsy to deny.
A denied call does NOT crash the loop: the model sees a standardized
"denied by approval handler" tool result and keeps going.

This extends the existing machinery (Tool.requires_approval +
AgentConfig.confirm_action): require_approval centralizes per-tool gating
at config level, and approval_handler is a richer confirm_action — it also
services ToolPolicy ``review`` decisions and tools marked
``requires_approval=True``.

No API key needed. Runs offline with a scripted provider.

Run: python examples/108_agent_hitl.py
"""

from __future__ import annotations

from typing import Any, List, Tuple

from selectools import Agent, AgentConfig, ApprovalRequest, Message, Role, ToolCall, tool
from selectools.agent.config_groups import ToolConfig
from selectools.usage import UsageStats


@tool()
def read_report() -> str:
    """Read the quarterly report (safe, ungated)."""
    return "Q2 revenue: $1.2M, churn 3%"


@tool()
def send_email(to: str, subject: str) -> str:
    """Send an email (dangerous, gated)."""
    return f"Email sent to {to}: {subject}"


@tool()
def execute_shell(command: str) -> str:
    """Run a shell command (dangerous, gated)."""
    return f"$ {command}\nok"


# ---------------------------------------------------------------------------
# Approval handler — in production this would page a human (Slack, CLI, web)
# ---------------------------------------------------------------------------


def approval_handler(request: ApprovalRequest) -> bool:
    print(f"  [approval] {request.preview}")
    print(f"  [approval] reason: {request.reason}")
    decision = request.tool_name == "send_email"  # approve email, deny shell
    print(f"  [approval] -> {'APPROVED' if decision else 'DENIED'}")
    return decision


# ---------------------------------------------------------------------------
# Scripted offline provider: calls all three tools, then answers
# ---------------------------------------------------------------------------


def _usage(model: str) -> UsageStats:
    return UsageStats(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0, model=model
    )


class ScriptedProvider:
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
                tool_calls=[
                    ToolCall(tool_name="read_report", parameters={}, id="tc-1"),
                    ToolCall(
                        tool_name="send_email",
                        parameters={"to": "boss@corp.com", "subject": "Q2 numbers"},
                        id="tc-2",
                    ),
                    ToolCall(
                        tool_name="execute_shell", parameters={"command": "rm -rf /"}, id="tc-3"
                    ),
                ],
            )
        else:
            denied = [m for m in messages if m.role == Role.TOOL and "denied" in (m.content or "")]
            msg = Message(
                role=Role.ASSISTANT,
                content=f"Report sent. {len(denied)} action(s) were denied by the reviewer.",
            )
        return msg, _usage(model)


def main() -> None:
    config = AgentConfig(
        max_iterations=3,
        parallel_tool_execution=False,
        tool=ToolConfig(
            require_approval=["send_email", "execute_shell"],
            approval_handler=approval_handler,
        ),
    )
    agent = Agent(
        tools=[read_report, send_email, execute_shell],
        provider=ScriptedProvider(),
        config=config,
    )
    result = agent.run("Email the Q2 numbers to my boss and clean up the disk.")

    print("\nTool results the model saw:")
    for m in agent._history:
        if m.role == Role.TOOL:
            print(f"  {m.tool_name}: {m.content}")
    print(f"\nFinal answer: {result.content}")

    tool_msgs = {m.tool_name: m.content for m in agent._history if m.role == Role.TOOL}
    assert tool_msgs["read_report"].startswith("Q2 revenue")  # ungated, no approval
    assert tool_msgs["send_email"].startswith("Email sent")  # approved
    assert "denied by approval handler" in tool_msgs["execute_shell"]  # denied


if __name__ == "__main__":
    main()
