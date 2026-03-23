#!/usr/bin/env python3
"""
Per-Tool Approval Gate — require human approval for dangerous tools.

Demonstrates:
- @tool(requires_approval=True) decorator flag
- confirm_action callback on AgentConfig
- Safe tools execute freely, dangerous tools pause for approval

Prerequisites:
    pip install selectools
    export OPENAI_API_KEY=your-key
"""

from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class ApprovalDemoProvider:
    """Provider that requests tools in sequence."""

    name = "mock"
    supports_streaming = False
    supports_async = True

    def __init__(self, tool_sequence: List[Tuple[str, Dict[str, str]]]) -> None:
        self._sequence = tool_sequence
        self._call_count = 0

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        idx = self._call_count - 1

        if idx < len(self._sequence):
            tool_name, tool_args = self._sequence[idx]
            if tool_name:
                return (
                    Message(
                        role=Role.ASSISTANT,
                        content=f"I'll use {tool_name}",
                        tool_calls=[
                            ToolCall(
                                tool_name=tool_name,
                                parameters=tool_args,
                                id=f"call_{self._call_count}",
                            )
                        ],
                    ),
                    UsageStats(100, 50, 150, 0.001, "mock", "mock"),
                )

        return (
            Message(role=Role.ASSISTANT, content="Done."),
            UsageStats(100, 50, 150, 0.001, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools — safe vs dangerous
# ---------------------------------------------------------------------------


@tool(description="Look up a customer record (read-only)")
def lookup_customer(customer_id: str) -> str:
    return f"Customer {customer_id}: Alice Smith, alice@example.com"


@tool(description="Check current account balance (read-only)")
def check_balance(customer_id: str) -> str:
    return f"Customer {customer_id}: balance = $1,250.00"


@tool(requires_approval=True, description="Issue a refund to customer")
def issue_refund(customer_id: str, amount: str) -> str:
    return f"Refund of {amount} issued to customer {customer_id}."


@tool(requires_approval=True, description="Close a customer account permanently")
def close_account(customer_id: str, reason: str) -> str:
    return f"Account {customer_id} closed: {reason}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Per-Tool Approval Gate Demo")
    print("=" * 70)

    all_tools = [lookup_customer, check_balance, issue_refund, close_account]

    # Show which tools need approval
    print("\n--- Tool approval status ---\n")
    for t in all_tools:
        flag = "REQUIRES APPROVAL" if t.requires_approval else "auto-execute"
        print(f"  {t.name:20s}  {flag}")

    # --- Demo 1: Safe tool runs without approval ---
    print("\n--- Demo 1: Safe tool executes freely ---\n")

    provider = ApprovalDemoProvider(
        [
            ("lookup_customer", {"customer_id": "C-123"}),
            ("", {}),
        ]
    )

    approval_log: List[str] = []

    def confirm_action(tool_name: str, tool_args: Dict[str, Any], reason: str) -> bool:
        approved = tool_name != "close_account"
        approval_log.append(f"{tool_name}: {'APPROVED' if approved else 'DENIED'}")
        print(f"    [APPROVAL] {tool_name}({tool_args}) -> {'APPROVED' if approved else 'DENIED'}")
        return approved

    agent = Agent(
        tools=all_tools,
        provider=provider,
        config=AgentConfig(
            max_iterations=3,
            confirm_action=confirm_action,
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Look up customer C-123")])
    print(f"\n  Result:       {result.content[:60]}")
    print(f"  Approvals:    {len(approval_log)} (safe tools skip approval)")

    # --- Demo 2: Dangerous tool triggers approval (approved) ---
    print("\n--- Demo 2: Dangerous tool triggers approval (APPROVED) ---\n")

    approval_log.clear()
    provider2 = ApprovalDemoProvider(
        [
            ("issue_refund", {"customer_id": "C-123", "amount": "$50.00"}),
            ("", {}),
        ]
    )

    agent2 = Agent(
        tools=all_tools,
        provider=provider2,
        config=AgentConfig(
            max_iterations=3,
            confirm_action=confirm_action,
        ),
    )

    result2 = agent2.run([Message(role=Role.USER, content="Refund $50 to C-123")])
    print(f"\n  Result:       {result2.content[:60]}")
    print(f"  Approvals:    {approval_log}")

    # --- Demo 3: Dangerous tool triggers approval (denied) ---
    print("\n--- Demo 3: Dangerous tool triggers approval (DENIED) ---\n")

    approval_log.clear()
    provider3 = ApprovalDemoProvider(
        [
            ("close_account", {"customer_id": "C-123", "reason": "requested"}),
            ("", {}),
        ]
    )

    agent3 = Agent(
        tools=all_tools,
        provider=provider3,
        config=AgentConfig(
            max_iterations=3,
            confirm_action=confirm_action,
        ),
    )

    result3 = agent3.run([Message(role=Role.USER, content="Close account C-123")])
    print(f"\n  Result:       {result3.content[:60]}")
    print(f"  Approvals:    {approval_log}")

    print("\n" + "=" * 70)
    print("  Key takeaways:")
    print("    - @tool(requires_approval=True) flags individual tools")
    print("    - confirm_action callback decides approve/deny at runtime")
    print("    - Safe tools never trigger the callback")
    print("    - Denied tools feed an error back to the LLM")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
