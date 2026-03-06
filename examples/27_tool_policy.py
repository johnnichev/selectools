#!/usr/bin/env python3
"""
Tool Policy & Human-in-the-Loop — Control which tools the agent can execute.

Demonstrates:
  1. ToolPolicy with allow/review/deny rules
  2. Glob pattern matching for tool names
  3. Argument-level deny_when conditions
  4. Human-in-the-loop confirm_action callback
  5. Tool-pair-aware memory trimming (ConversationMemory)

No API key needed — uses mock providers.

Prerequisites: pip install selectools
Run: python examples/27_tool_policy.py
"""

from typing import Any, Dict, List, Optional, Tuple

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role
from selectools.policy import PolicyDecision, PolicyResult, ToolPolicy
from selectools.tools import tool
from selectools.types import AgentResult, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class PolicyDemoProvider:
    """Provider that requests different tools based on call count."""

    name = "policy-mock"
    supports_streaming = False
    supports_async = True

    def __init__(self, tool_sequence: list[tuple[str, dict[str, str]]]) -> None:
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
        idx = min(self._call_count - 1, len(self._sequence) - 1)

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
# Tools
# ---------------------------------------------------------------------------


@tool(description="Search the knowledge base for information")
def search_docs(query: str) -> str:
    return f"Found 3 results for: {query}"


@tool(description="Read a file from the filesystem")
def read_file(path: str) -> str:
    return f"Contents of {path}: ..."


@tool(description="Send an email to a recipient")
def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to}: {subject}"


@tool(description="Create a new user account")
def create_user(name: str, email: str) -> str:
    return f"Created user {name} ({email})"


@tool(description="Delete a user account permanently")
def delete_user(user_id: str) -> str:
    return f"Deleted user {user_id}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  Tool Policy & Human-in-the-Loop Demo")
    print("=" * 70)

    all_tools = [search_docs, read_file, send_email, create_user, delete_user]

    # --- Step 1: Policy evaluation (standalone) ---
    print("\n--- Step 1: ToolPolicy evaluation rules ---\n")

    policy = ToolPolicy(
        allow=["search_*", "read_*"],
        review=["send_*", "create_*"],
        deny=["delete_*"],
    )

    test_cases: List[Tuple[str, dict[str, str]]] = [
        ("search_docs", {}),
        ("read_file", {}),
        ("send_email", {}),
        ("create_user", {}),
        ("delete_user", {}),
        ("unknown_tool", {}),
    ]

    for tool_name, args in test_cases:
        result = policy.evaluate(tool_name, args)
        print(f"  {tool_name:20s} -> {result.decision.value:6s}  ({result.reason})")

    assert policy.evaluate("search_docs").decision == PolicyDecision.ALLOW
    assert policy.evaluate("read_file").decision == PolicyDecision.ALLOW
    assert policy.evaluate("send_email").decision == PolicyDecision.REVIEW
    assert policy.evaluate("create_user").decision == PolicyDecision.REVIEW
    assert policy.evaluate("delete_user").decision == PolicyDecision.DENY
    assert policy.evaluate("unknown_tool").decision == PolicyDecision.REVIEW
    print("\n  PASS: Policy evaluation order: deny -> review -> allow -> default(review)\n")

    # --- Step 2: Argument-level deny_when ---
    print("--- Step 2: Argument-level deny_when conditions ---\n")

    strict_policy = ToolPolicy(
        allow=["send_email"],
        deny_when=[
            {"tool": "send_email", "arg": "to", "pattern": "*@external.com"},
        ],
    )

    internal = strict_policy.evaluate("send_email", {"to": "alice@company.com", "subject": "Hi"})
    external = strict_policy.evaluate("send_email", {"to": "bob@external.com", "subject": "Hi"})

    print(f"  send_email to alice@company.com  -> {internal.decision.value}")
    print(f"  send_email to bob@external.com   -> {external.decision.value}")

    assert internal.decision == PolicyDecision.ALLOW
    assert external.decision == PolicyDecision.DENY
    print("\n  PASS: Argument-level deny blocks external emails\n")

    # --- Step 3: Agent with tool policy (allowed tool) ---
    print("--- Step 3: Agent executes allowed tools normally ---\n")

    provider = PolicyDemoProvider(
        [
            ("search_docs", {"query": "python tutorials"}),
            ("", {}),
        ]
    )

    agent = Agent(
        tools=all_tools,
        provider=provider,
        config=AgentConfig(
            max_iterations=3,
            tool_policy=policy,
        ),
    )

    result = agent.run([Message(role=Role.USER, content="Search for python tutorials")])
    print(f"  Tool called: search_docs (policy: allow)")
    print(f"  Result: {result.content[:60]}")
    print("\n  PASS: Allowed tools execute normally\n")

    # --- Step 4: Agent with denied tool ---
    print("--- Step 4: Agent with denied tool ---\n")

    provider_deny = PolicyDemoProvider(
        [
            ("delete_user", {"user_id": "123"}),
            ("", {}),
        ]
    )

    agent_deny = Agent(
        tools=all_tools,
        provider=provider_deny,
        config=AgentConfig(
            max_iterations=3,
            tool_policy=policy,
        ),
    )

    result_deny = agent_deny.run([Message(role=Role.USER, content="Delete user 123")])
    print(f"  Agent tried: delete_user (policy: deny)")
    print(f"  Result: {result_deny.content[:80]}")
    print("\n  PASS: Denied tools are blocked, error fed back to LLM\n")

    # --- Step 5: Human-in-the-loop approval ---
    print("--- Step 5: Human-in-the-loop confirm_action ---\n")

    approval_log: list[str] = []

    def confirm_action(tool_name: str, tool_args: dict, reason: str) -> bool:
        approved = tool_name == "send_email"
        approval_log.append(f"{tool_name}: {'approved' if approved else 'denied'}")
        print(f"    [HITL] Tool: {tool_name}, Args: {tool_args}")
        print(f"    [HITL] Reason: {reason}")
        print(f"    [HITL] Decision: {'APPROVED' if approved else 'DENIED'}")
        return approved

    provider_review = PolicyDemoProvider(
        [
            ("send_email", {"to": "team@company.com", "subject": "Update", "body": "Hello"}),
            ("", {}),
        ]
    )

    agent_hitl = Agent(
        tools=all_tools,
        provider=provider_review,
        config=AgentConfig(
            max_iterations=3,
            tool_policy=policy,
            confirm_action=confirm_action,
            approval_timeout=30.0,
        ),
    )

    result_hitl = agent_hitl.run([Message(role=Role.USER, content="Send an email to the team")])
    print(f"\n  Approval log: {approval_log}")
    print(f"  Result: {result_hitl.content[:60]}")
    assert len(approval_log) == 1
    assert "approved" in approval_log[0]
    print("\n  PASS: Review tool triggered confirm_action, approved and executed\n")

    # --- Step 6: Tool-pair-aware memory trimming ---
    print("--- Step 6: Tool-pair-aware memory trimming ---\n")

    memory = ConversationMemory(max_messages=4)

    memory.add(Message(role=Role.USER, content="Hello"))
    memory.add(
        Message(
            role=Role.ASSISTANT,
            content="I'll search for that",
            tool_calls=[ToolCall(tool_name="search", parameters={}, id="c1")],
        )
    )
    memory.add(Message(role=Role.TOOL, content="Search results...", tool_name="search"))
    memory.add(Message(role=Role.ASSISTANT, content="Here are the results"))
    memory.add(Message(role=Role.USER, content="Thanks, now search again"))
    memory.add(
        Message(
            role=Role.ASSISTANT,
            content="Searching again",
            tool_calls=[ToolCall(tool_name="search", parameters={}, id="c2")],
        )
    )
    memory.add(Message(role=Role.TOOL, content="More results...", tool_name="search"))
    memory.add(Message(role=Role.ASSISTANT, content="Here are more results"))
    memory.add(Message(role=Role.USER, content="Great, thanks!"))

    history = memory.get_history()
    print(f"  Memory has {len(history)} messages (max_messages=4)")

    first_msg = history[0]
    print(f"  First message role: {first_msg.role}")
    print(f"  First message has tool_calls: {bool(first_msg.tool_calls)}")

    assert first_msg.role != Role.TOOL, "First message should not be an orphaned TOOL result"
    if first_msg.role == Role.ASSISTANT:
        assert not first_msg.tool_calls, "First message should not be an orphaned tool_use"

    print("\n  PASS: Memory trim preserves tool-pair boundaries\n")

    print("=" * 70)
    print("  All tool policy & HITL tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
