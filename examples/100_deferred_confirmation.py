#!/usr/bin/env python3
"""
Deferred confirmation flow for chat-channel destructive tools (issue #58).

The built-in approval gate blocks the agent loop waiting for a "yes" — wrong
for WhatsApp/Telegram/Slack agents, where the user's confirmation arrives as
a SEPARATE webhook turn. The pending module provides the out-of-loop shape:

1. Turn 1: the destructive tool does NOT execute. It stashes a pending
   action (preview + executor closure + TTL) via stash_pending() and returns
   a PendingConfirmation, so the LLM asks the user to confirm.
2. Turn 2: the next webhook goes through ChannelAgent.ask_channel(). A
   confirming reply ("sim", "yes", "sí") atomically claims and runs the
   stashed executor — the LLM is bypassed. A cancel ("não", "no") drops it.
   Anything else drops the pending and flows to the agent normally, so a
   casual later "yes" can never fire a stale destructive action.

Safety: the confirmation is bound to ONE exact side effect (args digest),
expires after a short TTL, is scoped to user/channel/conversation, and
duplicate webhook delivery executes ONCE.

No API key needed. Runs offline with a scripted provider.

Run: python examples/100_deferred_confirmation.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from selectools import Agent, AgentConfig, Message, Role, ToolCall, tool
from selectools.pending import (
    ChannelAgent,
    InMemoryPendingStore,
    PendingConfirmation,
    stash_pending,
)
from selectools.providers.base import Provider
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# A destructive tool that defers instead of executing
# ---------------------------------------------------------------------------

_EXPENSES: Dict[str, Dict[str, Any]] = {
    "exp-1": {"amount": 47.0, "description": "mercado"},
}


@tool()
def delete_expense(expense_id: str) -> PendingConfirmation:
    """Delete an expense (destructive — requires user confirmation)."""
    expense = _EXPENSES[expense_id]
    preview = f"Delete R${expense['amount']:.2f} — {expense['description']}"

    def execute() -> str:
        _EXPENSES.pop(expense_id, None)
        return f"Deleted: {preview}"

    # Stash the side effect; it runs only when a LATER turn confirms it.
    # The channel scope (store, user, conversation) is injected via the same
    # ContextVar mechanism emit_artifact() uses.
    stash_pending(
        kind="delete_expense",
        preview=preview,
        executor=execute,
        args={"expense_id": expense_id},  # digest-bound to THIS expense
    )
    return PendingConfirmation(
        action="delete_expense",
        preview=preview,
        user_prompt="Reply 'yes' to confirm or 'no' to cancel.",
    )


# ---------------------------------------------------------------------------
# Scripted offline provider
# ---------------------------------------------------------------------------

_DUMMY_USAGE = UsageStats(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    cost_usd=0.0,
    model="offline",
    provider="scripted",
)


class ScriptedProvider(Provider):
    """Calls delete_expense once, then relays the confirmation prompt."""

    name = "scripted"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self.default_model = "offline"
        self._calls = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._calls += 1
        if self._calls == 1:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="delete_expense",
                            parameters={"expense_id": "exp-1"},
                            id="call-1",
                        )
                    ],
                ),
                _DUMMY_USAGE,
            )
        if self._calls == 2:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="You're about to delete R$47.00 (mercado). Reply 'yes' to confirm.",
                ),
                _DUMMY_USAGE,
            )
        return (
            Message(role=Role.ASSISTANT, content="Nothing pending — how can I help?"),
            _DUMMY_USAGE,
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Simulated two-turn webhook flow
# ---------------------------------------------------------------------------


def main() -> None:
    agent = Agent(
        tools=[delete_expense],
        provider=ScriptedProvider(),
        config=AgentConfig(max_iterations=3),
    )
    store = InMemoryPendingStore()  # swap for RedisPendingStore in multi-instance
    channel = ChannelAgent(agent, store=store)

    user_id = "whatsapp:+5547999999999"

    # ---- Webhook turn 1: the destructive request --------------------------
    print("user> delete my mercado expense")
    first = channel.ask_channel(user_id, "delete my mercado expense")
    print(f"bot > {first.content}")
    pending = store.get(user_id)
    assert pending is not None and pending.status == "pending"
    assert "exp-1" in _EXPENSES, "nothing executed yet"
    print(
        f"      (pending: {pending.kind} | preview: {pending.preview!r} "
        f"| ttl until {pending.expires_at:.0f})"
    )

    # ---- Duplicate webhook delivery: only ONE execution --------------------
    # ---- Webhook turn 2: the user confirms in a separate turn --------------
    print("user> sim")
    second = channel.ask_channel(user_id, "sim")
    print(f"bot > {second.content}")
    assert "exp-1" not in _EXPENSES, "executor ran exactly here"

    print("user> sim   (webhook redelivery)")
    third = channel.ask_channel(user_id, "sim")
    print(f"bot > {third.content}")  # falls through: nothing pending anymore

    print("\nDone: previewed on turn 1, executed on turn 2, redelivery safe.")


if __name__ == "__main__":
    main()
