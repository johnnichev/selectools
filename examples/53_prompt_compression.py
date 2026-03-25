#!/usr/bin/env python3
"""
Prompt Compression — prevent context-window overflow in long conversations.

Demonstrates:
- compress_context=True — enable proactive context compression
- compress_threshold=0.75 — trigger when context is 75 % full
- compress_keep_recent=4 — keep last N turns verbatim
- PROMPT_COMPRESSED step in AgentTrace
- Observer event: on_prompt_compressed(run_id, before_tokens, after_tokens, count)
- Memory vs history: self.memory is never modified; only the per-call view

How it works:
  Before each LLM call, selectools estimates the token count.  If the fill
  rate exceeds compress_threshold, older messages are summarised into a single
  [Compressed context] system message.  Recent turns (compress_keep_recent)
  are always kept verbatim so the LLM retains immediate context.

Prerequisites:
    pip install selectools
    # Uses a stub provider — no API keys needed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from unittest.mock import patch

from selectools import Agent, AgentConfig, Message, Role, Tool, UsageStats
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.token_estimation import TokenEstimate
from selectools.trace import StepType

# ---------------------------------------------------------------------------
# Stub provider (no real API calls)
# ---------------------------------------------------------------------------


class StubProvider:
    """Returns scripted responses in order."""

    name = "stub"
    supports_streaming = False
    supports_async = False

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._idx = 0

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        usage = UsageStats(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cost_usd=0.0,
            model=model,
            provider="stub",
        )
        return Message(role=Role.ASSISTANT, content=text), usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_tool() -> Tool:
    return Tool(name="noop", description="no-op", parameters=[], function=lambda: "ok")


def _make_estimate(total_tokens: int) -> TokenEstimate:
    ctx = 100_000
    return TokenEstimate(
        0, total_tokens, 0, total_tokens, ctx, ctx - total_tokens, "stub", "heuristic"
    )


def _fake_history(n_turns: int) -> List[Message]:
    """n_turns user/assistant pairs."""
    msgs: List[Message] = []
    for i in range(n_turns):
        msgs.append(Message(role=Role.USER, content=f"Tell me about topic {i}."))
        msgs.append(Message(role=Role.ASSISTANT, content=f"Topic {i} is about X, Y, and Z."))
    return msgs


class CompressionObserver(AgentObserver):
    """Records on_prompt_compressed events."""

    def __init__(self) -> None:
        self.events: list = []

    def on_prompt_compressed(
        self,
        run_id: str,
        before_tokens: int,
        after_tokens: int,
        messages_compressed: int,
    ) -> None:
        self.events.append(
            {
                "before": before_tokens,
                "after": after_tokens,
                "compressed": messages_compressed,
            }
        )
        print(
            f"  [observer] Compressed {messages_compressed} messages: "
            f"{before_tokens:,} → {after_tokens:,} tokens"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _separator(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def main() -> None:
    print("=== Prompt Compression Demo ===")

    # ------------------------------------------------------------------ #
    # 1. Disabled by default
    # ------------------------------------------------------------------ #
    _separator("1. Disabled by default")

    agent = Agent(
        tools=[_noop_tool()],
        provider=StubProvider(["hello"]),
        config=AgentConfig(model="gpt-4o-mini"),  # compress_context defaults to False
    )
    result = agent.run("hi")
    compressed = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    print(f"  compress_context default: {agent.config.compress_context}")
    print(f"  PROMPT_COMPRESSED steps:  {len(compressed)}  (expected 0)")

    # ------------------------------------------------------------------ #
    # 2. Fires when context is nearly full
    # ------------------------------------------------------------------ #
    _separator("2. Compression fires when threshold exceeded")

    observer = CompressionObserver()
    provider = StubProvider(["Summary of old messages.", "Final answer."])
    config = AgentConfig(
        model="gpt-4o-mini",
        compress_context=True,
        compress_threshold=0.85,  # trigger at 85 % fill
        compress_keep_recent=1,  # keep last 1 turn verbatim
        observers=[observer],
    )
    agent2 = Agent(tools=[_noop_tool()], provider=provider, config=config)

    # Pre-load 5 turns of history
    agent2.memory = ConversationMemory(max_messages=50)
    for msg in _fake_history(5):
        agent2.memory.add(msg)

    print(f"  Memory messages before run: {len(agent2.memory)}")
    print(f"  Threshold: {config.compress_threshold} (85 %)")

    # Patch estimate so the first call looks near-full (90 k / 100 k = 90 %)
    def _toggle_estimate(*args, **kwargs):
        if not hasattr(_toggle_estimate, "_called"):
            _toggle_estimate._called = True
            return _make_estimate(90_000)
        return _make_estimate(2_000)

    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens", side_effect=_toggle_estimate
    ):
        result2 = agent2.run("What have we covered so far?")

    compressed_steps = [s for s in result2.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    print(f"\n  PROMPT_COMPRESSED steps: {len(compressed_steps)}")
    if compressed_steps:
        step = compressed_steps[0]
        print(f"  Before → After: {step.prompt_tokens:,} → {step.completion_tokens:,} tokens")
        print(f"  Summary: {step.summary}")
    print(f"  Memory UNCHANGED: {len(agent2.memory)} messages (history view was compressed)")

    # ------------------------------------------------------------------ #
    # 3. keep_recent preserves recent turns
    # ------------------------------------------------------------------ #
    _separator("3. compress_keep_recent controls verbatim window")

    print(
        """
  config = AgentConfig(
      compress_context=True,
      compress_threshold=0.75,  # trigger at 75 % fill
      compress_keep_recent=4,   # always keep last 4 turns verbatim
  )

  With a 10-turn history:
    - Turns 0-5  → summarised into [Compressed context] system message
    - Turns 6-9  → kept verbatim (last 4 turns × 2 messages each = 8 msgs)

  Effect: LLM sees full recent context + condensed summary of the rest.
    """
    )

    # ------------------------------------------------------------------ #
    # 4. Observer integration
    # ------------------------------------------------------------------ #
    _separator("4. Observer event: on_prompt_compressed")

    print(
        """
  class MyObserver(AgentObserver):
      def on_prompt_compressed(
          self,
          run_id: str,
          before_tokens: int,
          after_tokens: int,
          messages_compressed: int,
      ) -> None:
          reduction = 1 - after_tokens / before_tokens
          print(f"Compressed {messages_compressed} messages, "
                f"{reduction:.0%} token reduction")

  agent = Agent(
      tools=[...],
      config=AgentConfig(
          compress_context=True,
          observers=[MyObserver()],
      ),
  )
    """
    )

    # ------------------------------------------------------------------ #
    # 5. Configuration reference
    # ------------------------------------------------------------------ #
    _separator("5. Configuration defaults")

    cfg = AgentConfig()
    print(f"  compress_context  = {cfg.compress_context}  (disabled by default)")
    print(f"  compress_threshold = {cfg.compress_threshold}  (trigger at 75 % fill)")
    print(f"  compress_keep_recent = {cfg.compress_keep_recent}  (keep last 4 turns verbatim)")

    print("\nDone.")


if __name__ == "__main__":
    main()
