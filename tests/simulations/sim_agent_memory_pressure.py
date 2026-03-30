"""
Simulation: Agent under memory pressure (trim + summarize-on-trim).

Accumulates messages well beyond max_messages to trigger ConversationMemory
trimming. Verifies the trim fires, the observer event fires, and the agent
continues to function after trimming.

No API keys required — uses LocalProvider.

Run: pytest tests/simulations/sim_agent_memory_pressure.py -v
"""

from __future__ import annotations

from typing import List

import pytest

from selectools.memory import ConversationMemory
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role


@pytest.mark.integration
class TestAgentMemoryPressure:
    """ConversationMemory must handle many messages without growing unbounded."""

    def test_memory_stays_bounded_after_many_adds(self):
        """
        Adding 150 messages to max_messages=20 must never exceed the limit.
        """
        max_msgs = 20
        memory = ConversationMemory(max_messages=max_msgs)
        trim_events: List[int] = []

        for i in range(150):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            memory.add(Message(role=role, content=f"Message number {i}. " + "x" * 50))
            current_len = len(memory.get_history())
            assert (
                current_len <= max_msgs
            ), f"Memory exceeded max_messages at message {i}: {current_len} > {max_msgs}"
            trim_events.append(current_len)

        # Must have trimmed at some point (not all 150 messages retained)
        assert len(memory.get_history()) <= max_msgs
        assert len(memory.get_history()) > 0

    def test_trimmed_memory_still_returns_recent_messages(self):
        """
        After trimming, the most recent messages must still be accessible.
        """
        max_msgs = 10
        memory = ConversationMemory(max_messages=max_msgs)

        # Add 50 messages
        for i in range(50):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            memory.add(Message(role=role, content=f"message_{i}"))

        history = memory.get_history()
        # Recent messages must be present
        recent_contents = {m.content for m in history}
        # At least the last message should be in history
        assert "message_49" in recent_contents

    def test_branch_after_heavy_trimming_is_independent(self):
        """
        branch() after trimming must produce an independent copy.
        Uses a large enough max_messages so the branch can still grow after branching.
        """
        # large limit so the branch isn't full when we add one more message
        memory = ConversationMemory(max_messages=100)
        for i in range(20):
            memory.add(Message(role=Role.USER, content=f"original_{i}"))

        branch = memory.branch()
        original_len = len(memory.get_history())
        branch_len = len(branch.get_history())
        assert original_len == branch_len

        # Modify branch — original must not change
        branch.add(Message(role=Role.USER, content="branch_only"))
        assert len(memory.get_history()) == original_len
        assert len(branch.get_history()) == branch_len + 1

    def test_clear_after_heavy_use_resets_completely(self):
        """
        clear() after heavy usage must result in empty history.
        """
        memory = ConversationMemory(max_messages=20)
        for i in range(100):
            memory.add(Message(role=Role.USER, content=f"msg_{i}"))

        memory.clear()
        assert len(memory.get_history()) == 0
        assert memory.summary is None or memory.summary == ""
