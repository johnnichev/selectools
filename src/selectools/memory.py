"""
Conversation memory management for maintaining multi-turn dialogue history.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .types import Message, Role


class ConversationMemory:
    """
    Maintains conversation history with configurable limits.

    Automatically manages message history across multiple agent interactions,
    implementing a sliding window that keeps the most recent messages when
    limits are exceeded.

    Example:
        >>> memory = ConversationMemory(max_messages=10)
        >>> agent = Agent(tools=[...], provider=provider, memory=memory)
        >>>
        >>> # First conversation turn
        >>> response1 = agent.run([Message(role=Role.USER, content="Hello")])
        >>>
        >>> # Second turn - history is automatically maintained
        >>> response2 = agent.run([Message(role=Role.USER, content="How are you?")])
        >>>
        >>> # Access full history
        >>> history = memory.get_history()
    """

    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize conversation memory with optional limits.

        Args:
            max_messages: Maximum number of messages to retain. When exceeded,
                oldest messages are removed. Default is 20.
            max_tokens: Optional maximum token count. When exceeded, oldest
                messages are removed. Requires messages to have token estimates.
                If None, token-based limiting is disabled.
        """
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._summary: Optional[str] = None
        self._last_trimmed: List[Message] = []

    def add(self, message: Message) -> None:
        """
        Add a single message to the conversation history.

        Automatically enforces configured limits by removing oldest messages
        if necessary.

        Args:
            message: The message to add to history.
        """
        self._messages.append(message)
        self._enforce_limits()

    def add_many(self, messages: List[Message]) -> None:
        """
        Add multiple messages to the conversation history.

        More efficient than calling add() multiple times as it only enforces
        limits once after all messages are added.

        Args:
            messages: List of messages to add to history.
        """
        self._messages.extend(messages)
        self._enforce_limits()

    def get_history(self) -> List[Message]:
        """
        Retrieve the complete conversation history.

        Returns:
            List of all messages in chronological order.
        """
        return list(self._messages)

    def get_recent(self, n: int) -> List[Message]:
        """
        Retrieve the N most recent messages.

        Args:
            n: Number of recent messages to retrieve.

        Returns:
            List of up to N most recent messages in chronological order.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        return self._messages[-n:] if len(self._messages) >= n else list(self._messages)

    def clear(self) -> None:
        """
        Clear all messages from the conversation history.

        Useful for starting a fresh conversation while reusing the same
        memory instance.
        """
        self._messages.clear()
        self._last_trimmed = []
        self._summary = None

    @property
    def summary(self) -> Optional[str]:
        """Current conversation summary produced by summarize-on-trim."""
        return self._summary

    @summary.setter
    def summary(self, value: Optional[str]) -> None:
        self._summary = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the conversation memory to a dictionary.

        Useful for debugging, logging, or persisting conversations.

        Returns:
            Dictionary containing configuration and all messages.
        """
        return {
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "message_count": len(self._messages),
            "messages": [msg.to_dict() for msg in self._messages],
            "summary": self._summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Reconstruct a ConversationMemory from a dictionary produced by to_dict().

        Restores config and messages without re-running ``_enforce_limits()``
        so the persisted state is preserved exactly.
        """
        mem = cls.__new__(cls)
        mem.max_messages = data["max_messages"]
        mem.max_tokens = data.get("max_tokens")
        mem._messages = [Message.from_dict(m) for m in data.get("messages", [])]
        mem._summary = data.get("summary")
        mem._last_trimmed = []
        mem._fix_tool_pair_boundary()
        return mem

    def _enforce_limits(self) -> None:
        """
        Enforce configured limits by removing oldest messages.

        Uses a tool-pair-aware sliding window: after trimming, the cut point
        is advanced past any orphaned tool results so the conversation always
        starts at a safe boundary (a user or system text message).

        Trimmed messages are stored in ``_last_trimmed`` for the agent to
        optionally summarize.
        """
        trimmed: List[Message] = []

        # Enforce message count limit
        if len(self._messages) > self.max_messages:
            excess = len(self._messages) - self.max_messages
            trimmed.extend(self._messages[:excess])
            self._messages = self._messages[excess:]

        # Enforce token count limit if configured
        if self.max_tokens is not None:
            while len(self._messages) > 1:
                total_tokens = sum(
                    getattr(msg, "estimate_tokens", lambda: len(msg.content or "") // 4)()
                    for msg in self._messages
                )

                if total_tokens <= self.max_tokens:
                    break

                trimmed.append(self._messages.pop(0))

        boundary_trimmed = self._fix_tool_pair_boundary()
        trimmed.extend(boundary_trimmed)
        self._last_trimmed = trimmed

    def _fix_tool_pair_boundary(self) -> List[Message]:
        """Advance past orphaned tool results / assistant tool_use messages.

        After a naive trim the first message might be a TOOL result whose
        matching ASSISTANT tool_use was dropped, or an ASSISTANT message
        whose tool results follow but the pair is incomplete.  Scan forward
        until we reach a safe starting point: a USER text message (without
        ``tool_call_id``) or a SYSTEM message.  Always keep at least one
        message.

        Returns:
            List of messages removed by boundary fixing.
        """
        removed: List[Message] = []
        while len(self._messages) > 1:
            first = self._messages[0]
            if first.role == Role.TOOL:
                removed.append(self._messages.pop(0))
                continue
            if first.role == Role.ASSISTANT and first.tool_calls:
                removed.append(self._messages.pop(0))
                continue
            break
        return removed

    def branch(self) -> "ConversationMemory":
        """Return an independent snapshot of this memory.

        The returned memory has the same configuration (``max_messages``,
        ``max_tokens``) and an independent copy of the current message list
        and summary.  Changes to either memory do not affect the other.

        Example::

            branch = agent.memory.branch()
            branch_agent = Agent(tools=agent.tools, provider=provider, memory=branch)
            result = branch_agent.run("What if we tried X instead?")
            # agent.memory is unchanged

        Returns:
            A new :class:`ConversationMemory` instance with an independent copy
            of the current state.
        """
        branched = ConversationMemory(
            max_messages=self.max_messages,
            max_tokens=self.max_tokens,
        )
        branched._messages = [
            (
                replace(
                    msg,
                    tool_calls=[
                        replace(tc, parameters=copy.deepcopy(tc.parameters))
                        for tc in msg.tool_calls
                    ],
                )
                if msg.tool_calls
                else msg
            )
            for msg in self._messages
        ]
        branched._summary = self._summary
        branched._last_trimmed = []
        return branched

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Always return True so memory object is truthy even when empty."""
        return True

    def __repr__(self) -> str:
        """Return a string representation of the memory state."""
        return (
            f"ConversationMemory(max_messages={self.max_messages}, "
            f"max_tokens={self.max_tokens}, "
            f"current_messages={len(self._messages)})"
        )


__all__ = ["ConversationMemory"]
