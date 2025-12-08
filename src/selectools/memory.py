"""
Conversation memory management for maintaining multi-turn dialogue history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import Message


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
        }

    def _enforce_limits(self) -> None:
        """
        Enforce configured limits by removing oldest messages.

        Implements a sliding window that keeps the most recent messages.
        Checks both message count and token count limits if configured.
        """
        # Enforce message count limit
        if len(self._messages) > self.max_messages:
            excess = len(self._messages) - self.max_messages
            self._messages = self._messages[excess:]

        # Enforce token count limit if configured
        if self.max_tokens is not None:
            while len(self._messages) > 1:  # Keep at least 1 message
                total_tokens = sum(
                    getattr(msg, "estimate_tokens", lambda: len(msg.content) // 4)()
                    for msg in self._messages
                )

                if total_tokens <= self.max_tokens:
                    break

                # Remove oldest message
                self._messages.pop(0)

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
