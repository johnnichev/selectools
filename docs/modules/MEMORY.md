# Memory Module

**File:** `src/selectools/memory.py`
**Classes:** `ConversationMemory`

## Table of Contents

1. [Overview](#overview)
2. [Memory Management](#memory-management)
3. [Integration with Agent](#integration-with-agent)
4. [Implementation](#implementation)
5. [Best Practices](#best-practices)

---

## Overview

The **ConversationMemory** class maintains dialogue history across multiple agent interactions, implementing a sliding window that keeps the most recent messages when limits are exceeded.

### Purpose

- **Multi-Turn Conversations**: Enable context retention across calls
- **Memory Management**: Prevent token limit explosions
- **History Access**: Retrieve conversation state for debugging/logging

---

## Memory Management

### Configuration

```python
memory = ConversationMemory(
    max_messages=20,    # Keep last 20 messages
    max_tokens=4000     # Optional token-based limit
)
```

### Sliding Window

```
Initial: []

Add: USER("Hello")
└─→ [USER("Hello")]

Add: ASSISTANT("Hi!")
└─→ [USER("Hello"), ASSISTANT("Hi!")]

Add: USER("What's 2+2?")
└─→ [USER("Hello"), ASSISTANT("Hi!"), USER("What's 2+2?")]

... continues until limit ...

At limit (max_messages=3):
[USER("Hello"), ASSISTANT("Hi!"), USER("What's 2+2?")]

Add: ASSISTANT("4")
└─→ Remove oldest: USER("Hello")
└─→ [ASSISTANT("Hi!"), USER("What's 2+2?"), ASSISTANT("4")]
```

### Implementation

```python
def _enforce_limits(self) -> None:
    # 1. Enforce message count limit
    if len(self._messages) > self.max_messages:
        excess = len(self._messages) - self.max_messages
        self._messages = self._messages[excess:]

    # 2. Enforce token count limit (if configured)
    if self.max_tokens is not None:
        while len(self._messages) > 1:  # Keep at least 1
            total_tokens = sum(
                estimate_tokens(msg.content)
                for msg in self._messages
            )

            if total_tokens <= self.max_tokens:
                break

            # Remove oldest message
            self._messages.pop(0)
```

---

## Integration with Agent

### With Memory

```python
from selectools import Agent, ConversationMemory, Message, Role

memory = ConversationMemory(max_messages=20)
agent = Agent(tools=[...], provider=provider, memory=memory)

# Turn 1
response1 = agent.run([
    Message(role=Role.USER, content="My name is Alice")
])

# Turn 2 - Context preserved
response2 = agent.run([
    Message(role=Role.USER, content="What's my name?")
])
# Agent knows: "Alice"
```

### Flow

```
run() called
    │
    ├─→ memory.get_history()
    │   └─→ Returns previous messages
    │
    ├─→ Append new user messages
    │
    ├─→ memory.add_many(new_messages)
    │
    ├─→ Execute agent loop
    │   ├─→ LLM sees full history
    │   ├─→ Tool calls append to history
    │   └─→ memory.add() for each message
    │
    ├─→ memory.add(final_response)
    │
    └─→ Return response
```

### Without Memory

```python
agent = Agent(tools=[...], provider=provider)  # No memory

# Each call is independent
response1 = agent.run([Message(role=Role.USER, content="My name is Alice")])
response2 = agent.run([Message(role=Role.USER, content="What's my name?")])
# Agent doesn't know - no memory
```

---

## Implementation

### Class Structure

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 20, max_tokens: Optional[int] = None):
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
```

### Core Methods

```python
def add(self, message: Message) -> None:
    """Add a single message to history."""
    self._messages.append(message)
    self._enforce_limits()

def add_many(self, messages: List[Message]) -> None:
    """Add multiple messages at once."""
    self._messages.extend(messages)
    self._enforce_limits()

def get_history(self) -> List[Message]:
    """Get full conversation history."""
    return list(self._messages)

def get_recent(self, n: int) -> List[Message]:
    """Get last N messages."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return self._messages[-n:] if len(self._messages) >= n else list(self._messages)

def clear(self) -> None:
    """Clear all messages."""
    self._messages.clear()
```

### Serialization

```python
def to_dict(self) -> Dict[str, Any]:
    """Serialize memory for logging/persistence."""
    return {
        "max_messages": self.max_messages,
        "max_tokens": self.max_tokens,
        "message_count": len(self._messages),
        "messages": [msg.to_dict() for msg in self._messages],
    }
```

---

## Best Practices

### 1. Choose Appropriate Limits

```python
# Short interactions (Q&A bot)
memory = ConversationMemory(max_messages=10)

# Standard conversations
memory = ConversationMemory(max_messages=20)

# Long-form dialogues
memory = ConversationMemory(max_messages=50)
```

### 2. Use Token Limits for Cost Control

```python
# Limit by tokens to prevent large prompts
memory = ConversationMemory(
    max_messages=100,     # High message count
    max_tokens=4000       # But limit tokens
)
```

### 3. Clear Memory Between Sessions

```python
# Start fresh conversation
memory.clear()
```

### 4. Access Recent Context

```python
# Get last 5 messages for display
recent = memory.get_recent(5)
for msg in recent:
    print(f"{msg.role}: {msg.content}")
```

### 5. Serialize for Persistence

```python
# Save conversation
import json
with open("conversation.json", "w") as f:
    json.dump(memory.to_dict(), f)

# Load conversation
with open("conversation.json", "r") as f:
    data = json.load(f)
    # Reconstruct memory from data
```

---

## Testing

```python
def test_memory_sliding_window():
    memory = ConversationMemory(max_messages=3)

    # Add 5 messages
    for i in range(5):
        memory.add(Message(role=Role.USER, content=f"Message {i}"))

    # Should only keep last 3
    history = memory.get_history()
    assert len(history) == 3
    assert history[0].content == "Message 2"
    assert history[2].content == "Message 4"

def test_memory_with_agent():
    memory = ConversationMemory(max_messages=10)
    agent = Agent(tools=[...], provider=LocalProvider(), memory=memory)

    # First turn
    agent.run([Message(role=Role.USER, content="Hello")])
    assert len(memory.get_history()) > 0

    # Second turn
    agent.run([Message(role=Role.USER, content="Goodbye")])
    assert len(memory.get_history()) > 1
```

---

## Common Pitfalls

### 1. Forgetting to Share Memory

```python
# ❌ Bad - Each agent has separate memory
agent1 = Agent(..., memory=ConversationMemory())
agent2 = Agent(..., memory=ConversationMemory())

# ✅ Good - Shared memory
memory = ConversationMemory()
agent1 = Agent(..., memory=memory)
agent2 = Agent(..., memory=memory)
```

### 2. Not Clearing Between Users

```python
# ❌ Bad - User A sees User B's history
def handle_user_a():
    agent.run([...])

def handle_user_b():
    agent.run([...])  # Sees User A's messages!

# ✅ Good - Clear between users
def handle_user(user_id):
    if user_id != previous_user:
        memory.clear()
    agent.run([...])
```

### 3. Setting Limits Too Low

```python
# ❌ Bad - Forgets context quickly
memory = ConversationMemory(max_messages=2)

# ✅ Good - Reasonable context
memory = ConversationMemory(max_messages=20)
```

---

## Future Enhancements

Potential improvements (not currently implemented):

1. **Summarization**: Auto-summarize old messages instead of dropping
2. **Importance-Based**: Keep important messages regardless of age
3. **Semantic Pruning**: Remove similar/redundant messages
4. **Persistence**: Auto-save/load from disk or database
5. **Compression**: Store messages in compact format

---

## Further Reading

- [Agent Module](AGENT.md) - How agents use memory
- [Types Module](../ARCHITECTURE.md#core-components) - Message data structure

---

**Next Steps:** Learn about usage tracking in the [Usage Module](USAGE.md).
