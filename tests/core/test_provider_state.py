"""Message.provider_state: opaque provider payload that round-trips with the message."""

from __future__ import annotations

from selectools.cache import CacheKeyBuilder
from selectools.memory import ConversationMemory
from selectools.types import Message, Role, ToolCall

_STATE = {"items": [{"type": "reasoning", "encrypted_content": "abc"}]}


def test_provider_state_defaults_to_none() -> None:
    msg = Message(role=Role.ASSISTANT, content="hi")
    assert msg.provider_state is None


def test_provider_state_round_trips_through_to_dict_and_from_dict() -> None:
    msg = Message(role=Role.ASSISTANT, content="", provider_state=_STATE)

    restored = Message.from_dict(msg.to_dict())

    assert restored.provider_state == _STATE


def test_from_dict_without_provider_state_key_restores_none() -> None:
    restored = Message.from_dict({"role": "assistant", "content": "old payload"})
    assert restored.provider_state is None


def test_memory_copy_deep_copies_provider_state() -> None:
    original = Message(role=Role.ASSISTANT, content="", provider_state={"items": [1]})

    copied = ConversationMemory._copy_message(original)
    copied.provider_state["items"].append(2)  # type: ignore[index]

    assert original.provider_state == {"items": [1]}


def test_cache_key_ignores_missing_provider_state() -> None:
    plain = [Message(role=Role.USER, content="q")]
    assert CacheKeyBuilder.build(
        model="m", system_prompt="s", messages=plain
    ) == CacheKeyBuilder.build(
        model="m",
        system_prompt="s",
        messages=[Message(role=Role.USER, content="q", provider_state=None)],
    )


def test_cache_key_changes_with_provider_state() -> None:
    call = ToolCall(tool_name="t", parameters={}, id="c1")
    with_state = [Message(role=Role.ASSISTANT, tool_calls=[call], provider_state=_STATE)]
    without = [Message(role=Role.ASSISTANT, tool_calls=[call])]

    assert CacheKeyBuilder.build(
        model="m", system_prompt="s", messages=with_state
    ) != CacheKeyBuilder.build(model="m", system_prompt="s", messages=without)
