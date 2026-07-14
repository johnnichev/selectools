"""Public contract tests for final_turn_only streaming + should_finalize (issue #174).

Consumers of `final_turn_only` in streaming chat surfaces rely on two
behaviors that were previously source-only knowledge. These tests ARE the
contract — a refactor that breaks them breaks integrators:

1. With ``final_turn_only=True``: the SYNTHESIS call never streams
   (``structured_synthesis_start`` fires iff it happens); ``single_pass``
   suppresses content chunks entirely; and on the reuse path — where the
   converged loop answer already streamed as ordinary loop chunks —
   ``StreamChunk(event="structured_reuse")`` signals that those chunks WERE
   the structured answer.
2. ``should_finalize(messages, last_response_text)`` receives the run's
   conversation view at convergence, including this turn's TOOL messages
   with ``tool_name`` and ``tool_result`` populated.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest

from selectools import Agent, AgentConfig, StructuredOutputConfig, tool
from selectools.types import AgentResult, Message, Role, StreamChunk, ToolCall
from selectools.usage import UsageStats
from tests.conftest import SharedFakeProvider


class NativeWithToolsProvider(SharedFakeProvider):
    """Fake advertising tools+json_schema support; accepts response_format."""

    supports_native_structured_output = True
    supports_native_structured_output_with_tools = True

    def complete(self, **kwargs: Any) -> "tuple[Message, UsageStats]":
        kwargs.pop("response_format", None)
        return super().complete(**kwargs)

    async def acomplete(self, **kwargs: Any) -> "tuple[Message, UsageStats]":
        kwargs.pop("response_format", None)
        return await super().acomplete(**kwargs)

    async def astream(self, **kwargs: Any):
        kwargs.pop("response_format", None)
        async for chunk in super().astream(**kwargs):
            yield chunk


def _tool_call_message() -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name="lookup", parameters={"query": "x"}, id="tc1")],
    )


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
JSON_ANSWER = '{"answer": "42"}'


@tool(description="A lookup tool.")
def lookup(query: str) -> str:
    return json.dumps({"found": query})


def _agent(provider: Any, **structured: Any) -> Agent:
    return Agent(
        [lookup],
        provider=provider,
        config=AgentConfig(
            model="fake-model",
            structured_output=StructuredOutputConfig(final_turn_only=True, **structured),
        ),
    )


async def _collect(agent: Agent, **kwargs: Any) -> tuple:
    chunks: List[StreamChunk] = []
    result: Optional[AgentResult] = None
    async for item in agent.astream("go", **kwargs):
        if isinstance(item, StreamChunk):
            chunks.append(item)
        else:
            result = item
    return chunks, result


class TestStreamingContract:
    @pytest.mark.asyncio
    async def test_single_pass_never_streams_json_content(self) -> None:
        """single_pass + astream: the structured answer must NOT leak as
        content chunks — it arrives only on the terminal AgentResult."""
        provider = NativeWithToolsProvider([_tool_call_message(), JSON_ANSWER])
        agent = _agent(provider, single_pass=True)
        chunks, result = await _collect(agent, response_format=SCHEMA)

        streamed = "".join(c.content or "" for c in chunks)
        assert "answer" not in streamed, f"JSON leaked into chunks: {streamed!r}"
        assert result is not None
        assert result.parsed == {"answer": "42"}
        assert result.structured_status == "ok"
        assert provider.calls == 2, "single_pass still avoids the synthesis call"

    @pytest.mark.asyncio
    async def test_single_pass_still_streams_tool_call_chunks(self) -> None:
        """Suppression covers content only — tool-call activity still streams."""
        provider = NativeWithToolsProvider([_tool_call_message(), JSON_ANSWER])
        agent = _agent(provider, single_pass=True)
        chunks, _result = await _collect(agent, response_format=SCHEMA)
        assert any(c.tool_calls for c in chunks), "tool-call chunks must still be visible"

    @pytest.mark.asyncio
    async def test_synthesis_path_prose_streams_json_does_not(self) -> None:
        """Non-single_pass: loop prose streams as chunks; the synthesis JSON
        never appears in any chunk."""
        provider = SharedFakeProvider([_tool_call_message(), "Here is your answer.", JSON_ANSWER])
        agent = _agent(provider)
        chunks, result = await _collect(agent, response_format=SCHEMA)

        streamed = "".join(c.content or "" for c in chunks)
        assert "Here is your answer." in streamed
        assert JSON_ANSWER not in streamed
        assert result is not None and result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_synthesis_start_event_fires_iff_synthesis_call_made(self) -> None:
        # Case A: prose answer -> synthesis call -> event fires
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _agent(provider)
        chunks, _ = await _collect(agent, response_format=SCHEMA)
        assert sum(1 for c in chunks if c.event == "structured_synthesis_start") == 1

        # Case B: reuse path (loop answer validates) -> no synthesis event,
        # but the structured_reuse signal fires — and the answer HAS already
        # streamed as ordinary loop chunks (documented caveat, pinned honest).
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER])
        agent = _agent(provider)
        chunks, result = await _collect(agent, response_format=SCHEMA)
        assert all(c.event != "structured_synthesis_start" for c in chunks)
        assert sum(1 for c in chunks if c.event == "structured_reuse") == 1
        streamed = "".join(c.content or "" for c in chunks)
        assert JSON_ANSWER in streamed, "reuse-path answer streams as loop output"
        assert result is not None and result.parsed == {"answer": "42"}

        # Case C: single_pass -> no synthesis call -> no event
        provider = NativeWithToolsProvider([_tool_call_message(), JSON_ANSWER])
        agent = _agent(provider, single_pass=True)
        chunks, _ = await _collect(agent, response_format=SCHEMA)
        assert all(c.event != "structured_synthesis_start" for c in chunks)

    @pytest.mark.asyncio
    async def test_event_chunks_carry_no_content(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _agent(provider)
        chunks, _ = await _collect(agent, response_format=SCHEMA)
        for c in chunks:
            if c.event:
                assert not c.content, "event chunks must be content-free"


class TestShouldFinalizeMessagesContract:
    def test_predicate_sees_tool_messages_with_name_and_result(self) -> None:
        seen: List[Message] = []

        def predicate(messages: List[Message], text: str) -> bool:
            seen.extend(messages)
            return False

        provider = SharedFakeProvider([_tool_call_message(), "converged prose"])
        agent = _agent(provider, should_finalize=predicate)
        agent.run("go", response_format=SCHEMA)

        tool_msgs = [m for m in seen if m.role == Role.TOOL]
        assert tool_msgs, "this turn's tool results must be visible to the predicate"
        for m in tool_msgs:
            assert m.tool_name == "lookup"
            assert m.tool_result, "tool_result must be populated on TOOL messages"
            assert json.loads(m.tool_result) == {"found": "x"}

    @pytest.mark.asyncio
    async def test_predicate_sees_tool_messages_in_astream(self) -> None:
        seen: List[Message] = []

        def predicate(messages: List[Message], text: str) -> bool:
            seen.extend(messages)
            return False

        provider = SharedFakeProvider([_tool_call_message(), "converged prose"])
        agent = _agent(provider, should_finalize=predicate)
        async for _item in agent.astream("go", response_format=SCHEMA):
            pass

        tool_msgs = [m for m in seen if m.role == Role.TOOL]
        assert tool_msgs
        assert all(m.tool_name == "lookup" and m.tool_result for m in tool_msgs)

    def test_predicate_sees_user_and_assistant_turns(self) -> None:
        seen: List[Message] = []

        def predicate(messages: List[Message], text: str) -> bool:
            seen.extend(messages)
            return False

        provider = SharedFakeProvider([_tool_call_message(), "converged prose"])
        agent = _agent(provider, should_finalize=predicate)
        agent.run("find x please", response_format=SCHEMA)

        roles = [m.role for m in seen]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles, "the tool_call assistant message is part of the view"
        user_msgs = [m for m in seen if m.role == Role.USER]
        assert any("find x please" in (m.content or "") for m in user_msgs)


class TestToolResultFieldInvariant:
    def test_every_tool_message_in_history_has_tool_result(self) -> None:
        """The invariant behind the predicate contract: TOOL messages always
        carry tool_result (defaulting to their content)."""
        first = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name="lookup", parameters={"query": "x"}, id="t1")],
        )
        provider = SharedFakeProvider([first, "Done"])
        agent = Agent([lookup], provider=provider, config=AgentConfig(model="fake-model"))
        agent.run("go")
        tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
        assert tool_msgs
        for m in tool_msgs:
            assert m.tool_name
            assert m.tool_result is not None

    def test_parallel_path_tool_messages_have_tool_result(self) -> None:
        @tool(description="Second lookup.")
        def lookup_b(query: str) -> str:
            return f"b:{query}"

        first = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(tool_name="lookup", parameters={"query": "x"}, id="t1"),
                ToolCall(tool_name="lookup_b", parameters={"query": "y"}, id="t2"),
            ],
        )
        provider = SharedFakeProvider([first, "Done"])
        agent = Agent(
            [lookup, lookup_b],
            provider=provider,
            config=AgentConfig(model="fake-model", parallel_tool_execution=True),
        )
        agent.run("go")
        tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        for m in tool_msgs:
            assert m.tool_result is not None, f"{m.tool_name} missing tool_result"

    def test_policy_denied_tool_messages_carry_tool_result(self) -> None:
        from selectools.policy import ToolPolicy

        first = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name="lookup", parameters={"query": "x"}, id="t1")],
        )
        provider = SharedFakeProvider([first, "Done"])
        agent = Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(model="fake-model", tool_policy=ToolPolicy(deny=["lookup"])),
        )
        agent.run("go")
        tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
        assert tool_msgs, "the denial must still produce a TOOL message"
        for m in tool_msgs:
            assert m.tool_result is not None, "policy denials must populate tool_result"

    def test_legacy_persisted_tool_messages_normalized_on_load(self) -> None:
        """Sessions saved before v1.2.x stored tool_result=None on error
        TOOL messages; from_dict must normalize so the contract holds for
        restored histories."""
        legacy = {
            "role": "tool",
            "content": "Error executing tool 'x': boom",
            "tool_name": "x",
            "tool_result": None,
            "tool_call_id": "t1",
        }
        restored = Message.from_dict(legacy)
        assert restored.tool_result == "Error executing tool 'x': boom"

    def test_error_tool_messages_also_carry_tool_result(self) -> None:
        @tool(description="Always fails.")
        def broken(query: str) -> str:
            raise ValueError("boom")

        first = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name="broken", parameters={"query": "x"}, id="t1")],
        )
        provider = SharedFakeProvider([first, "Done"])
        agent = Agent([broken], provider=provider, config=AgentConfig(model="fake-model"))
        agent.run("go")
        tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
        assert tool_msgs
        for m in tool_msgs:
            assert m.tool_result is not None, "error results must also populate tool_result"
