"""Tests for the structured-output follow-ups (issues #164 and #166).

Pins four behaviors added on top of v1.1.0's StructuredOutputConfig:

- reuse_loop_answer (default True): in final_turn_only mode, when the loop's
  converged answer already validates against the schema, it is used directly
  and the extra synthesis call is skipped.
- single_pass (opt-in, #166): with a provider that supports tools+json_schema,
  the schema rides natively on the loop calls, so the converged answer IS the
  structured object — one pass, no synthesis call.
- should_finalize predicate (#164): when the converged answer does not
  validate, the caller can skip the synthesis call for turns that need no
  structured output.
- structured_status / structured_error on AgentResult (#164): "ok",
  "validation_failed", or "skipped" — no more guessing from a bare
  parsed=None.
"""

from __future__ import annotations

from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union

import pytest

from selectools import Agent, AgentConfig, StructuredOutputConfig, tool
from selectools.types import AgentResult, Message, Role, StreamChunk, ToolCall
from selectools.usage import UsageStats
from tests.conftest import SharedFakeProvider

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

JSON_ANSWER = '{"answer": "42"}'


@tool(description="A tool the loop can call.")
def lookup(query: str) -> str:
    return f"result for {query}"


class NativeWithToolsProvider(SharedFakeProvider):
    """Advertises tools+json_schema support; records per-call response_format."""

    supports_native_structured_output = True
    supports_native_structured_output_with_tools = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.response_formats: List[Optional[Dict[str, Any]]] = []

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.response_formats.append(kwargs.pop("response_format", None))
        return super().complete(**kwargs)

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.response_formats.append(kwargs.pop("response_format", None))
        return await super().acomplete(**kwargs)

    async def astream(self, **kwargs: Any) -> AsyncIterable[Union[str, ToolCall]]:
        self.response_formats.append(kwargs.pop("response_format", None))
        async for chunk in super().astream(**kwargs):
            yield chunk


def _tool_call_message() -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name="lookup", parameters={"query": "x"}, id="tc1")],
    )


def _final_turn_agent(provider: Any, **structured: Any) -> Agent:
    return Agent(
        [lookup],
        provider=provider,
        config=AgentConfig(
            model="fake-model",
            structured_output=StructuredOutputConfig(final_turn_only=True, **structured),
        ),
    )


class TestConfigDefaults:
    def test_new_fields_defaults(self) -> None:
        cfg = StructuredOutputConfig()
        assert cfg.reuse_loop_answer is True
        assert cfg.single_pass is False
        assert cfg.should_finalize is None


class TestStructuredStatus:
    def test_ok_status_on_success(self) -> None:
        provider = SharedFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go", response_format=SCHEMA)
        assert result.structured_status == "ok"
        assert result.structured_error is None

    def test_validation_failed_status_after_retry_exhaustion(self) -> None:
        provider = SharedFakeProvider(["not json at all"])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed is None
        assert result.structured_status == "validation_failed"
        assert result.structured_error

    def test_none_status_without_response_format(self) -> None:
        provider = SharedFakeProvider(["plain answer"])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go")
        assert result.structured_status is None
        assert result.structured_error is None

    def test_validation_failed_in_final_turn_only_synthesis(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", "garbage"])
        agent = _final_turn_agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed is None
        assert result.structured_status == "validation_failed"

    @pytest.mark.asyncio
    async def test_arun_ok_status(self) -> None:
        provider = SharedFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = await agent.arun("go", response_format=SCHEMA)
        assert result.structured_status == "ok"


class TestReuseLoopAnswer:
    def test_valid_loop_answer_skips_synthesis_call(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 2, "no synthesis call when the loop answer validates"
        assert result.parsed == {"answer": "42"}
        assert result.structured_status == "ok"

    def test_reuse_disabled_still_synthesizes(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER, JSON_ANSWER])
        agent = _final_turn_agent(provider, reuse_loop_answer=False)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3
        assert result.parsed == {"answer": "42"}

    def test_invalid_loop_answer_falls_back_to_synthesis(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose answer", JSON_ANSWER])
        agent = _final_turn_agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3
        assert result.parsed == {"answer": "42"}
        assert result.structured_status == "ok"

    @pytest.mark.asyncio
    async def test_astream_reuse_skips_synthesis(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider)
        result: Optional[AgentResult] = None
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, AgentResult):
                result = chunk
        assert provider.calls == 2
        assert result is not None and result.parsed == {"answer": "42"}


class TestShouldFinalize:
    def test_predicate_false_skips_synthesis(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "just a chat reply"])
        agent = _final_turn_agent(provider, should_finalize=lambda messages, text: False)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 2, "predicate False must skip the synthesis call"
        assert result.parsed is None
        assert result.structured_status == "skipped"
        assert result.content == "just a chat reply"

    def test_predicate_true_synthesizes(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _final_turn_agent(provider, should_finalize=lambda messages, text: True)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3
        assert result.parsed == {"answer": "42"}

    def test_predicate_receives_messages_and_text(self) -> None:
        seen: Dict[str, Any] = {}

        def predicate(messages: List[Message], text: str) -> bool:
            seen["n_messages"] = len(messages)
            seen["text"] = text
            return False

        provider = SharedFakeProvider([_tool_call_message(), "the converged reply"])
        agent = _final_turn_agent(provider, should_finalize=predicate)
        agent.run("go", response_format=SCHEMA)
        assert seen["text"] == "the converged reply"
        assert seen["n_messages"] >= 1

    def test_predicate_not_consulted_when_answer_validates(self) -> None:
        calls: List[str] = []

        def predicate(messages: List[Message], text: str) -> bool:
            calls.append(text)
            return False

        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider, should_finalize=predicate)
        result = agent.run("go", response_format=SCHEMA)
        assert result.structured_status == "ok"
        assert calls == [], "a validating answer wins before the predicate runs"

    @pytest.mark.asyncio
    async def test_arun_predicate_false_skips(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "chat reply"])
        agent = _final_turn_agent(provider, should_finalize=lambda m, t: False)
        result = await agent.arun("go", response_format=SCHEMA)
        assert provider.calls == 2
        assert result.structured_status == "skipped"

    @pytest.mark.asyncio
    async def test_astream_predicate_false_skips(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "chat reply"])
        agent = _final_turn_agent(provider, should_finalize=lambda m, t: False)
        result: Optional[AgentResult] = None
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, AgentResult):
                result = chunk
        assert provider.calls == 2
        assert result is not None and result.structured_status == "skipped"


class TestSinglePass:
    def test_single_pass_carries_schema_on_loop_calls(self) -> None:
        provider = NativeWithToolsProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider, single_pass=True)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.response_formats == [SCHEMA, SCHEMA]
        assert provider.calls == 2, "structured object comes from the final loop turn"
        assert result.parsed == {"answer": "42"}
        assert result.structured_status == "ok"

    def test_single_pass_ignored_without_with_tools_support(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _final_turn_agent(provider, single_pass=True)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3, "falls back to the separate synthesis call"
        assert result.parsed == {"answer": "42"}

    def test_single_pass_invalid_answer_falls_back_to_synthesis(self) -> None:
        provider = NativeWithToolsProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _final_turn_agent(provider, single_pass=True)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3
        assert result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_arun_single_pass(self) -> None:
        provider = NativeWithToolsProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider, single_pass=True)
        result = await agent.arun("go", response_format=SCHEMA)
        assert provider.calls == 2
        assert result.parsed == {"answer": "42"}


class TestSynthesisStreamSignal:
    @pytest.mark.asyncio
    async def test_astream_emits_synthesis_start_event(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = _final_turn_agent(provider)
        events: List[str] = []
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, StreamChunk) and chunk.event:
                events.append(chunk.event)
        assert "structured_synthesis_start" in events

    @pytest.mark.asyncio
    async def test_no_synthesis_event_when_synthesis_skipped(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER])
        agent = _final_turn_agent(provider)
        events: List[str] = []
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, StreamChunk) and chunk.event:
                events.append(chunk.event)
        assert "structured_synthesis_start" not in events
        # the reuse path signals that the streamed loop answer WAS the
        # structured answer (#174)
        assert events == ["structured_reuse"]


# ── Review follow-ups (PR #169 self-review) ─────────────────────────────


class TestReuseGateConservative:
    """reuse must only fire when the WHOLE converged answer is a validating
    JSON object — never on a fragment embedded in prose, and never on a
    dict-schema answer missing required keys."""

    def test_embedded_json_fragment_does_not_trigger_reuse(self) -> None:
        prose = 'The lookup returned {"answer": "x"} so I am done here.'
        provider = SharedFakeProvider([_tool_call_message(), prose, JSON_ANSWER])
        agent = _final_turn_agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3, "embedded fragments must go through synthesis"
        assert result.parsed == {"answer": "42"}

    def test_dict_schema_missing_required_keys_does_not_reuse(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), '{"other": 1}', JSON_ANSWER])
        agent = _final_turn_agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3, "answer missing required keys must synthesize"
        assert result.parsed == {"answer": "42"}

    def test_validating_answer_wins_even_with_reuse_disabled(self) -> None:
        """A schema-valid converged answer must never be 'skipped' by the
        predicate, regardless of reuse_loop_answer."""
        provider = SharedFakeProvider([_tool_call_message(), JSON_ANSWER, JSON_ANSWER])
        agent = _final_turn_agent(
            provider,
            reuse_loop_answer=False,
            should_finalize=lambda m, t: False,
        )
        result = agent.run("go", response_format=SCHEMA)
        assert provider.calls == 3, "reuse off: valid answer synthesizes, never skips"
        assert result.structured_status == "ok"
        assert result.parsed == {"answer": "42"}


class TestSinglePassParserGuard:
    def test_native_json_answer_is_not_hijacked_by_text_parser(self) -> None:
        """single_pass converged answers are native JSON; a schema whose keys
        look like a tool call (name/parameters) must not be intercepted by
        the ToolCallParser fallback."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "parameters": {"type": "object"}},
            "required": ["name", "parameters"],
        }
        answer = '{"name": "recommendation", "parameters": {"answer": "42"}}'
        provider = NativeWithToolsProvider([_tool_call_message(), answer])
        agent = _final_turn_agent(provider, single_pass=True)
        result = agent.run("go", response_format=schema)
        assert provider.calls == 2
        assert result.parsed == {"name": "recommendation", "parameters": {"answer": "42"}}
        assert not any(tc.tool_name == "recommendation" for tc in result.tool_calls), (
            "the JSON answer must not be executed as a tool call"
        )


class TestEarlyExitStatus:
    def test_max_iterations_reports_not_attempted(self) -> None:
        provider = SharedFakeProvider([_tool_call_message()])  # tools forever
        agent = Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(model="fake-model", max_iterations=2),
        )
        result = agent.run("go", response_format=SCHEMA)
        assert "Maximum iterations" in result.content
        assert result.structured_status == "not_attempted"

    def test_budget_exceeded_reports_not_attempted(self) -> None:
        provider = SharedFakeProvider([_tool_call_message()])
        agent = Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(model="fake-model", max_total_tokens=1),
        )
        result = agent.run("go", response_format=SCHEMA)
        assert "budget" in result.content.lower()
        assert result.structured_status == "not_attempted"

    def test_early_exit_without_response_format_stays_none(self) -> None:
        provider = SharedFakeProvider([_tool_call_message()])
        agent = Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(model="fake-model", max_iterations=2),
        )
        result = agent.run("go")
        assert result.structured_status is None
