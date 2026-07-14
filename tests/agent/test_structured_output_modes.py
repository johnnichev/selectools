"""Agent-loop tests for structured-output modes (issue #159).

Pins the three behaviors introduced by ``StructuredOutputConfig``:

- ``native=True``: when the provider advertises native structured-output
  support, the JSON schema is sent to the provider instead of injected into
  the system prompt.
- ``final_turn_only=True``: the schema stays out of tool-loop turns entirely;
  after the loop converges, one synthesis call (no tools) produces the
  schema-validated final answer. Works from run(), arun(), and astream() —
  and in astream() the synthesis JSON is NOT leaked as content chunks.
- Defaults preserve v1.0 behavior for providers without native support.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union

import pytest

from selectools import Agent, AgentConfig, tool
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


class NativeFakeProvider(SharedFakeProvider):
    """Fake provider that advertises native structured output and records
    the response_format it receives per call."""

    supports_native_structured_output = True
    supports_native_structured_output_with_tools = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.response_formats: List[Optional[Dict[str, Any]]] = []

    def complete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Message, UsageStats]:
        self.response_formats.append(response_format)
        return super().complete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def acomplete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Message, UsageStats]:
        self.response_formats.append(response_format)
        return await super().acomplete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def astream(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        self.response_formats.append(response_format)
        async for chunk in super().astream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        ):
            yield chunk


class NativeNoToolsProvider(NativeFakeProvider):
    """Native support, but NOT alongside tools (Gemini-style)."""

    supports_native_structured_output_with_tools = False


def _tool_call_message(text: str = "") -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=text,
        tool_calls=[ToolCall(tool_name="lookup", parameters={"query": "x"}, id="tc1")],
    )


class TestConfigPlumbing:
    def test_defaults(self) -> None:
        config = AgentConfig()
        assert config.structured_output is not None
        assert config.structured_output.native is True
        assert config.structured_output.final_turn_only is False

    def test_dict_unpack(self) -> None:
        config = AgentConfig(structured_output={"final_turn_only": True})
        assert config.structured_output.final_turn_only is True

    def test_exported_from_selectools(self) -> None:
        from selectools import StructuredOutputConfig

        assert StructuredOutputConfig().native is True


class TestPromptInjectionFallback:
    """Providers without native support keep the v1.0 prompt-injection path."""

    def test_schema_instruction_still_injected(self) -> None:
        provider = SharedFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go", response_format=SCHEMA)
        assert "MUST respond with valid JSON" in provider.last_system_prompt
        assert result.parsed == {"answer": "42"}

    def test_native_flag_off_forces_injection_even_with_native_provider(self) -> None:
        provider = NativeFakeProvider([JSON_ANSWER])
        agent = Agent(
            [],
            provider=provider,
            config=AgentConfig(model="fake-model", structured_output={"native": False}),
        )
        agent.run("go", response_format=SCHEMA)
        assert "MUST respond with valid JSON" in provider.last_system_prompt
        assert provider.response_formats == [None]


class TestNativeMode:
    def test_native_provider_receives_schema_not_prompt(self) -> None:
        provider = NativeFakeProvider([JSON_ANSWER])
        agent = Agent([lookup], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go", response_format=SCHEMA)
        assert provider.response_formats == [SCHEMA]
        assert "MUST respond with valid JSON" not in provider.last_system_prompt
        assert result.parsed == {"answer": "42"}

    def test_native_without_tools_support_falls_back_when_agent_has_tools(self) -> None:
        provider = NativeNoToolsProvider([JSON_ANSWER])
        agent = Agent([lookup], provider=provider, config=AgentConfig(model="fake-model"))
        agent.run("go", response_format=SCHEMA)
        assert provider.response_formats == [None]
        assert "MUST respond with valid JSON" in provider.last_system_prompt

    def test_native_without_tools_support_engages_for_toolless_agent(self) -> None:
        provider = NativeNoToolsProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        agent.run("go", response_format=SCHEMA)
        assert provider.response_formats == [SCHEMA]

    def test_validation_retry_still_works_in_native_mode(self) -> None:
        provider = NativeFakeProvider(["not json at all", JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_arun_native_passes_schema(self) -> None:
        provider = NativeFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result = await agent.arun("go", response_format=SCHEMA)
        assert provider.response_formats == [SCHEMA]
        assert result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_astream_native_passes_schema(self) -> None:
        provider = NativeFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        result: Optional[AgentResult] = None
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, AgentResult):
                result = chunk
        assert provider.response_formats == [SCHEMA]
        assert result is not None and result.parsed == {"answer": "42"}


class TestFinalTurnOnly:
    def _agent(self, provider: Any) -> Agent:
        return Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(model="fake-model", structured_output={"final_turn_only": True}),
        )

    def test_run_schema_scoped_to_synthesis_call(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "The answer is 42.", JSON_ANSWER])
        agent = self._agent(provider)
        result = agent.run("go", response_format=SCHEMA)

        assert provider.calls == 3
        assert result.parsed == {"answer": "42"}
        assert result.content == JSON_ANSWER
        # Synthesis call carried the schema instruction; loop turns did not
        assert "MUST respond with valid JSON" in provider.last_system_prompt

    def test_loop_turns_have_no_schema_instruction(self) -> None:
        prompts: List[str] = []

        class SpyProvider(SharedFakeProvider):
            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                prompts.append(kwargs.get("system_prompt", ""))
                return super().complete(**kwargs)

        provider = SpyProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = self._agent(provider)
        agent.run("go", response_format=SCHEMA)

        assert len(prompts) == 3
        assert "MUST respond with valid JSON" not in prompts[0]
        assert "MUST respond with valid JSON" not in prompts[1]
        assert "MUST respond with valid JSON" in prompts[2]

    def test_synthesis_call_has_no_tools(self) -> None:
        tools_seen: List[Any] = []

        class SpyProvider(SharedFakeProvider):
            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                tools_seen.append(kwargs.get("tools"))
                return super().complete(**kwargs)

        provider = SpyProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = self._agent(provider)
        agent.run("go", response_format=SCHEMA)

        assert tools_seen[0], "loop turns keep tools"
        assert not tools_seen[-1], "synthesis call must not offer tools"

    def test_text_parser_active_during_loop_turns(self) -> None:
        """final_turn_only re-enables the TOOL_CALL text parser in the loop."""
        text_tool_call = 'TOOL_CALL\n{"tool": "lookup", "parameters": {"query": "x"}}'
        provider = SharedFakeProvider([text_tool_call, "prose", JSON_ANSWER])
        agent = self._agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed == {"answer": "42"}
        assert any(tc.tool_name == "lookup" for tc in result.tool_calls)

    def test_validation_retry_during_synthesis(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", "garbage", JSON_ANSWER])
        agent = self._agent(provider)
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed == {"answer": "42"}

    def test_synthesis_happens_even_at_iteration_budget_edge(self) -> None:
        provider = SharedFakeProvider(["prose answer", JSON_ANSWER])
        agent = Agent(
            [lookup],
            provider=provider,
            config=AgentConfig(
                model="fake-model",
                max_iterations=1,
                structured_output={"final_turn_only": True},
            ),
        )
        result = agent.run("go", response_format=SCHEMA)
        assert result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_arun_final_turn_only(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "prose", JSON_ANSWER])
        agent = self._agent(provider)
        result = await agent.arun("go", response_format=SCHEMA)
        assert result.parsed == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_astream_streams_prose_but_not_synthesis_json(self) -> None:
        provider = SharedFakeProvider([_tool_call_message(), "The answer is 42.", JSON_ANSWER])
        agent = self._agent(provider)

        streamed = ""
        result: Optional[AgentResult] = None
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, StreamChunk):
                streamed += chunk.content or ""
            elif isinstance(chunk, AgentResult):
                result = chunk

        assert result is not None
        assert result.parsed == {"answer": "42"}
        assert result.content == JSON_ANSWER
        assert "The answer is 42." in streamed
        assert JSON_ANSWER not in streamed, "synthesis JSON must not leak as chunks"


class TestDefaultStreamingBehaviorPinned:
    @pytest.mark.asyncio
    async def test_astream_default_mode_still_streams_json_content(self) -> None:
        """Without final_turn_only, the JSON answer streams as content (documented)."""
        provider = SharedFakeProvider([JSON_ANSWER])
        agent = Agent([], provider=provider, config=AgentConfig(model="fake-model"))
        streamed = ""
        async for chunk in agent.astream("go", response_format=SCHEMA):
            if isinstance(chunk, StreamChunk):
                streamed += chunk.content or ""
        assert json.loads(streamed) == {"answer": "42"}
