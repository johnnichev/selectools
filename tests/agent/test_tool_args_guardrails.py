"""Agent-loop integration tests for tool-args guardrails (issue #158).

Output guardrails only ever saw ``response_msg.content``; tool-call
arguments flowed to execution unchecked. These tests pin the new opt-in
``GuardrailsPipeline.tool_args`` stage across run(), arun(), and astream().
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    GuardrailsPipeline,
)
from selectools.trace import StepType
from selectools.types import AgentResult, Message, Role, ToolCall
from tests.conftest import SharedToolCallProvider

RECEIVED_ARGS: List[Dict[str, Any]] = []


@tool(description="Echo tool used to capture the arguments it receives.")
def echo(payload: str) -> str:
    RECEIVED_ARGS.append({"payload": payload})
    return f"echoed: {payload}"


class _RedactSecret(Guardrail):
    name = "redact-secret"
    action = GuardrailAction.REWRITE

    def check(self, content: str) -> GuardrailResult:
        if "secret-token" in content:
            return GuardrailResult(
                passed=False,
                content=content.replace("secret-token", "[REDACTED]"),
                reason="secret found",
                guardrail_name=self.name,
            )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


class _BlockSecret(Guardrail):
    name = "block-secret"
    action = GuardrailAction.BLOCK

    def check(self, content: str) -> GuardrailResult:
        if "secret-token" in content:
            return GuardrailResult(passed=False, content=content, reason="secret found")
        return GuardrailResult(passed=True, content=content)


def _tool_call_provider() -> SharedToolCallProvider:
    tc = ToolCall(tool_name="echo", parameters={"payload": "use secret-token now"}, id="tc1")
    return SharedToolCallProvider([([tc], "")])


def _agent(guardrails: GuardrailsPipeline, provider: Any = None) -> Agent:
    return Agent(
        [echo],
        provider=provider or _tool_call_provider(),
        config=AgentConfig(guardrails=guardrails, model="fake-model"),
    )


@pytest.fixture(autouse=True)
def _clear_received() -> None:
    RECEIVED_ARGS.clear()


class TestRunPath:
    def test_rewrite_sanitizes_args_before_execution(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]))
        agent.run("go")
        assert RECEIVED_ARGS == [{"payload": "use [REDACTED] now"}]

    def test_block_raises_guardrail_error(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_BlockSecret()]))
        with pytest.raises(GuardrailError, match="secret found"):
            agent.run("go")
        assert RECEIVED_ARGS == []

    def test_no_tool_args_guardrails_leaves_args_untouched(self) -> None:
        agent = _agent(GuardrailsPipeline(output=[_RedactSecret()]))
        agent.run("go")
        assert RECEIVED_ARGS == [{"payload": "use secret-token now"}]

    def test_trace_records_guardrail_step(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]))
        result = agent.run("go")
        assert result.trace is not None
        guardrail_steps = [s for s in result.trace.steps if s.type == StepType.GUARDRAIL]
        assert any("redact-secret" in (s.summary or "") for s in guardrail_steps)

    def test_result_tool_args_reflect_sanitized_values(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]))
        result = agent.run("go")
        assert result.tool_args == {"payload": "use [REDACTED] now"}


class TestAsyncPaths:
    @pytest.mark.asyncio
    async def test_arun_rewrite_sanitizes_args(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]))
        await agent.arun("go")
        assert RECEIVED_ARGS == [{"payload": "use [REDACTED] now"}]

    @pytest.mark.asyncio
    async def test_arun_block_raises(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_BlockSecret()]))
        with pytest.raises(GuardrailError):
            await agent.arun("go")
        assert RECEIVED_ARGS == []

    @pytest.mark.asyncio
    async def test_astream_rewrite_sanitizes_args(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]))
        async for chunk in agent.astream("go"):
            if isinstance(chunk, AgentResult):
                break
        assert RECEIVED_ARGS == [{"payload": "use [REDACTED] now"}]

    @pytest.mark.asyncio
    async def test_astream_block_raises(self) -> None:
        agent = _agent(GuardrailsPipeline(tool_args=[_BlockSecret()]))
        with pytest.raises(GuardrailError):
            async for _chunk in agent.astream("go"):
                pass
        assert RECEIVED_ARGS == []


class TestParserExtractedToolCalls:
    def test_text_parsed_tool_calls_are_guarded_too(self) -> None:
        """Tool calls extracted by the text ToolCallParser get the same screening."""
        text_response = Message(
            role=Role.ASSISTANT,
            content=(
                'TOOL_CALL\n{"tool": "echo", "parameters": {"payload": "use secret-token now"}}'
            ),
        )
        from tests.conftest import SharedFakeProvider

        provider = SharedFakeProvider([text_response, "Done"])
        agent = _agent(GuardrailsPipeline(tool_args=[_RedactSecret()]), provider=provider)
        agent.run("go")
        assert RECEIVED_ARGS == [{"payload": "use [REDACTED] now"}]
