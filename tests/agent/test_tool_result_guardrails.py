"""Agent-loop integration tests for tool-results guardrails (issue #165).

tool_args (#160) gated a tool's incoming arguments; this stage gates the
RETURN value before it re-enters the model context — the other half of the
tool-boundary guardrail surface. Covers single and parallel execution across
run(), arun(), and astream().
"""

from __future__ import annotations

from typing import Any

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
from tests.conftest import SharedFakeProvider


@tool(description="Fetches remote content (returns a payload with a secret).")
def fetch(url: str) -> str:
    return f"fetched from {url}: secret-token payload"


@tool(description="Another fetch tool for parallel runs.")
def fetch_b(url: str) -> str:
    return f"B fetched {url} with secret-token"


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


def _one_call_provider() -> SharedFakeProvider:
    first = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name="fetch", parameters={"url": "http://x"}, id="tc1")],
    )
    return SharedFakeProvider([first, "Done"])


def _two_call_provider() -> SharedFakeProvider:
    first = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(tool_name="fetch", parameters={"url": "http://x"}, id="tc1"),
            ToolCall(tool_name="fetch_b", parameters={"url": "http://y"}, id="tc2"),
        ],
    )
    return SharedFakeProvider([first, "Done"])


def _agent(provider: Any, guardrails: GuardrailsPipeline, **cfg: Any) -> Agent:
    return Agent(
        [fetch, fetch_b],
        provider=provider,
        config=AgentConfig(guardrails=guardrails, model="fake-model", **cfg),
    )


def _tool_messages(provider: SharedFakeProvider) -> list:
    return [m for m in provider.last_messages if m.role == Role.TOOL]


class TestRunPath:
    def test_rewrite_sanitizes_result_before_history(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert len(tool_msgs) == 1
        assert "[REDACTED]" in tool_msgs[0].content
        assert "secret-token" not in tool_msgs[0].content

    def test_block_raises_guardrail_error(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_BlockSecret()]))
        with pytest.raises(GuardrailError, match="secret found"):
            agent.run("go")

    def test_no_tool_results_guardrails_passes_raw(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(output=[_RedactSecret()]))
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert "secret-token" in tool_msgs[0].content

    def test_parallel_execution_guards_all_results(self) -> None:
        provider = _two_call_provider()
        agent = _agent(
            provider,
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            parallel_tool_execution=True,
        )
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert len(tool_msgs) == 2
        for m in tool_msgs:
            assert "secret-token" not in m.content
            assert "[REDACTED]" in m.content

    def test_trace_records_guardrail_step(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        result = agent.run("go")
        steps = [s for s in result.trace.steps if s.type == StepType.GUARDRAIL]
        assert any("redact-secret" in (s.summary or "") for s in steps)


class TestAsyncPaths:
    @pytest.mark.asyncio
    async def test_arun_rewrite(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        await agent.arun("go")
        tool_msgs = _tool_messages(provider)
        assert "secret-token" not in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_arun_parallel_rewrite(self) -> None:
        provider = _two_call_provider()
        agent = _agent(
            provider,
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            parallel_tool_execution=True,
        )
        await agent.arun("go")
        for m in _tool_messages(provider):
            assert "secret-token" not in m.content

    @pytest.mark.asyncio
    async def test_arun_block_raises(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_BlockSecret()]))
        with pytest.raises(GuardrailError):
            await agent.arun("go")

    @pytest.mark.asyncio
    async def test_astream_rewrite(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        async for chunk in agent.astream("go"):
            if isinstance(chunk, AgentResult):
                break
        tool_msgs = _tool_messages(provider)
        assert "secret-token" not in tool_msgs[0].content
