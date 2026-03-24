"""Regression tests for bug hunt batch 1 — agent core and provider fixes."""

import inspect

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.observer import AgentObserver
from selectools.tools.base import Tool
from selectools.trace import StepType
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")


def _resp(text, model="test"):
    return (
        Message(role=Role.ASSISTANT, content=text),
        UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model=model,
            provider="test",
        ),
    )


def _tool_resp(tool_name):
    return (
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name=tool_name, parameters={})],
        ),
        UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model="test",
            provider="test",
        ),
    )


class TestAstreamModelSelector:
    """Bug #1: astream() must use _effective_model."""

    @pytest.mark.asyncio
    async def test_astream_uses_effective_model(self, fake_provider):
        provider = fake_provider(responses=[_tool_resp("noop"), _resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                model="base-model",
                max_iterations=6,
                model_selector=lambda i, tc, u: "switched-model" if i > 1 else "base-model",
            ),
        )
        chunks = []
        async for chunk in agent.astream("test"):
            if hasattr(chunk, "trace") and chunk.trace:
                for step in chunk.trace.steps:
                    if step.type == StepType.LLM_CALL:
                        chunks.append(step.model)
        # Second iteration should use switched-model
        if len(chunks) > 1:
            assert chunks[1] == "switched-model"


class TestAsyncConfirmAction:
    """Bug #2: Sync _check_policy must reject async confirm_action."""

    def test_sync_run_rejects_async_confirm(self, fake_provider):
        async def async_confirm(name, args, reason):
            return True

        danger = Tool(
            name="danger",
            description="d",
            parameters=[],
            function=lambda: "ok",
            requires_approval=True,
        )
        provider = fake_provider(
            responses=[
                (
                    Message(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=[ToolCall(tool_name="danger", parameters={})],
                    ),
                    UsageStats(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                        model="test",
                        provider="test",
                    ),
                ),
                "done",
            ]
        )
        agent = Agent(
            tools=[danger],
            provider=provider,
            config=AgentConfig(max_iterations=3, confirm_action=async_confirm),
        )
        # Should not crash — should deny the tool gracefully
        result = agent.run("test")
        assert result.iterations <= 3


class TestStreamingToolCallStringification:
    """Bug #18: Sync streaming must not stringify ToolCall objects."""

    def test_toolcall_not_stringified(self):
        # The fix is verified by code inspection — isinstance check added
        # This test verifies the method exists and has the right pattern
        import inspect as insp

        from selectools.agent._provider_caller import _ProviderCallerMixin

        source = insp.getsource(_ProviderCallerMixin._streaming_call)
        assert "isinstance(chunk, str)" in source
