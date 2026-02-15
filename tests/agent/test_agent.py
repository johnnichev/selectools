"""
Tests for agent.py — AgentConfig, Agent.run(), Agent.arun(), Agent.reset().

Uses FakeProvider stubs — no API keys required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools import AgentResult, UsageStats
from selectools.memory import ConversationMemory
from selectools.types import ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal provider stub that returns queued responses."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.last_system_prompt = None

    def complete(
        self,
        *,
        model,
        system_prompt,
        messages,
        tools=None,
        temperature=0.0,
        max_tokens=1000,
        timeout=None,
    ):
        self.last_system_prompt = system_prompt
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            model=model or "fake",
            provider="fake",
        )
        if isinstance(response, str):
            response = Message(role=Role.ASSISTANT, content=response)
        return response, usage


def _noop_tool():
    return Tool(name="noop", description="no-op", parameters=[], function=lambda: "ok")


def _add_tool():
    def add(a: int, b: int) -> str:
        return json.dumps({"sum": a + b})

    return Tool(
        name="add",
        description="Add two integers",
        parameters=[
            ToolParameter(name="a", param_type=int, description="first"),
            ToolParameter(name="b", param_type=int, description="second"),
        ],
        function=add,
    )


# ===========================================================================
# Feature 1: Custom System Prompt Injection
# ===========================================================================


class TestCustomSystemPrompt:
    def test_default_system_prompt_used_when_not_set(self):
        """Default PromptBuilder instructions should be used when system_prompt is None."""
        provider = FakeProvider(responses=["Hello"])
        agent = Agent(tools=[_noop_tool()], provider=provider)
        agent.run([Message(role=Role.USER, content="hi")])

        assert "TOOL_CALL" in provider.last_system_prompt
        assert "tool call contract" in provider.last_system_prompt.lower()

    def test_custom_system_prompt_replaces_default(self):
        """When system_prompt is set in AgentConfig, it should replace the default instructions."""
        custom = "You are a helpful routing agent. Always route to the correct service."
        provider = FakeProvider(responses=["Routed!"])
        config = AgentConfig(system_prompt=custom, model="fake")
        agent = Agent(tools=[_noop_tool()], provider=provider, config=config)
        agent.run([Message(role=Role.USER, content="route this")])

        assert "routing agent" in provider.last_system_prompt
        assert "Always route to the correct service" in provider.last_system_prompt

    def test_custom_system_prompt_includes_tool_schemas(self):
        """Custom system prompt should still include tool schemas in the final prompt."""
        custom = "Custom instructions here."
        provider = FakeProvider(responses=["Done"])
        config = AgentConfig(system_prompt=custom, model="fake")
        tool = _add_tool()
        agent = Agent(tools=[tool], provider=provider, config=config)
        agent.run([Message(role=Role.USER, content="add")])

        # The system prompt should contain both the custom text AND the tool schemas
        assert "Custom instructions here" in provider.last_system_prompt
        assert "add" in provider.last_system_prompt
        assert "parameters" in provider.last_system_prompt

    def test_explicit_prompt_builder_takes_precedence(self):
        """If a PromptBuilder is passed directly, it should take precedence over system_prompt."""
        from selectools.prompt import PromptBuilder

        explicit_builder = PromptBuilder(base_instructions="Explicit builder wins.")
        provider = FakeProvider(responses=["Done"])
        config = AgentConfig(system_prompt="This should NOT appear", model="fake")
        agent = Agent(
            tools=[_noop_tool()],
            provider=provider,
            prompt_builder=explicit_builder,
            config=config,
        )
        agent.run([Message(role=Role.USER, content="test")])

        assert "Explicit builder wins" in provider.last_system_prompt
        assert "This should NOT appear" not in provider.last_system_prompt


# ===========================================================================
# Feature 2: Structured Decision Result (AgentResult)
# ===========================================================================


class TestAgentResult:
    def test_run_returns_agent_result(self):
        """run() should return an AgentResult, not a raw Message."""
        provider = FakeProvider(responses=["Hello there"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))
        result = agent.run([Message(role=Role.USER, content="hi")])

        assert isinstance(result, AgentResult)

    def test_backward_compatible_content_property(self):
        """result.content should work the same as the old result.content (Message.content)."""
        provider = FakeProvider(responses=["The answer is 42"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))
        result = agent.run([Message(role=Role.USER, content="question")])

        assert result.content == "The answer is 42"

    def test_backward_compatible_role_property(self):
        """result.role should return Role.ASSISTANT."""
        provider = FakeProvider(responses=["Done"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))
        result = agent.run([Message(role=Role.USER, content="test")])

        assert result.role == Role.ASSISTANT

    def test_no_tool_call_metadata(self):
        """When no tool is called, tool_name and tool_args should be None/empty."""
        provider = FakeProvider(responses=["Direct response"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))
        result = agent.run([Message(role=Role.USER, content="hi")])

        assert result.tool_name is None
        assert result.tool_args == {}
        assert result.tool_calls == []
        assert result.iterations == 1

    def test_tool_call_metadata_populated(self):
        """When a tool is called, AgentResult should contain full tool call metadata."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "add", "parameters": {"a": 2, "b": 3}}',
                "The sum is 5.",
            ]
        )
        agent = Agent(
            tools=[_add_tool()],
            provider=provider,
            config=AgentConfig(max_iterations=3, model="fake"),
        )
        result = agent.run([Message(role=Role.USER, content="Add 2 and 3")])

        assert result.content == "The sum is 5."
        assert result.tool_name == "add"
        assert result.tool_args == {"a": 2, "b": 3}
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "add"
        assert result.tool_calls[0].parameters == {"a": 2, "b": 3}

    def test_multiple_tool_calls_tracked(self):
        """When multiple tools are called, all should be tracked in tool_calls."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "add", "parameters": {"a": 1, "b": 2}}',
                'TOOL_CALL: {"tool_name": "add", "parameters": {"a": 3, "b": 4}}',
                "Results: 3 and 7.",
            ]
        )
        agent = Agent(
            tools=[_add_tool()],
            provider=provider,
            config=AgentConfig(max_iterations=5, model="fake"),
        )
        result = agent.run([Message(role=Role.USER, content="Add 1+2 and 3+4")])

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].parameters == {"a": 1, "b": 2}
        assert result.tool_calls[1].parameters == {"a": 3, "b": 4}
        # Last tool call details
        assert result.tool_name == "add"
        assert result.tool_args == {"a": 3, "b": 4}
        assert result.iterations == 3

    def test_message_attribute_is_accessible(self):
        """The underlying Message object should be accessible via .message."""
        provider = FakeProvider(responses=["Hello"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))
        result = agent.run([Message(role=Role.USER, content="hi")])

        assert isinstance(result.message, Message)
        assert result.message.content == "Hello"
        assert result.message.role == Role.ASSISTANT


# ===========================================================================
# Feature 3: Reusable Agent Instances
# ===========================================================================


class TestReusableAgent:
    def test_agent_reusable_without_reset(self):
        """Agent should work correctly across multiple run() calls even without reset()."""
        provider = FakeProvider(responses=["Response 1", "Response 2", "Response 3"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))

        r1 = agent.run([Message(role=Role.USER, content="First")])
        r2 = agent.run([Message(role=Role.USER, content="Second")])
        r3 = agent.run([Message(role=Role.USER, content="Third")])

        assert r1.content == "Response 1"
        assert r2.content == "Response 2"
        assert r3.content == "Response 3"

    def test_reset_clears_usage(self):
        """reset() should clear usage tracking."""
        provider = FakeProvider(responses=["Done", "Done again"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))

        agent.run([Message(role=Role.USER, content="First")])
        assert agent.total_tokens > 0

        agent.reset()
        assert agent.total_tokens == 0
        assert agent.total_cost == 0.0

    def test_reset_clears_history(self):
        """reset() should clear internal conversation history."""
        provider = FakeProvider(responses=["Done", "Done again"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))

        agent.run([Message(role=Role.USER, content="First")])
        assert len(agent._history) > 0

        agent.reset()
        assert len(agent._history) == 0

    def test_reset_clears_memory(self):
        """reset() should clear conversation memory if set."""
        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(responses=["Done"])
        agent = Agent(
            tools=[_noop_tool()], provider=provider, memory=memory, config=AgentConfig(model="fake")
        )

        agent.run([Message(role=Role.USER, content="Hello")])
        assert len(memory) > 0

        agent.reset()
        assert len(memory) == 0

    def test_reset_clears_analytics(self):
        """reset() should reset analytics when enabled."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "noop", "parameters": {}}',
                "Done",
            ]
        )
        config = AgentConfig(model="fake", enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[_noop_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="test")])
        analytics = agent.get_analytics()
        assert analytics is not None

        agent.reset()
        new_analytics = agent.get_analytics()
        assert new_analytics is not None
        # The analytics object should be a fresh instance
        assert new_analytics is not analytics

    def test_agent_works_after_reset(self):
        """Agent should produce correct results after a reset()."""
        provider = FakeProvider(responses=["First", "Second"])
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(model="fake"))

        r1 = agent.run([Message(role=Role.USER, content="A")])
        assert r1.content == "First"

        agent.reset()

        r2 = agent.run([Message(role=Role.USER, content="B")])
        assert r2.content == "Second"
        assert r2.iterations == 1


# ===========================================================================
# Async tests
# ===========================================================================


class FakeAsyncProvider:
    """Provider stub with async support."""

    name = "fake-async"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def acomplete(
        self,
        *,
        model,
        system_prompt,
        messages,
        tools=None,
        temperature=0.0,
        max_tokens=1000,
        timeout=None,
    ):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            model=model or "fake",
            provider="fake",
        )
        if isinstance(response, str):
            response = Message(role=Role.ASSISTANT, content=response)
        return response, usage

    def complete(self, **kwargs):
        raise RuntimeError("Should use acomplete")


@pytest.mark.asyncio
async def test_arun_returns_agent_result():
    """arun() should return AgentResult with the same structure as run()."""
    provider = FakeAsyncProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "noop", "parameters": {}}',
            "All done.",
        ]
    )
    agent = Agent(
        tools=[_noop_tool()],
        provider=provider,
        config=AgentConfig(max_iterations=3, model="fake"),
    )
    result = await agent.arun([Message(role=Role.USER, content="test")])

    assert isinstance(result, AgentResult)
    assert result.content == "All done."
    assert result.tool_name == "noop"
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
