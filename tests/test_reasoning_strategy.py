"""Tests for reasoning_strategy on AgentConfig (v0.17.6)."""

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.prompt import REASONING_STRATEGIES, PromptBuilder
from selectools.tools.base import Tool
from selectools.types import Message, Role
from selectools.usage import UsageStats

_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")


def _resp(text):
    return (
        Message(role=Role.ASSISTANT, content=text),
        UsageStats(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.0001,
            model="test",
            provider="test",
        ),
    )


# ── PromptBuilder unit tests ──────────────────────────────────────────


class TestPromptBuilderStrategy:
    def test_none_strategy_no_extra_text(self):
        pb = PromptBuilder(reasoning_strategy=None)
        prompt = pb.build([_DUMMY])
        for strategy_text in REASONING_STRATEGIES.values():
            assert strategy_text not in prompt

    def test_react_strategy_injected(self):
        pb = PromptBuilder(reasoning_strategy="react")
        prompt = pb.build([_DUMMY])
        assert "Thought → Action → Observation" in prompt
        assert "ReAct" in prompt

    def test_cot_strategy_injected(self):
        pb = PromptBuilder(reasoning_strategy="cot")
        prompt = pb.build([_DUMMY])
        assert "Chain of Thought" in prompt
        assert "step by step" in prompt

    def test_plan_then_act_strategy_injected(self):
        pb = PromptBuilder(reasoning_strategy="plan_then_act")
        prompt = pb.build([_DUMMY])
        assert "Plan Then Act" in prompt
        assert "numbered plan" in prompt

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid reasoning_strategy 'invalid'"):
            PromptBuilder(reasoning_strategy="invalid")

    def test_invalid_strategy_shows_valid_options(self):
        with pytest.raises(ValueError, match="cot"):
            PromptBuilder(reasoning_strategy="bad")

    def test_strategy_appended_after_tools(self):
        pb = PromptBuilder(reasoning_strategy="react")
        prompt = pb.build([_DUMMY])
        tool_pos = prompt.index("Available tools")
        strategy_pos = prompt.index("ReAct")
        assert strategy_pos > tool_pos

    def test_custom_base_instructions_with_strategy(self):
        pb = PromptBuilder(
            base_instructions="Custom instructions.",
            reasoning_strategy="cot",
        )
        prompt = pb.build([_DUMMY])
        assert "Custom instructions." in prompt
        assert "Chain of Thought" in prompt


# ── Agent integration tests ───────────────────────────────────────────


class TestAgentReasoningStrategy:
    def test_agent_default_no_strategy(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        agent = Agent(tools=[_DUMMY], provider=provider)
        assert agent.prompt_builder.reasoning_strategy is None
        for strategy_text in REASONING_STRATEGIES.values():
            assert strategy_text not in agent._system_prompt

    def test_agent_react_in_system_prompt(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(reasoning_strategy="react"),
        )
        assert "ReAct" in agent._system_prompt

    def test_agent_cot_in_system_prompt(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(reasoning_strategy="cot"),
        )
        assert "Chain of Thought" in agent._system_prompt

    def test_agent_plan_then_act_in_system_prompt(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(reasoning_strategy="plan_then_act"),
        )
        assert "Plan Then Act" in agent._system_prompt

    def test_agent_invalid_strategy_raises(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        with pytest.raises(ValueError, match="Invalid reasoning_strategy"):
            Agent(
                tools=[_DUMMY],
                provider=provider,
                config=AgentConfig(reasoning_strategy="invalid"),
            )

    def test_strategy_survives_add_tool(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(reasoning_strategy="react"),
        )
        extra = Tool(name="extra", description="extra", parameters=[], function=lambda: "ok")
        agent.add_tool(extra)
        assert "ReAct" in agent._system_prompt

    def test_strategy_survives_remove_tool(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        extra = Tool(name="extra", description="extra", parameters=[], function=lambda: "ok")
        agent = Agent(
            tools=[_DUMMY, extra],
            provider=provider,
            config=AgentConfig(reasoning_strategy="cot"),
        )
        agent.remove_tool("extra")
        assert "Chain of Thought" in agent._system_prompt

    def test_run_with_strategy(self, fake_provider):
        provider = fake_provider(responses=[_resp("I'll think step by step. The answer is 42.")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(reasoning_strategy="react"),
        )
        result = agent.run("What is the answer?")
        assert result.content == "I'll think step by step. The answer is 42."

    def test_custom_prompt_builder_overrides_config_strategy(self, fake_provider):
        provider = fake_provider(responses=[_resp("hello")])
        custom_pb = PromptBuilder(reasoning_strategy="cot")
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            prompt_builder=custom_pb,
            config=AgentConfig(reasoning_strategy="react"),
        )
        assert "Chain of Thought" in agent._system_prompt
        assert "ReAct" not in agent._system_prompt


# ── REASONING_STRATEGIES dict tests ──────────────────────────────────


class TestReasoningStrategiesDict:
    def test_has_three_strategies(self):
        assert len(REASONING_STRATEGIES) == 3

    def test_keys(self):
        assert set(REASONING_STRATEGIES.keys()) == {"react", "cot", "plan_then_act"}

    def test_all_values_are_strings(self):
        for key, value in REASONING_STRATEGIES.items():
            assert isinstance(value, str), f"{key} value is not a string"
