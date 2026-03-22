"""Tests for token budget per run (R1) and cost attribution (R7)."""

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.observer import AgentObserver
from selectools.tools.base import Tool
from selectools.trace import StepType
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")


def _usage(total_tokens=150, cost_usd=0.001):
    return UsageStats(
        prompt_tokens=total_tokens // 2,
        completion_tokens=total_tokens - total_tokens // 2,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        model="test",
        provider="test",
    )


def _resp(text, total_tokens=150, cost_usd=0.001):
    """Create a (Message, UsageStats) tuple for fake provider."""
    return (
        Message(role=Role.ASSISTANT, content=text),
        _usage(total_tokens=total_tokens, cost_usd=cost_usd),
    )


def _tool_resp(tool_name, total_tokens=150, cost_usd=0.001):
    """Create a tool-call response tuple."""
    return (
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name=tool_name, parameters={})],
        ),
        _usage(total_tokens=total_tokens, cost_usd=cost_usd),
    )


class TestTokenBudget:
    """R1: Agent stops when max_total_tokens or max_cost_usd is exceeded."""

    def test_token_limit_stops_on_second_iteration(self, fake_provider):
        """Budget exceeded check fires at the start of iteration 2."""
        provider = fake_provider(
            responses=[
                _tool_resp("noop", total_tokens=500),
                _resp("final", total_tokens=500),
            ]
        )
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=10, max_total_tokens=400),
        )
        result = agent.run("test")
        # Iter 1 uses 500 tokens (>= 400), so iter 2 budget check fires
        assert result.iterations == 2
        assert "budget exceeded" in result.content.lower()
        budget_steps = [s for s in result.trace.steps if s.type == StepType.BUDGET_EXCEEDED]
        assert len(budget_steps) == 1

    def test_cost_limit_stops_run(self, fake_provider):
        """Agent stops when cumulative cost exceeds max_cost_usd."""
        provider = fake_provider(
            responses=[
                _tool_resp("noop", cost_usd=0.10),
                _resp("done", cost_usd=0.10),
            ]
        )
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=10, max_cost_usd=0.05),
        )
        result = agent.run("test")
        assert "cost budget exceeded" in result.content.lower()

    def test_budget_not_exceeded_completes_normally(self, fake_provider):
        """Agent completes normally when budget is not exceeded."""
        provider = fake_provider(responses=[_resp("Final answer", total_tokens=15)])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6, max_total_tokens=10000),
        )
        result = agent.run("test")
        assert result.iterations == 1
        assert "budget" not in result.content.lower()

    def test_none_budget_no_enforcement(self, fake_provider):
        """None budget values mean no enforcement (backward compat)."""
        provider = fake_provider(responses=[_resp("answer", total_tokens=15000, cost_usd=5.0)])
        agent = Agent(tools=[_DUMMY], provider=provider, config=AgentConfig(max_iterations=6))
        result = agent.run("test")
        assert result.iterations == 1
        assert "budget" not in result.content.lower()

    def test_budget_observer_event(self, fake_provider):
        """Observer receives on_budget_exceeded when budget is hit."""
        events = []

        class BudgetObserver(AgentObserver):
            def on_budget_exceeded(self, run_id, reason, tokens_used, cost_used):
                events.append(
                    {"reason": reason, "tokens_used": tokens_used, "cost_used": cost_used}
                )

        provider = fake_provider(
            responses=[
                _tool_resp("noop", total_tokens=1000),
                _resp("r2", total_tokens=1000),
            ]
        )
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                max_iterations=10, max_total_tokens=500, observers=[BudgetObserver()]
            ),
        )
        agent.run("test")
        assert len(events) == 1
        assert "token budget" in events[0]["reason"].lower()


class TestCostAttribution:
    """R7: cost_usd populated on LLM_CALL trace steps."""

    def test_cost_usd_on_trace_step(self, fake_provider):
        """LLM_CALL trace steps include cost_usd from provider."""
        provider = fake_provider(responses=[_resp("answer", cost_usd=0.0042)])
        agent = Agent(tools=[_DUMMY], provider=provider, config=AgentConfig(max_iterations=6))
        result = agent.run("test")
        llm_steps = [s for s in result.trace.steps if s.type == StepType.LLM_CALL]
        assert len(llm_steps) >= 1
        assert llm_steps[0].cost_usd == 0.0042
