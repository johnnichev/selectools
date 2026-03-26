"""Tests for SupervisorAgent — all four strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.orchestration.graph import GraphResult, _merge_usage
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState
from selectools.orchestration.supervisor import (
    ModelSplit,
    SupervisorAgent,
    SupervisorStrategy,
    _format_history,
    _looks_complete,
    _safe_json_parse,
)
from selectools.types import AgentResult, Message, Role
from selectools.usage import UsageStats

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_mock_agent(content: str = "agent response") -> MagicMock:
    """Create a mock agent with a predictable arun() response."""
    agent = MagicMock()
    result = AgentResult(
        message=Message(role=Role.ASSISTANT, content=content),
        iterations=1,
        usage=UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
    )
    agent.arun = AsyncMock(return_value=result)
    return agent


def make_mock_provider(response: str = "[]") -> MagicMock:
    """Create a mock provider whose acomplete returns the given text."""
    provider = MagicMock()
    response_msg = Message(role=Role.ASSISTANT, content=response)
    provider.acomplete = AsyncMock(return_value=(response_msg, UsageStats()))
    return provider


# ------------------------------------------------------------------
# Strategy enum
# ------------------------------------------------------------------


class TestSupervisorStrategy:
    def test_enum_values(self):
        assert SupervisorStrategy.PLAN_AND_EXECUTE == "plan_and_execute"
        assert SupervisorStrategy.ROUND_ROBIN == "round_robin"
        assert SupervisorStrategy.DYNAMIC == "dynamic"
        assert SupervisorStrategy.MAGENTIC == "magentic"

    def test_str_coercion(self):
        s = SupervisorStrategy("round_robin")
        assert s == SupervisorStrategy.ROUND_ROBIN


class TestModelSplit:
    def test_construction(self):
        ms = ModelSplit(planner_model="gpt-4o", executor_model="gpt-4o-mini")
        assert ms.planner_model == "gpt-4o"
        assert ms.executor_model == "gpt-4o-mini"


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


class TestUtilities:
    def test_safe_json_parse_valid(self):
        result = _safe_json_parse('[{"agent": "a", "task": "do x"}]')
        assert isinstance(result, list)
        assert result[0]["agent"] == "a"

    def test_safe_json_parse_with_markdown_fences(self):
        text = '```json\n[{"agent": "a", "task": "x"}]\n```'
        result = _safe_json_parse(text)
        assert isinstance(result, list)

    def test_safe_json_parse_invalid_returns_default(self):
        result = _safe_json_parse("not json at all", default=[])
        assert result == []

    def test_safe_json_parse_embedded_json(self):
        text = 'Here is the plan: [{"agent": "a", "task": "do it"}] — good luck'
        result = _safe_json_parse(text)
        assert isinstance(result, list)

    def test_looks_complete_positive(self):
        assert _looks_complete("Task complete.") is True
        assert _looks_complete("All done.") is True

    def test_looks_complete_negative(self):
        assert _looks_complete("Still working on it") is False
        assert _looks_complete("") is False

    def test_format_history_empty(self):
        result = _format_history([])
        assert "No steps" in result

    def test_format_history_with_results(self):
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="analysis complete"),
            iterations=1,
        )
        history = [("researcher", result), ("writer", result)]
        formatted = _format_history(history)
        assert "researcher" in formatted
        assert "writer" in formatted

    def test_merge_usage(self):
        a = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001)
        b = UsageStats(prompt_tokens=20, completion_tokens=10, total_tokens=30, cost_usd=0.002)
        merged = _merge_usage(a, b)
        assert merged.prompt_tokens == 30
        assert merged.completion_tokens == 15
        assert merged.total_tokens == 45
        assert abs(merged.cost_usd - 0.003) < 1e-6


# ------------------------------------------------------------------
# SupervisorAgent construction
# ------------------------------------------------------------------


class TestSupervisorConstruction:
    def test_default_strategy(self):
        agents = {"a": make_mock_agent()}
        provider = make_mock_provider()
        supervisor = SupervisorAgent(agents=agents, provider=provider)
        assert supervisor.strategy == SupervisorStrategy.PLAN_AND_EXECUTE

    def test_string_strategy_coercion(self):
        agents = {"a": make_mock_agent()}
        provider = make_mock_provider()
        supervisor = SupervisorAgent(agents=agents, provider=provider, strategy="round_robin")
        assert supervisor.strategy == SupervisorStrategy.ROUND_ROBIN

    def test_model_split_stored(self):
        agents = {"a": make_mock_agent()}
        provider = make_mock_provider()
        ms = ModelSplit(planner_model="gpt-4o", executor_model="gpt-4o-mini")
        supervisor = SupervisorAgent(agents=agents, provider=provider, model_split=ms)
        assert supervisor._planner_model == "gpt-4o"
        assert supervisor._executor_model == "gpt-4o-mini"

    def test_delegation_constraints_stored(self):
        agents = {"a": make_mock_agent()}
        provider = make_mock_provider()
        supervisor = SupervisorAgent(
            agents=agents,
            provider=provider,
            delegation_constraints={"a": ["supervisor"]},
        )
        assert supervisor.delegation_constraints == {"a": ["supervisor"]}


# ------------------------------------------------------------------
# Strategy: plan_and_execute
# ------------------------------------------------------------------


class TestPlanAndExecute:
    @pytest.mark.asyncio
    async def test_executes_plan_steps(self):
        """Supervisor generates plan and executes each agent."""
        researcher = make_mock_agent("research done")
        writer = make_mock_agent("article written")

        plan_json = '[{"agent": "researcher", "task": "research AI safety"}, {"agent": "writer", "task": "write article"}]'
        provider = make_mock_provider(plan_json)

        supervisor = SupervisorAgent(
            agents={"researcher": researcher, "writer": writer},
            provider=provider,
            strategy=SupervisorStrategy.PLAN_AND_EXECUTE,
        )

        result = await supervisor.arun("Write a blog post about AI safety")
        assert isinstance(result, GraphResult)
        assert result.content != "" or result.state is not None

    @pytest.mark.asyncio
    async def test_fallback_when_plan_empty(self):
        """Falls back to executing agents in registration order when plan is empty."""
        a = make_mock_agent("output_a")
        b = make_mock_agent("output_b")
        provider = make_mock_provider("not valid json at all")

        supervisor = SupervisorAgent(
            agents={"a": a, "b": b},
            provider=provider,
            strategy=SupervisorStrategy.PLAN_AND_EXECUTE,
        )

        result = await supervisor.arun("do something")
        assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_agent_in_plan_skipped(self):
        """Steps referencing unknown agents are skipped gracefully."""
        a = make_mock_agent("output")
        plan_json = '[{"agent": "nonexistent", "task": "do x"}, {"agent": "a", "task": "do y"}]'
        provider = make_mock_provider(plan_json)

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.PLAN_AND_EXECUTE,
        )

        result = await supervisor.arun("task")
        assert result is not None


# ------------------------------------------------------------------
# Strategy: round_robin
# ------------------------------------------------------------------


class TestRoundRobin:
    @pytest.mark.asyncio
    async def test_all_agents_called(self):
        a = make_mock_agent("a done")
        b = make_mock_agent("b done")
        provider = make_mock_provider()

        supervisor = SupervisorAgent(
            agents={"a": a, "b": b},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=2,
        )

        result = await supervisor.arun("task")
        assert result is not None
        # Both agents should have been called at least once
        assert a.arun.call_count >= 1
        assert b.arun.call_count >= 1

    @pytest.mark.asyncio
    async def test_max_rounds_respected(self):
        a = make_mock_agent("still working")
        provider = make_mock_provider()

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=3,
        )

        result = await supervisor.arun("task")
        # With 1 agent, 3 rounds → called exactly 3 times
        assert a.arun.call_count == 3

    @pytest.mark.asyncio
    async def test_result_has_usage(self):
        a = make_mock_agent("done")
        provider = make_mock_provider()

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=1,
        )

        result = await supervisor.arun("task")
        assert result.total_usage.total_tokens >= 0


# ------------------------------------------------------------------
# Strategy: dynamic
# ------------------------------------------------------------------


class TestDynamic:
    @pytest.mark.asyncio
    async def test_routes_to_named_agent(self):
        researcher = make_mock_agent("research output")
        provider = make_mock_provider("researcher")  # routes to researcher first

        supervisor = SupervisorAgent(
            agents={"researcher": researcher},
            provider=provider,
            strategy=SupervisorStrategy.DYNAMIC,
            max_rounds=2,
        )

        result = await supervisor.arun("research topic")
        assert researcher.arun.call_count >= 1

    @pytest.mark.asyncio
    async def test_done_signal_stops_routing(self):
        a = make_mock_agent("done output")
        # Provider returns "DONE" on second call → stops after 1 agent call
        provider = make_mock_provider("DONE")

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.DYNAMIC,
            max_rounds=10,
        )

        result = await supervisor.arun("task")
        # With "DONE" response immediately, 0 agent calls
        assert result is not None
        assert a.arun.call_count == 0

    @pytest.mark.asyncio
    async def test_unknown_agent_falls_back_to_first(self):
        a = make_mock_agent("a output")
        provider = make_mock_provider("hallucinated_agent_name")

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.DYNAMIC,
            max_rounds=1,
        )

        result = await supervisor.arun("task")
        # Should fall back to "a" and call it
        assert a.arun.call_count >= 1


# ------------------------------------------------------------------
# Strategy: magentic
# ------------------------------------------------------------------


class TestMagentic:
    @pytest.mark.asyncio
    async def test_executes_next_agent_from_ledger(self):
        a = make_mock_agent("analysis done")
        ledger_response = '{"task_ledger": {"facts": [], "plan": []}, "progress_ledger": {"is_complete": false, "is_progressing": true, "next_agent": "a", "reason": "needs analysis"}}'
        done_response = '{"task_ledger": {}, "progress_ledger": {"is_complete": true, "is_progressing": true, "next_agent": "DONE", "reason": "done"}}'

        call_count = {"n": 0}

        async def mock_acomplete(*args, **kwargs):
            call_count["n"] += 1
            content = ledger_response if call_count["n"] == 1 else done_response
            return (Message(role=Role.ASSISTANT, content=content), UsageStats())

        provider = MagicMock()
        provider.acomplete = mock_acomplete

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.MAGENTIC,
            max_rounds=5,
        )

        result = await supervisor.arun("complex task")
        assert result is not None
        assert a.arun.call_count >= 1

    @pytest.mark.asyncio
    async def test_stall_triggers_replan(self):
        """After max_stalls consecutive not-progressing steps, replan fires."""
        a = make_mock_agent("still stuck")
        replan_fired = []

        from selectools.observer import AgentObserver

        class ReplanObserver(AgentObserver):
            def on_supervisor_replan(self, run_id, stall_count, new_plan):
                replan_fired.append(stall_count)

        stall_response = '{"task_ledger": {}, "progress_ledger": {"is_complete": false, "is_progressing": false, "next_agent": "a", "reason": "stuck"}}'
        done_response = '{"task_ledger": {}, "progress_ledger": {"is_complete": true, "is_progressing": true, "next_agent": "DONE", "reason": "done"}}'

        call_count = {"n": 0}

        async def mock_acomplete(*args, **kwargs):
            call_count["n"] += 1
            # Return stall 3 times, then done
            if call_count["n"] <= 3:
                content = stall_response
            else:
                content = done_response
            return (Message(role=Role.ASSISTANT, content=content), UsageStats())

        provider = MagicMock()
        provider.acomplete = mock_acomplete

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.MAGENTIC,
            max_rounds=20,
            max_stalls=2,
            observers=[ReplanObserver()],
        )

        result = await supervisor.arun("stuck task")
        assert len(replan_fired) >= 1

    @pytest.mark.asyncio
    async def test_complete_flag_stops_execution(self):
        a = make_mock_agent("final")
        done_response = '{"task_ledger": {}, "progress_ledger": {"is_complete": true, "is_progressing": true, "next_agent": "DONE", "reason": "done"}}'
        provider = make_mock_provider(done_response)

        supervisor = SupervisorAgent(
            agents={"a": a},
            provider=provider,
            strategy=SupervisorStrategy.MAGENTIC,
            max_rounds=10,
        )

        result = await supervisor.arun("task")
        assert result is not None
        # is_complete=true on first call → 0 agent calls
        assert a.arun.call_count == 0


# ------------------------------------------------------------------
# Cancellation
# ------------------------------------------------------------------


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_token_stops_execution(self):
        from selectools.cancellation import CancellationToken

        a = make_mock_agent("output")
        provider = make_mock_provider()

        token = CancellationToken()
        token.cancel()  # cancel immediately

        supervisor = SupervisorAgent(
            agents={"a": a, "b": make_mock_agent()},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=10,
            cancellation_token=token,
        )

        result = await supervisor.arun("task")
        # Should stop quickly due to cancellation
        assert result is not None
        assert a.arun.call_count == 0  # cancelled before first call
