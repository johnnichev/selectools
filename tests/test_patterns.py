"""Tests for selectools.patterns — v0.19.1 Advanced Agent Patterns."""

from __future__ import annotations

import asyncio
from dataclasses import is_dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.patterns import (
    DebateAgent,
    DebateResult,
    DebateRound,
    PlanAndExecuteAgent,
    PlanStep,
    ReflectionRound,
    ReflectiveAgent,
    ReflectiveResult,
    Subtask,
    TeamLeadAgent,
    TeamLeadResult,
)
from selectools.types import Message, Role

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(response: str = "done") -> MagicMock:
    """Return a mock Agent whose arun() returns a result with the given content."""
    agent = MagicMock()
    result = MagicMock()
    result.content = response
    result.usage = None
    agent.arun = AsyncMock(return_value=result)
    return agent


def _make_plan_agent(plan_json: str) -> MagicMock:
    """Return a mock planner agent that returns a JSON plan string."""
    return _make_agent(response=plan_json)


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_defaults(self):
        step = PlanStep(executor_name="writer", task="write intro")
        assert step.status == "pending"
        assert step.result is None

    def test_mutation(self):
        step = PlanStep(executor_name="writer", task="write intro")
        step.status = "done"
        step.result = "intro text"
        assert step.status == "done"
        assert step.result == "intro text"

    def test_is_dataclass(self):
        assert is_dataclass(PlanStep)


# ---------------------------------------------------------------------------
# PlanAndExecuteAgent
# ---------------------------------------------------------------------------


class TestPlanAndExecuteAgent:
    def test_construction_requires_executors(self):
        planner = _make_agent()
        with pytest.raises(ValueError, match="at least one executor"):
            PlanAndExecuteAgent(planner=planner, executors={})

    def test_run_calls_planner_then_executors(self):
        plan_json = '[{"executor": "writer", "task": "write something"}]'
        planner = _make_plan_agent(plan_json)
        writer = _make_agent("written content")

        agent = PlanAndExecuteAgent(planner=planner, executors={"writer": writer})
        result = agent.run("Write a blog post")

        planner.arun.assert_called_once()
        writer.arun.assert_called_once()
        assert result is not None

    def test_run_fallback_when_plan_empty(self):
        planner = _make_plan_agent("[]")
        writer = _make_agent("fallback content")

        agent = PlanAndExecuteAgent(planner=planner, executors={"writer": writer})
        result = agent.run("Write something")

        # Fallback: writer should still be called
        writer.arun.assert_called_once()

    def test_run_unknown_executor_skipped(self):
        plan_json = '[{"executor": "unknown_agent", "task": "do something"}]'
        planner = _make_plan_agent(plan_json)
        writer = _make_agent("content")

        agent = PlanAndExecuteAgent(planner=planner, executors={"writer": writer})
        result = agent.run("Write something")

        # unknown_agent is not in executors — step is silently skipped.
        # The plan parse yields no valid steps, so the fallback in _call_planner
        # runs all executors (writer) instead.
        assert result is not None
        # writer is called via the _call_planner fallback (no valid executor in plan)
        writer.arun.assert_called_once()

    def test_run_returns_graph_result_like(self):
        plan_json = '[{"executor": "writer", "task": "write"}]'
        planner = _make_plan_agent(plan_json)
        writer = _make_agent("output")

        agent = PlanAndExecuteAgent(planner=planner, executors={"writer": writer})
        result = agent.run("task")
        # Result should have a content attribute
        assert hasattr(result, "content")

    def test_plan_stored_in_state(self):
        plan_json = '[{"executor": "writer", "task": "write intro"}]'
        planner = _make_plan_agent(plan_json)
        writer = _make_agent("content")

        agent = PlanAndExecuteAgent(planner=planner, executors={"writer": writer})
        result = agent.run("task")
        assert "__plan__" in result.state.data

    def test_replanner_argument_accepted(self):
        planner = _make_agent()
        writer = _make_agent("content")
        # Should not raise
        agent = PlanAndExecuteAgent(
            planner=planner,
            executors={"writer": writer},
            replanner=True,
            max_replan_attempts=1,
        )
        assert agent.replanner is True

    def test_arun_is_async(self):
        planner = _make_plan_agent('[{"executor": "w", "task": "t"}]')
        agent = PlanAndExecuteAgent(planner=planner, executors={"w": _make_agent()})
        import inspect

        assert inspect.iscoroutinefunction(agent.arun)


# ---------------------------------------------------------------------------
# ReflectiveAgent
# ---------------------------------------------------------------------------


class TestReflectiveAgent:
    def test_construction(self):
        actor = _make_agent()
        critic = _make_agent()
        agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=5, stop_condition="ok")
        assert agent.max_reflections == 5
        assert agent.stop_condition == "ok"

    def test_run_single_round_approved(self):
        actor = _make_agent("great draft")
        critic = _make_agent("This is approved and looks good.")

        agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=3)
        result = agent.run("Write a press release")

        assert result.approved is True
        assert result.total_rounds == 1
        assert result.final_draft == "great draft"

    def test_run_max_reflections_reached(self):
        actor = _make_agent("draft")
        critic = _make_agent("Needs more work.")  # never says "approved"

        agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=3)
        result = agent.run("Write something")

        assert result.approved is False
        assert result.total_rounds == 3

    def test_result_has_rounds(self):
        actor = _make_agent("draft")
        critic = _make_agent("needs work")

        agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=2)
        result = agent.run("task")

        assert len(result.rounds) == 2
        for i, r in enumerate(result.rounds):
            assert r.round_number == i

    def test_result_approved_flag(self):
        actor = _make_agent("draft")
        # First call returns not approved, second returns approved
        critic = MagicMock()
        r1, r2 = MagicMock(), MagicMock()
        r1.content = "needs work"
        r2.content = "This is now approved."
        critic.arun = AsyncMock(side_effect=[r1, r2])

        agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=3)
        result = agent.run("task")

        assert result.approved is True
        assert result.total_rounds == 2

    def test_stop_condition_case_insensitive(self):
        actor = _make_agent("draft")
        critic = _make_agent("APPROVED — looks great!")

        agent = ReflectiveAgent(actor=actor, critic=critic, stop_condition="approved")
        result = agent.run("task")

        assert result.approved is True
        assert result.total_rounds == 1

    def test_arun_is_async(self):
        import inspect

        agent = ReflectiveAgent(actor=_make_agent(), critic=_make_agent())
        assert inspect.iscoroutinefunction(agent.arun)


# ---------------------------------------------------------------------------
# DebateAgent
# ---------------------------------------------------------------------------


class TestDebateAgent:
    def test_construction(self):
        agents = {"pro": _make_agent(), "con": _make_agent()}
        judge = _make_agent()
        da = DebateAgent(agents=agents, judge=judge, max_rounds=2)
        assert da.max_rounds == 2

    def test_requires_at_least_two_agents(self):
        with pytest.raises(ValueError, match="at least 2 agents"):
            DebateAgent(agents={"only_one": _make_agent()}, judge=_make_agent())

    def test_run_calls_each_agent_each_round(self):
        optimist = _make_agent("I think it's great")
        skeptic = _make_agent("I have concerns")
        judge = _make_agent("After considering both sides...")

        da = DebateAgent(
            agents={"optimist": optimist, "skeptic": skeptic},
            judge=judge,
            max_rounds=3,
        )
        result = da.run("Should we adopt microservices?")

        assert optimist.arun.call_count == 3
        assert skeptic.arun.call_count == 3

    def test_run_calls_judge_once(self):
        optimist = _make_agent("pro argument")
        skeptic = _make_agent("con argument")
        judge = _make_agent("synthesized conclusion")

        da = DebateAgent(
            agents={"optimist": optimist, "skeptic": skeptic},
            judge=judge,
            max_rounds=2,
        )
        result = da.run("topic")

        judge.arun.assert_called_once()

    def test_result_has_rounds(self):
        da = DebateAgent(
            agents={"a": _make_agent("arg a"), "b": _make_agent("arg b")},
            judge=_make_agent("conclusion"),
            max_rounds=3,
        )
        result = da.run("topic")

        assert len(result.rounds) == 3
        for i, r in enumerate(result.rounds):
            assert r.round_number == i
            assert "a" in r.arguments
            assert "b" in r.arguments

    def test_prior_context_passed_to_agents_in_later_rounds(self):
        optimist = _make_agent("pro")
        skeptic = _make_agent("con")
        judge = _make_agent("verdict")

        da = DebateAgent(
            agents={"optimist": optimist, "skeptic": skeptic},
            judge=judge,
            max_rounds=2,
        )
        da.run("topic")

        # Round 2 call should include "Round 1" in the prompt
        call_args_round2 = optimist.arun.call_args_list[1]
        messages = call_args_round2[0][0]  # first positional arg
        assert any("Round 1" in (m.content or "") for m in messages)

    def test_result_conclusion_from_judge(self):
        da = DebateAgent(
            agents={"a": _make_agent("arg"), "b": _make_agent("counter")},
            judge=_make_agent("final answer"),
            max_rounds=1,
        )
        result = da.run("question")
        assert result.conclusion == "final answer"

    def test_arun_is_async(self):
        import inspect

        da = DebateAgent(
            agents={"a": _make_agent(), "b": _make_agent()},
            judge=_make_agent(),
        )
        assert inspect.iscoroutinefunction(da.arun)

    def test_max_rounds_zero_raises(self):
        with pytest.raises(ValueError, match="max_rounds"):
            DebateAgent(
                agents={"a": _make_agent(), "b": _make_agent()},
                judge=_make_agent(),
                max_rounds=0,
            )

    @pytest.mark.asyncio
    async def test_arun_returns_debate_result(self):
        optimist = _make_agent("optimist arg")
        skeptic = _make_agent("skeptic arg")
        judge = _make_agent("my verdict")

        da = DebateAgent(
            agents={"optimist": optimist, "skeptic": skeptic},
            judge=judge,
            max_rounds=1,
        )
        result = await da.arun("topic")

        assert result.conclusion == "my verdict"
        assert result.total_rounds == 1
        optimist.arun.assert_called_once()
        skeptic.arun.assert_called_once()
        judge.arun.assert_called_once()


# ---------------------------------------------------------------------------
# TeamLeadAgent
# ---------------------------------------------------------------------------


class TestTeamLeadAgent:
    def _make_lead(
        self,
        plan_json: str,
        review_json: str = '{"complete": true, "reassignments": [], "synthesis": "done"}',
    ) -> MagicMock:
        lead = MagicMock()
        plan_result = MagicMock()
        plan_result.content = plan_json
        review_result = MagicMock()
        review_result.content = review_json
        synthesis_result = MagicMock()
        synthesis_result.content = "synthesized answer"
        lead.arun = AsyncMock(side_effect=[plan_result, review_result, synthesis_result] * 5)
        return lead

    def test_construction_sequential(self):
        agent = TeamLeadAgent(
            lead=_make_agent(),
            team={"a": _make_agent()},
            delegation_strategy="sequential",
        )
        assert agent.delegation_strategy == "sequential"

    def test_construction_parallel(self):
        agent = TeamLeadAgent(
            lead=_make_agent(),
            team={"a": _make_agent()},
            delegation_strategy="parallel",
        )
        assert agent.delegation_strategy == "parallel"

    def test_construction_dynamic(self):
        agent = TeamLeadAgent(
            lead=_make_agent(),
            team={"a": _make_agent()},
            delegation_strategy="dynamic",
        )
        assert agent.delegation_strategy == "dynamic"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="delegation_strategy"):
            TeamLeadAgent(lead=_make_agent(), team={"a": _make_agent()}, delegation_strategy="bad")

    def test_sequential_executes_in_order(self):
        plan_json = (
            '[{"assignee": "analyst", "task": "analyze"}, {"assignee": "writer", "task": "write"}]'
        )
        lead = self._make_lead(plan_json)
        analyst = _make_agent("analysis done")
        writer = _make_agent("writing done")

        agent = TeamLeadAgent(
            lead=lead,
            team={"analyst": analyst, "writer": writer},
            delegation_strategy="sequential",
        )
        result = agent.run("Investigate and report")

        analyst.arun.assert_called_once()
        writer.arun.assert_called_once()

    def test_dynamic_marks_complete_early(self):
        plan_json = '[{"assignee": "analyst", "task": "analyze"}]'
        complete_review = '{"complete": true, "reassignments": [], "synthesis": "all done"}'
        lead = self._make_lead(plan_json, complete_review)
        analyst = _make_agent("analysis done")

        agent = TeamLeadAgent(
            lead=lead,
            team={"analyst": analyst},
            delegation_strategy="dynamic",
        )
        result = agent.run("task")

        assert result.content in ("all done", "synthesized answer")

    def test_result_subtasks_populated(self):
        plan_json = '[{"assignee": "writer", "task": "write"}]'
        lead = self._make_lead(plan_json)
        writer = _make_agent("written")

        agent = TeamLeadAgent(
            lead=lead,
            team={"writer": writer},
            delegation_strategy="sequential",
        )
        result = agent.run("task")

        assert len(result.subtasks) >= 1
        assert result.subtasks[0].assignee == "writer"

    def test_subtask_is_dataclass(self):
        assert is_dataclass(Subtask)

    def test_team_lead_result_is_dataclass(self):
        assert is_dataclass(TeamLeadResult)

    def test_arun_is_async(self):
        import inspect

        agent = TeamLeadAgent(lead=_make_agent(), team={"a": _make_agent()})
        assert inspect.iscoroutinefunction(agent.arun)

    def test_empty_team_raises(self):
        with pytest.raises(ValueError, match="at least one team member"):
            TeamLeadAgent(lead=_make_agent(), team={})

    @pytest.mark.asyncio
    async def test_arun_sequential_returns_result(self):
        plan_json = '[{"assignee": "writer", "task": "write"}]'
        lead = self._make_lead(plan_json)
        writer = _make_agent("written")

        agent = TeamLeadAgent(
            lead=lead,
            team={"writer": writer},
            delegation_strategy="sequential",
        )
        result = await agent.arun("Write something")

        assert result is not None
        assert len(result.subtasks) >= 1
        writer.arun.assert_called_once()

    def test_reassignment_bounded(self):
        plan_json = '[{"assignee": "analyst", "task": "analyze"}]'
        # Lead always asks for reassignment but bounded by max_reassignments
        reassign_review = '{"complete": false, "reassignments": [{"assignee": "analyst", "task": "redo"}], "synthesis": ""}'
        complete_review = '{"complete": true, "reassignments": [], "synthesis": "done"}'

        lead = MagicMock()
        plan_result = MagicMock()
        plan_result.content = plan_json
        reassign_result = MagicMock()
        reassign_result.content = reassign_review
        complete_result = MagicMock()
        complete_result.content = complete_review
        synthesis_result = MagicMock()
        synthesis_result.content = "synthesized"

        # Plan → reassign → reassign → complete → synthesis
        lead.arun = AsyncMock(
            side_effect=[
                plan_result,
                reassign_result,
                reassign_result,
                complete_result,
                synthesis_result,
            ]
            * 3
        )

        analyst = _make_agent("analysis")
        agent = TeamLeadAgent(
            lead=lead,
            team={"analyst": analyst},
            delegation_strategy="dynamic",
            max_reassignments=2,
        )
        result = agent.run("task")
        assert result is not None
