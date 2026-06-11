"""Tests for planning-as-config (PlanningConfig + agent/_planning.py)."""

from __future__ import annotations

import asyncio
import warnings
from typing import List, Tuple

import pytest

import selectools
from selectools import Agent, AgentConfig, PlanningConfig, tool
from selectools.agent._planning import _complexity_score
from selectools.memory import ConversationMemory
from selectools.patterns import PlanStep
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool(description="No-op tool for agent construction")
def _noop(x: str) -> str:
    return x


class _ScriptedProvider(LocalProvider):
    """Returns scripted responses in order and records every call."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Tuple[str, str]] = []  # (model, last message content)

    def complete(self, *, model, system_prompt, messages, **kwargs):  # type: ignore[override]
        self.calls.append((model, messages[-1].content or ""))
        text = self._responses.pop(0) if self._responses else "exhausted"
        usage = UsageStats(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0.01,
            model=model,
            provider="local",
        )
        return Message(role=Role.ASSISTANT, content=text), usage

    async def astream(self, *, model, system_prompt, messages, tools=None, **kwargs):  # type: ignore[override]
        msg, _ = self.complete(model=model, system_prompt=system_prompt, messages=messages)
        for token in (msg.content or "").split():
            yield token + " "


_PLAN_JSON = (
    '[{"executor": "executor", "task": "research the topic"},'
    ' {"executor": "executor", "task": "write the summary"}]'
)
_COMPLEX_PROMPT = "Research vector databases, then write a summary, and finally review it."
_SIMPLE_PROMPT = "What is 2+2?"


def _planning_agent(provider: _ScriptedProvider, planning: PlanningConfig) -> Agent:
    return Agent([_noop], provider=provider, config=AgentConfig(planning=planning))


# ---------------------------------------------------------------------------
# PlanningConfig
# ---------------------------------------------------------------------------


class TestPlanningConfig:
    def test_defaults(self):
        cfg = PlanningConfig()
        assert cfg.enabled is False
        assert cfg.provider is None
        assert cfg.model is None
        assert cfg.auto_approve is True
        assert cfg.reasoning is True
        assert cfg.always is False
        assert cfg.min_complexity == 2

    def test_agent_config_default_has_no_planning(self):
        assert AgentConfig().planning is None

    def test_auto_approve_false_requires_handler(self):
        with pytest.raises(ValueError, match="plan_approval_handler"):
            PlanningConfig(enabled=True, auto_approve=False)

    def test_auto_approve_false_with_handler_ok(self):
        cfg = PlanningConfig(enabled=True, auto_approve=False, plan_approval_handler=lambda p: True)
        assert cfg.plan_approval_handler is not None

    def test_min_complexity_validation(self):
        with pytest.raises(ValueError, match="min_complexity"):
            PlanningConfig(min_complexity=0)

    def test_dict_unpack(self):
        config = AgentConfig(planning={"enabled": True, "always": True})
        assert isinstance(config.planning, PlanningConfig)
        assert config.planning.enabled is True
        assert config.planning.always is True

    def test_exported_at_top_level(self):
        assert selectools.PlanningConfig is PlanningConfig
        assert "PlanningConfig" in selectools.__all__


# ---------------------------------------------------------------------------
# Complexity heuristic
# ---------------------------------------------------------------------------


class TestComplexityHeuristic:
    def test_simple_question_scores_one(self):
        assert _complexity_score(_SIMPLE_PROMPT) == 1

    def test_sequence_connectives_raise_score(self):
        assert _complexity_score(_COMPLEX_PROMPT) >= 2

    def test_numbered_list_raises_score(self):
        assert _complexity_score("Do this:\n1. fetch data\n2. clean it\n3. plot it") >= 2

    def test_long_input_raises_score(self):
        assert _complexity_score("word " * 200) >= 2


# ---------------------------------------------------------------------------
# Disabled = no-op regression
# ---------------------------------------------------------------------------


class TestDisabledNoOp:
    def test_no_planning_field_runs_normally(self):
        provider = _ScriptedProvider(["normal answer"])
        agent = Agent([_noop], provider=provider, config=AgentConfig())
        result = agent.run(_COMPLEX_PROMPT)
        assert result.content == "normal answer"
        assert len(provider.calls) == 1

    def test_enabled_false_runs_normally(self):
        provider = _ScriptedProvider(["normal answer"])
        agent = _planning_agent(provider, PlanningConfig(enabled=False))
        result = agent.run(_COMPLEX_PROMPT)
        assert result.content == "normal answer"
        assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Complexity gate
# ---------------------------------------------------------------------------


class TestComplexityGate:
    def test_simple_input_skips_planning(self):
        provider = _ScriptedProvider(["four"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_SIMPLE_PROMPT)
        assert result.content == "four"
        assert len(provider.calls) == 1  # no planner call

    def test_always_forces_planning_for_simple_input(self):
        provider = _ScriptedProvider([_PLAN_JSON, "step1", "step2", "final"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True, always=True))
        result = agent.run(_SIMPLE_PROMPT)
        assert result.content == "final"
        assert len(provider.calls) == 4  # plan + 2 steps + synthesis


# ---------------------------------------------------------------------------
# Plan -> execute -> synthesize
# ---------------------------------------------------------------------------


class TestPlannedExecution:
    def test_complex_input_plans_then_executes(self):
        provider = _ScriptedProvider([_PLAN_JSON, "research done", "summary written", "FINAL"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_COMPLEX_PROMPT)

        assert result.content == "FINAL"
        assert len(provider.calls) == 4
        # call 1 = planner, calls 2-3 = steps, call 4 = synthesis
        assert "JSON execution plan" in provider.calls[0][1]
        assert "research the topic" in provider.calls[1][1]
        assert "write the summary" in provider.calls[2][1]
        assert "Original task" in provider.calls[3][1]
        assert result.iterations == 3  # 2 steps + synthesis

    def test_reasoning_contains_plan(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_COMPLEX_PROMPT)
        assert result.reasoning is not None
        assert "research the topic" in result.reasoning
        assert "write the summary" in result.reasoning

    def test_reasoning_false_omits_plan(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True, reasoning=False))
        result = agent.run(_COMPLEX_PROMPT)
        assert result.reasoning is None or "research the topic" not in result.reasoning

    def test_trace_metadata_carries_plan(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_COMPLEX_PROMPT)
        meta = result.trace.metadata["planning"]
        assert meta["steps_executed"] == 2
        assert meta["plan"][0]["task"] == "research the topic"

    def test_usage_aggregated_across_planner_and_steps(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_COMPLEX_PROMPT)
        # 4 provider calls at $0.01 each, all folded into agent + result usage
        assert result.usage.total_cost_usd == pytest.approx(0.04)
        assert agent.usage.total_cost_usd == pytest.approx(0.04)

    def test_arun_plans_then_executes(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "FINAL-ASYNC"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = asyncio.run(agent.arun(_COMPLEX_PROMPT))
        assert result.content == "FINAL-ASYNC"
        assert len(provider.calls) == 4

    def test_memory_persists_planned_turn(self):
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        memory = ConversationMemory(max_messages=20)
        agent = Agent(
            [_noop],
            provider=provider,
            config=AgentConfig(planning=PlanningConfig(enabled=True)),
            memory=memory,
        )
        agent.run(_COMPLEX_PROMPT)
        history = memory.get_history()
        assert any(m.role == Role.USER and m.content == _COMPLEX_PROMPT for m in history)
        assert any(m.role == Role.ASSISTANT and m.content == "final" for m in history)


# ---------------------------------------------------------------------------
# Approval handler
# ---------------------------------------------------------------------------


class TestApproval:
    def test_handler_approves_plan(self):
        seen: List[List[PlanStep]] = []

        def handler(plan: List[PlanStep]):
            seen.append(plan)
            return True

        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", "final"])
        agent = _planning_agent(
            provider,
            PlanningConfig(enabled=True, auto_approve=False, plan_approval_handler=handler),
        )
        result = agent.run(_COMPLEX_PROMPT)
        assert result.content == "final"
        assert len(seen) == 1
        assert [s.task for s in seen[0]] == ["research the topic", "write the summary"]

    def test_handler_rejects_plan_falls_back_with_warning(self):
        provider = _ScriptedProvider([_PLAN_JSON, "normal answer"])
        agent = _planning_agent(
            provider,
            PlanningConfig(enabled=True, auto_approve=False, plan_approval_handler=lambda p: False),
        )
        with pytest.warns(UserWarning, match="rejected"):
            result = agent.run(_COMPLEX_PROMPT)
        # Fallback: planner called once, then the normal agent loop ran.
        assert result.content == "normal answer"
        assert len(provider.calls) == 2

    def test_handler_edits_plan(self):
        def handler(plan: List[PlanStep]):
            return [PlanStep(executor_name="executor", task="single edited step")]

        provider = _ScriptedProvider([_PLAN_JSON, "edited result", "final"])
        agent = _planning_agent(
            provider,
            PlanningConfig(enabled=True, auto_approve=False, plan_approval_handler=handler),
        )
        result = agent.run(_COMPLEX_PROMPT)
        assert result.content == "final"
        assert len(provider.calls) == 3  # plan + 1 edited step + synthesis
        assert "single edited step" in provider.calls[1][1]
        assert result.trace.metadata["planning"]["plan"][0]["task"] == "single edited step"


# ---------------------------------------------------------------------------
# Planner provider/model overrides
# ---------------------------------------------------------------------------


class TestPlannerOverrides:
    def test_planner_provider_and_model_override(self):
        planner_provider = _ScriptedProvider([_PLAN_JSON])
        main_provider = _ScriptedProvider(["a", "b", "final"])
        agent = _planning_agent(
            main_provider,
            PlanningConfig(enabled=True, provider=planner_provider, model="planner-model"),
        )
        result = agent.run(_COMPLEX_PROMPT)
        assert result.content == "final"
        # Planner call went to the override provider with the override model.
        assert len(planner_provider.calls) == 1
        assert planner_provider.calls[0][0] == "planner-model"
        # Steps + synthesis used the agent's own provider and model.
        assert len(main_provider.calls) == 3
        assert all(model != "planner-model" for model, _ in main_provider.calls)


# ---------------------------------------------------------------------------
# Streaming interplay
# ---------------------------------------------------------------------------


class TestStreamingSkip:
    def test_astream_warns_once_and_streams_normally(self):
        provider = _ScriptedProvider(["streamed answer", "streamed again"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))

        async def consume():
            return [chunk async for chunk in agent.astream(_COMPLEX_PROMPT)]

        with pytest.warns(UserWarning, match="streaming"):
            chunks = asyncio.run(consume())
        assert chunks  # normal streaming still produced output

        # Second streaming run does not warn again (one-time per agent).
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            asyncio.run(consume())

    def test_run_with_stream_handler_warns_and_skips_planning(self):
        provider = _ScriptedProvider(["handler answer"])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        with pytest.warns(UserWarning, match="streaming"):
            result = agent.run(_COMPLEX_PROMPT, stream_handler=lambda s: None)
        assert result.content == "handler answer"
        assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Structured output interplay
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_response_format_applied_to_synthesis(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        provider = _ScriptedProvider([_PLAN_JSON, "a", "b", '{"answer": "structured final"}'])
        agent = _planning_agent(provider, PlanningConfig(enabled=True))
        result = agent.run(_COMPLEX_PROMPT, response_format=schema)
        assert result.parsed == {"answer": "structured final"}
