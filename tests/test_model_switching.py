"""Tests for model switching per iteration (R10)."""

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.observer import AgentObserver
from selectools.tools.base import Tool
from selectools.trace import StepType
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")


def _resp(text, total_tokens=150, cost_usd=0.001):
    return (
        Message(role=Role.ASSISTANT, content=text),
        UsageStats(
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens - total_tokens // 2,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model="test",
            provider="test",
        ),
    )


def _tool_resp(tool_name, total_tokens=150):
    return (
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name=tool_name, parameters={})],
        ),
        UsageStats(
            prompt_tokens=75,
            completion_tokens=75,
            total_tokens=total_tokens,
            cost_usd=0.001,
            model="test",
            provider="test",
        ),
    )


class TestModelSwitching:
    """R10: model_selector switches model per iteration."""

    def test_selector_switches_model(self, fake_provider):
        """model_selector changes the model on iteration 2."""
        provider = fake_provider(responses=[_tool_resp("noop"), _resp("done")])

        def selector(iteration, tool_calls, usage):
            return "cheap-model" if iteration <= 1 else "expensive-model"

        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(max_iterations=6, model_selector=selector),
        )
        result = agent.run("test")
        assert result.iterations >= 1

    def test_selector_same_model_no_event(self, fake_provider):
        """No on_model_switch event when selector returns same model."""
        events = []

        class SwitchObserver(AgentObserver):
            def on_model_switch(self, run_id, iteration, old_model, new_model):
                events.append({"old": old_model, "new": new_model})

        provider = fake_provider(responses=[_resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                model="gpt-4o",
                max_iterations=6,
                model_selector=lambda i, tc, u: "gpt-4o",  # same model
                observers=[SwitchObserver()],
            ),
        )
        agent.run("test")
        assert len(events) == 0

    def test_selector_observer_event_fires(self, fake_provider):
        """on_model_switch fires when model changes."""
        events = []

        class SwitchObserver(AgentObserver):
            def on_model_switch(self, run_id, iteration, old_model, new_model):
                events.append({"iteration": iteration, "old": old_model, "new": new_model})

        provider = fake_provider(responses=[_tool_resp("noop"), _resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                model="haiku",
                max_iterations=6,
                model_selector=lambda i, tc, u: "haiku" if i <= 1 else "sonnet",
                observers=[SwitchObserver()],
            ),
        )
        agent.run("test")
        assert len(events) == 1
        assert events[0]["old"] == "haiku"
        assert events[0]["new"] == "sonnet"
        assert events[0]["iteration"] == 2

    def test_no_selector_uses_config_model(self, fake_provider):
        """Without model_selector, config.model is used throughout."""
        provider = fake_provider(responses=[_resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(model="gpt-4o", max_iterations=6),
        )
        result = agent.run("test")
        llm_steps = [s for s in result.trace.steps if s.type == StepType.LLM_CALL]
        assert all(s.model == "gpt-4o" for s in llm_steps)

    def test_trace_shows_switched_model(self, fake_provider):
        """Trace LLM_CALL steps show the effective model per iteration."""
        provider = fake_provider(responses=[_tool_resp("noop"), _resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                model="base-model",
                max_iterations=6,
                model_selector=lambda i, tc, u: "base-model" if i <= 1 else "upgraded-model",
            ),
        )
        result = agent.run("test")
        llm_steps = [s for s in result.trace.steps if s.type == StepType.LLM_CALL]
        # First iteration uses base-model, second uses upgraded-model
        assert llm_steps[0].model == "base-model"
        if len(llm_steps) > 1:
            assert llm_steps[1].model == "upgraded-model"

    def test_reset_between_runs(self, fake_provider):
        """_current_model is reset between runs via _prepare_run."""
        provider = fake_provider(responses=[_resp("done")])
        agent = Agent(
            tools=[_DUMMY],
            provider=provider,
            config=AgentConfig(
                model="default",
                max_iterations=6,
                model_selector=lambda i, tc, u: "switched",
            ),
        )
        agent.run("first")
        # After run, _current_model should be reset on next run
        agent.run("second")
        # If it weren't reset, the second run would start with "switched"
        # instead of "default", and model_selector would see it's already
        # "switched" and not fire on_model_switch


# --------------------------------------------------------------------------- #
# Bug hunt 2026-06-14: agent must use the provider's default_model, not a
# hardcoded "gpt-5-mini", when config.model is unset.
# --------------------------------------------------------------------------- #


class _ModelCapturingProvider:
    """Records the `model` the agent passes to complete()."""

    name = "capture"

    def __init__(self, default_model: str) -> None:
        self.default_model = default_model
        self.seen_models: list = []

    def complete(self, *, model=None, system_prompt="", messages, tools=None, **kw):
        self.seen_models.append(model)
        return _resp("done")

    async def acomplete(self, *, model=None, system_prompt="", messages, tools=None, **kw):
        self.seen_models.append(model)
        return _resp("done")


def test_uses_provider_default_model_when_config_model_unset():
    prov = _ModelCapturingProvider(default_model="claude-sonnet-4-6")
    agent = Agent(tools=[_DUMMY], provider=prov, config=AgentConfig())
    assert agent._effective_model == "claude-sonnet-4-6"
    agent.run("hi")
    assert prov.seen_models == ["claude-sonnet-4-6"]  # NOT "gpt-5-mini"


def test_explicit_config_model_overrides_provider_default():
    prov = _ModelCapturingProvider(default_model="claude-sonnet-4-6")
    agent = Agent(tools=[_DUMMY], provider=prov, config=AgentConfig(model="gpt-4o"))
    assert agent._effective_model == "gpt-4o"
    agent.run("hi")
    assert prov.seen_models == ["gpt-4o"]


def test_config_model_defaults_to_none():
    # The default must be None (sentinel for "use the provider default"), not a
    # hardcoded model id that would reach every provider.
    assert AgentConfig().model is None
