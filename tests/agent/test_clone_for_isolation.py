"""Tests for the public ``Agent.clone_for_isolation()`` API (promoted in v0.25).

The method was underscore-private (``_clone_for_isolation``) but load-bearing
across module boundaries (evals, serve, a2a, planning). It is now public
``@beta``; the old name remains as a deprecated alias for one release window.
"""

from __future__ import annotations

import warnings

import pytest

from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.providers.stubs import LocalProvider


@tool(description="No-op tool")
def _noop() -> str:
    return "ok"


def _make_agent(**config_kwargs) -> Agent:
    return Agent(
        tools=[_noop],
        provider=LocalProvider(),
        config=AgentConfig(**config_kwargs),
    )


class TestCloneForIsolationPublic:
    def test_is_public_and_beta(self) -> None:
        assert hasattr(Agent, "clone_for_isolation")
        assert getattr(Agent.clone_for_isolation, "__stability__", None) == "beta"

    def test_shares_tools_and_provider(self) -> None:
        agent = _make_agent()
        clone = agent.clone_for_isolation()
        assert clone.tools is agent.tools
        assert clone.provider is agent.provider

    def test_copies_config_and_observer_list(self) -> None:
        class _Obs(AgentObserver):
            pass

        obs = _Obs()
        agent = _make_agent(observers=[obs])
        clone = agent.clone_for_isolation()

        assert clone.config is not agent.config
        assert clone.config.observers is not agent.config.observers
        assert clone.config.observers == [obs]

    def test_empty_default_observer_list_not_shared(self) -> None:
        """Regression: with the default observers=[] the clone must still get
        its OWN list. The old truthiness guard skipped the copy for empty
        lists, so parent and all clones shared one list object and an
        observer appended on a clone fired on siblings."""
        agent = _make_agent()
        clone = agent.clone_for_isolation()

        assert clone.config.observers is not agent.config.observers

        class _Obs(AgentObserver):
            pass

        clone.config.observers.append(_Obs())
        assert agent.config.observers == []

    def test_fresh_history_and_usage_memory_dropped(self) -> None:
        agent = Agent(
            tools=[_noop],
            provider=LocalProvider(),
            config=AgentConfig(enable_analytics=True),
            memory=ConversationMemory(max_messages=10),
        )
        agent._history = [Message(role=Role.USER, content="old")]
        clone = agent.clone_for_isolation()

        assert clone._history == []
        assert clone.usage.total_tokens == 0
        assert clone.memory is None
        assert clone.analytics is None
        assert agent._history, "Cloning must not mutate the parent's history"

    def test_no_warning_on_public_name(self) -> None:
        agent = _make_agent()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            agent.clone_for_isolation()


class TestCloneForIsolationDeprecatedAlias:
    def test_alias_warns_and_delegates(self) -> None:
        agent = _make_agent()
        with pytest.warns(DeprecationWarning, match="clone_for_isolation"):
            clone = agent._clone_for_isolation()
        assert clone._history == []
        assert clone.tools is agent.tools

    def test_alias_marked_deprecated(self) -> None:
        assert getattr(Agent._clone_for_isolation, "__stability__", None) == "deprecated"
        assert getattr(Agent._clone_for_isolation, "__deprecated_since__", None) == "1.0.0"
