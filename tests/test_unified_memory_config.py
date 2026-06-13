"""Tests for UnifiedMemory AgentConfig wiring — MemoryConfig(unified=True).

Config validation for the unified MemoryConfig fields, plus integration tests
that run a real Agent on offline providers and assert the STM -> LTM
promotion, episodic, and context-injection paths are exercised.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from selectools import Agent, AgentConfig, MemoryConfig, tool
from selectools.knowledge import KnowledgeMemory
from selectools.memory import ConversationMemory
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.unified_memory import InMemoryKnowledgeStore, UnifiedMemory
from selectools.usage import UsageStats


@tool(description="Echo the given text back")
def echo(text: str) -> str:
    return text


class CapturingProvider(LocalProvider):
    """LocalProvider that records the messages of every complete() call."""

    def __init__(self) -> None:
        self.calls: List[List[Message]] = []

    def complete(self, *, messages: List[Message], **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.calls.append(list(messages))
        return super().complete(messages=messages, **kwargs)


def make_agent(provider: Any = None, **memory_kwargs: Any) -> Agent:
    return Agent(
        tools=[echo],
        provider=provider or LocalProvider(),
        config=AgentConfig(
            max_iterations=1,
            memory=MemoryConfig(unified=True, **memory_kwargs),
        ),
    )


def lt_contents(memory: UnifiedMemory) -> List[str]:
    return [e.content for e in memory.long_term.store.query(limit=1000)]


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


class TestMemoryConfigValidation:
    def test_default_is_off(self) -> None:
        config = MemoryConfig()
        assert config.unified is False
        assert config.unified_memory is None
        assert config._unified_enabled is False

    def test_agentconfig_default_has_unified_off(self) -> None:
        config = AgentConfig()
        assert isinstance(config.memory, MemoryConfig)
        assert config.memory._unified_enabled is False

    def test_roadmap_config_shape(self) -> None:
        config = AgentConfig(
            memory=MemoryConfig(
                unified=True,
                importance_threshold=0.7,
                short_term_limit=100,
                long_term_limit=1000,
                episodic_retention_days=30,
                auto_promote=True,
            )
        )
        assert config.memory.unified is True
        assert config.memory.importance_threshold == 0.7

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"importance_threshold": 1.5},
            {"importance_threshold": -0.1},
            {"short_term_limit": 0},
            {"long_term_limit": 0},
            {"episodic_retention_days": 0},
            {"context_max_tokens": 0},
        ],
    )
    def test_invalid_params_rejected_when_unified(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(unified=True, **kwargs)

    def test_params_inert_when_unified_off(self) -> None:
        # Out-of-range tier params are not validated while unified is off.
        config = MemoryConfig(importance_threshold=1.5, short_term_limit=0)
        assert config._unified_enabled is False

    def test_mutually_exclusive_with_knowledge_memory(self, tmp_path: Any) -> None:
        km = KnowledgeMemory(directory=str(tmp_path / "km"), store=InMemoryKnowledgeStore())
        with pytest.raises(ValueError, match="mutually exclusive"):
            MemoryConfig(unified=True, knowledge_memory=km)

    def test_mutually_exclusive_with_entity_memory_and_kg(self) -> None:
        sentinel: Any = object()
        with pytest.raises(ValueError, match="mutually exclusive"):
            MemoryConfig(unified=True, entity_memory=sentinel)
        with pytest.raises(ValueError, match="mutually exclusive"):
            MemoryConfig(unified=True, knowledge_graph=sentinel)

    def test_instance_implies_enabled(self) -> None:
        config = MemoryConfig(unified_memory=UnifiedMemory())
        assert config.unified is False
        assert config._unified_enabled is True

    def test_dict_unpack_via_agentconfig(self) -> None:
        config = AgentConfig(memory={"unified": True, "short_term_limit": 4})
        assert isinstance(config.memory, MemoryConfig)
        assert config.memory.unified is True
        assert config.memory.short_term_limit == 4

    def test_dict_unpack_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(memory={"unified": True, "importance_threshold": 2.0})


# ----------------------------------------------------------------------
# Agent construction
# ----------------------------------------------------------------------


class TestAgentConstruction:
    def test_agent_builds_unified_memory_from_config(self) -> None:
        agent = make_agent(short_term_limit=8, importance_threshold=0.8, auto_promote=False)
        assert isinstance(agent.unified_memory, UnifiedMemory)
        assert agent.unified_memory.short_term.max_messages == 8
        assert agent.unified_memory.importance_threshold == 0.8
        assert agent.unified_memory.auto_promote is False
        assert agent.memory is None

    def test_injected_instance_used_as_is(self) -> None:
        memory = UnifiedMemory(short_term_limit=4)
        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(memory=MemoryConfig(unified_memory=memory)),
        )
        assert agent.unified_memory is memory

    def test_default_agent_has_no_unified_memory(self) -> None:
        agent = Agent(tools=[echo], provider=LocalProvider())
        assert agent.unified_memory is None

    def test_conflicts_with_memory_param(self) -> None:
        with pytest.raises(ValueError, match="short-term tier"):
            Agent(
                tools=[echo],
                provider=LocalProvider(),
                config=AgentConfig(memory=MemoryConfig(unified=True)),
                memory=ConversationMemory(),
            )

    def test_conflicts_with_session_store(self, tmp_path: Any) -> None:
        from selectools.sessions import JsonFileSessionStore

        store = JsonFileSessionStore(directory=str(tmp_path / "sessions"))
        with pytest.raises(ValueError, match="session_store"):
            Agent(
                tools=[echo],
                provider=LocalProvider(),
                config=AgentConfig(
                    memory=MemoryConfig(unified=True),
                    session_store=store,
                    session_id="s1",
                ),
            )

    def test_legacy_knowledge_memory_path_unchanged(self, tmp_path: Any) -> None:
        km = KnowledgeMemory(directory=str(tmp_path / "km"), store=InMemoryKnowledgeStore())
        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(memory=MemoryConfig(knowledge_memory=km)),
        )
        assert agent.unified_memory is None
        assert "remember" in agent._tools_by_name


# ----------------------------------------------------------------------
# Agent integration (offline providers)
# ----------------------------------------------------------------------


class TestAgentIntegration:
    def test_turn_persisted_across_tiers(self) -> None:
        agent = make_agent()
        result = agent.run([Message(role=Role.USER, content="My name is Alice")])
        unified = agent.unified_memory
        assert unified is not None
        history = unified.short_term.get_history()
        assert len(history) == 2
        assert history[0].content == "My name is Alice"
        assert history[1].content == result.content
        assert len(unified.episodic) == 1

    def test_stm_to_ltm_promotion_on_age_out(self) -> None:
        # Window of 2 messages: turn 1 ages out when turn 2 lands.
        agent = make_agent(short_term_limit=2, importance_threshold=0.7)
        agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
        unified = agent.unified_memory
        assert unified is not None
        assert lt_contents(unified) == []  # still inside the window

        agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
        promoted = lt_contents(unified)
        assert any("My name is Alice Smith" in c for c in promoted)
        entries = unified.long_term.store.query(limit=10)
        assert any(e.category == "identity" for e in entries)

    def test_mundane_turns_not_promoted(self) -> None:
        agent = make_agent(short_term_limit=2)
        agent.run([Message(role=Role.USER, content="What time is it?")])
        agent.run([Message(role=Role.USER, content="And what day is it?")])
        unified = agent.unified_memory
        assert unified is not None
        assert lt_contents(unified) == []

    def test_auto_promote_false_disables_promotion(self) -> None:
        agent = make_agent(short_term_limit=2, auto_promote=False)
        agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
        agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
        unified = agent.unified_memory
        assert unified is not None
        assert lt_contents(unified) == []

    def test_promoted_knowledge_injected_as_context(self) -> None:
        provider = CapturingProvider()
        agent = make_agent(provider=provider, short_term_limit=2)
        agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
        agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
        agent.run([Message(role=Role.USER, content="Do you know my name?")])

        last_call = provider.calls[-1]
        system_msgs = [m for m in last_call if m.role == Role.SYSTEM]
        assert any(
            "My name is Alice Smith" in (m.content or "")
            and "[Important Knowledge]" in (m.content or "")
            for m in system_msgs
        ), "promoted fact should be injected via unified context"
        # Episodic section present too (turns recorded earlier today).
        assert any("[Recent Episodes]" in (m.content or "") for m in system_msgs)
        # The conversation tier must NOT be duplicated into the system context.
        assert all("[Conversation]" not in (m.content or "") for m in system_msgs)

    def test_short_term_history_sent_as_messages(self) -> None:
        provider = CapturingProvider()
        agent = make_agent(provider=provider, short_term_limit=10)
        agent.run([Message(role=Role.USER, content="First message")])
        agent.run([Message(role=Role.USER, content="Second message")])

        last_call = provider.calls[-1]
        user_contents = [m.content for m in last_call if m.role == Role.USER]
        assert "First message" in user_contents
        assert "Second message" in user_contents
        # System context messages never leak into the short-term tier.
        unified = agent.unified_memory
        assert unified is not None
        assert all(m.role in (Role.USER, Role.ASSISTANT) for m in unified.short_term.get_history())

    def test_context_persists_across_unrelated_turns(self) -> None:
        # Promotion happens because of importance, not recency of mention.
        agent = make_agent(short_term_limit=2)
        agent.run([Message(role=Role.USER, content="I live in Porto Alegre")])  # 0.6 < 0.7
        agent.run([Message(role=Role.USER, content="My favourite tea is green tea")])  # 0.75
        agent.run([Message(role=Role.USER, content="ok")])
        agent.run([Message(role=Role.USER, content="ok again")])
        unified = agent.unified_memory
        assert unified is not None
        promoted = lt_contents(unified)
        assert any("favourite tea" in c for c in promoted)
        assert not any("Porto Alegre" in c for c in promoted)

    def test_recall_reaches_promoted_facts(self) -> None:
        agent = make_agent(short_term_limit=2)
        agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
        agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
        unified = agent.unified_memory
        assert unified is not None
        results = unified.recall("Alice name")
        assert results and results[0].source == "long_term"

    def test_reset_clears_short_term_keeps_long_term(self) -> None:
        agent = make_agent(short_term_limit=2)
        agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
        agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
        unified = agent.unified_memory
        assert unified is not None
        assert lt_contents(unified)

        agent.reset()
        assert len(unified.short_term) == 0
        assert len(unified.episodic) == 0
        assert lt_contents(unified)  # long-term survives reset

    def test_clone_for_isolation_drops_unified_memory(self) -> None:
        agent = make_agent()
        clone = agent.clone_for_isolation()
        assert clone.unified_memory is None
        assert agent.unified_memory is not None

    @pytest.mark.asyncio
    async def test_arun_drives_unified_memory(self) -> None:
        class AsyncCapturingProvider(CapturingProvider):
            supports_async = True

            async def acomplete(
                self, *, messages: List[Message], **kwargs: Any
            ) -> Tuple[Message, UsageStats]:
                return self.complete(messages=messages, **kwargs)

        agent = make_agent(provider=AsyncCapturingProvider(), short_term_limit=2)
        await agent.arun([Message(role=Role.USER, content="My name is Alice Smith")])
        await agent.arun([Message(role=Role.USER, content="What should I eat for lunch?")])
        unified = agent.unified_memory
        assert unified is not None
        assert any("Alice Smith" in c for c in lt_contents(unified))
        assert len(unified.episodic) == 2
