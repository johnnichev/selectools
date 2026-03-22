"""
Tests for KnowledgeMemory (knowledge.py) and memory_tools.

Tests cover:
- remember() with daily log + persistent storage
- get_recent_logs and get_persistent_facts
- build_context output
- Truncation at max_context_chars
- prune_old_logs
- Round-trip serialization
- make_remember_tool binding
- Agent integration (context injection, auto-add remember tool)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from selectools.knowledge import KnowledgeMemory
from selectools.toolbox.memory_tools import make_remember_tool
from selectools.types import Message, Role
from selectools.usage import UsageStats


def _usage(model: str = "fake") -> UsageStats:
    return UsageStats(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.0001,
        model=model,
        provider="fake",
    )


# ======================================================================
# KnowledgeMemory — remember()
# ======================================================================


class TestKnowledgeRemember:
    def test_remember_creates_daily_log(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        result = km.remember("User prefers dark mode", category="preference")
        assert result  # returns entry ID

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = tmp_path / f"{today}.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "User prefers dark mode" in content
        assert "[preference]" in content

    def test_remember_persistent_writes_memory_md(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("User name is Alice", category="fact", persistent=True)

        mem_path = tmp_path / "MEMORY.md"
        assert mem_path.exists()
        content = mem_path.read_text()
        assert "User name is Alice" in content
        assert "[fact]" in content

    def test_remember_non_persistent_no_memory_md(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("Temporary note", persistent=False)

        mem_path = tmp_path / "MEMORY.md"
        assert not mem_path.exists()

    def test_remember_appends_to_existing_log(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("First note")
        km.remember("Second note")

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = tmp_path / f"{today}.log"
        content = log_path.read_text()
        assert "First note" in content
        assert "Second note" in content

    def test_remember_default_category(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("A note")

        today = datetime.now().strftime("%Y-%m-%d")
        content = (tmp_path / f"{today}.log").read_text()
        assert "[general]" in content


# ======================================================================
# KnowledgeMemory — get_recent_logs
# ======================================================================


class TestKnowledgeRecentLogs:
    def test_reads_today_log(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("Today's note")

        logs = km.get_recent_logs()
        assert "Today's note" in logs

    def test_reads_multiple_days(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=3)
        # Write a "yesterday" log manually
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday_path = tmp_path / f"{yesterday}.log"
        yesterday_path.write_text("[2024-01-01 12:00:00] [general] Yesterday note\n")

        km.remember("Today's note")
        logs = km.get_recent_logs()
        assert "Today's note" in logs
        assert "Yesterday note" in logs

    def test_no_logs_returns_empty(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        assert km.get_recent_logs() == ""

    def test_custom_days_parameter(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=1)
        km.remember("Today")
        logs = km.get_recent_logs(days=1)
        assert "Today" in logs


# ======================================================================
# KnowledgeMemory — get_persistent_facts
# ======================================================================


class TestKnowledgePersistentFacts:
    def test_reads_memory_md(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("Important fact", persistent=True)
        facts = km.get_persistent_facts()
        assert "Important fact" in facts

    def test_no_memory_md_returns_empty(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        assert km.get_persistent_facts() == ""


# ======================================================================
# KnowledgeMemory — build_context
# ======================================================================


class TestKnowledgeBuildContext:
    def test_empty_returns_empty_string(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        assert km.build_context() == ""

    def test_includes_persistent_and_recent(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("A persistent fact", persistent=True)
        km.remember("A daily note")

        ctx = km.build_context()
        assert "A persistent fact" in ctx
        assert "A daily note" in ctx

    def test_only_persistent(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        (tmp_path / "MEMORY.md").write_text("- [fact] Standalone fact\n")

        ctx = km.build_context()
        assert "[Long-term Memory]" in ctx
        assert "Standalone fact" in ctx

    def test_only_recent(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("Just a note")

        ctx = km.build_context()
        assert "Just a note" in ctx

    def test_truncation(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), max_context_chars=50)
        km.remember("A" * 100, persistent=True)

        ctx = km.build_context()
        assert len(ctx) < 200  # should be truncated
        assert "truncated" in ctx


# ======================================================================
# KnowledgeMemory — prune_old_logs
# ======================================================================


class TestKnowledgePrune:
    def test_prunes_old_logs(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=1)
        # Create an "old" log
        old_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        old_path = tmp_path / f"{old_date}.log"
        old_path.write_text("old note\n")

        km.remember("Today's note")
        removed = km.prune_old_logs()
        assert removed == 1
        assert not old_path.exists()

    def test_preserves_recent_logs(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=7)
        km.remember("Today's note")
        removed = km.prune_old_logs()
        assert removed == 0

    def test_preserves_memory_md(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=0)
        km.remember("Fact", persistent=True)
        km.prune_old_logs(keep_days=0)
        assert (tmp_path / "MEMORY.md").exists()

    def test_ignores_non_log_files(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=0)
        (tmp_path / "notes.txt").write_text("some notes")
        removed = km.prune_old_logs(keep_days=0)
        assert removed == 0
        assert (tmp_path / "notes.txt").exists()


# ======================================================================
# KnowledgeMemory — serialization
# ======================================================================


class TestKnowledgeSerialization:
    def test_round_trip(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path), recent_days=3, max_context_chars=8000)
        d = km.to_dict()
        restored = KnowledgeMemory.from_dict(d)
        assert restored._directory == str(tmp_path)
        assert restored._recent_days == 3
        assert restored._max_context_chars == 8000

    def test_defaults(self) -> None:
        km = KnowledgeMemory.from_dict({})
        assert km._directory == "./memory"
        assert km._recent_days == 2


# ======================================================================
# make_remember_tool
# ======================================================================


class TestMakeRememberTool:
    def test_creates_tool(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        tool = make_remember_tool(km)
        assert tool.name == "remember"
        assert "remember" in tool.description.lower() or "store" in tool.description.lower()

    def test_tool_invocation(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        tool = make_remember_tool(km)
        result = tool.function(content="Test fact", category="fact", persistent="true")
        assert result  # returns entry ID or confirmation

        facts = km.get_persistent_facts()
        assert "Test fact" in facts

    def test_tool_default_params(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        tool = make_remember_tool(km)
        result = tool.function(content="Simple note")
        assert result  # returns entry ID or confirmation

    def test_tool_persistent_false(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        tool = make_remember_tool(km)
        tool.function(content="Ephemeral note", persistent="false")
        assert not (tmp_path / "MEMORY.md").exists()


# ======================================================================
# Agent integration
# ======================================================================


class TestKnowledgeAgentIntegration:
    def _make_tool(self):
        from selectools.tools import Tool

        return Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")

    def test_knowledge_context_injected(self, tmp_path) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class RecordingProvider:
            name = "recording"
            supports_streaming = False
            supports_async = False

            def __init__(self):
                self.last_messages: List[Message] = []

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                self.last_messages = list(messages)
                return Message(role=Role.ASSISTANT, content="ok"), _usage(model)

        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("User prefers Python", persistent=True)
        km.remember("Today we discussed AI")

        recording = RecordingProvider()
        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_memory=km),
        )
        agent.run("Hello")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert any("Knowledge]" in m.content for m in system_msgs)

    def test_no_knowledge_no_injection(self, tmp_path) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class RecordingProvider:
            name = "recording"
            supports_streaming = False
            supports_async = False

            def __init__(self):
                self.last_messages: List[Message] = []

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                self.last_messages = list(messages)
                return Message(role=Role.ASSISTANT, content="ok"), _usage(model)

        recording = RecordingProvider()
        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(),
        )
        agent.run("Hello")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert not any("Knowledge]" in m.content for m in system_msgs)

    def test_remember_tool_auto_added(self, tmp_path) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class SimpleProvider:
            name = "simple"
            supports_streaming = False
            supports_async = False

            def complete(self, **kw):
                return Message(role=Role.ASSISTANT, content="ok"), _usage()

        km = KnowledgeMemory(directory=str(tmp_path))
        agent = Agent(
            tools=[self._make_tool()],
            provider=SimpleProvider(),
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_memory=km),
        )
        assert "remember" in agent._tools_by_name

    def test_remember_tool_not_duplicated(self, tmp_path) -> None:
        """If user already provides a remember tool, don't add another."""
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory
        from selectools.tools import Tool

        class SimpleProvider:
            name = "simple"
            supports_streaming = False
            supports_async = False

            def complete(self, **kw):
                return Message(role=Role.ASSISTANT, content="ok"), _usage()

        km = KnowledgeMemory(directory=str(tmp_path))
        custom_remember = Tool(
            name="remember",
            description="Custom remember",
            parameters=[],
            function=lambda: "custom",
        )
        agent = Agent(
            tools=[self._make_tool(), custom_remember],
            provider=SimpleProvider(),
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_memory=km),
        )
        # Should keep the custom one, not replace with auto-generated
        remember_tools = [t for t in agent.tools if t.name == "remember"]
        assert len(remember_tools) == 1
        assert remember_tools[0].description == "Custom remember"

    def test_empty_knowledge_no_context(self, tmp_path) -> None:
        """Empty knowledge memory produces no SYSTEM context message."""
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class RecordingProvider:
            name = "recording"
            supports_streaming = False
            supports_async = False

            def __init__(self):
                self.last_messages: List[Message] = []

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                self.last_messages = list(messages)
                return Message(role=Role.ASSISTANT, content="ok"), _usage(model)

        km = KnowledgeMemory(directory=str(tmp_path))
        recording = RecordingProvider()
        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_memory=km),
        )
        agent.run("Hello")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert not any("Knowledge]" in m.content for m in system_msgs)
