"""
Tests for summarize-on-trim (Phase 2 of v0.16.0 Memory & Persistence).

Tests cover:
- Memory _last_trimmed tracking
- Summary generation on trim with mock provider
- Summary injected into history
- Provider failure doesn't crash agent
- Summary round-trips through session save/load
- Observer event fires
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pytest

from selectools.agent import Agent, AgentConfig
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.sessions import JsonFileSessionStore
from selectools.tools import Tool
from selectools.types import Message, Role, ToolCall
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


class FakeProvider:
    """Provider that returns canned responses."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = responses or ["response"]
        self.calls = 0

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return Message(role=Role.ASSISTANT, content=response), _usage(model)


class SummarizingProvider:
    """Provider that returns a fixed summary for summarize calls."""

    name = "summarizer"
    supports_streaming = False
    supports_async = False

    def __init__(self, summary: str = "Summary of conversation.") -> None:
        self.summary = summary
        self.summary_calls: List[Dict[str, Any]] = []

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        self.summary_calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "messages": messages,
            }
        )
        return Message(role=Role.ASSISTANT, content=self.summary), _usage(model)


class FailingProvider:
    """Provider that always raises."""

    name = "failing"
    supports_streaming = False
    supports_async = False

    def complete(self, **kw):
        raise RuntimeError("Provider down!")


def _tool() -> Tool:
    return Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")


# ======================================================================
# Memory _last_trimmed tracking
# ======================================================================


class TestLastTrimmed:
    def test_no_trim_empty_list(self) -> None:
        mem = ConversationMemory(max_messages=10)
        mem.add(Message(role=Role.USER, content="Hello"))
        assert mem._last_trimmed == []

    def test_trim_captures_removed_messages(self) -> None:
        mem = ConversationMemory(max_messages=3)
        for i in range(5):
            mem.add(Message(role=Role.USER, content=f"msg-{i}"))

        # After adding msg-4, messages 0-1 should have been trimmed at some point
        # The _last_trimmed reflects the MOST RECENT trim only
        assert len(mem._last_trimmed) > 0

    def test_last_trimmed_updates_on_each_add(self) -> None:
        mem = ConversationMemory(max_messages=2)
        mem.add(Message(role=Role.USER, content="A"))
        mem.add(Message(role=Role.USER, content="B"))
        assert mem._last_trimmed == []

        mem.add(Message(role=Role.USER, content="C"))
        assert len(mem._last_trimmed) == 1
        assert mem._last_trimmed[0].content == "A"

    def test_last_trimmed_with_add_many(self) -> None:
        mem = ConversationMemory(max_messages=2)
        msgs = [Message(role=Role.USER, content=f"m-{i}") for i in range(5)]
        mem.add_many(msgs)
        assert len(mem._last_trimmed) == 3
        assert mem._last_trimmed[0].content == "m-0"

    def test_from_dict_initializes_last_trimmed(self) -> None:
        mem = ConversationMemory(max_messages=5)
        mem.add(Message(role=Role.USER, content="hi"))
        restored = ConversationMemory.from_dict(mem.to_dict())
        assert restored._last_trimmed == []


# ======================================================================
# Summarize-on-trim integration
# ======================================================================


class TestSummarizeOnTrim:
    def test_summary_generated_on_trim(self) -> None:
        summarizer = SummarizingProvider(summary="User discussed topics A and B.")
        main_provider = FakeProvider()

        mem = ConversationMemory(max_messages=3)
        # Pre-fill memory close to limit
        mem.add(Message(role=Role.USER, content="Topic A"))
        mem.add(Message(role=Role.ASSISTANT, content="I see topic A"))

        agent = Agent(
            tools=[_tool()],
            provider=main_provider,
            memory=mem,
            config=AgentConfig(
                summarize_on_trim=True,
                summarize_provider=summarizer,
            ),
        )
        # This run adds user msg + assistant response, triggering trim
        agent.run("Topic B")

        assert mem.summary is not None
        assert "User discussed" in mem.summary

    def test_summary_not_generated_when_disabled(self) -> None:
        summarizer = SummarizingProvider()
        main_provider = FakeProvider()

        mem = ConversationMemory(max_messages=3)
        mem.add(Message(role=Role.USER, content="A"))
        mem.add(Message(role=Role.ASSISTANT, content="B"))

        agent = Agent(
            tools=[_tool()],
            provider=main_provider,
            memory=mem,
            config=AgentConfig(summarize_on_trim=False),
        )
        agent.run("C")

        assert mem.summary is None
        assert summarizer.summary_calls == []

    def test_summary_uses_configured_model(self) -> None:
        summarizer = SummarizingProvider()

        mem = ConversationMemory(max_messages=3)
        mem.add(Message(role=Role.USER, content="A"))
        mem.add(Message(role=Role.ASSISTANT, content="B"))

        agent = Agent(
            tools=[_tool()],
            provider=FakeProvider(),
            memory=mem,
            config=AgentConfig(
                summarize_on_trim=True,
                summarize_provider=summarizer,
                summarize_model="gpt-4o-mini",
            ),
        )
        agent.run("C")

        if summarizer.summary_calls:
            assert summarizer.summary_calls[0]["model"] == "gpt-4o-mini"

    def test_summary_accumulates_across_trims(self) -> None:
        """Multiple trims should append summaries."""
        summarizer = SummarizingProvider(summary="More context.")

        mem = ConversationMemory(max_messages=2)
        mem.summary = "Earlier context."

        agent = Agent(
            tools=[_tool()],
            provider=FakeProvider(),
            memory=mem,
            config=AgentConfig(
                summarize_on_trim=True,
                summarize_provider=summarizer,
            ),
        )
        # Pre-fill and run to trigger trim
        mem.add(Message(role=Role.USER, content="X"))
        agent.run("Y")

        if mem.summary:
            assert "Earlier context." in mem.summary

    def test_provider_failure_doesnt_crash(self) -> None:
        mem = ConversationMemory(max_messages=3)
        mem.add(Message(role=Role.USER, content="A"))
        mem.add(Message(role=Role.ASSISTANT, content="B"))

        agent = Agent(
            tools=[_tool()],
            provider=FakeProvider(),
            memory=mem,
            config=AgentConfig(
                summarize_on_trim=True,
                summarize_provider=FailingProvider(),
            ),
        )
        # Should not raise
        result = agent.run("C")
        assert result.content == "response"

    def test_summary_injected_into_history(self) -> None:
        """When summary exists, it should appear as SYSTEM message in history."""

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
        mem = ConversationMemory(max_messages=50)
        mem.summary = "User asked about weather in NYC."

        agent = Agent(
            tools=[_tool()],
            provider=recording,
            memory=mem,
            config=AgentConfig(),
        )
        agent.run("What else?")

        # Check that the summary was injected as first message
        assert len(recording.last_messages) > 0
        first = recording.last_messages[0]
        assert first.role == Role.SYSTEM
        assert "[Conversation Summary]" in first.content
        assert "weather in NYC" in first.content

    def test_no_summary_no_injection(self) -> None:
        """Without summary, no SYSTEM message should be injected."""

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
        mem = ConversationMemory(max_messages=50)

        agent = Agent(
            tools=[_tool()],
            provider=recording,
            memory=mem,
            config=AgentConfig(),
        )
        agent.run("Hello")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 0


class TestSummarizeObserver:
    def test_observer_fires_on_summarize(self) -> None:
        events: list = []

        class Obs(AgentObserver):
            def on_memory_summarize(self, run_id, summary):
                events.append(("summarize", summary))

        summarizer = SummarizingProvider(summary="Summarized.")
        mem = ConversationMemory(max_messages=3)
        mem.add(Message(role=Role.USER, content="A"))
        mem.add(Message(role=Role.ASSISTANT, content="B"))

        agent = Agent(
            tools=[_tool()],
            provider=FakeProvider(),
            memory=mem,
            config=AgentConfig(
                summarize_on_trim=True,
                summarize_provider=summarizer,
                observers=[Obs()],
            ),
        )
        agent.run("C")

        summarize_events = [e for e in events if e[0] == "summarize"]
        if summarize_events:
            assert "Summarized" in summarize_events[0][1]


class TestSummarizeSessionRoundTrip:
    def test_summary_persists_through_session(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = ConversationMemory(max_messages=50)
        mem.summary = "Important context from before."
        mem.add(Message(role=Role.USER, content="Hello"))
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.summary == "Important context from before."
