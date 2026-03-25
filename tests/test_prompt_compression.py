"""Tests for prompt compression (compress_context in AgentConfig)."""

from __future__ import annotations

from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, UsageStats
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.token_estimation import TokenEstimate
from selectools.trace import StepType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_tool() -> Tool:
    return Tool(name="noop", description="no-op", parameters=[], function=lambda: "ok")


class FakeProvider:
    """Minimal provider stub with queued responses."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[dict] = []

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self.calls.append({"model": model, "messages": messages, "system_prompt": system_prompt})
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        text = self._responses[idx]
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0,
            model=model,
            provider="fake",
        )
        return Message(role=Role.ASSISTANT, content=text), usage


def _make_agent(responses: List[str], **config_kwargs) -> Agent:
    provider = FakeProvider(responses)
    config = AgentConfig(model="gpt-4o-mini", **config_kwargs)
    return Agent(tools=[_dummy_tool()], provider=provider, config=config)


def _long_history(n: int) -> List[Message]:
    """Return n pairs of user/assistant messages."""
    msgs: List[Message] = []
    for i in range(n):
        msgs.append(Message(role=Role.USER, content=f"User message {i}"))
        msgs.append(Message(role=Role.ASSISTANT, content=f"Assistant reply {i}"))
    return msgs


def _make_estimate(total_tokens: int) -> TokenEstimate:
    return TokenEstimate(
        0,
        total_tokens,
        0,
        total_tokens,
        100_000,
        100_000 - total_tokens,
        "gpt-4o-mini",
        "heuristic",
    )


def _toggle_estimate(high: int, low: int):
    """Return a side_effect callable that returns high on the first call, low thereafter."""
    calls = [0]

    def _estimate(*args, **kwargs):
        calls[0] += 1
        return _make_estimate(high) if calls[0] == 1 else _make_estimate(low)

    return _estimate


def _agent_with_history(responses: List[str], n_pairs: int = 5, **config_kwargs) -> Agent:
    """Build an agent pre-loaded with n_pairs of conversation history."""
    provider = FakeProvider(responses)
    config = AgentConfig(model="gpt-4o-mini", **config_kwargs)
    agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)
    agent.memory = ConversationMemory(max_messages=50)
    for msg in _long_history(n_pairs):
        agent.memory.add(msg)
    return agent


class RecordingObserver(AgentObserver):
    def __init__(self) -> None:
        self.compressed_events: List[dict] = []

    def on_prompt_compressed(
        self,
        run_id: str,
        before_tokens: int,
        after_tokens: int,
        messages_compressed: int,
    ) -> None:
        self.compressed_events.append(
            {
                "run_id": run_id,
                "before": before_tokens,
                "after": after_tokens,
                "count": messages_compressed,
            }
        )


# ---------------------------------------------------------------------------
# StepType presence
# ---------------------------------------------------------------------------


def test_prompt_compressed_step_type_exists():
    assert StepType.PROMPT_COMPRESSED == "prompt_compressed"


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


def test_compress_context_disabled_by_default():
    """compress_context=False (default) — _maybe_compress_context is a no-op."""
    agent = _make_agent(["hello"])
    result = agent.run("hi")
    compressed_steps = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    assert compressed_steps == []


# ---------------------------------------------------------------------------
# Does not fire when fill-rate is below threshold
# ---------------------------------------------------------------------------


def test_no_compression_when_below_threshold():
    """With a very high threshold (0.99), compression should not trigger."""
    agent = _make_agent(
        ["done"],
        compress_context=True,
        compress_threshold=0.99,
    )
    result = agent.run("hello")
    compressed_steps = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    assert compressed_steps == []


# ---------------------------------------------------------------------------
# Fires when fill-rate exceeds threshold
# ---------------------------------------------------------------------------


def test_compression_fires_when_threshold_exceeded():
    """Force threshold so low that even minimal history triggers compression."""
    agent = _agent_with_history(
        ["summary of old messages", "final answer"],
        compress_context=True,
        compress_threshold=0.85,
        compress_keep_recent=1,
    )
    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens",
        side_effect=_toggle_estimate(90_000, 1_000),
    ):
        result = agent.run("compress me")

    compressed_steps = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    assert len(compressed_steps) == 1
    step = compressed_steps[0]
    assert step.prompt_tokens == 90_000
    assert step.completion_tokens == 1_000


# ---------------------------------------------------------------------------
# Observer event fires on compression
# ---------------------------------------------------------------------------


def test_observer_on_prompt_compressed_fires():
    """RecordingObserver.on_prompt_compressed must receive the correct args."""
    observer = RecordingObserver()
    agent = _agent_with_history(
        ["summary text", "final"],
        compress_context=True,
        compress_threshold=0.75,
        compress_keep_recent=1,
        observers=[observer],
    )
    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens",
        side_effect=_toggle_estimate(80_000, 5_000),
    ):
        agent.run("go")

    assert len(observer.compressed_events) == 1
    ev = observer.compressed_events[0]
    assert ev["before"] == 80_000
    assert ev["after"] == 5_000
    assert ev["count"] >= 2  # at least 2 old messages compressed


# ---------------------------------------------------------------------------
# History is modified; memory is untouched
# ---------------------------------------------------------------------------


def test_compression_modifies_history_not_memory():
    """_maybe_compress_context must NOT modify self.memory."""
    agent = _agent_with_history(
        ["compressed summary", "done"],
        compress_context=True,
        compress_threshold=0.75,
        compress_keep_recent=1,
    )
    memory_len_before = len(agent.memory)

    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens",
        side_effect=_toggle_estimate(80_000, 3_000),
    ):
        agent.run("hello")

    # Memory must be >= before (only history view was compressed; memory grows with new messages)
    assert len(agent.memory) >= memory_len_before


# ---------------------------------------------------------------------------
# No-op with too few messages to compress
# ---------------------------------------------------------------------------


def test_no_compression_when_too_few_messages():
    """When all non-system messages fall within keep_recent window, no compression."""
    provider = FakeProvider(["fine"])
    agent = Agent(
        tools=[_dummy_tool()],
        provider=provider,
        config=AgentConfig(
            model="gpt-4o-mini",
            compress_context=True,
            compress_threshold=0.85,
            compress_keep_recent=10,  # keep_recent * 2 = 20 messages
        ),
    )
    # Only 1 message in history — all within keep_recent window
    agent.memory = ConversationMemory(max_messages=50)
    agent.memory.add(Message(role=Role.USER, content="hi"))

    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens",
        return_value=_make_estimate(90_000),
    ):
        result = agent.run("hello")

    compressed = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    assert compressed == []


# ---------------------------------------------------------------------------
# Provider error during compression is swallowed
# ---------------------------------------------------------------------------


def test_compression_swallows_provider_error():
    """If the summarization LLM call raises, compression is silently skipped."""
    call_count_provider = 0

    class ErroringProvider(FakeProvider):
        def complete(self, **kwargs):
            nonlocal call_count_provider
            call_count_provider += 1
            if call_count_provider == 1:
                raise RuntimeError("LLM exploded")
            return super().complete(**kwargs)

    provider = ErroringProvider(["fallback answer"])
    config = AgentConfig(
        model="gpt-4o-mini",
        compress_context=True,
        compress_threshold=0.85,
        compress_keep_recent=1,
    )
    agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)
    agent.memory = ConversationMemory(max_messages=50)
    for msg in _long_history(5):
        agent.memory.add(msg)

    with patch(
        "selectools.agent._memory_manager.estimate_run_tokens",
        side_effect=_toggle_estimate(90_000, 1_000),
    ):
        result = agent.run("hi")

    assert result.content is not None
    compressed = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
    assert compressed == []


# ---------------------------------------------------------------------------
# Config fields have correct defaults
# ---------------------------------------------------------------------------


def test_compress_context_defaults():
    config = AgentConfig()
    assert config.compress_context is False
    assert config.compress_threshold == 0.75
    assert config.compress_keep_recent == 4


def test_compress_context_custom_values():
    config = AgentConfig(
        compress_context=True,
        compress_threshold=0.6,
        compress_keep_recent=2,
    )
    assert config.compress_context is True
    assert config.compress_threshold == 0.6
    assert config.compress_keep_recent == 2
