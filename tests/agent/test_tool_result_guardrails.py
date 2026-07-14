"""Agent-loop integration tests for tool-results guardrails (issue #165).

tool_args (#160) gated a tool's incoming arguments; this stage gates the
RETURN value before it re-enters the model context — the other half of the
tool-boundary guardrail surface. Covers single and parallel execution across
run(), arun(), and astream().
"""

from __future__ import annotations

from typing import Any

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    GuardrailsPipeline,
)
from selectools.trace import StepType
from selectools.types import AgentResult, Message, Role, ToolCall
from tests.conftest import SharedFakeProvider


@tool(description="Fetches remote content (returns a payload with a secret).")
def fetch(url: str) -> str:
    return f"fetched from {url}: secret-token payload"


@tool(description="Another fetch tool for parallel runs.")
def fetch_b(url: str) -> str:
    return f"B fetched {url} with secret-token"


class _RedactSecret(Guardrail):
    name = "redact-secret"
    action = GuardrailAction.REWRITE

    def check(self, content: str) -> GuardrailResult:
        if "secret-token" in content:
            return GuardrailResult(
                passed=False,
                content=content.replace("secret-token", "[REDACTED]"),
                reason="secret found",
                guardrail_name=self.name,
            )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


class _BlockSecret(Guardrail):
    name = "block-secret"
    action = GuardrailAction.BLOCK

    def check(self, content: str) -> GuardrailResult:
        if "secret-token" in content:
            return GuardrailResult(passed=False, content=content, reason="secret found")
        return GuardrailResult(passed=True, content=content)


def _one_call_provider() -> SharedFakeProvider:
    first = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name="fetch", parameters={"url": "http://x"}, id="tc1")],
    )
    return SharedFakeProvider([first, "Done"])


def _two_call_provider() -> SharedFakeProvider:
    first = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(tool_name="fetch", parameters={"url": "http://x"}, id="tc1"),
            ToolCall(tool_name="fetch_b", parameters={"url": "http://y"}, id="tc2"),
        ],
    )
    return SharedFakeProvider([first, "Done"])


def _agent(provider: Any, guardrails: GuardrailsPipeline, **cfg: Any) -> Agent:
    return Agent(
        [fetch, fetch_b],
        provider=provider,
        config=AgentConfig(guardrails=guardrails, model="fake-model", **cfg),
    )


def _tool_messages(provider: SharedFakeProvider) -> list:
    return [m for m in provider.last_messages if m.role == Role.TOOL]


class TestRunPath:
    def test_rewrite_sanitizes_result_before_history(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert len(tool_msgs) == 1
        assert "[REDACTED]" in tool_msgs[0].content
        assert "secret-token" not in tool_msgs[0].content

    def test_block_raises_guardrail_error(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_BlockSecret()]))
        with pytest.raises(GuardrailError, match="secret found"):
            agent.run("go")

    def test_no_tool_results_guardrails_passes_raw(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(output=[_RedactSecret()]))
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert "secret-token" in tool_msgs[0].content

    def test_parallel_execution_guards_all_results(self) -> None:
        provider = _two_call_provider()
        agent = _agent(
            provider,
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            parallel_tool_execution=True,
        )
        agent.run("go")
        tool_msgs = _tool_messages(provider)
        assert len(tool_msgs) == 2
        for m in tool_msgs:
            assert "secret-token" not in m.content
            assert "[REDACTED]" in m.content

    def test_trace_records_guardrail_step(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        result = agent.run("go")
        steps = [s for s in result.trace.steps if s.type == StepType.GUARDRAIL]
        assert any("redact-secret" in (s.summary or "") for s in steps)


class TestAsyncPaths:
    @pytest.mark.asyncio
    async def test_arun_rewrite(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        await agent.arun("go")
        tool_msgs = _tool_messages(provider)
        assert "secret-token" not in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_arun_parallel_rewrite(self) -> None:
        provider = _two_call_provider()
        agent = _agent(
            provider,
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            parallel_tool_execution=True,
        )
        await agent.arun("go")
        for m in _tool_messages(provider):
            assert "secret-token" not in m.content

    @pytest.mark.asyncio
    async def test_arun_block_raises(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_BlockSecret()]))
        with pytest.raises(GuardrailError):
            await agent.arun("go")

    @pytest.mark.asyncio
    async def test_astream_rewrite(self) -> None:
        provider = _one_call_provider()
        agent = _agent(provider, GuardrailsPipeline(tool_results=[_RedactSecret()]))
        async for chunk in agent.astream("go"):
            if isinstance(chunk, AgentResult):
                break
        tool_msgs = _tool_messages(provider)
        assert "secret-token" not in tool_msgs[0].content


# ── Review follow-ups (PR #168 self-review) ─────────────────────────────


class _TerminalEventRecorder:
    """Observer capturing tool lifecycle terminal events."""

    def __init__(self) -> None:
        self.starts: list = []
        self.ends: list = []
        self.errors: list = []

    def on_tool_start(self, run_id, call_id, tool_name, parameters) -> None:
        self.starts.append(tool_name)

    def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms) -> None:
        self.ends.append((tool_name, result))

    def on_tool_error(self, run_id, call_id, tool_name, error, parameters, duration_ms) -> None:
        self.errors.append(tool_name)

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


class TestBlockStateCoherence:
    """A block must abort the run WITHOUT corrupting conversation state."""

    def test_block_leaves_memory_coherent(self) -> None:
        """The assistant tool_call message must be followed by a TOOL result
        in memory even when the result was blocked — otherwise the next run
        replays a dangling tool_call and the provider 400s."""
        from selectools.memory import ConversationMemory

        provider = _one_call_provider()
        memory = ConversationMemory(max_messages=50)
        agent = Agent(
            [fetch, fetch_b],
            provider=provider,
            memory=memory,
            config=AgentConfig(
                guardrails=GuardrailsPipeline(tool_results=[_BlockSecret()]),
                model="fake-model",
            ),
        )
        with pytest.raises(GuardrailError):
            agent.run("go")

        history = memory.get_history()
        dangling = [
            i
            for i, m in enumerate(history)
            if m.role == Role.ASSISTANT
            and m.tool_calls
            and not any(later.role == Role.TOOL for later in history[i + 1 :])
        ]
        assert dangling == [], "assistant tool_call without a TOOL result wedges the next turn"
        tool_msgs = [m for m in history if m.role == Role.TOOL]
        assert tool_msgs, "the blocked call still needs a TOOL message"
        assert "blocked" in tool_msgs[-1].content.lower()
        assert "secret-token" not in tool_msgs[-1].content

    def test_block_fires_terminal_observer_event(self) -> None:
        obs = _TerminalEventRecorder()
        provider = _one_call_provider()
        agent = Agent(
            [fetch, fetch_b],
            provider=provider,
            config=AgentConfig(
                guardrails=GuardrailsPipeline(tool_results=[_BlockSecret()]),
                model="fake-model",
                observers=[obs],
            ),
        )
        with pytest.raises(GuardrailError):
            agent.run("go")
        assert obs.starts, "on_tool_start fired"
        assert obs.ends or obs.errors, "a terminal tool event must fire on block"
        if obs.ends:
            assert "secret-token" not in obs.ends[0][1], "raw content must not leak to observers"

    def test_parallel_block_preserves_sibling_results(self) -> None:
        """A block on one tool must not discard the sibling's completed
        result from durable state (memory). For memory-less agents the
        run-level exception rollback intentionally truncates _history."""
        from selectools.memory import ConversationMemory

        provider = _two_call_provider()

        class _BlockOnlyB(Guardrail):
            name = "block-b"
            action = GuardrailAction.BLOCK

            def check(self, content: str) -> GuardrailResult:
                if content.startswith("B fetched"):
                    return GuardrailResult(passed=False, content=content, reason="B blocked")
                return GuardrailResult(passed=True, content=content)

        memory = ConversationMemory(max_messages=50)
        agent = Agent(
            [fetch, fetch_b],
            provider=provider,
            memory=memory,
            config=AgentConfig(
                guardrails=GuardrailsPipeline(tool_results=[_BlockOnlyB()]),
                model="fake-model",
                parallel_tool_execution=True,
            ),
        )
        with pytest.raises(GuardrailError):
            agent.run("go")
        tool_msgs = [m for m in memory.get_history() if m.role == Role.TOOL]
        assert any("fetched from http://x" in m.content for m in tool_msgs), (
            "sibling A's completed result must be recorded before the raise"
        )
        assert not any("B fetched" in m.content for m in tool_msgs), (
            "blocked content must not reach memory"
        )
        assert any("blocked" in m.content.lower() for m in tool_msgs), (
            "the blocked call still gets a marker TOOL message"
        )

    def test_tool_body_guardrail_error_degrades_like_any_tool_error(self) -> None:
        """A GuardrailError raised INSIDE a tool body (e.g. a nested agent's
        own guardrails) is a tool failure, not an outer policy block — it must
        degrade to an error-result message, not abort the run."""

        @tool(description="Tool whose body raises GuardrailError.")
        def nested_agent_tool(query: str) -> str:
            raise GuardrailError(guardrail_name="inner", reason="inner block")

        first = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(tool_name="nested_agent_tool", parameters={"query": "x"}, id="tc1")
            ],
        )
        provider = SharedFakeProvider([first, "Recovered"])
        agent = Agent(
            [nested_agent_tool],
            provider=provider,
            config=AgentConfig(model="fake-model"),  # NO tool_results guardrails
        )
        result = agent.run("go")
        assert result.content == "Recovered"
        tool_msgs = [m for m in provider.last_messages if m.role == Role.TOOL]
        assert any("inner block" in m.content for m in tool_msgs)


class TestCacheHitGuarding:
    def test_cached_result_still_goes_through_guardrails(self) -> None:
        """A cache entry written before a guardrail was configured must not
        bypass the stage on later hits."""
        from selectools.cache import InMemoryCache

        @tool(description="Cacheable fetch.", cacheable=True, cache_ttl=300)
        def cached_fetch(url: str) -> str:
            return f"cached payload with secret-token from {url}"

        def _mk_provider() -> SharedFakeProvider:
            first = Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(tool_name="cached_fetch", parameters={"url": "http://x"}, id="tc1")
                ],
            )
            return SharedFakeProvider([first, "Done"])

        cache = InMemoryCache()
        # Prime the cache WITHOUT guardrails (pre-guardrail deployment).
        agent1 = Agent(
            [cached_fetch],
            provider=_mk_provider(),
            config=AgentConfig(model="fake-model", cache=cache),
        )
        agent1.run("go")

        # Redeploy with a rewrite guardrail; the hit must be sanitized.
        provider2 = _mk_provider()
        agent2 = Agent(
            [cached_fetch],
            provider=provider2,
            config=AgentConfig(
                model="fake-model",
                cache=cache,
                guardrails=GuardrailsPipeline(tool_results=[_RedactSecret()]),
            ),
        )
        agent2.run("go")
        tool_msgs = [m for m in provider2.last_messages if m.role == Role.TOOL]
        assert tool_msgs and "secret-token" not in tool_msgs[0].content
        assert "[REDACTED]" in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_cached_result_guarded_in_arun(self) -> None:
        from selectools.cache import InMemoryCache

        @tool(description="Cacheable fetch async.", cacheable=True, cache_ttl=300)
        def cached_fetch_a(url: str) -> str:
            return f"cached payload with secret-token from {url}"

        def _mk_provider() -> SharedFakeProvider:
            first = Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(tool_name="cached_fetch_a", parameters={"url": "http://x"}, id="tc1")
                ],
            )
            return SharedFakeProvider([first, "Done"])

        cache = InMemoryCache()
        agent1 = Agent(
            [cached_fetch_a],
            provider=_mk_provider(),
            config=AgentConfig(model="fake-model", cache=cache),
        )
        await agent1.arun("go")

        provider2 = _mk_provider()
        agent2 = Agent(
            [cached_fetch_a],
            provider=provider2,
            config=AgentConfig(
                model="fake-model",
                cache=cache,
                guardrails=GuardrailsPipeline(tool_results=[_RedactSecret()]),
            ),
        )
        await agent2.arun("go")
        tool_msgs = [m for m in provider2.last_messages if m.role == Role.TOOL]
        assert tool_msgs and "secret-token" not in tool_msgs[0].content
