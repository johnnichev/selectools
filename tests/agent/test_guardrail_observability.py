"""Observer events for guardrail trips (issue #167).

Guardrail outcomes were applied (raise / mutate / log) but never surfaced
through the observer infrastructure, so consumers wiring Langfuse or the
audit logger could not measure hit-rates. These tests pin the new
``on_guardrail_triggered(run_id, stage, guardrail_name, action, detail)``
event across all four stages and the block path.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    GuardrailsPipeline,
)
from selectools.observer import AgentObserver, AsyncAgentObserver
from selectools.types import AgentResult, Message, Role, ToolCall
from tests.conftest import SharedFakeProvider


@tool(description="Fetches content containing a secret.")
def fetch(url: str) -> str:
    return f"fetched {url}: secret-token"


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


class _Recorder(AgentObserver):
    def __init__(self) -> None:
        self.events: List[Tuple[str, str, str, Optional[str]]] = []

    def on_guardrail_triggered(
        self,
        run_id: str,
        stage: str,
        guardrail_name: str,
        action: str,
        detail: Optional[str] = None,
    ) -> None:
        self.events.append((stage, guardrail_name, action, detail))


class _AsyncRecorder(AsyncAgentObserver):
    blocking = True  # await inline so events are visible right after the run

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[str, str, str]] = []

    async def a_on_guardrail_triggered(
        self,
        run_id: str,
        stage: str,
        guardrail_name: str,
        action: str,
        detail: Optional[str] = None,
    ) -> None:
        self.events.append((stage, guardrail_name, action))


def _tool_call_provider() -> SharedFakeProvider:
    first = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name="fetch", parameters={"url": "use secret-token"}, id="tc1")],
    )
    return SharedFakeProvider([first, "Done"])


def _agent(guardrails: GuardrailsPipeline, observer: Any, provider: Any = None) -> Agent:
    return Agent(
        [fetch],
        provider=provider or SharedFakeProvider(["reply with secret-token"]),
        config=AgentConfig(guardrails=guardrails, model="fake-model", observers=[observer]),
    )


class TestStageEvents:
    def test_input_rewrite_event(self) -> None:
        obs = _Recorder()
        agent = _agent(GuardrailsPipeline(input=[_RedactSecret()]), obs)
        agent.run("please use secret-token")
        assert ("input", "redact-secret", "rewrite", None) in obs.events

    def test_output_rewrite_event(self) -> None:
        obs = _Recorder()
        agent = _agent(GuardrailsPipeline(output=[_RedactSecret()]), obs)
        agent.run("go")
        assert ("output", "redact-secret", "rewrite", None) in obs.events

    def test_tool_args_rewrite_event(self) -> None:
        obs = _Recorder()
        agent = _agent(
            GuardrailsPipeline(tool_args=[_RedactSecret()]), obs, provider=_tool_call_provider()
        )
        agent.run("go")
        assert ("tool_args", "redact-secret", "rewrite", None) in obs.events

    def test_tool_results_rewrite_event(self) -> None:
        obs = _Recorder()
        agent = _agent(
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            obs,
            provider=_tool_call_provider(),
        )
        agent.run("go")
        assert ("tool_results", "redact-secret", "rewrite", None) in obs.events

    def test_no_events_when_nothing_triggers(self) -> None:
        obs = _Recorder()
        agent = _agent(
            GuardrailsPipeline(input=[_RedactSecret()], output=[_RedactSecret()]),
            obs,
            provider=SharedFakeProvider(["clean reply"]),
        )
        agent.run("clean input")
        assert obs.events == []


class TestBlockEvents:
    def test_input_block_event_and_raise(self) -> None:
        obs = _Recorder()
        agent = _agent(GuardrailsPipeline(input=[_BlockSecret()]), obs)
        with pytest.raises(GuardrailError):
            agent.run("please use secret-token")
        assert ("input", "block-secret", "block", "secret found") in obs.events

    def test_output_block_event_and_raise(self) -> None:
        obs = _Recorder()
        agent = _agent(GuardrailsPipeline(output=[_BlockSecret()]), obs)
        with pytest.raises(GuardrailError):
            agent.run("go")
        assert ("output", "block-secret", "block", "secret found") in obs.events

    def test_tool_results_block_event_and_raise(self) -> None:
        obs = _Recorder()
        agent = _agent(
            GuardrailsPipeline(tool_results=[_BlockSecret()]),
            obs,
            provider=_tool_call_provider(),
        )
        with pytest.raises(GuardrailError):
            agent.run("go")
        assert ("tool_results", "block-secret", "block", "secret found") in obs.events


class TestAsyncEvents:
    @pytest.mark.asyncio
    async def test_arun_fires_async_event(self) -> None:
        obs = _AsyncRecorder()
        agent = _agent(GuardrailsPipeline(output=[_RedactSecret()]), obs)
        await agent.arun("go")
        assert ("output", "redact-secret", "rewrite") in obs.events

    @pytest.mark.asyncio
    async def test_astream_fires_async_event(self) -> None:
        obs = _AsyncRecorder()
        agent = _agent(
            GuardrailsPipeline(tool_results=[_RedactSecret()]),
            obs,
            provider=_tool_call_provider(),
        )
        async for chunk in agent.astream("go"):
            if isinstance(chunk, AgentResult):
                break
        assert ("tool_results", "redact-secret", "rewrite") in obs.events


class TestObserverBase:
    def test_base_observer_has_noop_method(self) -> None:
        AgentObserver().on_guardrail_triggered("rid", "input", "g", "rewrite", None)

    def test_audit_logger_writes_guardrail_event(self, tmp_path: Any) -> None:
        import json

        from selectools.audit import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path), daily_rotation=False)
        agent = _agent(GuardrailsPipeline(output=[_RedactSecret()]), logger)
        agent.run("go")
        log = tmp_path / "audit.jsonl"
        entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        guardrail_entries = [e for e in entries if e.get("event") == "guardrail_triggered"]
        assert guardrail_entries
        assert guardrail_entries[0]["stage"] == "output"
        assert guardrail_entries[0]["guardrail_name"] == "redact-secret"
        assert guardrail_entries[0]["action"] == "rewrite"


# ── Review follow-ups (PR #170 self-review) ─────────────────────────────


class TestAsyncBlockEvents:
    """Block events must reach async observers in arun/astream too — blocks
    are the most security-relevant guardrail outcome."""

    @pytest.mark.asyncio
    async def test_arun_input_block_fires_async_event(self) -> None:
        obs = _AsyncRecorder()
        agent = _agent(GuardrailsPipeline(input=[_BlockSecret()]), obs)
        with pytest.raises(GuardrailError):
            await agent.arun("please use secret-token")
        assert ("input", "block-secret", "block") in obs.events

    @pytest.mark.asyncio
    async def test_arun_output_block_fires_async_event(self) -> None:
        obs = _AsyncRecorder()
        agent = _agent(GuardrailsPipeline(output=[_BlockSecret()]), obs)
        with pytest.raises(GuardrailError):
            await agent.arun("go")
        assert ("output", "block-secret", "block") in obs.events

    @pytest.mark.asyncio
    async def test_astream_tool_results_block_fires_async_event(self) -> None:
        obs = _AsyncRecorder()
        agent = _agent(
            GuardrailsPipeline(tool_results=[_BlockSecret()]),
            obs,
            provider=_tool_call_provider(),
        )
        with pytest.raises(GuardrailError):
            async for _chunk in agent.astream("go"):
                pass
        assert ("tool_results", "block-secret", "block") in obs.events


class TestTripFidelity:
    def test_rewrite_before_block_emits_both_events(self) -> None:
        """A rewrite that mutated content before a later guardrail blocked
        must still be recorded — compliance audits rely on it."""
        obs = _Recorder()
        agent = _agent(
            GuardrailsPipeline(output=[_RedactSecret(), _BlockAll()]),
            obs,
        )
        with pytest.raises(GuardrailError):
            agent.run("go")
        stages_names_actions = [(s, n, a) for s, n, a, _d in obs.events]
        assert ("output", "redact-secret", "rewrite") in stages_names_actions
        assert ("output", "block-all", "block") in stages_names_actions

    def test_comma_in_guardrail_name_stays_one_event(self) -> None:
        class _CommaName(Guardrail):
            name = "pii, strict"
            action = GuardrailAction.REWRITE

            def check(self, content: str) -> GuardrailResult:
                return GuardrailResult(
                    passed=False,
                    content=content + " [checked]",
                    reason="always",
                    guardrail_name=self.name,
                )

        obs = _Recorder()
        agent = _agent(GuardrailsPipeline(output=[_CommaName()]), obs)
        agent.run("go")
        matching = [e for e in obs.events if e[1] == "pii, strict"]
        assert matching, f"expected a single event for the comma name, got {obs.events}"
        assert matching[0][2] == "rewrite"

    def test_block_exception_carries_agent_trace(self) -> None:
        agent = _agent(GuardrailsPipeline(output=[_BlockSecret()]), _Recorder())
        with pytest.raises(GuardrailError) as excinfo:
            agent.run("go")
        trace = getattr(excinfo.value, "agent_trace", None)
        assert trace is not None, "blocks must surface the run trace on the exception"
        from selectools.trace import StepType

        assert any(s.type == StepType.GUARDRAIL for s in trace.steps)


class _BlockAll(Guardrail):
    name = "block-all"
    action = GuardrailAction.BLOCK

    def check(self, content: str) -> GuardrailResult:
        return GuardrailResult(passed=False, content=content, reason="blocked all")
