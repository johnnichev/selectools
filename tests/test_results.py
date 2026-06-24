"""Tests for selectools.results — ToolResult base class + Artifact side-channel (issue #59)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, asdict, dataclass
from typing import Any, ClassVar, List, Tuple

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.base import Provider
from selectools.results import (
    Ambiguous,
    Artifact,
    NotFound,
    ToolResult,
    emit_artifact,
)
from selectools.tools import tool
from selectools.types import AgentResult, Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    cost_usd=0.0,
    model="test",
    provider="test",
)


class _ToolThenDoneProvider(Provider):
    """First call returns the given tool calls, second call returns plain text."""

    name = "tool-then-done"
    supports_streaming = False
    supports_async = True

    def __init__(self, tool_calls: List[ToolCall]) -> None:
        self.default_model = "test"
        self._tool_calls = tool_calls
        self._call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        if self._call_count == 1:
            return (
                Message(role=Role.ASSISTANT, content="", tool_calls=list(self._tool_calls)),
                _DUMMY_USAGE,
            )
        return Message(role=Role.ASSISTANT, content="done"), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# ToolResult base class
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_base_kind_is_empty_classvar(self) -> None:
        assert ToolResult.kind == ""

    def test_builtin_kinds(self) -> None:
        assert Ambiguous.kind == "ambiguous"
        assert NotFound.kind == "not_found"

    def test_builtins_are_frozen(self) -> None:
        nf = NotFound(entity="customer", query="acme")
        with pytest.raises(FrozenInstanceError):
            nf.entity = "other"  # type: ignore[misc]

    def test_asdict_drops_classvar_kind(self) -> None:
        """Documents the footgun: ClassVar fields are excluded from asdict()."""
        nf = NotFound(entity="customer", query="acme")
        assert "kind" not in asdict(nf)

    def test_user_subclass(self) -> None:
        @dataclass(frozen=True)
        class BudgetExceeded(ToolResult):
            kind: ClassVar[str] = "budget_exceeded"
            limit: float = 0.0

        r = BudgetExceeded(limit=10.0)
        assert r.kind == "budget_exceeded"
        assert isinstance(r, ToolResult)


# ---------------------------------------------------------------------------
# Serializer round-trip: kind MUST survive Tool._serialize_result
# ---------------------------------------------------------------------------


class TestSerializerRoundTrip:
    def _serialize(self, result: Any) -> dict:
        @tool()
        def dummy() -> str:
            """Dummy."""
            return "ok"

        return json.loads(dummy._serialize_result(result))

    def test_ambiguous_kind_survives_serialization(self) -> None:
        r = Ambiguous(
            entity="customer",
            query="acme",
            matches=[{"id": 1, "name": "Acme Corp"}, {"id": 2, "name": "Acme Inc"}],
        )
        data = self._serialize(r)
        assert data["kind"] == "ambiguous"
        assert data["entity"] == "customer"
        assert data["query"] == "acme"
        assert data["matches"] == [{"id": 1, "name": "Acme Corp"}, {"id": 2, "name": "Acme Inc"}]

    def test_not_found_kind_survives_serialization(self) -> None:
        r = NotFound(entity="invoice", query="INV-42")
        data = self._serialize(r)
        assert data == {"kind": "not_found", "entity": "invoice", "query": "INV-42"}

    def test_user_subclass_kind_survives_serialization(self) -> None:
        @dataclass(frozen=True)
        class RateLimited(ToolResult):
            kind: ClassVar[str] = "rate_limited"
            retry_after: int = 0

        data = self._serialize(RateLimited(retry_after=30))
        assert data["kind"] == "rate_limited"
        assert data["retry_after"] == 30

    def test_kind_is_first_key_for_llm_readability(self) -> None:
        @tool()
        def dummy() -> str:
            """Dummy."""
            return "ok"

        raw = dummy._serialize_result(NotFound(entity="x", query="y"))
        assert raw.startswith('{"kind"')

    def test_tool_execute_emits_kind_in_llm_string(self) -> None:
        """End-to-end: the string the LLM sees contains the kind discriminator."""

        @tool()
        def lookup(query: str) -> NotFound:
            """Look something up."""
            return NotFound(entity="customer", query=query)

        out = lookup.execute({"query": "acme"})
        assert json.loads(out)["kind"] == "not_found"

    def test_plain_dataclass_serialization_unchanged(self) -> None:
        @dataclass
        class Plain:
            x: int

        data = self._serialize(Plain(x=1))
        assert data == {"x": 1}


# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------


class TestArtifact:
    def test_minimal_construction(self) -> None:
        a = Artifact(url="https://example.com/chart.png")
        assert a.url == "https://example.com/chart.png"
        assert a.mime_type is None
        assert a.filename is None
        assert a.sha256 is None
        assert a.size is None
        assert a.role is None
        assert a.retention is None

    def test_enriched_construction(self) -> None:
        a = Artifact(
            url="https://example.com/report.pdf",
            mime_type="application/pdf",
            filename="report.pdf",
            sha256="ab" * 32,
            size=1024,
            role="primary",
            retention="30d",
        )
        assert a.size == 1024
        assert a.role == "primary"
        assert a.retention == "30d"

    def test_frozen(self) -> None:
        a = Artifact(url="https://example.com/x.png")
        with pytest.raises(FrozenInstanceError):
            a.url = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# emit_artifact + AgentResult.artifacts
# ---------------------------------------------------------------------------


class TestEmitArtifact:
    def test_noop_outside_agent_run(self) -> None:
        """Outside a run there is no collector — must not raise."""
        a = emit_artifact("https://example.com/orphan.png", mime_type="image/png")
        assert isinstance(a, Artifact)
        assert a.url == "https://example.com/orphan.png"

    def test_agent_result_artifacts_default_empty(self) -> None:
        result = AgentResult(message=Message(role=Role.ASSISTANT, content="hi"))
        assert result.artifacts == []

    def _make_agent(self, tools: list, tool_calls: List[ToolCall], **config_kwargs: Any) -> Agent:
        return Agent(
            tools=tools,
            provider=_ToolThenDoneProvider(tool_calls),
            config=AgentConfig(max_iterations=3, **config_kwargs),
        )

    def test_sync_run_collects_artifact(self) -> None:
        @tool()
        def render_chart(title: str) -> str:
            """Render a chart."""
            emit_artifact(
                f"https://example.com/{title}.png",
                mime_type="image/png",
                filename=f"{title}.png",
                role="primary",
            )
            return "chart rendered"

        agent = self._make_agent(
            [render_chart],
            [ToolCall(tool_name="render_chart", parameters={"title": "sales"}, id="c1")],
        )
        result = agent.run("chart please")
        assert len(result.artifacts) == 1
        assert result.artifacts[0].url == "https://example.com/sales.png"
        assert result.artifacts[0].mime_type == "image/png"
        assert result.artifacts[0].role == "primary"

    def test_sync_run_with_tool_timeout_executor_path(self) -> None:
        """Tool runs in the timeout ThreadPoolExecutor — contextvar must propagate."""

        @tool()
        def render_chart(title: str) -> str:
            """Render a chart."""
            emit_artifact(f"https://example.com/{title}.png", mime_type="image/png")
            return "chart rendered"

        agent = self._make_agent(
            [render_chart],
            [ToolCall(tool_name="render_chart", parameters={"title": "t"}, id="c1")],
            tool_timeout_seconds=10.0,
        )
        result = agent.run("chart please")
        assert [a.url for a in result.artifacts] == ["https://example.com/t.png"]

    def test_parallel_tool_execution_collects_all_artifacts(self) -> None:
        """Tools dispatched via the parallel ThreadPoolExecutor must propagate."""

        @tool()
        def render_chart(title: str) -> str:
            """Render a chart."""
            emit_artifact(f"https://example.com/{title}.png")
            return f"rendered {title}"

        agent = self._make_agent(
            [render_chart],
            [
                ToolCall(tool_name="render_chart", parameters={"title": "a"}, id="c1"),
                ToolCall(tool_name="render_chart", parameters={"title": "b"}, id="c2"),
            ],
            parallel_tool_execution=True,
        )
        result = agent.run("two charts")
        urls = sorted(a.url for a in result.artifacts)
        assert urls == ["https://example.com/a.png", "https://example.com/b.png"]

    @pytest.mark.asyncio
    async def test_arun_sync_tool_collects_artifact(self) -> None:
        """Sync tool through aexecute's run_in_executor path must propagate."""

        @tool()
        def render_chart(title: str) -> str:
            """Render a chart."""
            emit_artifact(f"https://example.com/{title}.png")
            return "rendered"

        agent = self._make_agent(
            [render_chart],
            [ToolCall(tool_name="render_chart", parameters={"title": "async"}, id="c1")],
        )
        result = await agent.arun("chart please")
        assert [a.url for a in result.artifacts] == ["https://example.com/async.png"]

    @pytest.mark.asyncio
    async def test_arun_async_tool_collects_artifact(self) -> None:
        @tool()
        async def render_chart(title: str) -> str:
            """Render a chart."""
            await asyncio.sleep(0)
            emit_artifact(f"https://example.com/{title}.png")
            return "rendered"

        agent = self._make_agent(
            [render_chart],
            [ToolCall(tool_name="render_chart", parameters={"title": "native"}, id="c1")],
        )
        result = await agent.arun("chart please")
        assert [a.url for a in result.artifacts] == ["https://example.com/native.png"]

    def test_consecutive_runs_do_not_leak_artifacts(self) -> None:
        @tool()
        def render_chart(title: str) -> str:
            """Render a chart."""
            emit_artifact(f"https://example.com/{title}.png")
            return "rendered"

        @tool()
        def quiet() -> str:
            """Emits nothing."""
            return "quiet"

        first = Agent(
            tools=[render_chart],
            provider=_ToolThenDoneProvider(
                [ToolCall(tool_name="render_chart", parameters={"title": "one"}, id="c1")]
            ),
            config=AgentConfig(max_iterations=3),
        ).run("chart")
        assert len(first.artifacts) == 1

        second = Agent(
            tools=[quiet],
            provider=_ToolThenDoneProvider([ToolCall(tool_name="quiet", parameters={}, id="c2")]),
            config=AgentConfig(max_iterations=3),
        ).run("nothing")
        assert second.artifacts == []
        assert len(first.artifacts) == 1


# ---------------------------------------------------------------------------
# Exports + stability markers
# ---------------------------------------------------------------------------


class TestExports:
    def test_top_level_exports(self) -> None:
        import selectools

        for name in ("ToolResult", "Ambiguous", "NotFound", "Artifact", "emit_artifact"):
            assert hasattr(selectools, name), f"selectools.{name} missing"
            assert name in selectools.__all__

    def test_beta_markers(self) -> None:
        for obj in (ToolResult, Ambiguous, NotFound, Artifact, emit_artifact):
            assert getattr(obj, "__stability__", None) == "beta"

    def test_not_found_docstring_states_epistemics(self) -> None:
        """not_found means "no match observed from this source at this time"."""
        doc = (NotFound.__doc__ or "").lower()
        assert "observed no match" in doc or "does not mean" in doc
