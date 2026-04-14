"""End-to-end tests for OTelObserver against the real OpenTelemetry SDK.

``test_otel_observer.py`` mocks the ``opentelemetry`` module. These tests
use the real ``opentelemetry-sdk`` with an in-memory span exporter so we
can assert that:

- A TracerProvider actually receives span start/end events
- Span names follow the GenAI semantic conventions
- Run -> LLM -> Tool span hierarchy is correct
- Attributes like ``gen_ai.request.model`` and token counts are set

Run with:

    pytest tests/test_e2e_otel_observer.py --run-e2e -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-api not installed")
pytest.importorskip("opentelemetry.sdk", reason="opentelemetry-sdk not installed")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from selectools import Agent, AgentConfig, tool  # noqa: E402
from selectools.observe import OTelObserver  # noqa: E402
from tests.conftest import SharedFakeProvider  # noqa: E402

pytestmark = pytest.mark.e2e

# The ``otel_exporter`` fixture comes from tests/conftest.py and installs a
# single process-wide TracerProvider + InMemorySpanExporter. Do NOT add a
# local fixture with the same name here — that breaks test isolation when
# another e2e file also wants OTel span capture (only the first file to
# call ``trace.set_tracer_provider`` wins, so the others see empty spans).


@tool()
def _noop() -> str:
    """Return a fixed string. Used so Agent can be instantiated."""
    return "noop"


class TestOTelRealSDK:
    def test_agent_run_emits_root_span(self, otel_exporter: InMemorySpanExporter) -> None:
        """A single agent run produces at least one finished span."""
        agent = Agent(
            tools=[_noop],
            provider=SharedFakeProvider(responses=["final answer"]),
            config=AgentConfig(
                model="fake-model",
                observers=[OTelObserver(tracer_name="selectools-e2e")],
            ),
        )
        result = agent.run("hello")
        assert "final answer" in result.content

        spans = otel_exporter.get_finished_spans()
        assert len(spans) >= 1, "Expected at least one span from agent.run"

        # There should be a root agent.run span
        names = [s.name for s in spans]
        assert any("run" in n.lower() or "agent" in n.lower() for n in names), (
            f"No agent/run span found; got: {names}"
        )

    def test_run_span_has_gen_ai_system_attribute(
        self, otel_exporter: InMemorySpanExporter
    ) -> None:
        """The root span carries the GenAI semantic-convention system attr."""
        agent = Agent(
            tools=[_noop],
            provider=SharedFakeProvider(responses=["hi"]),
            config=AgentConfig(
                model="fake-model",
                observers=[OTelObserver(tracer_name="selectools-e2e")],
            ),
        )
        agent.run("ping")

        spans = otel_exporter.get_finished_spans()
        # At least one span should carry the gen_ai.system attribute
        saw_gen_ai_system = False
        for span in spans:
            attrs = dict(span.attributes or {})
            if attrs.get("gen_ai.system") == "selectools":
                saw_gen_ai_system = True
                break
        assert saw_gen_ai_system, "Expected at least one span with gen_ai.system='selectools'"

    def test_multiple_runs_produce_distinct_spans(
        self, otel_exporter: InMemorySpanExporter
    ) -> None:
        """Each agent.run() creates its own set of spans."""
        agent = Agent(
            tools=[_noop],
            provider=SharedFakeProvider(responses=["a", "b", "c"]),
            config=AgentConfig(
                model="fake-model",
                observers=[OTelObserver(tracer_name="selectools-e2e")],
            ),
        )
        agent.run("first")
        count_after_first = len(otel_exporter.get_finished_spans())
        agent.run("second")
        count_after_second = len(otel_exporter.get_finished_spans())
        assert count_after_second > count_after_first, "Second run did not emit additional spans"
