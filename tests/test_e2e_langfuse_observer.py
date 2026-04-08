"""End-to-end tests for LangfuseObserver against a real Langfuse instance.

``test_langfuse_observer.py`` mocks the langfuse SDK. This file talks to a
real Langfuse backend — either Langfuse Cloud or a self-hosted instance.

Required env vars (tests skip if missing):
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional; defaults to Langfuse Cloud)

Run with:

    pytest tests/test_e2e_langfuse_observer.py --run-e2e -v

Note: this test does NOT attempt to read traces back from Langfuse (that
requires API access and timing). It just verifies the SDK accepts our
event sequence without throwing and that ``flush()`` completes cleanly.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("langfuse", reason="langfuse not installed")

from selectools import Agent, AgentConfig, tool  # noqa: E402
from selectools.observe import LangfuseObserver  # noqa: E402
from tests.conftest import SharedFakeProvider  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def langfuse_or_skip() -> None:
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        pytest.skip("LANGFUSE_PUBLIC_KEY not set — skipping Langfuse e2e")
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        pytest.skip("LANGFUSE_SECRET_KEY not set — skipping Langfuse e2e")


@tool()
def _noop() -> str:
    """Return a fixed string."""
    return "noop"


class TestLangfuseRealBackend:
    def test_agent_run_emits_trace_without_errors(self, langfuse_or_skip: None) -> None:
        """A full agent run pushes a real trace to Langfuse and flushes cleanly."""
        observer = LangfuseObserver()
        agent = Agent(
            tools=[_noop],
            provider=SharedFakeProvider(responses=["final answer"]),
            config=AgentConfig(
                model="fake-model",
                observers=[observer],
            ),
        )
        result = agent.run("hello")
        assert "final answer" in result.content
        # Force flush — should not raise
        observer._langfuse.flush()
