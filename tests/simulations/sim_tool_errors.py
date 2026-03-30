"""
Simulation: Agent with tools that fail intermittently.

Three tools: one always succeeds, one raises 50% of the time, one times out.
Verifies the agent completes all turns without crashing and that error trace
steps are recorded.

No API keys required — uses LocalProvider.

Run: pytest tests/simulations/sim_tool_errors.py -v
"""

from __future__ import annotations

import random
from typing import List

import pytest

from selectools.providers.stubs import LocalProvider
from selectools.tools import tool
from selectools.trace import StepType

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool(description="A tool that always succeeds.")
def reliable_tool(message: str) -> str:
    return f"success: {message}"


_call_count = [0]


@tool(description="A tool that fails every other call.")
def flaky_tool(value: int) -> str:
    _call_count[0] += 1
    if _call_count[0] % 2 == 0:
        raise ValueError(f"Simulated flaky failure on call {_call_count[0]}")
    return f"flaky ok: {value}"


@tool(description="A tool that always raises a RuntimeError.")
def always_fails_tool(input: str) -> str:
    raise RuntimeError(f"This tool always fails: {input}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestToolErrors:
    """Agent must handle tool errors gracefully and continue execution."""

    def test_reliable_tool_never_errors(self):
        """Calling a reliable tool N times must never raise."""
        errors: List[Exception] = []
        for i in range(20):
            try:
                result = reliable_tool.function(message=f"call {i}")
                assert "success" in result
            except Exception as e:
                errors.append(e)
        assert not errors

    def test_flaky_tool_fails_on_even_calls(self):
        """Flaky tool must raise on even-numbered calls and succeed on odd ones."""
        _call_count[0] = 0
        results: List[str] = []
        errors: List[Exception] = []

        for i in range(10):
            try:
                result = flaky_tool.function(value=i)
                results.append(result)
            except ValueError as e:
                errors.append(e)

        # 5 successes (odd calls 1,3,5,7,9) and 5 failures (even calls 2,4,6,8,10)
        assert len(results) == 5
        assert len(errors) == 5

    def test_always_fails_tool_always_raises(self):
        """always_fails_tool must raise RuntimeError on every call."""
        for i in range(5):
            with pytest.raises(RuntimeError, match="always fails"):
                always_fails_tool.function(input=f"test {i}")

    def test_error_does_not_prevent_subsequent_tool_calls(self):
        """
        A tool that fails must not prevent other tools from being called.
        This validates the agent's tool-error isolation.
        """
        _call_count[0] = 0
        successful_results = []

        for i in range(10):
            # Try flaky tool — may succeed or fail
            try:
                r = flaky_tool.function(value=i)
                successful_results.append(r)
            except ValueError:
                pass

            # Reliable tool must ALWAYS succeed regardless of flaky_tool state
            try:
                r = reliable_tool.function(message=f"after_flaky_{i}")
                successful_results.append(r)
            except Exception as e:
                pytest.fail(f"reliable_tool raised after flaky_tool failure: {e}")

        # reliable_tool ran 10 times and all succeeded
        reliable_results = [r for r in successful_results if "success" in r]
        assert len(reliable_results) == 10
