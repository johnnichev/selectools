"""Tests for selectools.toolbox.reasoning_tools."""

from __future__ import annotations

import pytest

from selectools.toolbox.reasoning_tools import (
    ReasoningStep,
    ReasoningTools,
    make_reasoning_tools,
)


def _call(tool, **kwargs):
    return tool.function(**kwargs)


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #


def test_make_reasoning_tools_returns_think_and_analyze():
    tools = make_reasoning_tools()
    names = {t.name for t in tools}
    assert names == {"think", "analyze"}


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        ReasoningTools(min_steps=-1)
    with pytest.raises(ValueError):
        ReasoningTools(max_steps=0)
    with pytest.raises(ValueError):
        ReasoningTools(min_steps=5, max_steps=3)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def test_think_records_step_and_returns_ack():
    r = ReasoningTools(min_steps=0, max_steps=10)
    think = r.think_tool()
    out = _call(think, thought="break the task into steps")
    assert "Recorded think step 1" in out
    assert r.count == 1
    assert r.steps[0] == ReasoningStep(index=1, kind="think", content="break the task into steps")


def test_think_and_analyze_share_one_budget():
    r = ReasoningTools(min_steps=0, max_steps=10)
    think, analyze = r.tools
    _call(think, thought="first")
    _call(analyze, analysis="evaluate first")
    assert r.count == 2
    assert [s.kind for s in r.steps] == ["think", "analyze"]
    assert r.steps[1].index == 2


def test_empty_content_is_not_recorded():
    r = ReasoningTools()
    think = r.think_tool()
    out = _call(think, thought="   ")
    assert "Nothing recorded" in out
    assert r.count == 0


# --------------------------------------------------------------------------- #
# max_steps enforcement (hard)
# --------------------------------------------------------------------------- #


def test_max_steps_is_enforced():
    r = ReasoningTools(min_steps=0, max_steps=2)
    think = r.think_tool()
    assert "Recorded think step 1" in _call(think, thought="a")
    assert "Recorded think step 2" in _call(think, thought="b")
    out = _call(think, thought="c")
    assert "Reasoning budget reached (2 steps)" in out
    assert "final answer" in out
    assert r.count == 2  # the over-budget call did not record


def test_max_steps_none_allows_unbounded():
    r = ReasoningTools(min_steps=0, max_steps=None)
    think = r.think_tool()
    for i in range(25):
        _call(think, thought=f"step {i}")
    assert r.count == 25


# --------------------------------------------------------------------------- #
# min_steps guidance (soft)
# --------------------------------------------------------------------------- #


def test_min_steps_nudges_until_met():
    r = ReasoningTools(min_steps=2, max_steps=10)
    think = r.think_tool()
    out1 = _call(think, thought="a")
    assert "Reason at least 1 more time" in out1
    out2 = _call(think, thought="b")
    assert "Continue reasoning or give your final answer" in out2


def test_min_steps_zero_never_nudges():
    r = ReasoningTools(min_steps=0)
    think = r.think_tool()
    out = _call(think, thought="a")
    assert "more time" not in out
    assert "Continue reasoning or give your final answer" in out


# --------------------------------------------------------------------------- #
# Reuse
# --------------------------------------------------------------------------- #


def test_reset_clears_chain():
    r = ReasoningTools(min_steps=0, max_steps=3)
    think = r.think_tool()
    _call(think, thought="a")
    _call(think, thought="b")
    assert r.count == 2
    r.reset()
    assert r.count == 0
    # Budget is fresh after reset.
    assert "Recorded think step 1" in _call(think, thought="c")


def test_tools_property_share_instance_state():
    r = ReasoningTools(min_steps=0)
    # Each access builds fresh Tool objects, but they record into the same log.
    _call(r.tools[0], thought="x")
    _call(r.tools[1], analysis="y")
    assert r.count == 2


def test_descriptions_mention_min_floor():
    r = ReasoningTools(min_steps=3)
    think = r.think_tool()
    assert "at least 3" in think.description


def test_step_parameters_are_required():
    tools = make_reasoning_tools()
    for t in tools:
        assert t.parameters[0].required is True
