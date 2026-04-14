"""Tests for loop detection module."""

from __future__ import annotations

import pytest

from selectools.loop_detection import (
    LoopDetectedError,
    LoopDetection,
    LoopDetector,
    LoopPolicy,
    PingPongDetector,
    RepeatDetector,
    StallDetector,
)
from selectools.types import ToolCall


def _tc(name: str, **params: object) -> ToolCall:
    return ToolCall(tool_name=name, parameters=dict(params))


# ---------------------------------------------------------------------------
# RepeatDetector
# ---------------------------------------------------------------------------


class TestRepeatDetector:
    def test_fires_when_same_call_repeats_n_times(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [_tc("search", q="cats")] * 3
        result = det.check(calls, [])
        assert result is not None
        assert result.detector_name == "repeat"
        assert result.details["tool"] == "search"
        assert result.details["count"] == 3

    def test_does_not_fire_below_threshold(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [_tc("search", q="cats")] * 2
        assert det.check(calls, []) is None

    def test_does_not_fire_when_args_differ(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [_tc("search", q="cats"), _tc("search", q="dogs"), _tc("search", q="cats")]
        assert det.check(calls, []) is None

    def test_does_not_fire_when_tool_differs(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [_tc("search", q="cats"), _tc("lookup", q="cats"), _tc("search", q="cats")]
        assert det.check(calls, []) is None

    def test_checks_only_tail_window(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [
            _tc("other", a=1),
            _tc("other", a=2),
            _tc("search", q="cats"),
            _tc("search", q="cats"),
            _tc("search", q="cats"),
        ]
        result = det.check(calls, [])
        assert result is not None
        assert result.details["tool"] == "search"

    def test_args_canonicalized_so_key_order_does_not_matter(self) -> None:
        det = RepeatDetector(threshold=3)
        calls = [
            ToolCall(tool_name="search", parameters={"q": "cats", "limit": 5}),
            ToolCall(tool_name="search", parameters={"limit": 5, "q": "cats"}),
            ToolCall(tool_name="search", parameters={"q": "cats", "limit": 5}),
        ]
        assert det.check(calls, []) is not None

    def test_threshold_must_be_at_least_two(self) -> None:
        with pytest.raises(ValueError):
            RepeatDetector(threshold=1)


# ---------------------------------------------------------------------------
# StallDetector
# ---------------------------------------------------------------------------


class TestStallDetector:
    def test_fires_when_same_result_repeats_n_times(self) -> None:
        det = StallDetector(threshold=3)
        calls = [_tc("poll"), _tc("poll"), _tc("poll")]
        results = ["pending", "pending", "pending"]
        result = det.check(calls, results)
        assert result is not None
        assert result.detector_name == "stall"
        assert result.details["tool"] == "poll"

    def test_does_not_fire_when_result_changes(self) -> None:
        det = StallDetector(threshold=3)
        calls = [_tc("poll"), _tc("poll"), _tc("poll")]
        results = ["pending", "pending", "done"]
        assert det.check(calls, results) is None

    def test_fires_even_if_args_differ(self) -> None:
        det = StallDetector(threshold=3)
        calls = [_tc("poll", id=1), _tc("poll", id=2), _tc("poll", id=3)]
        results = ["not-ready", "not-ready", "not-ready"]
        assert det.check(calls, results) is not None

    def test_empty_results_does_not_fire(self) -> None:
        det = StallDetector(threshold=3)
        assert det.check([_tc("poll")], []) is None


# ---------------------------------------------------------------------------
# PingPongDetector
# ---------------------------------------------------------------------------


class TestPingPongDetector:
    def test_fires_on_abab_pattern(self) -> None:
        det = PingPongDetector(cycle_length=2, repetitions=3)
        calls = [
            _tc("read"),
            _tc("write"),
            _tc("read"),
            _tc("write"),
            _tc("read"),
            _tc("write"),
        ]
        result = det.check(calls, [])
        assert result is not None
        assert result.detector_name == "ping_pong"
        assert result.details["cycle"] == ["read", "write"]

    def test_does_not_fire_below_repetitions(self) -> None:
        det = PingPongDetector(cycle_length=2, repetitions=3)
        calls = [_tc("read"), _tc("write"), _tc("read"), _tc("write")]
        assert det.check(calls, []) is None

    def test_does_not_fire_on_non_cyclic_sequence(self) -> None:
        det = PingPongDetector(cycle_length=2, repetitions=3)
        calls = [
            _tc("read"),
            _tc("write"),
            _tc("read"),
            _tc("parse"),
            _tc("read"),
            _tc("write"),
        ]
        assert det.check(calls, []) is None

    def test_longer_cycle_length(self) -> None:
        det = PingPongDetector(cycle_length=3, repetitions=2)
        calls = [
            _tc("a"),
            _tc("b"),
            _tc("c"),
            _tc("a"),
            _tc("b"),
            _tc("c"),
        ]
        assert det.check(calls, []) is not None


# ---------------------------------------------------------------------------
# LoopDetector facade
# ---------------------------------------------------------------------------


class TestLoopDetector:
    def test_default_factory_enables_all_three(self) -> None:
        det = LoopDetector.default()
        assert len(det.detectors) == 3
        names = {type(d).__name__ for d in det.detectors}
        assert names == {"RepeatDetector", "StallDetector", "PingPongDetector"}

    def test_check_returns_first_detection(self) -> None:
        det = LoopDetector(detectors=[RepeatDetector(threshold=3)])
        calls = [_tc("search", q="x")] * 3
        result = det.check(calls, [])
        assert result is not None
        assert result.detector_name == "repeat"

    def test_check_returns_none_when_no_detector_fires(self) -> None:
        det = LoopDetector(detectors=[RepeatDetector(threshold=3)])
        calls = [_tc("a"), _tc("b")]
        assert det.check(calls, []) is None

    def test_default_policy_is_raise(self) -> None:
        det = LoopDetector.default()
        assert det.policy == LoopPolicy.RAISE

    def test_explicit_policy_inject_message(self) -> None:
        det = LoopDetector(detectors=[RepeatDetector()], policy=LoopPolicy.INJECT_MESSAGE)
        assert det.policy == LoopPolicy.INJECT_MESSAGE


# ---------------------------------------------------------------------------
# LoopDetection / LoopDetectedError
# ---------------------------------------------------------------------------


class TestLoopDetectionError:
    def test_detection_as_error_returns_loop_detected_error(self) -> None:
        detection = LoopDetection(
            detector_name="repeat",
            message="Same call 3 times",
            details={"tool": "search", "count": 3},
        )
        err = detection.as_error()
        assert isinstance(err, LoopDetectedError)
        assert err.detector == "repeat"
        assert err.details == {"tool": "search", "count": 3}

    def test_loop_detected_error_is_selectools_error(self) -> None:
        from selectools.exceptions import SelectoolsError

        err = LoopDetectedError("stuck", detector="repeat", details={})
        assert isinstance(err, SelectoolsError)


# ---------------------------------------------------------------------------
# Integration: Agent core loop wiring
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    def _make_agent_with_repeating_provider(
        self,
        *,
        loop_detector: "LoopDetector | None",
        parallel_tool_execution: bool = False,
    ):
        from selectools.agent import Agent, AgentConfig
        from selectools.tools import Tool
        from tests.conftest import SharedToolCallProvider

        noop_tool = Tool(
            name="search",
            description="search",
            parameters=[],
            function=lambda: "pending",
        )
        repeat_call = ToolCall(tool_name="search", parameters={}, id="c1")
        provider = SharedToolCallProvider(responses=[([repeat_call], "") for _ in range(10)])
        agent = Agent(
            tools=[noop_tool],
            provider=provider,
            config=AgentConfig(
                model="fake-model",
                max_iterations=10,
                loop_detector=loop_detector,
                parallel_tool_execution=parallel_tool_execution,
            ),
        )
        return agent

    def test_loop_detector_raises_on_repeat_sync(self) -> None:
        agent = self._make_agent_with_repeating_provider(
            loop_detector=LoopDetector.default(),
        )
        with pytest.raises(LoopDetectedError) as exc:
            agent.run("do the thing")
        assert exc.value.detector == "repeat"

    def test_no_loop_detector_means_no_detection(self) -> None:
        # Without detector, agent runs until max_iterations hits
        agent = self._make_agent_with_repeating_provider(loop_detector=None)
        result = agent.run("do the thing")
        assert result is not None  # no exception

    def test_loop_detector_raises_on_repeat_async(self) -> None:
        import asyncio

        agent = self._make_agent_with_repeating_provider(
            loop_detector=LoopDetector.default(),
        )
        with pytest.raises(LoopDetectedError):
            asyncio.run(agent.arun("do the thing"))

    def test_loop_detector_observer_notified(self) -> None:
        from selectools.observer import AgentObserver

        captured: list[tuple] = []

        class RecordingObserver(AgentObserver):
            def on_tool_loop_detected(
                self,
                run_id: str,
                detector_name: str,
                details: dict,
            ) -> None:
                captured.append((run_id, detector_name, details))

        from selectools.agent import Agent, AgentConfig
        from selectools.tools import Tool
        from tests.conftest import SharedToolCallProvider

        noop_tool = Tool(
            name="search",
            description="search",
            parameters=[],
            function=lambda: "pending",
        )
        repeat_call = ToolCall(tool_name="search", parameters={}, id="c1")
        provider = SharedToolCallProvider(responses=[([repeat_call], "") for _ in range(10)])
        agent = Agent(
            tools=[noop_tool],
            provider=provider,
            config=AgentConfig(
                model="fake-model",
                max_iterations=10,
                loop_detector=LoopDetector.default(),
                observers=[RecordingObserver()],
                parallel_tool_execution=False,
            ),
        )
        with pytest.raises(LoopDetectedError):
            agent.run("do the thing")
        assert len(captured) == 1
        assert captured[0][1] == "repeat"

    def test_structured_retries_not_counted_as_loops(self) -> None:
        # Loop check runs after tool execution only; structured retries
        # happen before tool execution. A zero-tool-call run must never
        # trigger loop detection regardless of retry count.
        det = LoopDetector.default()
        assert det.check([], []) is None

    def test_inject_message_policy_continues_without_raising(self) -> None:
        # With INJECT_MESSAGE policy the agent should see a system notice
        # in history and keep looping until max_iterations.
        detector = LoopDetector(
            detectors=[RepeatDetector(threshold=3)],
            policy=LoopPolicy.INJECT_MESSAGE,
            inject_message="LOOP-NOTICE-SENTINEL",
        )
        agent = self._make_agent_with_repeating_provider(loop_detector=detector)
        result = agent.run("do the thing")
        assert result is not None
        # The corrective system message must have been injected at least once.
        from selectools.types import Role

        system_messages = [
            m
            for m in agent._history
            if m.role == Role.SYSTEM and m.content == "LOOP-NOTICE-SENTINEL"
        ]
        assert len(system_messages) >= 1
