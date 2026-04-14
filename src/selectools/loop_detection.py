"""Tool-call loop detection for agent runs.

Detects three pathological patterns that waste tokens and latency:

- **Repeat**: the same ``(tool_name, arguments)`` is called N consecutive times.
- **Stall**: the same ``(tool_name, result)`` is observed N consecutive times —
  the agent is polling but the world is not changing.
- **Ping-pong**: a short cycle of tool names (e.g. ``read -> write -> read -> write``)
  repeats M times without advancing.

Detection is intentionally cheap: each detector inspects only the tail of the
tool-call list. Hashing uses canonicalized JSON for arguments and SHA-256 for
result bodies so that key ordering and large payloads do not affect behavior.

The module is wired into the agent core loop via ``AgentConfig.loop_detector``.
Structured-validation retries are **not** counted as loop iterations — the
check runs only after successful tool execution.

See ``docs/specs/2026-04-13-loop-detection.md`` for the full spec.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .exceptions import SelectoolsError
from .stability import beta
from .types import ToolCall


class LoopPolicy(Enum):
    """How the agent should respond when a loop is detected."""

    RAISE = "raise"
    INJECT_MESSAGE = "inject_message"


@beta
class LoopDetectedError(SelectoolsError):
    """Raised when a ``LoopDetector`` with ``LoopPolicy.RAISE`` fires."""

    def __init__(self, message: str, *, detector: str, details: Dict[str, Any]):
        self.detector = detector
        self.details = details
        super().__init__(message)


@beta
@dataclass
class LoopDetection:
    """Structured result of a single loop detection."""

    detector_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def as_error(self) -> LoopDetectedError:
        return LoopDetectedError(
            self.message,
            detector=self.detector_name,
            details=dict(self.details),
        )


def _canonical_args(tool_call: ToolCall) -> str:
    return json.dumps(tool_call.parameters, sort_keys=True, default=str)


def _result_hash(result: str) -> str:
    return hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()


@beta
class BaseDetector(ABC):
    """Abstract base for individual loop detectors."""

    @abstractmethod
    def check(
        self,
        tool_calls: Sequence[ToolCall],
        tool_results: Sequence[str],
    ) -> Optional[LoopDetection]:
        """Return a ``LoopDetection`` if the pattern is present, else ``None``."""


@beta
class RepeatDetector(BaseDetector):
    """Fires when the same ``(tool_name, arguments)`` appears ``threshold`` consecutive times."""

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("RepeatDetector.threshold must be >= 2")
        self.threshold = threshold

    def check(
        self,
        tool_calls: Sequence[ToolCall],
        tool_results: Sequence[str],
    ) -> Optional[LoopDetection]:
        if len(tool_calls) < self.threshold:
            return None
        window = tool_calls[-self.threshold :]
        first = window[0]
        first_args = _canonical_args(first)
        for call in window[1:]:
            if call.tool_name != first.tool_name or _canonical_args(call) != first_args:
                return None
        return LoopDetection(
            detector_name="repeat",
            message=(
                f"Tool '{first.tool_name}' called {self.threshold} times with identical arguments."
            ),
            details={
                "tool": first.tool_name,
                "count": self.threshold,
                "arguments": dict(first.parameters),
            },
        )


@beta
class StallDetector(BaseDetector):
    """Fires when the same ``(tool_name, result)`` appears ``threshold`` consecutive times."""

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("StallDetector.threshold must be >= 2")
        self.threshold = threshold

    def check(
        self,
        tool_calls: Sequence[ToolCall],
        tool_results: Sequence[str],
    ) -> Optional[LoopDetection]:
        if len(tool_results) < self.threshold or len(tool_calls) < self.threshold:
            return None
        call_window = tool_calls[-self.threshold :]
        result_window = tool_results[-self.threshold :]
        first_name = call_window[0].tool_name
        first_hash = _result_hash(result_window[0])
        for call, result in zip(call_window[1:], result_window[1:]):
            if call.tool_name != first_name:
                return None
            if _result_hash(result) != first_hash:
                return None
        return LoopDetection(
            detector_name="stall",
            message=(
                f"Tool '{first_name}' returned the same result {self.threshold} times — "
                f"agent is polling without progress."
            ),
            details={
                "tool": first_name,
                "count": self.threshold,
                "result_hash": first_hash[:16],
            },
        )


@beta
class PingPongDetector(BaseDetector):
    """Fires when a cycle of length ``cycle_length`` repeats ``repetitions`` times."""

    def __init__(self, cycle_length: int = 2, repetitions: int = 3) -> None:
        if cycle_length < 2:
            raise ValueError("PingPongDetector.cycle_length must be >= 2")
        if repetitions < 2:
            raise ValueError("PingPongDetector.repetitions must be >= 2")
        self.cycle_length = cycle_length
        self.repetitions = repetitions

    def check(
        self,
        tool_calls: Sequence[ToolCall],
        tool_results: Sequence[str],
    ) -> Optional[LoopDetection]:
        window_size = self.cycle_length * self.repetitions
        if len(tool_calls) < window_size:
            return None
        window = tool_calls[-window_size:]
        cycle = [call.tool_name for call in window[: self.cycle_length]]
        # Cycle itself must have at least two distinct names — otherwise it's a Repeat.
        if len(set(cycle)) < 2:
            return None
        for rep in range(1, self.repetitions):
            offset = rep * self.cycle_length
            segment = [call.tool_name for call in window[offset : offset + self.cycle_length]]
            if segment != cycle:
                return None
        return LoopDetection(
            detector_name="ping_pong",
            message=(f"Tools {cycle} cycled {self.repetitions} times without advancing."),
            details={
                "cycle": cycle,
                "repetitions": self.repetitions,
            },
        )


@beta
@dataclass
class LoopDetector:
    """Facade that runs a set of ``BaseDetector`` instances after each tool round."""

    detectors: List[BaseDetector]
    policy: LoopPolicy = LoopPolicy.RAISE
    inject_message: str = (
        "You have been calling tools in a repetitive pattern that is not making progress. "
        "Step back, reconsider the goal, and try a different approach."
    )

    def check(
        self,
        tool_calls: Sequence[ToolCall],
        tool_results: Sequence[str],
    ) -> Optional[LoopDetection]:
        for detector in self.detectors:
            result = detector.check(tool_calls, tool_results)
            if result is not None:
                return result
        return None

    @classmethod
    def default(cls) -> "LoopDetector":
        return cls(
            detectors=[
                RepeatDetector(threshold=3),
                StallDetector(threshold=3),
                PingPongDetector(cycle_length=2, repetitions=3),
            ],
            policy=LoopPolicy.RAISE,
        )
