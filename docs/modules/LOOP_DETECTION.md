---
description: "Detect pathological tool-call patterns and surface them as errors"
tags:
  - runtime
  - reliability
  - beta
---

# Loop Detection

**Import:** `from selectools import LoopDetector`
**Stability:** beta

```python title="loop_detection_demo.py"
from selectools import Agent, AgentConfig, LoopDetector, LoopDetectedError
from selectools.providers.stubs import LocalProvider
from selectools.tools import tool


@tool(description="Search the web")
def search(query: str) -> str:
    return "no results"


agent = Agent(
    tools=[search],
    provider=LocalProvider(),
    config=AgentConfig(
        max_iterations=10,
        loop_detector=LoopDetector.default(),
    ),
)

try:
    agent.run("Search for something")
except LoopDetectedError as exc:
    print(f"Stopped: {exc.detector} — {exc.details}")
```

!!! tip "See Also"
    - [Agent](AGENT.md) — `AgentConfig.loop_detector` field
    - [Observer](OBSERVER.md) — `on_tool_loop_detected` callback
    - [Cancellation](CANCELLATION.md) — cooperative agent stopping

**Added in:** v0.22.0
**File:** `src/selectools/loop_detection.py`
**Classes:** `LoopDetector`, `LoopDetection`, `LoopDetectedError`, `LoopPolicy`, `BaseDetector`, `RepeatDetector`, `StallDetector`, `PingPongDetector`

## Overview

`LoopDetector` flags three repetitive tool-call patterns that waste tokens and latency:

| Pattern | Fires When |
|---------|-----------|
| **Repeat** | The same `(tool_name, arguments)` is called N times in a row (default N=3) |
| **Stall** | The same `(tool_name, result)` appears N times in a row — agent is polling without progress |
| **Ping-pong** | A cycle of length K repeats M times without advancing (default K=2, M=3) |

The detector runs **after each tool-execution round** (not after LLM calls or structured-validation retries). On detection, it notifies observers, records a trace step, and either raises `LoopDetectedError` or injects a corrective system message.

Loop detection is **opt-in** — `AgentConfig.loop_detector` defaults to `None` and existing agents are unchanged.

## Quick Start

```python
from selectools import Agent, AgentConfig, LoopDetector

# Enable all three detectors with default thresholds
config = AgentConfig(
    max_iterations=20,
    loop_detector=LoopDetector.default(),
)

agent = Agent(tools=[...], provider=provider, config=config)
result = agent.run("Research this topic")
```

## Detectors

### RepeatDetector

Fires when the same tool is called with identical arguments N times in a row. Argument comparison uses canonicalized JSON, so key ordering does not matter.

```python
from selectools import LoopDetector, RepeatDetector

detector = LoopDetector(detectors=[RepeatDetector(threshold=3)])
```

Use case: agent stuck retrying the same failing API call.

### StallDetector

Fires when a tool returns the same result N times in a row, regardless of arguments. Result comparison uses SHA-256 so large payloads are cheap to compare.

```python
from selectools import LoopDetector, StallDetector

detector = LoopDetector(detectors=[StallDetector(threshold=3)])
```

Use case: agent polling a status endpoint that never changes.

### PingPongDetector

Fires when a cycle of tool names repeats. Default: a 2-tool cycle (e.g., `read -> write -> read -> write`) repeating 3 times.

```python
from selectools import LoopDetector, PingPongDetector

# Detect 3-tool cycles repeating at least 2 times
detector = LoopDetector(detectors=[PingPongDetector(cycle_length=3, repetitions=2)])
```

Use case: agent alternating between tools without making progress.

## Policies

| Policy | Behavior |
|--------|----------|
| `LoopPolicy.RAISE` (default) | Raises `LoopDetectedError` when the loop fires |
| `LoopPolicy.INJECT_MESSAGE` | Appends a corrective system message to history and continues looping |

```python
from selectools import LoopDetector, LoopPolicy

detector = LoopDetector(
    detectors=[...],
    policy=LoopPolicy.INJECT_MESSAGE,
    inject_message="Stop repeating the same action. Try a different approach.",
)
```

Use `INJECT_MESSAGE` when you want the agent to self-correct; use `RAISE` when you want to fail fast and let the caller decide.

## Observer Callback

```python
from selectools import AgentObserver

class MyObserver(AgentObserver):
    def on_tool_loop_detected(
        self,
        run_id: str,
        detector_name: str,
        details: dict,
    ) -> None:
        log.warning(f"[{run_id}] loop={detector_name} details={details}")
```

The callback name is `on_tool_loop_detected` (not `on_loop_detected`) to avoid a name collision with the existing graph-level `on_loop_detected` that tracks node-cycle stalls in `AgentGraph`.

## Trace Step

Loop detections are recorded as `StepType.GRAPH_LOOP_DETECTED` in the execution trace:

```python
for step in result.trace.steps:
    if step.type == "graph_loop_detected":
        print(step.summary)
```

## Custom Detectors

Subclass `BaseDetector` to add your own pattern:

```python
from selectools import BaseDetector, LoopDetection, LoopDetector

class MyDetector(BaseDetector):
    def check(self, tool_calls, tool_results):
        if some_condition(tool_calls):
            return LoopDetection(
                detector_name="my_detector",
                message="Custom pattern detected",
                details={"info": "..."},
            )
        return None

detector = LoopDetector(detectors=[MyDetector(), RepeatDetector()])
```

## Interaction with Other Limits

- **Structured-validation retries** (`RetryConfig.max_retries`) do NOT count toward loop detection. The check runs after tool execution only.
- **`max_iterations`** still applies — if the detector is not configured, agents stop at `max_iterations` as before.
- **Terminal tools** check AFTER loop detection: a repetitive terminal tool still surfaces as a loop rather than returning a misleading "success".

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `AgentConfig.loop_detector` | `Optional[LoopDetector]` | `None` | Enable loop detection. `None` = disabled (backward compatible). |

## See Also

- [Agent](AGENT.md) — `AgentConfig` reference
- [Observer](OBSERVER.md) — lifecycle callbacks

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 95 | [`95_loop_detection.py`](https://github.com/johnnichev/selectools/blob/main/examples/95_loop_detection.py) | Loop detector with all three patterns |
