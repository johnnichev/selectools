# Tool-Call Loop Detection

**Stack:** Python 3.9+, src-layout, pytest, stdlib only (no new deps)
**Date:** 2026-04-13
**Status:** Draft

## Problem

Agents stuck in repetitive tool-call cycles silently burn tokens until `max_iterations` is exhausted. Every major competitor (PraisonAI, Agno, LangGraph) ships loop detection. Selectools has `StepType.GRAPH_LOOP_DETECTED` and `StepType.GRAPH_STALL` enum values defined in `trace.py` but zero detection logic wired into the core loop.

Three failure patterns are common in production:

1. **Repeat** — agent calls the same tool with identical arguments N times in a row (e.g., retrying a failing API call)
2. **Poll-no-progress** — agent calls a tool repeatedly but the output never changes (e.g., polling a status endpoint)
3. **Ping-pong** — agent alternates between two tools without advancing (e.g., read → write → read → write with identical content)

Without detection, users only discover these after hitting the iteration ceiling, wasting latency and cost.

## Solution

A new `src/selectools/loop_detection.py` module providing three composable detectors behind a single `LoopDetector` facade. The facade is wired into the core agent loop (all three entry points: `run`, `arun`, `astream`) and fires via the existing observer framework when a loop is detected. Configurable via `AgentConfig`.

The detector runs **after each tool execution round**, inspecting `ctx.all_tool_calls` for patterns. On detection it:
1. Adds a `TraceStep(type=StepType.GRAPH_LOOP_DETECTED)` (or `GRAPH_STALL`)
2. Notifies observers via new `on_loop_detected(run_id, detector_name, details)` callback
3. Either raises `LoopDetectedError` (default) or injects a corrective system message (configurable)

## Acceptance Criteria

- [ ] `RepeatDetector` fires when the same `(tool_name, args_hash)` appears N consecutive times (default N=3, configurable)
- [ ] `StallDetector` fires when the same `(tool_name, result_hash)` appears N consecutive times regardless of args (default N=3, configurable)
- [ ] `PingPongDetector` fires when a cycle of length ≤ K repeats M times (default K=2, M=3, configurable)
- [ ] `LoopDetector` facade composes one or more detectors; default configuration enables all three
- [ ] Detection runs after each tool-execution round in `run()`, `arun()`, and `astream()` — identical behavior across all three
- [ ] On detection: `TraceStep` added to `ctx.trace` with `type=StepType.GRAPH_LOOP_DETECTED` and descriptive summary
- [ ] On detection: observers notified via `on_tool_loop_detected(run_id, detector_name, details_dict)` (renamed from `on_loop_detected` to avoid collision with existing graph-level `on_loop_detected` callback)
- [ ] Default action on detection: raise `LoopDetectedError(message, detector, details)` (subclass of `AgentError`)
- [ ] Alternative action: inject corrective message (e.g., "You are repeating the same action. Try a different approach.") and continue — configurable via `LoopPolicy.INJECT_MESSAGE`
- [ ] `AgentConfig` gains `loop_detector: Optional[LoopDetector] = None` — None means no detection (backward compatible)
- [ ] Convenience factory: `LoopDetector.default()` returns all-three-enabled detector with sensible thresholds
- [ ] Structured retries (`ctx.structured_retries`) are NOT counted as loop iterations — only tool-call iterations
- [ ] Pure stdlib — no new dependencies
- [ ] ≥95% test coverage on the new module
- [ ] Stability marker: `@beta` on all public classes

## Non-Goals

- Graph-level cycle detection (node A → node B → node A in `AgentGraph`) — that's a different feature for `graph.py`
- Automatic fix/recovery beyond injecting a message — the agent may still loop after the message
- Modifying `max_iterations` or the iteration counter — loop detection is orthogonal to budget
- Token-cost accounting for wasted loops — out of scope
- Async-only or sync-only — must work in all three entry points identically

## Technical Approach

### New file: `src/selectools/loop_detection.py`

```
LoopPolicy (enum)          — RAISE | INJECT_MESSAGE
LoopDetectedError          — AgentError subclass

BaseDetector (ABC)
  ├── RepeatDetector        — window over (name, args_hash) tuples
  ├── StallDetector          — window over (name, result_hash) tuples
  └── PingPongDetector       — sliding-window cycle detection

LoopDetector (facade)
  - detectors: List[BaseDetector]
  - policy: LoopPolicy
  - check(tool_calls, tool_results) -> Optional[LoopDetection]
  - default() -> LoopDetector  (class method)

LoopDetection (dataclass)  — detector_name, message, details dict
```

### Modified files

| File | Change |
|------|--------|
| `agent/config.py` | Add `loop_detector: Optional[LoopDetector] = None` field |
| `agent/core.py` | After tool execution in all 3 loops, call `self._check_loop_detection(ctx)` |
| `agent/core.py` | New `_check_loop_detection(ctx)` method: runs detector, adds trace step, notifies observers, raises or injects |
| `agent/observer.py` | Add `on_tool_loop_detected(run_id, detector_name, details)` with no-op default |
| `agent/observer.py` | Add `a_on_tool_loop_detected(...)` async variant on `AsyncAgentObserver` |
| `trace.py` | No changes — `GRAPH_LOOP_DETECTED` and `GRAPH_STALL` already exist |
| `__init__.py` | Export `LoopDetector`, `LoopDetectedError`, `LoopPolicy` with `@beta` |

### Detection algorithm

Each detector maintains no state — it receives the full `ctx.all_tool_calls` list and the most recent tool results, then applies a windowed check:

- **RepeatDetector**: hash `(call.tool_name, json_canonical(call.arguments))` for last N calls; fire if all identical
- **StallDetector**: hash `(call.tool_name, sha256(result_text))` for last N (tool_call, result) pairs; fire if all identical
- **PingPongDetector**: extract `tool_name` sequence from last K×M calls; check if the sequence of length K repeats M times

Hashing uses `hashlib.sha256` on canonicalized JSON (`json.dumps(args, sort_keys=True)`) for args, and on raw result text for results.

### Integration point in core loop

After `_execute_tools_parallel()` / `_execute_single_tool()` returns, before `ctx.iteration += 1`:

```python
if self.config.loop_detector is not None:
    detection = self._check_loop_detection(ctx)
    if detection is not None:
        # trace step + observer notification already done inside
        if self.config.loop_detector.policy == LoopPolicy.RAISE:
            raise detection.as_error()
        # else: corrective message already injected
```

This placement ensures structured retries (which `continue` before reaching tool execution) never trigger loop detection.

## Dependencies

- `StepType.GRAPH_LOOP_DETECTED` and `GRAPH_STALL` — already exist in `trace.py`
- `AgentObserver` base class — already exists in `observer.py`
- `AgentError` base class — for `LoopDetectedError` inheritance
- `_RunContext.all_tool_calls` — already accumulates all calls

## Risks

| Risk | Mitigation |
|------|-----------|
| False positives on legitimate retries (e.g., retry-after-rate-limit) | Thresholds configurable; default N=3 is conservative; tools that intentionally retry should use `RetryConfig`, not repeated tool calls |
| Performance overhead on large tool-call lists | Detectors only inspect the tail window (last N or K×M entries), not the full list |
| Three code paths (run/arun/astream) diverge | Shared `_check_loop_detection(ctx)` helper — single implementation, called from all three |
| Result hashing requires tool results available | StallDetector needs `(call, result)` pairs — store result hashes in `_RunContext` alongside `all_tool_calls` |
