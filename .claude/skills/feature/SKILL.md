---
name: feature
description: End-to-end feature implementation — source, exports, tests, docs, examples, and notebook
argument-hint: <feature-description>
---

# Feature Implementation

Implement the following feature: $ARGUMENTS

## Live Project State

- Version: !`grep -m1 __version__ src/selectools/__init__.py`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Last example: !`ls examples/*.py | tail -1`
- Next example number: use the next zero-padded number after the last one above
- StepTypes: !`python3 -c "from selectools.trace import StepType; print(len(StepType))" 2>/dev/null`
- Observer events: !`python3 -c "from selectools.observer import AgentObserver; import inspect; print(len([m for m in dir(AgentObserver) if m.startswith('on_')]))" 2>/dev/null`

## 1. Cross-Feature Impact Analysis

Before writing code, determine:
- Does this touch `agent/core.py`? If so, follow the execution flow:
  `_prepare_run → cancellation check → budget check → model selection → on_iteration_start → provider call → _process_response → guardrails → parser → policy → coherence → tool execution → post-tool cancellation check → on_iteration_end`
- Does `AgentConfig` in `agent/config.py` need new fields?
- Does `__init__.py` need new public exports?
- Does `trace.py` need new `StepType` values? (currently 16)
- Does `observer.py` need new events? If so, add to ALL FOUR classes:
  - `AgentObserver` (no-op default)
  - `AsyncAgentObserver` (async no-op)
  - `LoggingObserver` (JSON emission via `_emit()`)
  - `SimpleStepObserver` (delegate to `self._cb()`)
- Does `AgentResult` need new fields?
- Update `tests/test_phase1_design_patterns.py` StepType count if adding new types

## 2. Write Source Code

**New module pattern**:
```python
"""Module docstring — one line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Lazy imports for optional deps
try:
    import some_lib
except ImportError:
    some_lib = None  # type: ignore[assignment]
```

**For agent loop changes** — add to the shared helpers (`_check_budget`, `_build_cancelled_result`, etc.) rather than duplicating in `run()`/`arun()`/`astream()`. Use `_RunContext` to carry per-run state.

**For provider caller changes** — use `self._effective_model` (not `self.config.model`) throughout `_provider_caller.py`.

**For tool changes** — add flags to both `Tool.__init__()` in `tools/base.py` AND `@tool()` decorator in `tools/decorators.py`.

## 3. Update Exports

Add to `src/selectools/__init__.py` in the appropriate section.

## 4. Format and Lint

Run `/lint` to format and check code quality.

## 5. Write Tests

See `/test` for detailed testing patterns. Key reminders:
- Agent requires at least one tool (use `_DUMMY`)
- Use `(Message, UsageStats)` tuples for controlled usage stats
- Test observer events fire correctly
- Test trace steps are recorded
- Test backward compatibility (None/default values work)

## 6. Write Example Script

Create `examples/NN_feature_name.py` (use next number from Live Project State above).

## 7. Write Documentation

See `/docs` for detailed documentation patterns.

## 8. Run Full Test Suite

```bash
pytest tests/ -x -q
```

ALL tests must pass. No exceptions.

## 9. Run Audit

Run `/audit` to verify all counts are consistent across docs. Fix any mismatches.

## 10. Commit (but DO NOT push)

Stage specific files and commit. Wait for user to approve before pushing.

## Defensive Patterns (from past bugs)

- `response_msg.content or ""` — providers can return None
- `elif response_format is None:` before parser — don't intercept structured output
- `_memory_add_many()` not `self.memory.add_many()` — ensures observer fires
- Provider `stream()`/`astream()` MUST pass `tools` param
- Never stringify `ToolCall` objects in streaming paths
- FallbackProvider observer wiring needs `threading.Lock` + refcount
- `astream()` must save/restore `_system_prompt` in finally block
- Budget check at TOP of iteration (before LLM call, not after)
- Cancellation check at TWO points: top of iteration AND after tool execution
- Model selection via `_effective_model` property (not `self.config.model`)
- `StreamChunk` has no `finished` field — don't pass `finished=True`
- `bandit`: mark safe SQL with `# nosec B608`, mark safe pass with `# nosec B110`
