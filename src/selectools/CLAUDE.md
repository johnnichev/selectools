# src/selectools/ — Source Code Conventions

## Style

- **Line length**: 100 characters
- **Formatter + linter**: Ruff (`ruff format` + `ruff check --fix`) — config in `pyproject.toml`
- **Type hints**: required on all public APIs — `mypy src/` must pass
- **No `Any` types** — always use explicit types
- **No comments** explaining what code does — only non-obvious intent

## Stability Markers (mandatory on all public symbols)

```python
from selectools.stability import stable, beta, deprecated
@beta            # New features in first release
@stable          # Core APIs, proven across releases
@deprecated(since="0.19", replacement="NewThing")  # Removal requires 2-minor window
```

## New Module Template

- `from __future__ import annotations` at top
- Module-level docstring describing purpose
- Lazy imports for optional deps: `try: import redis except ImportError: redis = None`

## Provider Protocol (providers/base.py)

Every provider must implement:
- `complete()` / `acomplete()` — sync/async completion
- `stream()` / `astream()` — sync/async streaming (yields `str | ToolCall`)
- `_format_messages()` — internal Message to provider-specific format
- **MUST pass `tools` to all stream/astream methods**
- **MUST handle `ToolCall` objects — never stringify them**

## Agent Loop (agent/core.py)

- Shared helpers: `_prepare_run()`, `_finalize_run()`, `_process_response()`, `_build_max_iterations_result()`
- `_RunContext` dataclass carries all per-run state
- **MUST use `self._effective_model`** not `self.config.model`
- **MUST use `response_msg.content or ""`** — providers can return None
- Observer events MUST fire in all exit paths (sync and async)
- `_system_prompt` must be saved before `_prepare_run()` and restored in `finally`

## Critical Pitfalls

- `datetime.now(timezone.utc)` not `datetime.utcnow()`
- `x = default if x is None else x` not `x = x or default` (zero/falsy trap)
- Module-level `ThreadPoolExecutor` singleton — never per-call
- `ConversationMemory.branch()` must deep-copy with `dataclasses.replace()`
- FallbackProvider must record success AFTER full stream consumption
- Early-exit builders must persist session/entity/KG state
