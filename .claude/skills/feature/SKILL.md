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

## 1. Cross-Feature Impact Analysis

Before writing code, determine:
- Does this touch `agent/core.py`? If so, follow the execution flow: input guardrails -> memory -> provider -> cache -> output guardrails -> parser -> policy -> coherence -> tool execution -> trace -> audit
- Does `AgentConfig` in `agent/config.py` need new fields?
- Does `__init__.py` need new public exports?
- Does `trace.py` need new `StepType` values?
- Does `observer.py` need new events? If so, add no-op default in `AgentObserver` class, implement in `LoggingObserver`, and use `_notify_observers()` helper in `agent/core.py`
- Does `AgentResult` need new fields?

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

**For RAG loaders** — add static methods to `DocumentLoader` in `src/selectools/rag/loaders.py` returning `List[Document]`. Include `metadata` param. Auto-populate `source` in metadata.

**For vector stores** — new file in `src/selectools/rag/stores/`. Inherit `VectorStore`. Implement `add_documents`, `search`, `delete`, `clear`. Register in `VectorStore.create()` factory and `stores/__init__.py`.

**For toolbox** — new file in `src/selectools/toolbox/`. Functions decorated with `@tool(description=...)`. Return `str`. Register in `toolbox/__init__.py`'s `get_all_tools()` and `get_tools_by_category()`.

## 3. Update Exports

Add to `src/selectools/__init__.py`:
```python
from selectools.new_module import NewClass
```

## 4. Format and Lint

Run `/lint` to format and check code quality.

## 5. Write Tests

See `/test` for detailed testing patterns.

## 6. Write Example Script

Create `examples/NN_feature_name.py` (use next number from Live Project State above).

## 7. Write Documentation

See `/docs` for detailed documentation patterns.

## 8. Run Full Test Suite

```bash
pytest tests/ -x -q
```

ALL tests must pass. No exceptions.

## Defensive Patterns (from past bugs)

- `response_msg.content or ""` — providers can return None
- `elif response_format is None:` before parser — don't intercept structured output
- `_memory_add_many()` not `self.memory.add_many()` — ensures observer fires
- Provider `stream()`/`astream()` MUST pass `tools` param
- Never stringify `ToolCall` objects in streaming paths
- FallbackProvider observer wiring needs `threading.Lock` + refcount
- `astream()` must save/restore `_system_prompt` in finally block (matches run/arun)
