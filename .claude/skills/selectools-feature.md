# Selectools Feature Implementation

End-to-end skill for implementing new features in the selectools codebase. Covers source code, agent integration, exports, tests, docs, examples, and notebook updates.

## Trigger

Use when implementing a new module, feature, or capability in selectools (e.g., new RAG loader, vector store, toolbox module, orchestration component).

## Context

- **Source layout**: `src/selectools/` with public API in `__init__.py`
- **Python**: 3.9+, line length 100, Black + isort, mypy enforced
- **No `any` types** — always explicit types
- **No narration comments** — only explain non-obvious intent
- **Optional deps**: All external dependencies must be lazy-imported with try/except ImportError
- **Latest example**: `examples/32_coherence_checking.py` (next is 33)
- **Latest test count**: 1183

## Implementation Steps

### 1. Cross-Feature Impact Analysis

Before writing code, determine:
- Does this touch `agent/core.py`? If so, follow the execution flow: input guardrails -> memory -> provider -> cache -> output guardrails -> parser -> policy -> coherence -> tool execution -> trace -> audit
- Does `AgentConfig` in `agent/config.py` need new fields?
- Does `__init__.py` need new public exports?
- Does `trace.py` need new `StepType` values?
- Does `observer.py` need new events? If so, guard with `if run_id:` and use `_notify_observers()` helper
- Does `AgentResult` need new fields?

### 2. Write Source Code

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

### 3. Update Exports

Add to `src/selectools/__init__.py`:
```python
from selectools.new_module import NewClass
```

### 4. Format and Lint

```bash
black src/ tests/ --line-length=100
isort src/ tests/ --profile=black --line-length=100
flake8 src/
mypy src/
```

### 5. Write Tests

See the `selectools-test` skill for detailed testing patterns.

### 6. Write Example Script

Create `examples/NN_feature_name.py` (next available number, zero-padded).

### 7. Write Documentation

See the `selectools-docs` skill for detailed documentation patterns.

### 8. Run Full Test Suite

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
