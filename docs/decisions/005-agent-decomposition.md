# ADR-005: Agent Decomposition via Mixins

**Status**: Accepted
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

The `Agent` class in `core.py` grew to 3128 lines with 10+ responsibilities: tool execution, provider calling, observer notification, fallback wiring, memory management, session persistence, entity extraction, knowledge graph extraction, guardrails, structured output, caching, retry logic, and the main run/arun/astream loops.

This violated Single Responsibility and made the file difficult to navigate, review, and test. With v0.17.0 (multi-agent orchestration) adding graph execution, the class would only grow further.

## Options Considered

### A. Mixin classes (chosen)
Split methods into mixin classes in separate files. Agent inherits from all mixins. Methods access shared state via `self`.

### B. Delegate objects
Create standalone helper classes (`ToolExecutor`, `ProviderCaller`, etc.) that receive an Agent reference. Agent holds them as `self._tool_executor = ToolExecutor(self)`.

### C. Functional decomposition
Extract methods into module-level functions that take Agent as the first parameter. Agent methods become thin wrappers.

## Decision

**Option A: Mixins**, with Agent inheriting from 4 mixin classes:

```
Agent(_ToolExecutorMixin, _ProviderCallerMixin, _LifecycleMixin, _MemoryManagerMixin)
```

| Mixin | File | Responsibilities |
|-------|------|------------------|
| `_ToolExecutorMixin` | `agent/_tool_executor.py` | Tool execution pipeline, policy, coherence, parallel execution, timeouts |
| `_ProviderCallerMixin` | `agent/_provider_caller.py` | LLM provider calls, caching, retry, streaming |
| `_LifecycleMixin` | `agent/_lifecycle.py` | Observer notification, fallback wiring, result truncation |
| `_MemoryManagerMixin` | `agent/_memory_manager.py` | Memory operations, session save, entity/KG extraction |

## Rationale

1. **Mixins preserve `self` access**: All methods already use `self.config`, `self.provider`, `self._history`, etc. Mixins require zero refactoring of method bodies — they move verbatim.

2. **Delegates add indirection**: Every method would need `self._agent.config` instead of `self.config`, plus forwarding methods on Agent for anything called by other mixins. This doubles the number of attribute accesses and makes stack traces harder to read.

3. **Functions lose encapsulation**: Module-level functions can't be overridden by subclasses, preventing future extension points.

4. **MRO is simple**: All mixins are leaf classes (no diamond inheritance). The MRO is `[Agent, _ToolExecutorMixin, _ProviderCallerMixin, _LifecycleMixin, _MemoryManagerMixin, object]`.

5. **Private by convention**: All mixins are prefixed with underscore. They're internal implementation details, not part of the public API.

## Consequences

- **Positive**: `core.py` went from 3128 to 1448 lines (-54%). Each mixin file is 140-970 lines with a focused responsibility.
- **Positive**: Code navigation is easier — "where is tool execution?" → `_tool_executor.py`.
- **Positive**: Monkeypatching in tests still works — patching `Agent._execute_single_tool` works the same as before since mixins are resolved at class definition.
- **Negative**: `TYPE_CHECKING` imports needed to avoid circular dependencies between mixin files and `core.py` (for `_RunContext`).
- **Negative**: IDE "go to definition" on `self._notify_observers` from `_tool_executor.py` may land in `_lifecycle.py` rather than showing it on Agent — depends on IDE MRO support.
- **Negative**: mypy reports attribute errors on mixin classes (they reference `self.config` etc. without defining them). These are expected and suppressed. The architecture fitness tests verify the composed Agent has all required attributes.
