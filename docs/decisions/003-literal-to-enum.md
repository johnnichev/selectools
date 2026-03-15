# ADR-003: Literal Types to str Enums (StepType, ModelType)

**Status**: Accepted
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

`StepType` (14 trace step types) and `ModelType` (5 model categories) were defined as `Literal` type aliases:

```python
StepType = Literal["llm_call", "tool_selection", "tool_execution", ...]
ModelType = Literal["chat", "embedding", "image", "audio", "multimodal"]
```

These provided type narrowing at check time but no runtime safety. A typo like `type="llm_cal"` would pass mypy (assignable to `str`) and fail silently at runtime — the trace step would be created with the wrong type, and `trace.filter(type="llm_call")` would miss it.

With v0.17.0 (multi-agent orchestration) adding 4+ new step types, the risk of typo-introduced bugs was increasing.

## Decision

Replace both with `class StepType(str, Enum)` and `class ModelType(str, Enum)`.

## Rationale

1. **Backward compatible**: `str, Enum` members compare equal to their string values. `StepType.LLM_CALL == "llm_call"` is `True`. Existing code using string comparisons continues to work.

2. **Fail-fast on typos**: `StepType("llm_cal")` raises `ValueError` at runtime. IDE autocompletion prevents the typo in the first place.

3. **Enumerable**: `list(StepType)` gives all valid values. Architecture fitness tests verify every member appears in at least one test and every ModelType is used in the registry.

4. **Hashable and serializable**: `str, Enum` members work as dict keys, in sets, and serialize to JSON as their string value via `dataclasses.asdict()`.

## Consequences

- **Positive**: 30+ string literals in `core.py` replaced with enum members. IDE autocompletion now works for step types.
- **Positive**: 146 `ModelInfo` records in `models.py` now use `ModelType.CHAT` etc., preventing invalid model type assignments.
- **Positive**: Architecture tests automatically verify enum coverage.
- **Negative**: Negligible — enum member access is microscopically slower than string comparison, but trace steps are created ~10 times per agent run, not in hot loops.
- **Migration**: None required. All string comparisons (`step.type == "llm_call"`) still work. New code should use enum members (`StepType.LLM_CALL`).
