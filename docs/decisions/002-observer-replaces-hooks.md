# ADR-002: Observer Pattern Replaces Hooks Dict

**Status**: Accepted (hooks deprecated, not yet removed)
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

The original notification system was a `hooks` dict on `AgentConfig` — string-keyed callbacks with 11 events. In v0.15.0, `AgentObserver` was added as a class-based alternative with 25 events and richer data (run_id, call_id, system_prompt). Both systems coexisted, requiring every notification site to call both `_call_hook()` and `_notify_observers()`.

By v0.16.4, this dual system was:
- **Error-prone**: New events were sometimes added to observers but forgotten in hooks.
- **Noisy**: ~55 `_call_hook` calls alongside ~55 `_notify_observers` calls in core.py.
- **Type-unsafe**: Hook names were strings — `hooks["on_toll_start"]` (typo) would silently do nothing.

## Decision

1. Create `_HooksAdapter(AgentObserver)` that wraps a hooks dict as an observer, mapping the 11 hook events to observer methods.
2. In `Agent.__init__`, when `config.hooks` is set, emit `DeprecationWarning` and prepend a `_HooksAdapter` to the observers list.
3. Remove all `_call_hook()` calls — hooks are now served through the single observer pipeline.
4. Add `AsyncAgentObserver` for async-native integrations (blocking and non-blocking modes).

## Rationale

- **Single notification path**: One pipeline to maintain, one place to add new events.
- **Backward compatible**: The adapter makes existing hooks work transparently. Users see a deprecation warning but nothing breaks.
- **Type safety**: Observer methods are real Python methods — typos are caught by IDE autocompletion and mypy.
- **Richer data**: Observers receive `run_id` for correlation, `call_id` for parallel tool matching, `system_prompt` for debugging — data that hooks never had.

## Consequences

- **Positive**: core.py lost ~55 lines of `_call_hook` calls. The notification logic is consolidated.
- **Positive**: `AsyncAgentObserver` enables async DB writes without blocking the agent loop.
- **Negative**: Users with existing `hooks` usage see `DeprecationWarning`. Migration is straightforward: create an `AgentObserver` subclass and move hook logic to the corresponding `on_*` methods.
- **Decision**: No removal timeline set. Hooks remain functional via the adapter indefinitely. Removal will be evaluated for v0.18.0+.
