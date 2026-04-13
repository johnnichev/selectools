# Selectools — Agent Instructions

Production-ready Python library for AI agents with tool calling, RAG, and multi-agent orchestration. OpenAI, Anthropic, Gemini, Ollama. Python 3.9+, src-layout, pytest, MkDocs Material.

- **Version**: `src/selectools/__init__.py`
- **PyPI**: https://pypi.org/project/selectools/
- **Docs**: https://selectools.dev

See `AGENTS.md` for commands, boundaries, and condensed landmines.
See subdirectory `CLAUDE.md` files for scoped rules: `tests/`, `src/selectools/`, `docs/`.

**Session workflow:** At ~40 messages or when context feels heavy: update `HANDOFF.md` with current state, run `/clear`, start fresh. This outperforms auto-compaction.

## Commands

```bash
pytest tests/ -x -q                                    # All tests
pytest tests/ -k "not e2e" -x -q                       # Skip E2E
black src/ tests/ --line-length=100                     # Format
isort src/ tests/ --profile=black --line-length=100     # Sort imports
flake8 src/ && mypy src/                                # Lint + types
bandit -r src/ -ll -q -c pyproject.toml                 # Security
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build       # Docs
```

## Stability Markers

Apply to every public class/function in `__init__.py` exports:

| Marker | When |
|--------|------|
| `@stable` | Core APIs, at least one release proven |
| `@beta` | First release, experimental |
| `@deprecated(since="0.X", replacement="Y")` | Removing/renaming. Keep 2 minor versions |

## StepType Reference (27 types)

`llm_call`, `tool_selection`, `tool_execution`, `cache_hit`, `error`, `structured_retry`, `guardrail`, `coherence_check`, `output_screening`, `session_load`, `session_save`, `memory_summarize`, `entity_extraction`, `kg_extraction`, `budget_exceeded`, `cancelled`, `prompt_compressed`, `graph_node_start`, `graph_node_end`, `graph_routing`, `graph_checkpoint`, `graph_interrupt`, `graph_resume`, `graph_parallel_start`, `graph_parallel_end`, `graph_stall`, `graph_loop_detected`

## Common Pitfalls (from past bugs)

1. **Provider streaming MUST pass `tools`**: All `stream()`/`astream()` methods MUST forward `tools` to the API.
2. **ToolCall objects MUST NOT be stringified**: Check `isinstance(chunk, str)` before appending in streaming paths.
3. **OpenAI `max_completion_tokens`**: GPT-5.x and o-series require `max_completion_tokens`, not `max_tokens`. See `_uses_max_completion_tokens()`.
4. **FallbackProvider.astream() error handling**: MUST include try/except with `_is_retriable`, `_record_failure`, `on_fallback`, `_record_success`.
5. **FallbackProvider + observers thread safety**: `_wire_fallback_observer` uses `threading.Lock` + refcount.
6. **Structured output vs text parser**: When `response_format` is set, `ToolCallParser` MUST NOT intercept valid JSON. Guard with `elif response_format is None:`.
7. **None content from providers**: MUST use `response_msg.content or ""`.
8. **Memory `on_memory_trim`**: Use `_memory_add_many()` helper, not `self.memory.add_many()`.
9. **Pre-commit YAML**: mkdocs.yml needs `args: ["--unsafe"]` for Python tags.
10. **MkDocs links**: Files outside `docs/` MUST use absolute GitHub URLs.
11. **`_system_prompt` restore in finally**: All execution methods save/restore in try/finally. The try block MUST wrap `_prepare_run()`, not just the iteration loop.
12. **`astream()` full parity**: Add features to shared helpers (`_prepare_run`, `_finalize_run`, `_process_response`), not individual methods. `_RunContext` carries per-run state.
13. **Hooks deprecated**: Use `AgentObserver`/`AsyncAgentObserver`, not `AgentConfig.hooks`.
14. **FallbackProvider streaming success**: Record success AFTER full consumption, not before.
15. **`_effective_model`**: All code MUST use `self._effective_model`, not `self.config.model`.
16. **Async observer events in all exit paths**: Early-exit builders only fire sync. Add `await _anotify_observers()` in async methods.
17. **`datetime.now(timezone.utc)`**: MUST use aware datetimes, not `datetime.utcnow()`.
18. **Async guardrails**: `arun()`/`astream()` use `_arun_input_guardrails()` with `skip_guardrails=True` in `_prepare_run()`.
19. **Fence eval judge prompts**: Wrap user fields with `<<<BEGIN_USER_CONTENT>>>` delimiters.
20. **Module-level ThreadPoolExecutor**: Use `_get_async_tool_executor()` singleton, not per-call.
21. **Python 3.10+ union syntax**: `_unwrap_type()` MUST handle both `types.UnionType` and `typing.Union`.
22. **Zero/falsy confusion**: MUST use `x = default if x is None else x`, not `x = x or default`. `0`, `""`, `[]` are valid.
23. **Early-exit builders MUST persist state**: `_build_max_iterations_result` etc. MUST call `_session_save()` and `_extract_entities()`/`_extract_kg_triples()`.
24. **`ConversationMemory.branch()` deep copy**: Use `dataclasses.replace()` on every Message. Restore `image_base64` explicitly (it's `init=False`).
25. **SVG badge XML escaping**: Use `xml.sax.saxutils.escape()` for label/value interpolation.
26. **`bandit` annotations**: Mark safe SQL with `# nosec B608`, safe subprocess with `# nosec B404`.
27. **`aclosing()` for async generators**: `async for item in provider.astream(...)` MUST be wrapped in `async with aclosing(gen) as gen:` so the provider generator is deterministically closed on exception. Use `selectools._async_utils.aclosing` (Python 3.9 backport of `contextlib.aclosing`).
28. **ContextVars propagation in `run_in_executor`**: Direct `loop.run_in_executor(None, fn)` drops `contextvars.Context` (OTel spans, Langfuse traces lost). Use `run_in_executor_copyctx(loop, executor, fn)` from `_async_utils.py`.
29. **Malformed tool-call JSON recovery**: Provider `json.loads()` failures MUST surface via `ToolCall.parse_error`, not silent `return {}`. Use shared `_parse_tool_args()` helper. Tool executor checks `parse_error` before tool lookup.
30. **Structured retry budget**: `RetryConfig.max_retries` controls structured-validation retries INDEPENDENTLY of `max_iterations`. Outer loop uses `max_iterations + ctx.structured_retries` so validation failures don't eat the tool budget.
