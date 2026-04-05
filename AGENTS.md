# Selectools -- Agent Instructions

## Commands

```bash
# Test (MUST pass before any commit)
pytest tests/ -x -q
pytest tests/ -k "not e2e" -x -q          # no API keys needed

# Format (MUST run before commit)
black src/ tests/ --line-length=100
isort src/ tests/ --profile=black --line-length=100

# Lint
flake8 src/
mypy src/

# Security (MUST pass before release tags)
bandit -r src/ -ll -q -c pyproject.toml

# Docs
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

## Stack

- Python 3.9+ (CI: 3.9, 3.10, 3.11, 3.12, 3.13)
- pip + setuptools, src-layout: `src/selectools/`
- pytest (4612 tests), Black + isort, flake8, mypy
- MkDocs Material for docs, deployed to GitHub Pages

## Boundaries

### Always (do without asking)
- Run `black` and `isort` on every modified file
- Run `pytest tests/ -k "not e2e" -x -q` after implementation changes
- Add type hints to all public APIs
- Add a regression test for every bug fix
- Apply `@beta` stability marker to new public symbols
- Use `response_msg.content or ""` when reading provider content

### Ask First
- Adding new fields to `AgentConfig`, `AgentResult`, or `AgentObserver`
- Adding new `StepType` values to `AgentTrace`
- Modifying `agent/core.py` (the main agent loop)
- Changing public API signatures on `@stable` symbols
- Adding new dependencies to `pyproject.toml`

### Never
- Push directly to main -- MUST use PR workflow
- Push to remote without explicit user request
- Call real LLM APIs in unit tests -- use `RecordingProvider` or mocks
- Use `datetime.utcnow()` -- MUST use `datetime.now(timezone.utc)`
- Use `x = x or default` when `0`, `""`, or `[]` are valid values
- Create `ThreadPoolExecutor()` per-call in async code -- use module singleton
- Use `self.config.model` in `astream()` -- MUST use `self._effective_model`
- Stringify `ToolCall` objects -- check `isinstance(chunk, str)` first
- Skip the `tools` parameter in provider `stream()`/`astream()` methods
- Add co-author trailer lines to commits

## Landmines

1. `Agent()` requires at least one tool as its first positional argument
2. `@tool()` decorator requires parentheses even with no arguments
3. Provider `stream()`/`astream()` MUST forward `tools` -- was a bug in ALL providers
4. OpenAI GPT-5.x/o-series/GPT-4.1 require `max_completion_tokens`, not `max_tokens`
5. `astream()` restores `_system_prompt` in `finally` -- omitting this leaks modified prompts
6. `astream()` shares helpers with `run()`/`arun()` via `_RunContext` -- add features to shared helpers, not individual methods
7. `FallbackProvider` streaming MUST call `_record_success()` AFTER full consumption
8. `ConversationMemory.branch()` MUST deep-copy Messages via `dataclasses.replace()` and restore `image_base64` explicitly
9. When `response_format` is set, `ToolCallParser` MUST NOT intercept valid JSON -- guard with `elif response_format is None:`
10. Eval judge prompts MUST fence user content with `<<<BEGIN_USER_CONTENT>>>`/`<<<END_USER_CONTENT>>>` delimiters
11. `_system_prompt` save MUST wrap `_prepare_run()`, not just the iteration loop
12. Early-exit result builders MUST call `_session_save()` and extraction methods
13. Async observer events MUST fire in all exit paths -- shared builders only fire sync observers
14. `_unwrap_type()` MUST handle both `types.UnionType` (3.10+) and `typing.Union`
15. SVG badge content MUST be XML-escaped via `xml.sax.saxutils.escape()`

## Patterns

- **Stability markers**: `@stable` (proven APIs), `@beta` (first release), `@deprecated(since="X.Y", replacement="Z")` with 2-minor-version deprecation window
- **Hooks are deprecated**: use `AgentObserver` or `AsyncAgentObserver`, not `AgentConfig.hooks`
- **Observer notifications**: use `_memory_add_many()` helper, not direct `self.memory.add_many()`
- **File naming**: source `snake_case.py`, tests `test_<module>.py`, examples `NN_name.py`, docs `UPPER_CASE.md`
- **Line length**: 100 characters everywhere
- **Comments**: only non-obvious intent, never explain what code does
- **mkdocs.yml YAML check**: needs `args: ["--unsafe"]` for Python emoji tags
- **MkDocs links**: files outside `docs/` MUST use absolute GitHub URLs
- **Every feature needs**: source + exports in `__init__.py` + tests + module docs + example

## Subagent Patterns

- **Bug hunts**: use `/bug-hunt` (parallel read-only audit) or `/ralph-bug-hunt module loops=N` (auto-fix loop)
- **Fan-out**: for 2+ independent tasks, dispatch parallel agents. Each agent MUST work on separate files
- **Worktrees**: use `isolation: "worktree"` for agents that need to make changes without conflicting with the main workspace
- **Quality gate**: after parallel agents complete, run `pytest tests/ -x -q` on the merged result before committing
