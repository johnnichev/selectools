---
name: lint
description: Run the full code quality pipeline — ruff format, ruff check, mypy, bandit
argument-hint: [check-only]
disable-model-invocation: true
---

# Code Quality Pipeline

The repo migrated off black/isort/flake8 to **ruff** (format + lint) plus **mypy**
and **bandit**. These are exactly what the pre-commit hooks and CI enforce, so
matching them here keeps `/release` from tripping at commit time.

## Mode

If "$ARGUMENTS" contains "check" or "check-only", run in **check mode** (no modifications). Otherwise, run in **fix mode** (auto-format and auto-fix, then verify).

## Fix Mode (default)

Run these in sequence, stopping on first failure:

1. `ruff check src/ tests/ --fix`   (lint + autofix; replaces flake8 + isort)
2. `ruff format src/ tests/`         (format; replaces black)
3. `mypy src/`
4. `bandit -c pyproject.toml -r src/`

## Check Mode

Read-only, no modifications:

1. `ruff check src/ tests/`
2. `ruff format src/ tests/ --check`
3. `mypy src/`
4. `bandit -c pyproject.toml -r src/`

## On Failure

- **ruff check failure**: Report the violations with file paths and line numbers. Fix them — do not blanket-suppress with `# noqa`.
- **ruff format failure (check mode)**: Report which files need formatting. Offer to re-run in fix mode.
- **mypy failure**: Report the type errors. Fix them — do not use `# type: ignore` unless truly unavoidable, and always give it an explicit error code (`# type: ignore[code]`) plus a comment explaining why.
- **bandit failure**: Report the finding. Fix it, or justify with a scoped `# nosec <ID>` and a comment.

## Success

When all four pass, report: "All clean — ruff check, ruff format, mypy, bandit passed."
