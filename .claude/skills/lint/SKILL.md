---
name: lint
description: Run the full code quality pipeline — black, isort, flake8, mypy
argument-hint: [check-only]
disable-model-invocation: true
---

# Code Quality Pipeline

## Mode

If "$ARGUMENTS" contains "check" or "check-only", run in **check mode** (no modifications). Otherwise, run in **fix mode** (auto-format, then check).

## Fix Mode (default)

Run these commands in sequence, stopping on first failure:

1. `black src/ tests/ --line-length=100`
2. `isort src/ tests/ --profile=black --line-length=100`
3. `flake8 src/`
4. `mypy src/`

Report results after all steps complete (or at first failure).

## Check Mode

Run these commands in sequence (read-only, no modifications):

1. `black src/ tests/ --line-length=100 --check`
2. `isort src/ tests/ --profile=black --line-length=100 --check`
3. `flake8 src/`
4. `mypy src/`

## On Failure

- **black/isort failure in check mode**: Report which files need formatting. Offer to re-run in fix mode.
- **flake8 failure**: Report the violations with file paths and line numbers. Fix them — do not suppress with `# noqa`.
- **mypy failure**: Report the type errors. Fix them — do not use `type: ignore` unless truly unavoidable, and explain why in a comment.

## Success

When all 4 pass, report: "All clean — black, isort, flake8, mypy passed."
