---
name: Good First Issue
about: A beginner-friendly contribution opportunity
title: "[Good First Issue] "
labels: good first issue, help wanted
assignees: ''
---

## What needs to be done

<!-- Clear description of the task -->

## Why this matters

<!-- How this helps users -->

## Getting started

1. Fork the repo and clone locally
2. `pip install -e ".[dev]"`
3. Run tests: `pytest tests/ -x -q`
4. Make your changes
5. Run lint: `black src/ tests/ --line-length=100 && isort src/ tests/ --profile=black --line-length=100`
6. Submit a PR

## Helpful context

<!-- Links to relevant source files, docs, or examples -->

## Acceptance criteria

- [ ] Tests pass
- [ ] Lint clean
- [ ] Example or test demonstrating the change
