---
name: docs
description: Write or update documentation — module docs, index, quickstart, architecture, examples, notebook
argument-hint: <module-or-topic>
---

# Documentation Writing

Write or update documentation for: $ARGUMENTS

## Live Counts (source of truth — update all docs to match)

- Models: !`grep -c "ModelInfo(" src/selectools/models.py`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Examples: !`ls examples/*.py | wc -l | tr -d ' '`
- Module docs: !`ls docs/modules/*.md | wc -l | tr -d ' '`
- StepTypes: !`python3 -c "from selectools.trace import StepType; print(len(StepType))" 2>/dev/null`
- Observer events: !`python3 -c "from selectools.observer import AgentObserver; import inspect; print(len([m for m in dir(AgentObserver) if m.startswith('on_')]))" 2>/dev/null`

## Documentation Checklist (ALL required for every feature)

### 1. Module Documentation

Create `docs/modules/<FEATURE>.md` with:

```markdown
# Feature Name

**Added in:** vX.Y.Z
**File:** `src/selectools/module.py`
**Classes:** `ClassName`

## Overview
Brief description.

## Quick Start
Minimal working example.

## API Reference
Class/method signatures with parameter tables.

## See Also
Links to related module docs.
```

### 2. Navigation Entry

Add to `mkdocs.yml` nav under the appropriate section:

```yaml
nav:
  - Core:          # Agent, Tools, Memory, Sessions, etc.
  - Runtime Controls:  # Budget, Cancellation, Token Estimation, Model Switching
  - Providers:     # Overview, Models, Usage
  - RAG:           # Pipeline, Hybrid Search, Chunking, etc.
  - Evaluation:    # Eval Framework
  - Integration:   # MCP Client/Server
  - Security:      # Guardrails, Audit, Screening
```

### 3. Landing Page Update

Update `docs/index.md` — add to feature table, update counts.

### 4. Quickstart Update

If user-facing, add a step to `docs/QUICKSTART.md`.

### 5. Architecture Update

If it adds a new system component, update `docs/ARCHITECTURE.md`.

### 6. Example Script

Create `examples/NN_feature_name.py` (next available number).
Follow existing style: docstring at top, self-contained, `main()` function with `if __name__` guard.

### 7. Count Sync

After all doc changes, update hardcoded counts in:
- `README.md` (test count, example count)
- `docs/index.md` (test count)
- `CLAUDE.md` (test count, example references)
- `CONTRIBUTING.md` + `docs/CONTRIBUTING.md` (test count)
- `landing/index.html` (test count badge)

### 8. CHANGELOG Sync

Always sync after updating CHANGELOG.md:
```bash
cp CHANGELOG.md docs/CHANGELOG.md
```

## Link Rules

- **Within docs/**: Use relative paths (`modules/AGENT.md`, `../ARCHITECTURE.md`)
- **Outside docs/** (ROADMAP.md, examples/, notebooks/): Use absolute GitHub URLs

## Style Conventions

- Use admonitions: `!!! tip`, `!!! warning`, `!!! example`
- Code examples should be complete and runnable
- No comments explaining obvious code — only non-obvious intent

## Verify Build

Always run after doc changes:
```bash
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```
