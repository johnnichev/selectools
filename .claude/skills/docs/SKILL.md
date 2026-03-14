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

## Documentation Checklist (ALL required for every feature)

### 1. Module Documentation

Create `docs/modules/<FEATURE>.md` with:

```markdown
# Feature Name

Brief description of what this feature does and why it matters.

## Quick Start
(Minimal working example)

## API Reference
(Class/method signatures with parameter tables)

## Examples
(Basic and advanced usage)

## Integration with Agent
(How to use via AgentConfig)
```

### 2. Navigation Entry

Add to `mkdocs.yml` nav under the appropriate tab section.

### 3. Landing Page Update

Update `docs/index.md` — add to feature table, update counts to match live values above.

### 4. Documentation Index

Update `docs/README.md` with the new module page.

### 5. Quickstart Update

If user-facing, add a step to `docs/QUICKSTART.md`.

### 6. Architecture Update

If it adds a new system component, update `docs/ARCHITECTURE.md`.

### 7. Notebook Section

Add a section to `notebooks/getting_started.ipynb` with interactive examples.

### 8. Example Script

Create `examples/NN_feature_name.py` (next available number, zero-padded).

## Link Rules

- **Within docs/**: Use relative paths (`modules/AGENT.md`, `../ARCHITECTURE.md`)
- **Outside docs/** (ROADMAP.md, examples/, notebooks/): Use absolute GitHub URLs
  - Example: `https://github.com/johnnichev/selectools/blob/main/examples/01_hello_world.py`

## Style Conventions

- Use admonitions for callouts: `!!! tip`, `!!! warning`, `!!! example`
- Use tabbed content for install/usage variants: `=== "Tab Name"`
- Use Material icons: `:material-icon-name:`
- Code examples should be complete and runnable
- No comments explaining obvious code — only non-obvious intent

## Verify Build

Always run after doc changes:
```bash
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

Check for broken links and warnings in the output.
