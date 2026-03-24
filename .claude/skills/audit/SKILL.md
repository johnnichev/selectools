---
name: audit
description: Cross-reference audit for stale counts, broken links, and doc drift across all files
---

# Cross-Reference Audit

## Live Counts (source of truth)

- Version (__init__.py): !`grep -m1 __version__ src/selectools/__init__.py`
- Version (pyproject.toml): !`grep -m1 "version" pyproject.toml`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Models: !`grep -c "ModelInfo(" src/selectools/models.py`
- Examples: !`ls examples/*.py | wc -l | tr -d ' '`
- Module docs: !`ls docs/modules/*.md | wc -l | tr -d ' '`
- Toolbox tools: !`grep -roh "@tool" src/selectools/toolbox/*.py | wc -l | tr -d ' '`
- Observer events (sync): !`python3 -c "from selectools.observer import AgentObserver; print(len([m for m in dir(AgentObserver) if m.startswith('on_')]))" 2>/dev/null`
- Observer events (async): !`python3 -c "from selectools.observer import AsyncAgentObserver; print(len([m for m in dir(AsyncAgentObserver) if m.startswith('a_on_')]))" 2>/dev/null`
- StepTypes: !`python3 -c "from selectools.trace import StepType; print(len(StepType))" 2>/dev/null`
- Knowledge store backends: !`echo "4 (File, SQLite, Redis, Supabase)"`
- Last example: !`ls examples/*.py | tail -1`

## Audit Procedure

### 1. Version Consistency
Verify `__init__.py` version matches `pyproject.toml` version.

### 2. Count Audit

Search each file for hardcoded counts. Report any mismatch against live values.

**Test count** — check:
- `README.md`, `docs/index.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/CONTRIBUTING.md`, `landing/index.html`

**Example count** — check:
- `README.md`, `CLAUDE.md`

**Model count** — check:
- `README.md`, `docs/index.md`, `CLAUDE.md`, `docs/modules/MODELS.md`

**Observer event count** — check:
- `CLAUDE.md`, `docs/modules/AGENT.md`, `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`

**StepType count** — check:
- `CLAUDE.md`, `tests/test_phase1_design_patterns.py`

### 3. Content Drift Audit

**CLAUDE.md**:
- Codebase Structure tree matches actual files (check for missing new files)
- TraceStep Types table has all StepType members
- Current Roadmap matches ROADMAP.md
- Observer counts match

**docs/ARCHITECTURE.md**:
- Version header is current
- Observer event counts current

**docs/modules/AGENT.md**:
- Lifecycle events table title matches actual count
- AsyncAgentObserver event count correct
- Hooks referenced as deprecated

**docs/modules/KNOWLEDGE.md**:
- Mentions all 4 store backends
- References KnowledgeEntry, KnowledgeStore protocol

**docs/modules/TOOLS.md**:
- Mentions `requires_approval` parameter
- Mentions `_serialize_result()` behavior

### 4. New Feature Doc Coverage

Verify each v0.17.3+ feature has:
- [ ] Module doc in `docs/modules/`
- [ ] Example script in `examples/`
- [ ] Entry in `mkdocs.yml` nav

Features to check: Budget, Cancellation, Token Estimation, Model Switching, SimpleStepObserver, Structured Results, Approval Gate, Reasoning Strategies, Tool Result Caching

### 5. Link Check
```bash
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```
Report any warnings.

### 6. CHANGELOG Sync
Verify `docs/CHANGELOG.md` matches `CHANGELOG.md`:
```bash
diff CHANGELOG.md docs/CHANGELOG.md
```

### 7. Private Docs (if accessible)
Check `.private/master-competitive-plan.md`, `.private/competitive-analysis.md`, and `.private/growth-plan.md` for stale test counts, example counts, version references, and competitive scorecard accuracy.

### 8. Output

Present results in two tables:

**Count mismatches:**

| Location | Field | Found | Expected | Status |
|----------|-------|-------|----------|--------|

**Content drift:**

| Location | Issue | Status |
|----------|-------|--------|

Only show mismatches/issues. If everything matches, say "All counts are consistent" and "No content drift detected."
