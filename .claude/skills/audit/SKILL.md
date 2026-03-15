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
- Observer events: !`python3 -c "import ast; t=ast.parse(open('src/selectools/observer.py').read()); print(len([n.name for c in ast.walk(t) if isinstance(c,ast.ClassDef) and c.name=='AgentObserver' for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('on_')]))" 2>/dev/null`
- StepTypes: !`python3 -c "import re; c=open('src/selectools/trace.py').read(); m=re.search(r'StepType = Literal\[(.*?)\]',c,re.DOTALL); print(len(re.findall(r'\"(\w+)\"',m.group(1))))" 2>/dev/null`
- Last example: !`ls examples/*.py | tail -1`

## Audit Procedure

### 1. Version Consistency
Verify `__init__.py` version matches `pyproject.toml` version. Report if they differ.

### 2. Count Audit

Search each file below for the relevant count. Report any mismatch against the live values above.

**Model count** — check:
- `README.md`, `docs/index.md`, `docs/QUICKSTART.md`, `docs/ARCHITECTURE.md`, `docs/modules/MODELS.md`, `CLAUDE.md`

**Test count** — check:
- `README.md`, `docs/index.md`, `CLAUDE.md`

**Example count** — check:
- `README.md`, `docs/index.md`, `CLAUDE.md`

**Observer event count** — check:
- `CLAUDE.md`, `docs/modules/AGENT.md`

**StepType count** — check:
- `CLAUDE.md`

**Module doc count** — check:
- `CLAUDE.md`

### 3. Skills Audit
Check `.claude/skills/*/SKILL.md` for any hardcoded counts that don't match live values.

### 4. Link Check
Run `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build` and report any warnings.

### 5. Output

Present results as a table:

| Location | Field | Found | Expected | Status |
|----------|-------|-------|----------|--------|

Only show mismatches. If everything matches, say "All counts are consistent."
