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
- Observer events (sync): !`python3 -c "import ast; t=ast.parse(open('src/selectools/observer.py').read()); print(len([n.name for c in ast.walk(t) if isinstance(c,ast.ClassDef) and c.name=='AgentObserver' for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('on_')]))" 2>/dev/null`
- Observer events (async): !`python3 -c "import ast; t=ast.parse(open('src/selectools/observer.py').read()); print(len([n.name for c in ast.walk(t) if isinstance(c,ast.ClassDef) and c.name=='AsyncAgentObserver' for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('a_on_')]))" 2>/dev/null`
- StepTypes: !`python3 -c "from selectools.trace import StepType; print(len(StepType))" 2>/dev/null`
- ModelTypes: !`python3 -c "from selectools.models import ModelType; print(len(ModelType))" 2>/dev/null`
- Agent mixin files: !`ls src/selectools/agent/_*.py 2>/dev/null | wc -l | tr -d ' '`
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

### 3. Content Drift Audit

Check these files for content accuracy against the actual codebase:

**CLAUDE.md** (root project instructions):
- Codebase Structure tree matches actual directory layout (`ls src/selectools/agent/`, `ls src/selectools/providers/`)
- Common Pitfalls section is current (hooks deprecated, mixin architecture, etc.)
- Current Roadmap reflects shipped releases
- TraceStep Types table matches `StepType` enum members

**docs/ARCHITECTURE.md**:
- Agent description mentions mixin decomposition
- Observer section mentions `AsyncAgentObserver`
- Hooks referenced as deprecated, not current

**docs/modules/AGENT.md**:
- Hook System section has deprecation warning
- `AsyncAgentObserver` is documented
- Terminal actions (`@tool(terminal=True)`, `stop_condition`) documented
- Mixin architecture mentioned in Implementation Details

**docs/modules/TOOLS.md**:
- `@tool()` decorator docs include `terminal` parameter
- `Tool` class docs mention `terminal` attribute

**docs/modules/MODELS.md**:
- `ModelType` shown as `str, Enum` (not `Literal`)

**docs/modules/PROVIDERS.md**:
- Mentions `_OpenAICompatibleBase` shared by OpenAI/Ollama
- Namespace imports documented (`from selectools.providers import ...`)

**docs/QUICKSTART.md**:
- Terminal tools step present
- Observer examples use `AgentObserver` (not hooks as primary)

**docs/README.md**:
- Hooks listed as deprecated
- `AsyncAgentObserver` mentioned

### 4. Skills Audit
Check `.claude/skills/*/SKILL.md` for any hardcoded counts that don't match live values.

### 5. Link Check
Run `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build` and report any warnings.

### 6. Output

Present results in two tables:

**Count mismatches:**

| Location | Field | Found | Expected | Status |
|----------|-------|-------|----------|--------|

**Content drift:**

| Location | Issue | Status |
|----------|-------|--------|

Only show mismatches/issues. If everything matches, say "All counts are consistent" and "No content drift detected."
