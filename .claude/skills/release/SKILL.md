---
name: release
description: Prepare and execute a selectools release — version bump, changelog, docs, git, PyPI
argument-hint: <version-number>
---

# Release Process

Preparing release version: $ARGUMENTS

## Live Project State

- Current version: !`grep -m1 __version__ src/selectools/__init__.py`
- pyproject.toml version: !`grep -m1 "version" pyproject.toml`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Examples: !`ls examples/*.py | wc -l | tr -d ' '`
- Models: !`grep -c "ModelInfo(" src/selectools/models.py`
- StepTypes: !`python3 -c "from selectools.trace import StepType; print(len(StepType))" 2>/dev/null`
- Observer events (sync): !`python3 -c "from selectools.observer import AgentObserver; print(len([m for m in dir(AgentObserver) if m.startswith('on_')]))" 2>/dev/null`
- Last example: !`ls examples/*.py | tail -1`

## CRITICAL: Git Workflow Rules

- **Never push without explicit user approval** — commit locally, then ask
- **Always use PRs** — never push directly to main
- **Keep feature work on one branch** — don't merge WIP to main
- **No co-author lines** in commits

---

## Phase 1: Quality Gate — Lint & Tests

Run `/lint` (fix mode) to auto-format and check code quality. ALL four checks must pass:
- black
- isort
- flake8
- mypy

Then run the full test suite:
```bash
pytest tests/ -x -q
```

**STOP if any lint or test failure. Fix before proceeding.**

---

## Phase 2: Version Bump

Update version in TWO files (must match):
- `src/selectools/__init__.py`: `__version__ = "X.Y.Z"`
- `pyproject.toml`: `version = "X.Y.Z"`

---

## Phase 3: CHANGELOG.md

Add entry at the top following existing format. Include:
- Feature summary with code examples
- Bug fixes (if any)
- New test count, example count
- Migration notes (if breaking)

Then sync:
```bash
cp CHANGELOG.md docs/CHANGELOG.md
```

---

## Phase 4: Documentation Sweep

This is the most commonly missed step. For EVERY new feature in this release, verify ALL of these exist. Use `/docs` guidance for each missing item.

### 4a. Feature-Level Doc Checklist

For each new feature, verify:
- [ ] **Module doc** exists in `docs/modules/<FEATURE>.md`
- [ ] **mkdocs.yml** nav includes the module doc
- [ ] **Example script** exists in `examples/`
- [ ] **docs/index.md** feature table updated (if user-facing)
- [ ] **docs/QUICKSTART.md** updated (if it changes the getting-started flow)
- [ ] **docs/ARCHITECTURE.md** updated (if it adds a new component)

### 4b. Cross-Cutting Doc Updates

These docs reference counts and features that change with every release:

| Document | What to check |
|----------|---------------|
| `README.md` | "What's New" section, feature table, stats (test count, example count, model count) |
| `ROADMAP.md` | Mark completed version ✅, update Implementation Order section |
| `CLAUDE.md` | Codebase Structure tree, StepType table, Observer counts, Current Roadmap section, Common Pitfalls |
| `docs/index.md` | Feature table, model count, test count |
| `CONTRIBUTING.md` + `docs/CONTRIBUTING.md` | Test count |
| `landing/index.html` | Badge counts (if exists) |

### 4c. Private Docs

Update `.private/` tracking docs:
- `.private/session.md` — update Current State table, record what shipped
- `.private/master-competitive-plan.md` — mark completed items, update scorecard/counts
- `.private/competitive-analysis.md` — update comparison matrix, test count, advantages list
- `.private/growth-plan.md` — update product features list

### 4d. Doc Build Verification

```bash
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

Report any warnings. Fix broken links.

---

## Phase 5: Cross-Reference Audit

Run `/audit` to catch anything missed in Phase 4. This is a HARD GATE — do not proceed to commit if there are count mismatches or content drift.

The audit checks:
- Version consistency (__init__.py vs pyproject.toml)
- Hardcoded counts across all docs (tests, examples, models, observers, StepTypes)
- Content drift (CLAUDE.md structure tree, StepType table, observer counts)
- New feature doc coverage (module docs, examples, mkdocs nav)
- CHANGELOG sync
- Link validity (mkdocs build)

**Fix ALL mismatches before proceeding.**

---

## Phase 6: Commit (DO NOT push yet)

```bash
git checkout -b release/vX.Y.Z   # or feat/<name> for feature releases
git add <specific-files>
git commit -m "release: vX.Y.Z — Feature Theme Name"
```

**Stop here and tell the user the commit is ready. Wait for explicit push approval.**

---

## Phase 7: After User Approves Push

```bash
git push -u origin HEAD
gh pr create --title "release: vX.Y.Z — Theme" --body "$(cat <<'EOF'
## Summary
- Feature 1
- Feature 2
- ...

## Checklist
- [ ] All tests pass (N tests)
- [ ] Lint clean (black, isort, flake8, mypy)
- [ ] Docs updated (README, ROADMAP, CHANGELOG, module docs, index, architecture)
- [ ] Audit passed (counts consistent)
- [ ] mkdocs build clean
EOF
)"
```

Wait for user to approve merge.

---

## Phase 8: After PR Merged

```bash
gh pr merge <number> --merge --delete-branch
git checkout main && git pull
git tag -a vX.Y.Z -m "vX.Y.Z — Feature Theme Name"
git push origin main --tags
```

---

## Phase 9: PyPI Publish (after user confirms)

```bash
rm -rf dist/
python3 -m build
python3 -m twine upload dist/*
```

---

## Phase 10: Post-Release Verification

- Verify GitHub Pages auto-deploys docs
- Verify PyPI page shows new version
- `pip install selectools==X.Y.Z` in a clean env

---

## Version Numbering

- **Patch** (0.X.Y): Bug fixes, small features
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes (not yet — still pre-1.0)
