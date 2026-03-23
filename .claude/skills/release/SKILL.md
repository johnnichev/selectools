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

## CRITICAL: Git Workflow Rules

- **Never push without explicit user approval** — commit locally, then ask
- **Always use PRs** — never push directly to main
- **Keep feature work on one branch** — don't merge WIP to main
- **No co-author lines** in commits

## 1. Pre-Release Checks

```bash
pytest tests/ -x -q
black src/ tests/ --line-length=100 --check
isort src/ tests/ --profile=black --line-length=100 --check
flake8 src/
mypy src/
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

## 2. Version Bump

Update version in TWO files (must match):
- `src/selectools/__init__.py`: `__version__ = "X.Y.Z"`
- `pyproject.toml`: `version = "X.Y.Z"`

## 3. CHANGELOG.md

Add entry at the top following existing format. Then sync:
```bash
cp CHANGELOG.md docs/CHANGELOG.md
```

## 4. README.md Updates

- Update "What's New" section
- Update feature table if new capabilities
- Update stats: test count, example count

## 5. ROADMAP.md + CLAUDE.md Updates

- Mark completed version with ✅
- Update any stale counts in CLAUDE.md

## 6. Count Audit

Run `/audit` to verify all hardcoded counts match live values.

## 7. Commit (DO NOT push yet)

```bash
git checkout -b release/vX.Y.Z   # or feat/<name> for feature releases
git add <specific-files>
git commit -m "release: vX.Y.Z — Feature Theme Name"
```

**Stop here and tell the user the commit is ready. Wait for explicit push approval.**

## 8. After User Approves Push

```bash
git push -u origin HEAD
gh pr create --title "release: vX.Y.Z" --body "..."
```

Wait for user to approve merge.

## 9. After PR Merged

```bash
gh pr merge <number> --merge --delete-branch
git checkout main && git pull
git tag -a vX.Y.Z -m "vX.Y.Z — Feature Theme Name"
git push origin main --tags
```

## 10. PyPI Publish (after user confirms)

```bash
rm -rf dist/
python3 -m build
python3 -m twine upload dist/*
```

## 11. Post-Release Verification

- Verify GitHub Pages auto-deploys docs
- Verify PyPI page shows new version
- `pip install selectools==X.Y.Z` in a clean env

## Version Numbering

- **Patch** (0.X.Y): Bug fixes, small features
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes (not yet — still pre-1.0)
