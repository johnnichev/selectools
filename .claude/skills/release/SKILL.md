---
name: release
description: Prepare and execute a selectools release — version bump, changelog, docs, git, PyPI
argument-hint: <version-number>
disable-model-invocation: true
---

# Release Process

Preparing release version: $ARGUMENTS

## Live Project State

- Current version: !`grep -m1 __version__ src/selectools/__init__.py`
- pyproject.toml version: !`grep -m1 "version" pyproject.toml`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Examples: !`ls examples/*.py | wc -l | tr -d ' '`
- Models: !`grep -c "ModelInfo(" src/selectools/models.py`

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

Add entry at the top:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added — Feature Theme Name

#### Feature Name (new `module/` subpackage)

- **`ClassName`**: Description of what it does and key capabilities.
- **Agent integration**: How it plugs into the agent via `AgentConfig(option=...)`.

### Changed

- **`StepType` literal**: Added new trace step types if any.
- **`AgentConfig`**: New fields added.

### Documentation

- **New module docs**: List new docs pages
- **New examples**: List new example scripts

### Tests

- **NN new tests** (total: NNNN): Summary of test coverage.
```

## 4. README.md Updates

- Update "What's New" section
- Update feature table if new capabilities
- Update stats: model count, test count, example count

## 5. ROADMAP.md Updates

- Mark completed features with checkmark
- Add entry to "Release History" section
- Update status from "In Progress" to "Complete"

## 6. Count Audit

Run `/audit` to verify all hardcoded counts match live values.

## 7. Git Workflow

```bash
git checkout -b release/vX.Y.Z
git add <specific-files>
git commit -m "release: vX.Y.Z — Feature Theme Name"
git push -u origin HEAD
gh pr create --title "release: vX.Y.Z" --body "..."
# After PR merged:
gh pr merge <number> --merge --delete-branch
git checkout main && git pull
git tag -a vX.Y.Z -m "vX.Y.Z — Feature Theme Name"
git push origin main --tags
```

## 8. PyPI Publish

```bash
rm -rf dist/
python3 -m build
python3 -m twine upload dist/*
```

## 9. Post-Release Verification

- Verify GitHub Pages auto-deploys docs
- Verify PyPI page shows new version
- Test install: `pip install selectools==X.Y.Z`

## Version Numbering

- **Patch** (0.X.1): Bug fixes, no new features
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes (not yet — still pre-1.0)
