# Selectools Release Process

Skill for preparing and executing a selectools release.

## Trigger

Use when bumping version, preparing a release, publishing to PyPI, or updating release artifacts.

## Release Steps

### 1. Pre-Release Checks

```bash
# All tests must pass
pytest tests/ -x -q

# Code quality
black src/ tests/ --line-length=100 --check
isort src/ tests/ --profile=black --line-length=100 --check
flake8 src/
mypy src/

# Docs build clean
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

### 2. Version Bump

Update version in TWO files (must match):
- `src/selectools/__init__.py`: `__version__ = "X.Y.Z"`
- `pyproject.toml`: `version = "X.Y.Z"`

### 3. CHANGELOG.md

Add entry at the top following the existing format:

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
- **Updated module docs**: List updated pages
- **New examples**: List new example scripts
- **Updated notebook**: Sections added

### Tests

- **NN new tests** (total: NNNN): Summary of test coverage.
```

### 4. README.md Updates

- Update "What's New" section
- Update feature table if new capabilities
- Update stats: model count, test count, example count

### 5. ROADMAP.md Updates

- Mark completed features with checkmark
- Move completed version to "Release History" section
- Update status from "In Progress" to "Complete"

### 6. Hardcoded Count Audit

Search ALL docs for stale counts:
- Model count (currently 146): `index.md`, `README.md`, `MODELS.md`, `QUICKSTART.md`, `ARCHITECTURE.md`
- Test count: `README.md`, `index.md`, `CHANGELOG.md`
- Example count: `README.md`, `index.md`
- Tool count (currently 24): `index.md`, `README.md`, `TOOLBOX.md`

### 7. Git Workflow

```bash
# Create release branch
git checkout -b release/vX.Y.Z

# Stage and commit
git add -A
git commit -m "release: vX.Y.Z — Feature Theme Name"

# Push and create PR
git push -u origin HEAD
gh pr create --title "release: vX.Y.Z" --body "..."

# After PR merged
gh pr merge <number> --merge --delete-branch
git checkout main && git pull

# Tag
git tag -a vX.Y.Z -m "vX.Y.Z — Feature Theme Name"
git push origin main --tags
```

### 8. PyPI Publish

```bash
python3 -m build
python3 -m twine upload dist/*
```

### 9. Post-Release Verification

- Verify GitHub Pages auto-deploys docs
- Verify PyPI page shows new version
- Test install: `pip install selectools==X.Y.Z`

## Version Numbering

- **Patch** (0.X.1): Bug fixes, no new features
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes (not yet — still pre-1.0)
