# How to Release a New Version to PyPI

This guide explains how to publish a new version of selectools to PyPI using the automated release scripts.

## Quick Start

```bash
# Test what would happen (dry-run)
python3 scripts/release.py --version 0.3.1 --dry-run

# Actually release
python3 scripts/release.py --version 0.3.1

# Or use the bash script
./scripts/release.sh 0.3.1
```

## Prerequisites (One-Time Setup)

### 1. Create PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `selectools-github-actions`
4. Scope: Select "Project: selectools" (or "Entire account")
5. Click "Add token"
6. **Copy the token** (starts with `pypi-...`) - you won't see it again!

### 2. Add Token to GitHub Secrets

1. Go to https://github.com/johnnichev/selectools/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Paste your PyPI token
5. Click "Add secret"

### 3. Optional: TestPyPI Token

Repeat the above steps for TestPyPI:
- Create token at: https://test.pypi.org/manage/account/token/
- Add to GitHub as: `TEST_PYPI_API_TOKEN`

## Release Process

### Step 1: Prepare Your Changes

Make sure all your changes are committed:

```bash
git status  # Should show clean working directory
git log     # Review recent commits
```

### Step 2: Choose Version Number

Follow [Semantic Versioning](https://semver.org/):

- **Patch** (0.3.0 → 0.3.1): Bug fixes, small improvements
- **Minor** (0.3.1 → 0.4.0): New features, backwards compatible
- **Major** (0.4.0 → 1.0.0): Breaking changes

### Step 3: Run Release Script

**Option A: Python Script (Recommended)**

```bash
# Dry run first to see what will happen
python3 scripts/release.py --version 0.3.1 --dry-run

# Actually release
python3 scripts/release.py --version 0.3.1

# With custom commit message
python3 scripts/release.py --version 0.3.1 --message "Add streaming support and bug fixes"
```

**Option B: Bash Script (Simpler)**

```bash
./scripts/release.sh 0.3.1

# With custom message
./scripts/release.sh 0.3.1 "Add streaming support"
```

### Step 4: Monitor GitHub Actions

The script will push a tag (e.g., `v0.3.1`) which triggers GitHub Actions.

1. Go to: https://github.com/johnnichev/selectools/actions
2. Watch the workflow run:
   - ✅ Test (runs all tests)
   - ✅ Build (builds package)
   - ✅ Publish to TestPyPI (if token configured)
   - ✅ Publish to PyPI (if token configured)

### Step 5: Verify Publication

After the workflow completes (usually 2-5 minutes):

1. Check PyPI: https://pypi.org/project/selectools/
2. Test installation:
   ```bash
   pip install --upgrade selectools
   python -c "import selectools; print(selectools.__version__)"
   ```

## What the Scripts Do

1. **Validate** version format and git status
2. **Update** `pyproject.toml` with new version
3. **Update** `CHANGELOG.md` with new entry (Python script only)
4. **Commit** changes with message
5. **Create** git tag (e.g., `v0.3.1`)
6. **Push** to GitHub (main branch + tag)
7. **Trigger** GitHub Actions workflow

GitHub Actions then:
- Runs all tests
- Builds wheel and source distribution
- Validates with `twine check`
- Publishes to PyPI

## Troubleshooting

### "Permission denied" when running scripts

```bash
chmod +x scripts/release.py scripts/release.sh
```

### "PYPI_API_TOKEN not found" in GitHub Actions

The publish step will be skipped. Add the token to GitHub secrets (see Prerequisites above).

### Version already exists on PyPI

PyPI doesn't allow re-uploading the same version. You must bump to a new version number.

### Tests fail in GitHub Actions

The publish step won't run if tests fail. Fix the tests and try again.

To retry with the same version, delete the tag:
```bash
git tag -d v0.3.1
git push origin :refs/tags/v0.3.1
# Fix issues, then re-run release script
```

### Want to undo a release

If you haven't pushed yet:
```bash
git reset --hard HEAD~1  # Undo commit
git tag -d v0.3.1        # Delete tag
```

If you already pushed:
```bash
# Can't undo PyPI publication, but can remove tag from GitHub
git push origin :refs/tags/v0.3.1
git tag -d v0.3.1
```

## Manual Release (Without Scripts)

If you prefer to do it manually:

```bash
# 1. Update version
# Edit pyproject.toml: version = "0.3.1"

# 2. Commit
git add pyproject.toml
git commit -m "Bump version to 0.3.1"

# 3. Tag
git tag v0.3.1

# 4. Push
git push origin main
git push origin v0.3.1

# GitHub Actions will automatically publish
```

## Release Checklist

- [ ] All changes committed and pushed to main
- [ ] Tests passing locally: `python tests/test_framework.py`
- [ ] Version number decided (semantic versioning)
- [ ] CHANGELOG.md updated (or will be auto-updated)
- [ ] PyPI API token configured in GitHub secrets
- [ ] Dry-run completed successfully
- [ ] Release script executed
- [ ] GitHub Actions workflow completed
- [ ] New version visible on PyPI
- [ ] Installation tested: `pip install --upgrade selectools`

## Version History

- **0.3.0** - Current version
- **0.2.0** - Previous release
- **0.1.0** - Initial release

## Questions?

- Check the [scripts/README.md](scripts/README.md) for more details
- Review the [.github/workflows/ci.yml](.github/workflows/ci.yml) workflow
- Open an issue if you encounter problems
