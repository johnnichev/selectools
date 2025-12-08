# Release Scripts

This directory contains scripts to help automate the release process for publishing selectools to PyPI.

## Prerequisites

Before using these scripts, ensure you have:

1. **PyPI API Token configured in GitHub**

   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Add it to GitHub repository secrets as `PYPI_API_TOKEN`
   - Go to: https://github.com/johnnichev/selectools/settings/secrets/actions

2. **Clean working directory**

   - Commit or stash any uncommitted changes before releasing

3. **On the main branch**
   - The scripts will warn you if you're not on main

## Option 1: Python Script (Recommended)

The Python script provides more features including dry-run mode and automatic changelog updates.

### Usage

```bash
# Basic release
python scripts/release.py --version 0.3.1

# With custom commit message
python scripts/release.py --version 0.4.0 --message "Add streaming support and bug fixes"

# Dry run (see what would happen without making changes)
python scripts/release.py --version 1.0.0 --dry-run
```

### Features

- ✅ Validates semantic versioning format
- ✅ Updates `pyproject.toml` version
- ✅ Updates `CHANGELOG.md` with new version entry
- ✅ Creates commit and tag
- ✅ Pushes to GitHub
- ✅ Dry-run mode for testing
- ✅ Interactive confirmations at each step

## Option 2: Bash Script (Quick & Simple)

A simpler bash script for quick releases.

### Usage

```bash
# Basic release
./scripts/release.sh 0.3.1

# With custom commit message
./scripts/release.sh 0.4.0 "Add new features"
```

### Features

- ✅ Validates semantic versioning format
- ✅ Updates `pyproject.toml` version
- ✅ Creates commit and tag
- ✅ Pushes to GitHub
- ✅ Interactive confirmations

## What Happens After Running the Script

1. **Local changes:**

   - Version updated in `pyproject.toml`
   - Changes committed
   - Git tag created (e.g., `v0.3.1`)
   - Pushed to GitHub

2. **GitHub Actions automatically:**

   - Runs all tests
   - Builds the package (wheel + source distribution)
   - Validates with `twine check`
   - Publishes to PyPI (if `PYPI_API_TOKEN` is set)
   - Publishes to TestPyPI (if `TEST_PYPI_API_TOKEN` is set)

3. **Monitor progress:**

   - https://github.com/johnnichev/selectools/actions

4. **Verify publication:**
   - https://pypi.org/project/selectools/
   - Test: `pip install --upgrade selectools`

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

Examples:

- `0.3.1` → Bug fixes
- `0.4.0` → New features
- `1.0.0` → First stable release

## Troubleshooting

### "Permission denied" error

Make the scripts executable:

```bash
chmod +x scripts/release.py scripts/release.sh
```

### "PYPI_API_TOKEN not found" in GitHub Actions

The publish step will be skipped. Add the token to GitHub secrets:

1. Go to https://github.com/johnnichev/selectools/settings/secrets/actions
2. Add `PYPI_API_TOKEN` with your PyPI token

### Version already exists on PyPI

PyPI doesn't allow re-uploading the same version. Bump to a new version number.

### Tests fail in GitHub Actions

The publish step won't run if tests fail. Fix the tests and push again, or delete the tag and re-release:

```bash
git tag -d v0.3.1
git push origin :refs/tags/v0.3.1
```

## Manual Release (Without Scripts)

If you prefer to do it manually:

```bash
# 1. Update version in pyproject.toml
# Edit: version = "0.3.1"

# 2. Commit
git add pyproject.toml
git commit -m "Bump version to 0.3.1"

# 3. Tag
git tag v0.3.1

# 4. Push
git push origin main
git push origin v0.3.1
```

## Other Scripts

- **`smoke_cli.py`**: Quick smoke tests for different providers
- **`test_memory_with_openai.py`**: Test conversation memory with OpenAI
