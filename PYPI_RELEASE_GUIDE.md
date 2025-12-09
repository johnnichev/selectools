# PyPI Release Guide - Adding Release Notes

## How to Add Release Notes on PyPI

PyPI displays release notes through **two mechanisms**:

### 1. Long Description (from README.md)

PyPI automatically shows your `README.md` as the project description on the main package page. This is configured in `pyproject.toml`:

```toml
[project]
readme = "README.md"
```

**✅ Already configured** - Your README will appear on https://pypi.org/project/selectools/

### 2. GitHub Releases (Recommended for Version Notes)

PyPI links to your GitHub repository, and you should create **GitHub Releases** for each version. Here's how:

## Step-by-Step Release Process

### Step 1: Commit Your Changes

```bash
git add -A
git commit -m "Release v0.7.0: Model Registry System

- Add model registry with 120 models
- IDE autocomplete for all models
- Rich metadata (pricing, context windows)
- Backward compatible migration
- Updated all examples and documentation
"
```

### Step 2: Create and Push Git Tag

```bash
git tag -a v0.7.0 -m "v0.7.0: Model Registry System"
git push origin main
git push origin v0.7.0
```

### Step 3: Create GitHub Release

1. **Go to GitHub**:

   - Navigate to https://github.com/johnnichev/selectools/releases
   - Click "Draft a new release"

2. **Configure Release**:

   - **Tag**: Select `v0.7.0` (the tag you just pushed)
   - **Release title**: `v0.7.0 - Model Registry System`
   - **Description**: Copy the contents from `RELEASE_NOTES_v0.7.0.md`

3. **Publish**:
   - Check "Set as the latest release"
   - Click "Publish release"

### Step 4: Build and Upload to PyPI

Your GitHub Actions workflow (`.github/workflows/publish.yml`) should automatically:

1. Trigger on the `v0.7.0` tag push
2. Run tests
3. Build the package
4. Publish to PyPI

**Or manually**:

```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/selectools-0.7.0*

# OR use the GitHub Actions (recommended)
# It triggers automatically on git tag push
```

### Step 5: Verify on PyPI

After a few minutes, check:

- **Main page**: https://pypi.org/project/selectools/
- **Release history**: https://pypi.org/project/selectools/#history
- **Version page**: https://pypi.org/project/selectools/0.7.0/

## What PyPI Shows

### Main Package Page

- Shows your `README.md` content
- Project links (GitHub, documentation)
- Latest version number
- Installation command

### Release History Tab

- Lists all versions with upload dates
- Links to each version's page
- Shows which is the latest

### Individual Version Pages

- Installation command for that specific version
- Release date
- File hashes
- Links to source code (GitHub tag)

## Best Practices

### 1. Use Semantic Versioning

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes only

### 2. Keep CHANGELOG.md Updated

Add entries to `CHANGELOG.md` for every release. PyPI doesn't directly show this, but:

- GitHub can render it
- Users can find it in your repo
- Many tools parse it

### 3. Write Good Release Notes

Include in GitHub Releases:

- ✅ What's new (features)
- ✅ What changed (modifications)
- ✅ What's fixed (bugs)
- ✅ Migration guide (if needed)
- ✅ Breaking changes (if any)
- ✅ Examples and use cases
- ✅ Links to documentation

### 4. Link GitHub and PyPI

In `pyproject.toml`, ensure these are set:

```toml
[project.urls]
Homepage = "https://github.com/johnnichev/selectools"
Repository = "https://github.com/johnnichev/selectools"
"Bug Tracker" = "https://github.com/johnnichev/selectools/issues"
Changelog = "https://github.com/johnnichev/selectools/blob/main/CHANGELOG.md"
```

PyPI will display these links on the package page.

### 5. Create Release Before PyPI Upload

This order works best:

1. Commit code
2. Create git tag
3. Push tag to GitHub
4. Create GitHub Release (with notes)
5. Let CI/CD upload to PyPI
6. GitHub Release will be linked

## Automation with GitHub Actions

Your `.github/workflows/publish.yml` should look like:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

## Quick Reference

### Complete Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with new version
- [ ] Update `ROADMAP.md` to mark features complete
- [ ] Create `RELEASE_NOTES_vX.X.X.md` file
- [ ] Run all tests (`pytest`)
- [ ] Commit changes
- [ ] Create git tag: `git tag -a vX.X.X -m "Version X.X.X"`
- [ ] Push main: `git push origin main`
- [ ] Push tag: `git push origin vX.X.X`
- [ ] Create GitHub Release with notes
- [ ] Wait for CI/CD to publish to PyPI
- [ ] Verify on https://pypi.org/project/selectools/
- [ ] Test install: `pip install --upgrade selectools`

## Viewing Release Notes

Users can find your release notes:

1. **PyPI main page**: Shows README.md
2. **GitHub Releases**: https://github.com/johnnichev/selectools/releases
3. **CHANGELOG.md**: https://github.com/johnnichev/selectools/blob/main/CHANGELOG.md
4. **Individual release**: https://github.com/johnnichev/selectools/releases/tag/v0.7.0

## Example: v0.7.0 Release

```bash
# 1. Ensure changes committed
git status

# 2. Create annotated tag
git tag -a v0.7.0 -m "v0.7.0: Model Registry System - IDE autocomplete for 120 models"

# 3. Push everything
git push origin main
git push origin v0.7.0

# 4. Go to GitHub → Releases → Draft new release
#    - Tag: v0.7.0
#    - Title: v0.7.0 - Model Registry System
#    - Body: Copy from RELEASE_NOTES_v0.7.0.md
#    - Publish

# 5. CI/CD will automatically:
#    - Build package
#    - Run tests
#    - Upload to PyPI

# 6. Verify
pip install --upgrade selectools
python -c "import selectools; print(selectools.__version__)"
```

## Tips

- **Preview**: Use "Save draft" on GitHub to preview release notes before publishing
- **Edit**: You can edit GitHub Releases after publishing
- **Delete**: If needed, you can delete a release (but not the git tag without extra steps)
- **Assets**: Attach additional files (PDFs, binaries) to GitHub Releases
- **Auto-generated notes**: GitHub can auto-generate notes from commits (but manual is better)

## Support

- PyPI Help: https://pypi.org/help/
- GitHub Releases Docs: https://docs.github.com/en/repositories/releasing-projects-on-github
- Twine Documentation: https://twine.readthedocs.io/
