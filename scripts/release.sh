#!/bin/bash
# Quick release script for selectools
# Usage: ./scripts/release.sh 0.3.1 "Optional commit message"

set -e

VERSION=$1
MESSAGE=$2

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh <version> [message]"
    echo "Example: ./scripts/release.sh 0.3.1 'Add new features'"
    exit 1
fi

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version format. Use semantic versioning (e.g., 0.3.1)"
    exit 1
fi

echo "🚀 Releasing version $VERSION"
echo ""

# Check we're on main
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're on branch '$BRANCH', not 'main'"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status --short
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "This will:"
echo "  1. Update version to $VERSION in pyproject.toml"
echo "  2. Commit changes"
echo "  3. Create tag v$VERSION"
echo "  4. Push to origin"
echo "  5. Trigger GitHub Actions to publish to PyPI"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Update version in pyproject.toml
echo "📝 Updating version in pyproject.toml..."
sed -i.bak "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Commit
echo "💾 Committing changes..."
git add pyproject.toml
if [ -n "$MESSAGE" ]; then
    git commit -m "$MESSAGE"
else
    git commit -m "Bump version to $VERSION"
fi

# Tag
echo "🏷️  Creating tag v$VERSION..."
git tag "v$VERSION"

# Push
echo "⬆️  Pushing to origin..."
git push origin main
git push origin "v$VERSION"

echo ""
echo "✅ Release process completed!"
echo ""
echo "GitHub Actions will now build and publish version $VERSION to PyPI."
echo "Monitor progress at: https://github.com/johnnichev/selectools/actions"
echo ""
echo "After publication, install with: pip install --upgrade selectools"

