#!/usr/bin/env python3
"""
Release script for publishing selectools to PyPI via GitHub Actions.

Usage:
    python scripts/release.py --version 0.3.1
    python scripts/release.py --version 0.4.0 --message "Add new features"
    python scripts/release.py --version 1.0.0 --dry-run
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ReleaseManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.pyproject_path = repo_root / "pyproject.toml"
        self.changelog_path = repo_root / "CHANGELOG.md"
        
    def get_current_version(self) -> str:
        """Extract current version from pyproject.toml"""
        content = self.pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        return match.group(1)
    
    def update_version(self, new_version: str) -> None:
        """Update version in pyproject.toml"""
        content = self.pyproject_path.read_text()
        updated = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version = "{new_version}"',
            content
        )
        self.pyproject_path.write_text(updated)
        print(f"✓ Updated version in pyproject.toml to {new_version}")
    
    def update_changelog(self, version: str, message: str = None) -> None:
        """Add new version entry to CHANGELOG.md"""
        if not self.changelog_path.exists():
            print("⚠ CHANGELOG.md not found, skipping changelog update")
            return
        
        content = self.changelog_path.read_text()
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Create new entry
        new_entry = f"\n## [{version}] - {date}\n\n"
        if message:
            new_entry += f"### Changed\n- {message}\n"
        else:
            new_entry += "### Changed\n- Bug fixes and improvements\n"
        
        # Insert after the first heading (usually "# Changelog")
        lines = content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_index = i + 1
                break
        
        lines.insert(insert_index, new_entry)
        self.changelog_path.write_text('\n'.join(lines))
        print(f"✓ Updated CHANGELOG.md with version {version}")
    
    def run_command(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command"""
        print(f"  Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.repo_root, check=check, capture_output=True, text=True)
    
    def check_git_status(self, dry_run: bool = False) -> None:
        """Ensure working directory is clean"""
        result = self.run_command(["git", "status", "--porcelain"])
        if result.stdout.strip():
            print("⚠ Warning: You have uncommitted changes:")
            print(result.stdout)
            if not dry_run:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
            else:
                print("(Skipping prompt in dry-run mode)")
    
    def check_on_main_branch(self, dry_run: bool = False) -> None:
        """Ensure we're on the main branch"""
        result = self.run_command(["git", "branch", "--show-current"])
        branch = result.stdout.strip()
        if branch != "main":
            print(f"⚠ Warning: You're on branch '{branch}', not 'main'")
            if not dry_run:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
            else:
                print("(Skipping prompt in dry-run mode)")
    
    def validate_version(self, version: str) -> None:
        """Validate semantic version format"""
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            raise ValueError(f"Invalid version format: {version}. Use semantic versioning (e.g., 0.3.1)")
    
    def git_commit(self, version: str, message: str = None) -> None:
        """Commit version changes"""
        files = ["pyproject.toml"]
        if self.changelog_path.exists():
            files.append("CHANGELOG.md")
        
        self.run_command(["git", "add"] + files)
        
        commit_msg = message or f"Bump version to {version}"
        self.run_command(["git", "commit", "-m", commit_msg])
        print(f"✓ Committed changes: {commit_msg}")
    
    def git_tag(self, version: str) -> None:
        """Create git tag"""
        tag = f"v{version}"
        self.run_command(["git", "tag", tag])
        print(f"✓ Created tag: {tag}")
    
    def git_push(self, version: str) -> None:
        """Push commits and tags to origin"""
        tag = f"v{version}"
        
        # Push main branch
        self.run_command(["git", "push", "origin", "main"])
        print("✓ Pushed to origin/main")
        
        # Push tag
        self.run_command(["git", "push", "origin", tag])
        print(f"✓ Pushed tag {tag} to origin")
    
    def release(self, new_version: str, message: str = None, dry_run: bool = False) -> None:
        """Execute the full release process"""
        print("\n🚀 Starting release process...\n")
        
        # Validate
        self.validate_version(new_version)
        current_version = self.get_current_version()
        
        print(f"Current version: {current_version}")
        print(f"New version: {new_version}")
        print()
        
        if current_version == new_version:
            print(f"⚠ Warning: Version {new_version} is the same as current version")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Pre-flight checks
        print("Running pre-flight checks...")
        self.check_on_main_branch(dry_run)
        self.check_git_status(dry_run)
        print()
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made\n")
            print("Would perform the following actions:")
            print(f"  1. Update version in pyproject.toml: {current_version} → {new_version}")
            print(f"  2. Update CHANGELOG.md")
            print(f"  3. Commit changes")
            print(f"  4. Create tag v{new_version}")
            print(f"  5. Push to origin/main")
            print(f"  6. Push tag v{new_version}")
            print("\nGitHub Actions would then:")
            print("  - Run tests")
            print("  - Build package")
            print("  - Publish to PyPI")
            return
        
        # Confirm
        print("This will:")
        print(f"  1. Update version to {new_version}")
        print(f"  2. Update CHANGELOG.md")
        print(f"  3. Commit and push changes")
        print(f"  4. Create and push tag v{new_version}")
        print(f"  5. Trigger GitHub Actions to publish to PyPI")
        print()
        response = input("Proceed? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        
        print()
        
        # Execute release steps
        try:
            self.update_version(new_version)
            self.update_changelog(new_version, message)
            self.git_commit(new_version, message)
            self.git_tag(new_version)
            self.git_push(new_version)
            
            print("\n✅ Release process completed successfully!\n")
            print(f"GitHub Actions will now build and publish version {new_version} to PyPI.")
            print(f"Monitor progress at: https://github.com/johnnichev/selectools/actions")
            print(f"\nAfter publication, the package will be available at:")
            print(f"  https://pypi.org/project/selectools/{new_version}/")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: Command failed: {e.cmd}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Release selectools to PyPI via GitHub Actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/release.py --version 0.3.1
  python scripts/release.py --version 0.4.0 --message "Add streaming support"
  python scripts/release.py --version 1.0.0 --dry-run

Prerequisites:
  1. Set up PYPI_API_TOKEN in GitHub repository secrets
  2. Ensure you're on the main branch
  3. Ensure working directory is clean (commit any changes first)
        """
    )
    
    parser.add_argument(
        "--version",
        required=True,
        help="New version number (semantic versioning, e.g., 0.3.1)"
    )
    
    parser.add_argument(
        "--message",
        help="Optional commit/changelog message describing changes"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    
    args = parser.parse_args()
    
    # Find repository root
    repo_root = Path(__file__).parent.parent
    
    # Execute release
    manager = ReleaseManager(repo_root)
    manager.release(args.version, args.message, args.dry_run)


if __name__ == "__main__":
    main()

