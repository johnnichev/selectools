"""
Comprehensive tests for env.py (load_env_if_present, load_default_env).

Tests cover:
- Loading from .env file
- Skipping comments and blank lines
- Not overwriting existing environment variables
- Handling missing files gracefully
- Handling malformed lines
- load_default_env() candidate resolution
- Stopping after first valid file
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from selectools.env import load_default_env, load_env_if_present


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """Create a temporary .env file."""
    env = tmp_path / ".env"
    env.write_text("TEST_VAR_A=hello\n" "TEST_VAR_B=world\n")
    return env


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove test env vars after each test."""
    yield
    for key in [
        "TEST_VAR_A",
        "TEST_VAR_B",
        "TEST_VAR_C",
        "TEST_VAR_D",
        "TEST_EXISTING",
        "MY_KEY",
        "COMMENTED_OUT",
    ]:
        os.environ.pop(key, None)


class TestLoadEnvIfPresent:
    """Tests for load_env_if_present()."""

    def test_loads_from_valid_file(self, env_file: Path) -> None:
        load_env_if_present([env_file])

        assert os.environ.get("TEST_VAR_A") == "hello"
        assert os.environ.get("TEST_VAR_B") == "world"

    def test_skips_nonexistent_files(self) -> None:
        fake = Path("/nonexistent/path/.env")
        load_env_if_present([fake])

    def test_skips_directories(self, tmp_path: Path) -> None:
        load_env_if_present([tmp_path])

    def test_skips_comments(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("# This is a comment\n" "TEST_VAR_A=value\n" "# COMMENTED_OUT=nope\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == "value"
        assert os.environ.get("COMMENTED_OUT") is None

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("\n" "TEST_VAR_A=a\n" "\n" "TEST_VAR_B=b\n" "\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == "a"
        assert os.environ.get("TEST_VAR_B") == "b"

    def test_skips_lines_without_equals(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_A=valid\n" "this_has_no_equals\n" "TEST_VAR_B=also_valid\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == "valid"
        assert os.environ.get("TEST_VAR_B") == "also_valid"

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        os.environ["TEST_EXISTING"] = "original"

        env = tmp_path / ".env"
        env.write_text("TEST_EXISTING=overwritten\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_EXISTING") == "original"

    def test_value_with_equals_sign(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("MY_KEY=value=with=equals\n")
        load_env_if_present([env])

        assert os.environ.get("MY_KEY") == "value=with=equals"

    def test_stops_after_first_valid_file(self, tmp_path: Path) -> None:
        first = tmp_path / "first.env"
        first.write_text("TEST_VAR_A=from_first\n")

        second = tmp_path / "second.env"
        second.write_text("TEST_VAR_A=from_second\nTEST_VAR_C=only_in_second\n")

        load_env_if_present([first, second])

        assert os.environ.get("TEST_VAR_A") == "from_first"
        assert os.environ.get("TEST_VAR_C") is None

    def test_empty_file(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("")
        load_env_if_present([env])

    def test_whitespace_stripped(self, tmp_path: Path) -> None:
        """Line is stripped, so leading/trailing whitespace on the whole line is removed."""
        env = tmp_path / ".env"
        env.write_text("  TEST_VAR_A=value  \n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == "value"

    def test_empty_value(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_A=\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == ""

    def test_empty_key_skipped(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("=no_key\nTEST_VAR_A=valid\n")
        load_env_if_present([env])

        assert os.environ.get("TEST_VAR_A") == "valid"

    def test_empty_candidates_list(self) -> None:
        load_env_if_present([])

    def test_skips_unreadable_file_gracefully(self, tmp_path: Path) -> None:
        """If a file can't be read, it should be silently skipped."""
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_A=value\n")
        env.chmod(0o000)

        try:
            load_env_if_present([env])
        finally:
            env.chmod(0o644)


class TestLoadDefaultEnv:
    """Tests for load_default_env()."""

    def test_callable_without_error(self) -> None:
        """load_default_env should not raise even if no .env exists."""
        load_default_env()

    def test_loads_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_D=from_cwd\n")

        load_default_env()

        assert os.environ.get("TEST_VAR_D") == "from_cwd"
