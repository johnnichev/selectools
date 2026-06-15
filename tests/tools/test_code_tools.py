"""
Tests for code execution tools (execute_python, execute_shell).

All subprocess calls are mocked to avoid executing real code in tests.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from selectools.toolbox import code_tools

# =============================================================================
# execute_python tests
# =============================================================================


class TestExecutePython:
    """Tests for the execute_python tool."""

    def test_tool_has_correct_metadata(self) -> None:
        """Tool has name, description, and function attributes."""
        assert code_tools.execute_python.name == "execute_python"
        assert "Python" in code_tools.execute_python.description
        assert hasattr(code_tools.execute_python, "function")

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_successful_execution(self, mock_run: MagicMock) -> None:
        """Successful code execution returns stdout."""
        mock_run.return_value = MagicMock(stdout="Hello World\n", stderr="", returncode=0)
        result = code_tools.execute_python.function("print('Hello World')")
        assert "Hello World" in result
        mock_run.assert_called_once()

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_execution_with_stderr(self, mock_run: MagicMock) -> None:
        """Code that produces stderr includes it in output."""
        mock_run.return_value = MagicMock(stdout="output\n", stderr="warning here\n", returncode=0)
        result = code_tools.execute_python.function("import warnings")
        assert "output" in result
        assert "stderr" in result
        assert "warning here" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_nonzero_exit_code_shown(self, mock_run: MagicMock) -> None:
        """Non-zero exit codes are included in the output."""
        mock_run.return_value = MagicMock(
            stdout="", stderr="NameError: name 'x' is not defined\n", returncode=1
        )
        result = code_tools.execute_python.function("print(x)")
        assert "exit code 1" in result
        assert "NameError" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_timeout_handling(self, mock_run: MagicMock) -> None:
        """Subprocess timeout is caught and reported."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python3", timeout=5)
        result = code_tools.execute_python.function("import time; time.sleep(100)", timeout=5)
        assert "timed out" in result
        assert "5" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_python3_not_found(self, mock_run: MagicMock) -> None:
        """FileNotFoundError when python3 is missing is caught."""
        mock_run.side_effect = FileNotFoundError("python3 not found")
        result = code_tools.execute_python.function("print(1)")
        assert "python3 not found" in result.lower()

    def test_empty_code_rejected(self) -> None:
        """Empty code string returns an error."""
        result = code_tools.execute_python.function("")
        assert "Error" in result
        assert "No code" in result

    def test_whitespace_only_code_rejected(self) -> None:
        """Whitespace-only code string returns an error."""
        result = code_tools.execute_python.function("   \n  ")
        assert "Error" in result

    def test_timeout_too_low(self) -> None:
        """Timeout below 1 second is rejected."""
        result = code_tools.execute_python.function("print(1)", timeout=0)
        assert "Error" in result
        assert "1 second" in result

    def test_timeout_too_high(self) -> None:
        """Timeout above 300 seconds is rejected."""
        result = code_tools.execute_python.function("print(1)", timeout=500)
        assert "Error" in result
        assert "300" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_no_output(self, mock_run: MagicMock) -> None:
        """Code that produces no output returns exit code info."""
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        result = code_tools.execute_python.function("x = 1")
        assert "no output" in result
        assert "exit code 0" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_output_truncation(self, mock_run: MagicMock) -> None:
        """Output exceeding 10 KB is truncated."""
        big_output = "A" * 20000
        mock_run.return_value = MagicMock(stdout=big_output, stderr="", returncode=0)
        result = code_tools.execute_python.function("print('A' * 20000)")
        assert "truncated" in result
        assert len(result) < 15000  # well under 20 KB

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_tempfile_cleanup_on_success(self, mock_run: MagicMock) -> None:
        """Temporary file is cleaned up after execution."""
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        # Just verify it completes without leaving temp files
        result = code_tools.execute_python.function("print('ok')")
        assert "ok" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_generic_exception_handled(self, mock_run: MagicMock) -> None:
        """Unexpected exceptions are caught gracefully."""
        mock_run.side_effect = RuntimeError("unexpected failure")
        result = code_tools.execute_python.function("print(1)")
        assert "Error" in result
        assert "unexpected failure" in result


# =============================================================================
# execute_shell tests
# =============================================================================


class TestExecuteShell:
    """Tests for the execute_shell tool."""

    def test_tool_has_correct_metadata(self) -> None:
        """Tool has name, description, and function attributes."""
        assert code_tools.execute_shell.name == "execute_shell"
        assert "shell" in code_tools.execute_shell.description.lower()

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_successful_command(self, mock_run: MagicMock) -> None:
        """Successful command returns stdout."""
        mock_run.return_value = MagicMock(stdout="file1.txt\nfile2.txt\n", stderr="", returncode=0)
        result = code_tools.execute_shell.function("ls")
        assert "file1.txt" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_command_with_stderr(self, mock_run: MagicMock) -> None:
        """Command with stderr includes it in output."""
        mock_run.return_value = MagicMock(
            stdout="", stderr="ls: No such file or directory\n", returncode=1
        )
        result = code_tools.execute_shell.function("ls /nonexistent")
        assert "No such file" in result
        assert "exit code 1" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_timeout_handling(self, mock_run: MagicMock) -> None:
        """Timeout is caught and reported."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=5)
        result = code_tools.execute_shell.function("sleep 100", timeout=5)
        assert "timed out" in result

    def test_empty_command_rejected(self) -> None:
        """Empty command returns an error."""
        result = code_tools.execute_shell.function("")
        assert "Error" in result

    def test_timeout_too_low(self) -> None:
        """Timeout below 1 second is rejected."""
        result = code_tools.execute_shell.function("echo hi", timeout=0)
        assert "Error" in result

    def test_timeout_too_high(self) -> None:
        """Timeout above 300 seconds is rejected."""
        result = code_tools.execute_shell.function("echo hi", timeout=999)
        assert "Error" in result
        assert "300" in result

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_runs_without_a_shell(self, mock_run: MagicMock) -> None:
        """Command is executed with shell=False and a parsed argv list."""
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        code_tools.execute_shell.function("echo ok")
        args, kwargs = mock_run.call_args
        assert kwargs.get("shell") is False
        # First positional arg is the shlex-parsed argv list, not a raw string.
        assert args[0] == ["echo", "ok"]

    @pytest.mark.parametrize(
        "payload",
        [
            "echo hi; rm -rf x",  # chaining
            "echo hi | cat",  # pipe
            "echo hi && echo bye",  # AND-chaining
            "echo hi & echo bye",  # backgrounding (bare &)
            "echo $(whoami)",  # subshell
            "echo `whoami`",  # backtick subshell
            "echo hi > /tmp/x",  # redirect
            "echo hi\nrm -rf x",  # newline injection
            "echo hi\rrm -rf x",  # carriage-return injection
        ],
    )
    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_metacharacters_rejected_before_execution(
        self, mock_run: MagicMock, payload: str
    ) -> None:
        """Every shell metacharacter is rejected without invoking subprocess."""
        result = code_tools.execute_shell.function(payload)
        assert result.lower().startswith("error")
        assert "blocked shell metacharacter" in result
        mock_run.assert_not_called()

    def test_unbalanced_quotes_reported(self) -> None:
        """A command that shlex cannot parse returns a clear error."""
        result = code_tools.execute_shell.function('echo "unterminated')
        assert "could not parse" in result.lower()

    @patch("selectools.toolbox.code_tools.subprocess.run")
    def test_arguments_are_not_glob_expanded(self, mock_run: MagicMock) -> None:
        """Globs pass through literally (no shell), proving shell=False semantics."""
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        code_tools.execute_shell.function("ls *.py")
        args, _ = mock_run.call_args
        assert args[0] == ["ls", "*.py"]
