"""End-to-end tests for code execution tools with real subprocesses.

Unlike ``test_code_tools.py`` (which mocks ``subprocess.run``), these tests
actually spawn ``python3`` and ``sh`` processes and assert on their real
output. They're the only place we verify that:

- The subprocess invocation string is well-formed
- Timeout handling works against a real blocking process
- The shell metacharacter blocklist matches what a real shell would execute
- Output truncation kicks in at the expected byte count

Run with:

    pytest tests/tools/test_e2e_code_tools.py --run-e2e -v
"""

from __future__ import annotations

import pytest

from selectools.toolbox import code_tools

pytestmark = pytest.mark.e2e


class TestExecutePythonReal:
    def test_hello_world_roundtrip(self) -> None:
        """Real python3 subprocess runs and stdout is captured."""
        result = code_tools.execute_python.function("print('hello e2e')")
        assert "hello e2e" in result

    def test_exception_shown_in_stderr_section(self) -> None:
        """Real python3 traceback lands in the stderr section of the output."""
        result = code_tools.execute_python.function("raise ValueError('boom')")
        assert "ValueError" in result
        assert "boom" in result
        assert "exit code" in result.lower()

    def test_real_timeout_expiry(self) -> None:
        """A real long-running process is killed after the timeout."""
        result = code_tools.execute_python.function("import time; time.sleep(10)", timeout=1)
        assert "timed out" in result.lower()

    def test_stdout_stderr_both_captured(self) -> None:
        """stdout and stderr are both captured from the real subprocess."""
        code = (
            "import sys\n" "sys.stdout.write('on stdout\\n')\n" "sys.stderr.write('on stderr\\n')\n"
        )
        result = code_tools.execute_python.function(code)
        assert "on stdout" in result
        assert "on stderr" in result

    def test_output_truncation_on_large_output(self) -> None:
        """Very large stdout is truncated (real process emits > 10KB)."""
        code = "print('x' * 20000)"  # 20KB of 'x'
        result = code_tools.execute_python.function(code)
        # Real output was 20KB; truncated to 10KB with a notice
        assert "truncated" in result.lower()


class TestExecuteShellReal:
    def test_echo_real_shell(self) -> None:
        """A real shell executes echo and returns stdout."""
        result = code_tools.execute_shell.function("echo hello-e2e")
        assert "hello-e2e" in result

    def test_nonexistent_command_returns_error(self) -> None:
        """A real shell rejects a nonexistent binary with non-zero exit."""
        result = code_tools.execute_shell.function("this-binary-does-not-exist-42")
        # Should include some indication of failure (stderr or exit code)
        assert "exit code" in result.lower() or "not found" in result.lower()

    def test_pipe_metacharacter_rejected_before_execution(self) -> None:
        """Shell metacharacters are rejected before subprocess is called."""
        result = code_tools.execute_shell.function("echo hi | cat")
        # Blocklist rejects the command; should not contain the piped output
        assert "error" in result.lower() or "reject" in result.lower()
