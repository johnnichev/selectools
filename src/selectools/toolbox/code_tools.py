"""
Code execution tools for running Python and shell commands.

These tools execute code in isolated subprocesses with configurable timeouts
and output truncation for safety.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 — code execution tool
import tempfile

from ..tools import tool

_MAX_OUTPUT_BYTES = 10 * 1024  # 10 KB

# Shell metacharacters that enable command chaining, subshells, or redirection.
# Commands containing any of these are rejected to prevent shell injection.
_SHELL_BLOCKLIST = [";", "|", "&&", "||", "`", "$(", "{", ">", ">>", "<", "<<", "2>"]


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate text to max_bytes, appending a notice if truncated."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n... (output truncated to 10 KB)"


@tool(description="Execute Python code and return stdout + stderr")
def execute_python(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a subprocess and return stdout + stderr.

    The code is written to a temporary file and executed with ``python3``.
    Output is truncated to 10 KB to avoid overwhelming the context window.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds (default: 30).

    Returns:
        Combined stdout and stderr, or an error message.
    """
    if not code or not code.strip():
        return "Error: No code provided."

    if timeout < 1:
        return "Error: Timeout must be at least 1 second."
    if timeout > 300:
        return "Error: Timeout must not exceed 300 seconds."

    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="selectools_exec_")
        with os.fdopen(fd, "w") as f:
            fd = None  # os.fdopen takes ownership
            f.write(code)

        result = subprocess.run(  # nosec B603 B607
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            if output_parts:
                output_parts.append("--- stderr ---")
            output_parts.append(result.stderr)

        if not output_parts:
            return f"(no output, exit code {result.returncode})"

        combined = "\n".join(output_parts)
        exit_info = f"\n(exit code {result.returncode})" if result.returncode != 0 else ""
        return _truncate(combined) + exit_info

    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {timeout} seconds."
    except FileNotFoundError:
        return "Error: python3 not found on PATH."
    except Exception as e:
        return f"Error executing Python code: {e}"
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@tool(description="Execute a shell command and return output")
def execute_shell(command: str, timeout: int = 30) -> str:
    """
    Execute a shell command and return combined stdout + stderr.

    **WARNING**: This tool executes arbitrary shell commands. It should be
    restricted via ``ToolPolicy`` to prevent misuse in untrusted contexts.
    Commands containing shell metacharacters (pipes, redirects, chaining)
    are rejected to mitigate injection attacks.

    Output is truncated to 10 KB to avoid overwhelming the context window.

    Args:
        command: Shell command to execute (no shell metacharacters allowed).
        timeout: Maximum execution time in seconds (default: 30).

    Returns:
        Combined stdout and stderr, or an error message.
    """
    if not command or not command.strip():
        return "Error: No command provided."

    if timeout < 1:
        return "Error: Timeout must be at least 1 second."
    if timeout > 300:
        return "Error: Timeout must not exceed 300 seconds."

    # Reject commands containing dangerous shell metacharacters
    for meta in _SHELL_BLOCKLIST:
        if meta in command:
            return (
                f"Error: Command contains blocked shell metacharacter {meta!r}. "
                "Shell chaining, pipes, redirects, and subshells are not allowed."
            )

    try:
        result = subprocess.run(  # nosec B602 B603 B607 — intentional shell tool
            command,
            shell=True,  # nosec B602
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            if output_parts:
                output_parts.append("--- stderr ---")
            output_parts.append(result.stderr)

        if not output_parts:
            return f"(no output, exit code {result.returncode})"

        combined = "\n".join(output_parts)
        exit_info = f"\n(exit code {result.returncode})" if result.returncode != 0 else ""
        return _truncate(combined) + exit_info

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {e}"
