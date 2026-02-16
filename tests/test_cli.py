"""
Comprehensive tests for CLI (cli.py).

Tests cover:
- build_parser() argument parsing
- _build_provider() factory function
- _default_tools() tool registration
- list_tools() output
- run_agent() with dry-run mode
- main() dispatch
- Unknown provider error
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from selectools.cli import _build_provider, _default_tools, build_parser, list_tools, main
from selectools.providers.openai_provider import OpenAIProvider
from selectools.providers.stubs import LocalProvider


class TestBuildParser:
    """Tests for build_parser() argument parsing."""

    def test_list_tools_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["list-tools"])
        assert args.command == "list-tools"
        assert args.func == "list"

    def test_run_command_required_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--prompt", "Hello world"])
        assert args.command == "run"
        assert args.func == "run"
        assert args.prompt == "Hello world"

    def test_run_command_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--prompt", "test"])
        assert args.provider == "openai"
        assert args.model == "gpt-4o"
        assert args.temperature == 0.0
        assert args.max_tokens == 500
        assert args.max_iterations == 4
        assert args.verbose is False
        assert args.stream is False
        assert args.dry_run is False
        assert args.timeout == 30.0
        assert args.retries == 2
        assert args.backoff == 1.0

    def test_run_command_custom_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--prompt",
                "test",
                "--provider",
                "anthropic",
                "--model",
                "claude-3",
                "--temperature",
                "0.7",
                "--max-tokens",
                "1000",
                "--max-iterations",
                "8",
                "--verbose",
                "--stream",
                "--dry-run",
                "--timeout",
                "60.0",
                "--retries",
                "5",
                "--backoff",
                "2.5",
            ]
        )
        assert args.provider == "anthropic"
        assert args.model == "claude-3"
        assert args.temperature == 0.7
        assert args.max_tokens == 1000
        assert args.max_iterations == 8
        assert args.verbose is True
        assert args.stream is True
        assert args.dry_run is True
        assert args.timeout == 60.0
        assert args.retries == 5
        assert args.backoff == 2.5

    def test_run_command_image_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--prompt", "describe", "--image", "photo.png"])
        assert args.image == "photo.png"

    def test_run_command_tool_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--prompt", "test", "--tool", "echo"])
        assert args.tool == "echo"

    def test_chat_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"
        assert args.func == "chat"

    def test_chat_command_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat"])
        assert args.provider == "openai"
        assert args.model == "gpt-4o"
        assert args.max_iterations == 6

    def test_no_command_raises(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestBuildProvider:
    """Tests for _build_provider() factory."""

    def test_local_provider(self) -> None:
        provider = _build_provider("local", "unused")
        assert isinstance(provider, LocalProvider)

    def test_openai_provider(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = _build_provider("openai", "gpt-4o")
            assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            _build_provider("nonexistent", "model")


class TestDefaultTools:
    """Tests for _default_tools() registration."""

    def test_returns_dict(self) -> None:
        tools = _default_tools()
        assert isinstance(tools, dict)

    def test_has_echo_tool(self) -> None:
        tools = _default_tools()
        assert "echo" in tools

    def test_echo_tool_works(self) -> None:
        tools = _default_tools()
        echo = tools["echo"]
        result = echo.execute({"text": "hello"})
        assert "hello" in result


class TestListTools:
    """Tests for list_tools() output."""

    def test_lists_tools(self, capsys: pytest.CaptureFixture[str]) -> None:
        tools = _default_tools()
        list_tools(tools)
        captured = capsys.readouterr()
        assert "echo" in captured.out
        assert "Echo back" in captured.out


class TestMain:
    """Tests for main() dispatch function."""

    def test_list_tools_dispatch(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["list-tools"])
        captured = capsys.readouterr()
        assert "echo" in captured.out

    def test_run_dry_run_with_local_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "run",
                "--provider",
                "local",
                "--prompt",
                "Hello",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert "TOOL_CALL" in captured.out or "tool" in captured.out.lower()

    def test_run_with_local_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "run",
                "--provider",
                "local",
                "--prompt",
                "Hello there",
                "--max-iterations",
                "1",
            ]
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_run_with_specific_tool(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "run",
                "--provider",
                "local",
                "--prompt",
                "Echo test",
                "--tool",
                "echo",
                "--max-iterations",
                "1",
            ]
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_run_with_stream(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "run",
                "--provider",
                "local",
                "--prompt",
                "Hello",
                "--stream",
                "--max-iterations",
                "1",
            ]
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0
