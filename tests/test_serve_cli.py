"""Tests for serve/cli.py — CLI arg parsing, _cmd_serve, _cmd_doctor, _auto_provider."""

from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from selectools.serve.cli import _auto_provider, _cmd_doctor, _cmd_serve, _serve_with_reload, main

# ---------------------------------------------------------------------------
# main() — argument parsing
# ---------------------------------------------------------------------------


class TestMainParsing:
    def test_no_args_prints_help(self, capsys):
        with patch("sys.argv", ["selectools"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "selectools" in captured.out.lower()

    def test_serve_subcommand_recognized(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            mock_serve.assert_called_once()

    def test_doctor_subcommand_recognized(self):
        with (
            patch("sys.argv", ["selectools", "doctor"]),
            patch("selectools.serve.cli._cmd_doctor") as mock_doctor,
        ):
            main()
            mock_doctor.assert_called_once()

    def test_serve_with_config(self):
        with (
            patch("sys.argv", ["selectools", "serve", "agent.yaml", "--port", "9000"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.config == "agent.yaml"
            assert args.port == 9000

    def test_serve_default_port(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.port == 8000

    def test_serve_default_host(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.host == "0.0.0.0"

    def test_serve_no_playground_flag(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder", "--no-playground"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.no_playground is True

    def test_serve_auth_token(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder", "--auth-token", "secret"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.auth_token == "secret"

    def test_serve_reload_flag(self):
        with (
            patch("sys.argv", ["selectools", "serve", "agent.yaml", "--reload"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.reload is True

    def test_serve_builder_flag(self):
        with (
            patch("sys.argv", ["selectools", "serve", "--builder"]),
            patch("selectools.serve.cli._cmd_serve") as mock_serve,
        ):
            main()
            args = mock_serve.call_args[0][0]
            assert args.builder is True


# ---------------------------------------------------------------------------
# _cmd_serve
# ---------------------------------------------------------------------------


class TestCmdServe:
    def test_no_config_no_builder_exits(self):
        args = argparse.Namespace(
            config=None,
            builder=False,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with pytest.raises(SystemExit):
            _cmd_serve(args)

    def test_builder_only_mode(self):
        args = argparse.Namespace(
            config=None,
            builder=True,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with patch("selectools.serve.cli._serve_builder") as mock_builder:
            _cmd_serve(args)
            mock_builder.assert_called_once_with("0.0.0.0", 8000, None)

    def test_reload_mode_calls_serve_with_reload(self):
        args = argparse.Namespace(
            config="agent.yaml",
            builder=False,
            reload=True,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with patch("selectools.serve.cli._serve_with_reload") as mock_reload:
            _cmd_serve(args)
            mock_reload.assert_called_once()

    def test_config_file_loads_yaml(self, tmp_path):
        config_file = tmp_path / "agent.yaml"
        config_file.write_text("provider: local\nmodel: test\n")
        args = argparse.Namespace(
            config=str(config_file),
            builder=False,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with patch("selectools.serve.app.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            _cmd_serve(args)
            mock_create.assert_called_once()
            mock_app.serve.assert_called_once()

    def test_config_file_not_found(self, capsys):
        args = argparse.Namespace(
            config="/nonexistent/path/agent.yaml",
            builder=False,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with pytest.raises(SystemExit):
            _cmd_serve(args)

    def test_template_name_loads_template(self):
        args = argparse.Namespace(
            config="customer_support",
            builder=False,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with (
            patch("selectools.serve.cli._auto_provider") as mock_prov,
            patch("selectools.serve.app.create_app") as mock_create,
        ):
            mock_prov.return_value = MagicMock()
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            _cmd_serve(args)
            mock_create.assert_called_once()

    def test_template_no_api_key_exits(self):
        args = argparse.Namespace(
            config="customer_support",
            builder=False,
            reload=False,
            host="0.0.0.0",
            port=8000,
            no_playground=False,
            auth_token=None,
        )
        with patch("selectools.serve.cli._auto_provider", return_value=None):
            with pytest.raises(SystemExit):
                _cmd_serve(args)


# ---------------------------------------------------------------------------
# _cmd_doctor
# ---------------------------------------------------------------------------


class TestCmdDoctor:
    def test_doctor_output(self, capsys):
        with patch.dict(os.environ, {}, clear=True):
            # Remove all API keys
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ]:
                os.environ.pop(key, None)
            _cmd_doctor()

        captured = capsys.readouterr()
        assert "Selectools Doctor" in captured.out
        assert "Version:" in captured.out
        assert "Python:" in captured.out
        assert "API Keys:" in captured.out
        assert "Optional Dependencies:" in captured.out
        assert "Provider Connectivity:" in captured.out
        assert "Diagnosis complete." in captured.out

    def test_doctor_shows_missing_keys(self, capsys):
        env = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "GEMINI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            # Clear all API keys
            for key in env:
                os.environ.pop(key, None)
            _cmd_doctor()

        captured = capsys.readouterr()
        assert "MISSING" in captured.out

    def test_doctor_shows_set_keys(self, capsys):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test123"},
            clear=False,
        ):
            # Mock the provider creation to avoid real API calls
            with patch("selectools.providers.openai_provider.OpenAIProvider") as mock_prov:
                mock_prov.return_value = MagicMock()
                _cmd_doctor()

        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_doctor_provider_connectivity_skipped(self, capsys):
        with patch.dict(os.environ, {}, clear=False):
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ]:
                os.environ.pop(key, None)
            _cmd_doctor()

        captured = capsys.readouterr()
        assert "skipped" in captured.out

    def test_doctor_optional_deps(self, capsys):
        _cmd_doctor()
        captured = capsys.readouterr()
        # Should mention at least some deps
        assert "YAML config loading" in captured.out or "yaml" in captured.out.lower()


# ---------------------------------------------------------------------------
# _auto_provider
# ---------------------------------------------------------------------------


class TestAutoProvider:
    def test_no_keys_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ]:
                os.environ.pop(key, None)
            result = _auto_provider()
        assert result is None

    def test_openai_key_returns_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            with patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = _auto_provider()
            assert result is not None

    def test_anthropic_key_returns_anthropic(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            try:
                with patch("selectools.providers.anthropic_provider.AnthropicProvider") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    result = _auto_provider()
                assert result is not None
            finally:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_gemini_key_returns_gemini(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ["GOOGLE_API_KEY"] = "ai-test"
            try:
                with patch("selectools.providers.gemini_provider.GeminiProvider") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    result = _auto_provider()
                assert result is not None
            finally:
                os.environ.pop("GOOGLE_API_KEY", None)

    def test_gemini_api_key_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ["GEMINI_API_KEY"] = "gem-test"
            try:
                with patch("selectools.providers.gemini_provider.GeminiProvider") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    result = _auto_provider()
                assert result is not None
            finally:
                os.environ.pop("GEMINI_API_KEY", None)


# ---------------------------------------------------------------------------
# _serve_with_reload
# ---------------------------------------------------------------------------


class TestServeWithReload:
    def test_no_watchfiles_exits(self):
        with patch.dict(sys.modules, {"watchfiles": None}):
            with patch("builtins.__import__", side_effect=ImportError("no watchfiles")):
                with pytest.raises((SystemExit, ImportError)):
                    _serve_with_reload()


# ---------------------------------------------------------------------------
# _serve_builder
# ---------------------------------------------------------------------------


class TestServeBuilder:
    @pytest.mark.skipif(
        not pytest.importorskip("starlette", reason="starlette not installed"),
        reason="starlette not installed",
    )
    def test_builder_with_uvicorn(self):
        from selectools.serve.cli import _serve_builder

        mock_uvicorn = MagicMock()
        mock_create = MagicMock()
        mock_create.return_value = MagicMock()

        with (
            patch.dict(sys.modules, {"uvicorn": mock_uvicorn}),
            patch(
                "selectools.serve._starlette_app.create_builder_app",
                mock_create,
            ),
        ):
            _serve_builder("0.0.0.0", 8000, None)
            mock_uvicorn.run.assert_called_once()

    def test_builder_fallback_to_stdlib(self):
        from selectools.serve.cli import _serve_builder

        with patch("selectools.serve.app.BuilderServer") as mock_builder_cls:
            mock_srv = MagicMock()
            mock_builder_cls.return_value = mock_srv
            # Remove uvicorn from sys.modules to force ImportError
            saved = sys.modules.pop("uvicorn", None)
            try:
                with patch.dict(sys.modules, {"uvicorn": None}):
                    _serve_builder("0.0.0.0", 8000, None)
                mock_builder_cls.assert_called_once_with(host="0.0.0.0", port=8000, auth_token=None)
                mock_srv.serve.assert_called_once()
            finally:
                if saved is not None:
                    sys.modules["uvicorn"] = saved
