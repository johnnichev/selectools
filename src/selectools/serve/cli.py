"""
CLI for selectools serve and selectools doctor.

Usage::

    selectools serve agent.yaml --port 8000 --playground
    selectools doctor
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def main() -> None:
    """Entry point for the selectools CLI."""
    parser = argparse.ArgumentParser(
        prog="selectools",
        description="Selectools CLI — serve agents and diagnose configuration.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Serve an agent as HTTP API")
    serve_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file or template name (optional with --builder)",
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-restart on source file changes (development). Requires: pip install watchfiles",
    )
    serve_parser.add_argument("--no-playground", action="store_true", help="Disable playground UI")
    serve_parser.add_argument(
        "--builder", action="store_true", help="Enable visual agent builder UI at /builder"
    )
    serve_parser.add_argument(
        "--auth-token",
        default=None,
        dest="auth_token",
        help=(
            "Protect the builder/playground with a token. "
            "Also reads BUILDER_AUTH_TOKEN env var or ~/.selectools/auth_token file."
        ),
    )

    # doctor
    subparsers.add_parser("doctor", help="Diagnose API keys, deps, and config")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "doctor":
        _cmd_doctor()
    else:
        parser.print_help()


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the agent server."""
    if getattr(args, "reload", False):
        _serve_with_reload()
        return

    from ..templates import from_yaml, list_templates, load_template
    from .app import _resolve_auth_token, create_app

    config_path = args.config
    enable_builder = getattr(args, "builder", False)
    auth_token = _resolve_auth_token(getattr(args, "auth_token", None))

    # Builder-only mode: no agent config required
    if config_path is None:
        if not enable_builder:
            print("Error: provide a config file/template name, or use --builder.")
            print("  selectools serve agent.yaml")
            print("  selectools serve --builder")
            sys.exit(1)
        _serve_builder(args.host, args.port, auth_token)
        return
    elif config_path in list_templates():
        provider = _auto_provider()
        if provider is None:
            print(
                "Error: No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
            )
            sys.exit(1)
        agent = load_template(config_path, provider=provider)
        print(f"Loaded template: {config_path}")
    elif os.path.exists(config_path):
        agent = from_yaml(config_path)
        print(f"Loaded config: {config_path}")
    else:
        print(f"Error: Config file not found and not a template name: {config_path}")
        print(f"Available templates: {', '.join(list_templates())}")
        sys.exit(1)

    app = create_app(
        agent,
        playground=not args.no_playground,
        builder=enable_builder,
        host=args.host,
        port=args.port,
        auth_token=auth_token,
    )
    app.serve()


def _serve_builder(host: str, port: int, auth_token: Any) -> None:
    """Start the builder server, preferring Starlette+uvicorn when available."""
    try:
        import uvicorn

        from ._starlette_app import create_builder_app

        print(f"Visual agent builder at http://{host}:{port}/builder  (Starlette/uvicorn)")
        print("Press Ctrl+C to stop.")
        uvicorn.run(create_builder_app(auth_token=auth_token), host=host, port=port)
    except ImportError:
        # Fall back to stdlib HTTP server when starlette/uvicorn not installed
        from .app import BuilderServer

        srv = BuilderServer(host=host, port=port, auth_token=auth_token)
        srv.serve()


def _serve_with_reload() -> None:
    """Restart the server whenever a .py file in selectools/serve/ changes."""
    try:
        from watchfiles import watch
    except ImportError:
        print("Hot reload requires watchfiles:  pip install watchfiles")
        sys.exit(1)

    import subprocess  # nosec B404 — used only for self-restarting the selectools CLI
    from pathlib import Path

    watch_dir = Path(__file__).parent  # selectools/serve/
    cmd = [c for c in sys.argv if c != "--reload"]

    print(f"[reload] watching {watch_dir}")
    print(f"[reload] {' '.join(cmd)}\n")

    proc = subprocess.Popen(cmd)  # nosec B603 — cmd is reconstructed from sys.argv
    try:
        for _ in watch(
            str(watch_dir),
            watch_filter=lambda _c, p: p.endswith((".py", ".html", ".css", ".js")),
        ):
            print("\n[reload] change detected — restarting…")
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc = subprocess.Popen(cmd)  # nosec B603
    except KeyboardInterrupt:
        proc.terminate()


def _cmd_doctor() -> None:
    """Diagnose configuration and connectivity."""
    print("Selectools Doctor")
    print("=" * 40)

    from .. import __version__

    print(f"Version: {__version__}")

    # Python
    print(f"Python: {sys.version.split()[0]}")

    # API Keys
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }
    print("\nAPI Keys:")
    for name, val in keys.items():
        status = "set" if val else "not set"
        icon = "OK" if val else "MISSING"
        print(f"  {name}: {icon}")

    # Optional deps
    print("\nOptional Dependencies:")
    deps = {
        "fastapi": "FastAPI serving",
        "flask": "Flask serving",
        "redis": "Redis cache/sessions",
        "chromadb": "Chroma vector store",
        "pinecone": "Pinecone vector store",
        "psycopg2": "Postgres checkpoints",
        "yaml": "YAML config loading",
    }
    for mod, desc in deps.items():
        try:
            __import__(mod)
            print(f"  {mod}: OK ({desc})")
        except ImportError:
            print(f"  {mod}: not installed ({desc})")

    # Provider connectivity
    print("\nProvider Connectivity:")
    if os.getenv("OPENAI_API_KEY"):
        try:
            from ..providers.openai_provider import OpenAIProvider

            OpenAIProvider()
            print("  OpenAI: OK (connected)")
        except Exception as e:
            print(f"  OpenAI: FAIL ({e})")
    else:
        print("  OpenAI: skipped (no key)")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from ..providers.anthropic_provider import AnthropicProvider

            AnthropicProvider()
            print("  Anthropic: OK (connected)")
        except Exception as e:
            print(f"  Anthropic: FAIL ({e})")
    else:
        print("  Anthropic: skipped (no key)")

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            from ..providers.gemini_provider import GeminiProvider

            GeminiProvider()
            print("  Gemini: OK (connected)")
        except Exception as e:
            print(f"  Gemini: FAIL ({e})")
    else:
        print("  Gemini: skipped (no key)")

    print("\nDiagnosis complete.")


def _auto_provider() -> Any:  # type: ignore[return]
    """Try to create a provider from available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        from ..providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if os.getenv("ANTHROPIC_API_KEY"):
        from ..providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        from ..providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    return None


if __name__ == "__main__":
    main()
