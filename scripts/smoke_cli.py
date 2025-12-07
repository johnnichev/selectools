"""
Lightweight smoke runner for the CLI across providers.

Skips providers when required API keys are missing. Uses low max_tokens and
single-iteration calls to keep usage minimal.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


PROVIDERS: List[Tuple[str, str, str, str]] = [
    ("openai", "gpt-4o", "OPENAI_API_KEY", "Hello from OpenAI"),
    ("anthropic", "claude-3-5-sonnet-20240620", "ANTHROPIC_API_KEY", "Hello from Anthropic"),
    ("gemini", "gemini-1.5-flash", "GEMINI_API_KEY", "Hello from Gemini"),
    ("local", "local", None, "Hello from Local"),
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # Ensure package is importable in subprocess even if not installed.
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root / 'src'}{os.pathsep}{existing_pp}" if existing_pp else str(project_root / "src")

    # Load a local .env if present so provider keys are available.
    dot_env = project_root / ".env"
    if dot_env.exists():
        for line in dot_env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key, value)

    for name, model, env_key, prompt in PROVIDERS:
        if env_key and not env.get(env_key):
            print(f"Skipping {name}: missing {env_key}")
            continue
        print(f"\n=== Smoke: {name} ({model}) ===")
        cmd = [
            sys.executable,
            "-m",
            "selectools.cli",
            "run",
            "--provider",
            name,
            "--model",
            model,
            "--prompt",
            prompt,
            "--max-iterations",
            "1",
            "--max-tokens",
            "64",
            "--timeout",
            "15",
        ]
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:  # noqa: BLE001
            print(f"{name} smoke FAILED: {exc}")


if __name__ == "__main__":
    main()

