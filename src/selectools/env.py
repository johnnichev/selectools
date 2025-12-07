"""
Lightweight environment variable loader for local development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_env_if_present(candidate_paths: Iterable[Path]) -> None:
    """Load key=value pairs from the first .env-style file that exists."""
    for env_path in candidate_paths:
        if not env_path.exists() or not env_path.is_file():
            continue
        try:
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key and key not in os.environ:
                    os.environ[key] = value
            break
        except Exception:
            # Fail quietly; explicit environment variables take precedence.
            continue


def load_default_env() -> None:
    """Load from common locations: cwd/.env and project root .env."""
    cwd = Path.cwd()
    default_candidates = [
        cwd / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    load_env_if_present(default_candidates)


__all__ = ["load_default_env", "load_env_if_present"]
