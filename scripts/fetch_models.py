"""Fetch available models from OpenAI, Anthropic, and Gemini APIs."""

from __future__ import annotations

import os
from typing import Any

import requests  # type: ignore[import-untyped]

from selectools.env import load_default_env

load_default_env()


def fetch_openai_models() -> None:
    """Print available OpenAI chat/GPT models."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI (no key)")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        response.raise_for_status()
        models: list[dict[str, Any]] = response.json()["data"]
        print("\n=== OpenAI Models ===")
        for m in sorted(models, key=lambda x: x["id"]):
            if "gpt" in m["id"] or "o1" in m["id"] or "o3" in m["id"]:
                print(f"{m['id']}")
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")


def fetch_anthropic_models() -> None:
    """Print available Anthropic models."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping Anthropic (no key)")
        return

    try:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)
        if hasattr(client, "models") and hasattr(client.models, "list"):
            models = client.models.list()
            print("\n=== Anthropic Models ===")
            for m in models:
                print(m.id)
        else:
            print("\n=== Anthropic Models ===")
            print("(Client does not support listing models, check docs)")
    except ImportError:
        print("Anthropic package not installed")
    except Exception as e:
        print(f"Error fetching Anthropic models: {e}")


def fetch_gemini_models() -> None:
    """Print available Gemini models."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Skipping Gemini (no key)")
        return

    try:
        from google import genai  # type: ignore[import-untyped]

        client = genai.Client(api_key=api_key)
        print("\n=== Gemini Models ===")
        for m in client.models.list():
            print(m.name)
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")


if __name__ == "__main__":
    fetch_openai_models()
    fetch_anthropic_models()
    fetch_gemini_models()
