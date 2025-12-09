#!/usr/bin/env python3
"""
Fetch current OpenAI models and their pricing information.

This script queries the OpenAI API to get all available models
and attempts to fetch their pricing information.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment
from selectools.env import load_default_env

load_default_env()

import json

from openai import OpenAI


def main():
    client = OpenAI()

    print("=" * 80)
    print("Fetching OpenAI Models")
    print("=" * 80)

    # Get all models
    models = client.models.list()

    # Filter for relevant models (GPT, embedding, etc.)
    chat_models = []
    embedding_models = []
    other_models = []

    for model in models.data:
        model_id = model.id
        if any(x in model_id for x in ["gpt-4", "gpt-3.5", "gpt-4o"]):
            chat_models.append(model_id)
        elif "embedding" in model_id:
            embedding_models.append(model_id)
        else:
            other_models.append(model_id)

    print(f"\n📊 Found {len(chat_models)} chat models")
    print(f"📊 Found {len(embedding_models)} embedding models")
    print(f"📊 Found {len(other_models)} other models")

    print("\n" + "=" * 80)
    print("GPT Chat Models (sorted)")
    print("=" * 80)
    for model_id in sorted(chat_models):
        print(f"  - {model_id}")

    print("\n" + "=" * 80)
    print("Embedding Models (sorted)")
    print("=" * 80)
    for model_id in sorted(embedding_models):
        print(f"  - {model_id}")

    # Print pricing note
    print("\n" + "=" * 80)
    print("Pricing Information")
    print("=" * 80)
    print(
        """
OpenAI pricing is available at: https://openai.com/api/pricing/

Current pricing (as of Dec 2024):

GPT-4o (2024-11-20):
  - Prompt: $2.50 / 1M tokens
  - Completion: $10.00 / 1M tokens

GPT-4o-mini (2024-07-18):
  - Prompt: $0.150 / 1M tokens
  - Completion: $0.600 / 1M tokens

GPT-4-turbo (2024-04-09):
  - Prompt: $10.00 / 1M tokens
  - Completion: $30.00 / 1M tokens

GPT-3.5-turbo:
  - Prompt: $0.50 / 1M tokens
  - Completion: $1.50 / 1M tokens

o1-preview (2024-09-12):
  - Prompt: $15.00 / 1M tokens
  - Completion: $60.00 / 1M tokens

o1-mini (2024-09-12):
  - Prompt: $3.00 / 1M tokens
  - Completion: $12.00 / 1M tokens
"""
    )

    # Export to JSON
    output = {
        "chat_models": sorted(chat_models),
        "embedding_models": sorted(embedding_models),
        "other_models": sorted(other_models),
        "total_models": len(models.data),
    }

    output_file = PROJECT_ROOT / "openai_models.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Model list exported to: {output_file}")


if __name__ == "__main__":
    main()
