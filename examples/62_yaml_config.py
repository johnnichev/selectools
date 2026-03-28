"""
Example 62: Loading an Agent from YAML config

Demonstrates the structured AgentConfig workflow (v0.19.0):
- Write a YAML config file describing the agent
- Load it with AgentConfig.from_yaml()
- Instantiate and run the agent

Uses LocalProvider so no API keys are needed.

Prerequisites: pyyaml
    pip install selectools pyyaml

Run:
    python examples/62_yaml_config.py
"""

import os
import tempfile

import yaml

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider


@tool(description="Convert text to uppercase")
def to_upper(text: str) -> str:
    """Convert input text to uppercase."""
    return text.upper()


@tool(description="Count words in text")
def word_count(text: str) -> str:
    """Return the number of words in the input."""
    count = len(text.split())
    return f"Word count: {count}"


def main() -> None:
    print("=" * 60)
    print("YAML Config Demo")
    print("=" * 60)

    # --- Step 1: Build a YAML config inline ---
    config_dict = {
        "name": "text-assistant",
        "model": "gpt-5-mini",
        "temperature": 0.2,
        "max_tokens": 512,
        "max_iterations": 3,
        "system_prompt": "You are a helpful text-processing assistant.",
        "verbose": False,
        "reasoning_strategy": "react",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name

    print(f"\n1. Wrote YAML config to: {config_path}")
    print(f"   Contents:")
    for key, value in config_dict.items():
        print(f"     {key}: {value}")

    # --- Step 2: Load config from YAML ---
    with open(config_path) as f:
        loaded = yaml.safe_load(f)

    config = AgentConfig(
        **{k: v for k, v in loaded.items() if k in AgentConfig.__dataclass_fields__}
    )

    print(f"\n2. Loaded AgentConfig from YAML:")
    print(f"   name={config.name}, model={config.model}")
    print(f"   temperature={config.temperature}, max_tokens={config.max_tokens}")
    print(f"   reasoning_strategy={config.reasoning_strategy}")

    # --- Step 3: Create and run the agent ---
    agent = Agent(
        tools=[to_upper, word_count],
        provider=LocalProvider(),
        config=config,
    )

    result = agent.run("Make this uppercase: hello world")

    print(f"\n3. Agent result:")
    print(f"   Content: {result.content}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Tools available: {[t.name for t in agent.tools]}")

    # Cleanup
    os.unlink(config_path)
    print(f"\n4. Cleaned up temp file.")
    print("\nDone! YAML-based config loaded and agent ran successfully.")


if __name__ == "__main__":
    main()
