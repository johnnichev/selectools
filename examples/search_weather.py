"""
Minimal search + weather examples using ToolRegistry and @tool.

Runs with the LocalProvider by default (no network calls).
"""

from __future__ import annotations

import argparse
import json
from typing import List

from selectools import Agent, AgentConfig, Message, Role, ToolRegistry, tool
from selectools.parser import ToolCallParser
from selectools.prompt import PromptBuilder
from selectools.providers.stubs import LocalProvider


registry = ToolRegistry()


@registry.tool(description="Return mock search results for a query.")
def search(query: str, top_k: int = 3) -> str:
    results = [{"title": f"Result {i+1} for {query}", "url": f"https://example.com/{i+1}"} for i in range(top_k)]
    return json.dumps({"results": results})


@registry.tool(description="Return mock current weather for a city.")
def weather(city: str, units: str = "metric") -> str:
    sample = {"city": city, "temp_c": 21.5, "temp_f": 70.7, "conditions": "sunny"}
    if units == "imperial":
        temp = {"temp_f": sample["temp_f"], "units": "F"}
    else:
        temp = {"temp_c": sample["temp_c"], "units": "C"}
    return json.dumps({"conditions": sample["conditions"], **temp})


def build_agent() -> Agent:
    provider = LocalProvider()
    config = AgentConfig(max_iterations=3, model="local", stream=False)
    return Agent(
        tools=registry.all(),
        provider=provider,
        prompt_builder=PromptBuilder(),
        parser=ToolCallParser(),
        config=config,
    )


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search + Weather demo (local provider)")
    parser.add_argument("--prompt", default="Find weather in Paris and search docs", help="User prompt")
    args = parser.parse_args(argv)

    agent = build_agent()
    response = agent.run([Message(role=Role.USER, content=args.prompt)])
    print(response.content)


if __name__ == "__main__":
    main()

