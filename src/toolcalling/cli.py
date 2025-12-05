"""
CLI entrypoint for the toolcalling library.

Examples:
    python -m toolcalling.cli list-tools
    python -m toolcalling.cli run --provider openai --model gpt-4o --prompt "Search docs" --tool echo
    python -m toolcalling.cli run --provider openai --prompt "Find the dog" --image assets/dog.png --tool detect_bounding_box
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .agent import Agent, AgentConfig
from .prompt import PromptBuilder
from .parser import ToolCallParser
from .types import Message, Role
from .tools import Tool, ToolParameter
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import AnthropicProvider, GeminiProvider, LocalProvider
from .examples.bbox import create_bounding_box_tool


def _build_provider(name: str, model: str):
    if name == "openai":
        return OpenAIProvider(default_model=model)
    if name == "anthropic":
        return AnthropicProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "local":
        return LocalProvider()
    raise ValueError(f"Unknown provider '{name}'.")


def _default_tools() -> Dict[str, Tool]:
    def echo(text: str) -> str:
        return json.dumps({"echo": text})

    echo_tool = Tool(
        name="echo",
        description="Echo back the provided text.",
        parameters=[ToolParameter(name="text", param_type=str, description="Text to echo")],
        function=echo,
    )
    bbox_tool = create_bounding_box_tool()
    return {echo_tool.name: echo_tool, bbox_tool.name: bbox_tool}


def list_tools(tools: Dict[str, Tool]) -> None:
    for tool in tools.values():
        schema = tool.schema()
        print(f"- {schema['name']}: {schema['description']}")


def run_agent(args, tools: Dict[str, Tool]) -> None:
    provider = _build_provider(args.provider, args.model)
    selected_tools: List[Tool] = list(tools.values())
    if args.tool and args.tool in tools:
        selected_tools = [tools[args.tool]]

    config = AgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )
    agent = Agent(
        tools=selected_tools,
        provider=provider,
        prompt_builder=PromptBuilder(),
        parser=ToolCallParser(),
        config=config,
    )

    messages = [Message(role=Role.USER, content=args.prompt, image_path=args.image)]
    response = agent.run(messages=messages)
    print(response.content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-model toolcalling CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tools", help="List available tools")
    list_parser.set_defaults(func="list")

    run_parser = subparsers.add_parser("run", help="Run the agent once")
    run_parser.add_argument("--provider", default="openai", help="Provider name (openai|anthropic|gemini|local)")
    run_parser.add_argument("--model", default="gpt-4o", help="Model name for the provider")
    run_parser.add_argument("--prompt", required=True, help="User prompt")
    run_parser.add_argument("--image", help="Optional image path for vision requests")
    run_parser.add_argument("--tool", help="Restrict to a single tool by name")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    run_parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens for the provider")
    run_parser.add_argument("--max-iterations", type=int, default=4, help="Max tool iterations")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")
    run_parser.set_defaults(func="run")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tools = _default_tools()

    if args.func == "list":
        list_tools(tools)
    elif args.func == "run":
        run_agent(args, tools)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
