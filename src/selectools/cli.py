"""
CLI entrypoint for the selectools library.

Examples:
    python -m selectools.cli list-tools
    python -m selectools.cli run --provider openai --model gpt-4o --prompt "Search docs" --tool echo
    python -m selectools.cli run --provider openai --prompt "Hello" --tool echo
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .agent import Agent, AgentConfig
from .parser import ToolCallParser
from .prompt import PromptBuilder
from .providers.anthropic_provider import AnthropicProvider
from .providers.base import Provider
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import LocalProvider
from .tools import Tool, ToolRegistry
from .types import Message, Role


def _build_provider(name: str, model: str) -> Provider:
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
    registry = ToolRegistry()

    @registry.tool(description="Echo back the provided text.")
    def echo(text: str) -> str:
        return json.dumps({"echo": text})

    return {tool.name: tool for tool in registry.all()}


def list_tools(tools: Dict[str, Tool]) -> None:
    for tool in tools.values():
        schema = tool.schema()
        print(f"- {schema['name']}: {schema['description']}")


def run_agent(args: argparse.Namespace, tools: Dict[str, Tool]) -> None:
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
        stream=args.stream,
        request_timeout=args.timeout,
        max_retries=args.retries,
        retry_backoff_seconds=args.backoff,
    )
    agent = Agent(
        tools=selected_tools,
        provider=provider,
        prompt_builder=PromptBuilder(),
        parser=ToolCallParser(),
        config=config,
    )

    if args.dry_run:
        prompt_preview = agent._system_prompt  # noqa: SLF001
        print(prompt_preview)
        return

    messages = [Message(role=Role.USER, content=args.prompt, image_path=args.image)]
    response = agent.run(
        messages=messages,
        stream_handler=_stream_printer if args.stream else None,
    )
    if not args.stream:
        print(response.content)


def interactive_chat(args: argparse.Namespace, tools: Dict[str, Tool]) -> None:
    provider = _build_provider(args.provider, args.model)
    config = AgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        stream=args.stream,
        request_timeout=args.timeout,
        max_retries=args.retries,
        retry_backoff_seconds=args.backoff,
    )
    agent = Agent(
        tools=list(tools.values()),
        provider=provider,
        prompt_builder=PromptBuilder(),
        parser=ToolCallParser(),
        config=config,
    )

    history: List[Message] = []
    print("Interactive chat. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        history.append(Message(role=Role.USER, content=user_input))
        response = agent.run(
            messages=history,
            stream_handler=_stream_printer if args.stream else None,
        )
        history.append(response.message)
        if not args.stream:
            print(f"\nAI: {response.content}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-model selectools CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tools", help="List available tools")
    list_parser.set_defaults(func="list")

    run_parser = subparsers.add_parser("run", help="Run the agent once")
    run_parser.add_argument(
        "--provider", default="openai", help="Provider name (openai|anthropic|gemini|local)"
    )
    run_parser.add_argument("--model", default="gpt-4o", help="Model name for the provider")
    run_parser.add_argument("--prompt", required=True, help="User prompt")
    run_parser.add_argument("--image", help="Optional image path for vision requests")
    run_parser.add_argument("--tool", help="Restrict to a single tool by name")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    run_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens for the provider"
    )
    run_parser.add_argument("--max-iterations", type=int, default=4, help="Max tool iterations")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")
    run_parser.add_argument(
        "--stream", action="store_true", help="Stream provider output to stdout"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print the composed system prompt and exit"
    )
    run_parser.add_argument(
        "--timeout", type=float, default=30.0, help="Request timeout in seconds"
    )
    run_parser.add_argument(
        "--retries", type=int, default=2, help="Number of provider retries on failure"
    )
    run_parser.add_argument(
        "--backoff", type=float, default=1.0, help="Backoff seconds between retries"
    )
    run_parser.set_defaults(func="run")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat with history")
    chat_parser.add_argument(
        "--provider", default="openai", help="Provider name (openai|anthropic|gemini|local)"
    )
    chat_parser.add_argument("--model", default="gpt-4o", help="Model name for the provider")
    chat_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    chat_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens for the provider"
    )
    chat_parser.add_argument(
        "--max-iterations", type=int, default=6, help="Max tool iterations per turn"
    )
    chat_parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")
    chat_parser.add_argument(
        "--stream", action="store_true", help="Stream provider output to stdout"
    )
    chat_parser.add_argument(
        "--timeout", type=float, default=30.0, help="Request timeout in seconds"
    )
    chat_parser.add_argument(
        "--retries", type=int, default=2, help="Number of provider retries on failure"
    )
    chat_parser.add_argument(
        "--backoff", type=float, default=1.0, help="Backoff seconds between retries"
    )
    chat_parser.set_defaults(func="chat")

    return parser


def _stream_printer(text: str) -> None:
    print(text, end="", flush=True)


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tools = _default_tools()

    if args.func == "list":
        list_tools(tools)
    elif args.func == "run":
        run_agent(args, tools)
    elif args.func == "chat":
        interactive_chat(args, tools)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
