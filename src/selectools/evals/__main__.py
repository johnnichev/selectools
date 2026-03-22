"""CLI entry point: python -m selectools.evals

Usage:
    python -m selectools.evals run cases.json [options]
    python -m selectools.evals compare cases.json --baseline ./baselines [options]

Options:
    --agent YAML       Agent config YAML file
    --html FILE        Write HTML report to FILE
    --junit FILE       Write JUnit XML to FILE
    --json FILE        Write JSON report to FILE
    --baseline DIR     Baseline directory for regression detection
    --concurrency N    Max parallel cases (default: 1)
    --name NAME        Suite name (default: "eval")
    --verbose          Print per-case results
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import BaselineStore, DatasetLoader, EvalSuite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m selectools.evals",
        description="Selectools Eval Framework — evaluate agents from the command line.",
    )
    sub = parser.add_subparsers(dest="command")

    # run command
    run_p = sub.add_parser("run", help="Run eval suite against an agent")
    run_p.add_argument("cases", help="Path to test cases file (JSON/YAML)")
    run_p.add_argument("--html", help="Write HTML report to file")
    run_p.add_argument("--junit", help="Write JUnit XML to file")
    run_p.add_argument("--json", dest="json_out", help="Write JSON report to file")
    run_p.add_argument("--baseline", help="Baseline directory for regression detection")
    run_p.add_argument("--concurrency", type=int, default=1, help="Max parallel cases")
    run_p.add_argument("--name", default="eval", help="Suite name")
    run_p.add_argument("--verbose", action="store_true", help="Print per-case results")
    run_p.add_argument(
        "--provider",
        default="local",
        choices=["local", "openai", "anthropic", "gemini", "ollama"],
        help="Provider to use (default: local)",
    )
    run_p.add_argument("--model", help="Model name")

    # compare command
    cmp_p = sub.add_parser("compare", help="Compare current run against baseline")
    cmp_p.add_argument("cases", help="Path to test cases file (JSON/YAML)")
    cmp_p.add_argument("--baseline", required=True, help="Baseline directory")
    cmp_p.add_argument("--name", default="eval", help="Suite name")
    cmp_p.add_argument("--provider", default="local")
    cmp_p.add_argument("--model", help="Model name")
    cmp_p.add_argument("--concurrency", type=int, default=1)
    cmp_p.add_argument("--save", action="store_true", help="Save as new baseline if no regression")

    return parser


def _create_agent(provider_name: str, model: str | None) -> "Agent":  # type: ignore[name-defined]  # noqa: F821
    """Create an agent with the specified provider."""
    from selectools import Agent, AgentConfig

    prov: Any = None
    mdl = model or "local"

    if provider_name == "local":
        from selectools.providers.stubs import LocalProvider

        prov = LocalProvider()
        mdl = model or "local"
    elif provider_name == "openai":
        from selectools.providers import OpenAIProvider

        prov = OpenAIProvider()
        mdl = model or "gpt-4.1-mini"
    elif provider_name == "anthropic":
        from selectools.providers import AnthropicProvider

        prov = AnthropicProvider()
        mdl = model or "claude-sonnet-4-6"
    elif provider_name == "gemini":
        from selectools.providers import GeminiProvider

        prov = GeminiProvider()
        mdl = model or "gemini-2.5-flash"
    elif provider_name == "ollama":
        from selectools.providers import OllamaProvider

        prov = OllamaProvider()
        mdl = model or "llama3"
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    return Agent(provider=prov, config=AgentConfig(model=mdl), tools=[])


def main() -> None:  # noqa: C901
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load cases
    cases = DatasetLoader.load(args.cases)
    print(f"Loaded {len(cases)} test cases from {args.cases}")

    # Create agent
    agent = _create_agent(args.provider, getattr(args, "model", None))

    # Run suite
    def on_progress(done: int, total: int) -> None:
        print(f"  [{done}/{total}]", end="\r", flush=True)

    suite = EvalSuite(
        agent=agent,
        cases=cases,
        name=args.name,
        max_concurrency=args.concurrency,
        on_progress=on_progress,
    )

    print(f"Running eval suite '{args.name}'...")
    report = suite.run()
    print()
    print(report.summary())
    print()

    if args.command == "run":
        if getattr(args, "verbose", False):
            for cr in report.case_results:
                status = cr.verdict.value.upper()
                name = cr.case.name or cr.case.input[:50]
                print(f"  [{status:5s}] {name} ({cr.latency_ms:.0f}ms)")
                for f in cr.failures:
                    print(f"         {f.evaluator_name}: {f.message}")
            print()

        if args.html:
            report.to_html(args.html)
            print(f"HTML report: {args.html}")
        if args.junit:
            report.to_junit_xml(args.junit)
            print(f"JUnit XML: {args.junit}")
        if args.json_out:
            report.to_json(args.json_out)
            print(f"JSON report: {args.json_out}")

        if args.baseline:
            store = BaselineStore(args.baseline)
            result = store.compare(report)
            if result.is_regression:
                print(f"\nREGRESSIONS DETECTED: {result.regressions}")
                print(f"Accuracy delta: {result.accuracy_delta:+.2%}")
                sys.exit(1)
            else:
                print(f"\nNo regressions (accuracy delta: {result.accuracy_delta:+.2%})")
                if result.improvements:
                    print(f"Improvements: {result.improvements}")
                store.save(report)
                print(f"Baseline saved to {args.baseline}/")

    elif args.command == "compare":
        store = BaselineStore(args.baseline)
        result = store.compare(report)

        if result.is_regression:
            print("REGRESSIONS DETECTED:")
            for name in result.regressions:
                print(f"  - {name}")
            print(f"Accuracy: {result.accuracy_delta:+.2%}")
            print(f"Latency p50: {result.latency_p50_delta:+.0f}ms")
            print(f"Cost: ${result.cost_delta:+.6f}")
            sys.exit(1)
        else:
            print("No regressions detected.")
            if result.improvements:
                print(f"Improvements: {result.improvements}")
            print(f"Accuracy: {result.accuracy_delta:+.2%}")
            if getattr(args, "save", False):
                store.save(report)
                print(f"Baseline updated at {args.baseline}/")

    # Exit with non-zero if accuracy is 0
    if report.accuracy == 0.0 and report.metadata.total_cases > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
