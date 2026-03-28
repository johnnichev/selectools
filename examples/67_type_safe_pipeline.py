"""
Example 67: Type-Safe Pipeline Contracts

Demonstrates type-safe step contracts in pipelines (v0.19.0):
- Steps infer input/output types from type hints
- Pipeline construction validates adjacent step types
- A mismatch between step N's output and step N+1's input emits a warning
- Correctly typed pipelines run without warnings

Run:
    python examples/67_type_safe_pipeline.py
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List

from selectools.pipeline import Pipeline, Step, step

# --- Well-typed steps ---


@step
def tokenize(text: str) -> List[str]:
    """Split text into tokens."""
    return text.lower().split()


@step
def count_tokens(tokens: List[str]) -> Dict[str, int]:
    """Count occurrences of each token."""
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


@step
def format_counts(counts: Dict[str, int]) -> str:
    """Format token counts as a readable string."""
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    lines = [f"  {word}: {count}" for word, count in sorted_counts[:5]]
    return "Top tokens:\n" + "\n".join(lines)


# --- Steps with a mismatched type ---


@step
def output_int(text: str) -> int:
    """Returns an int (word count)."""
    return len(text.split())


@step
def expects_str(value: str) -> str:
    """This step expects a str, not an int."""
    return f"Got string: {value}"


def main() -> None:
    print("=" * 60)
    print("Type-Safe Pipeline Contracts Demo")
    print("=" * 60)

    # --- Demo 1: Correctly typed pipeline (no warnings) ---
    print("\n--- Demo 1: Correct types (str -> List -> Dict -> str) ---")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        good_pipeline = tokenize | count_tokens | format_counts
        if not caught:
            print("No warnings -- types are compatible!")
        else:
            for w in caught:
                print(f"WARNING: {w.message}")

    result = good_pipeline.run("the quick brown fox jumps over the lazy dog the fox")
    print(f"Pipeline: {good_pipeline}")
    print(f"Result:\n{result.output}")

    # --- Demo 2: Type mismatch (int output fed into str input) ---
    print("\n--- Demo 2: Type mismatch (int -> str) ---")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad_pipeline = output_int | expects_str
        mismatches = [w for w in caught if "type mismatch" in str(w.message).lower()]
        if mismatches:
            print(f"Caught {len(mismatches)} type mismatch warning(s):")
            for w in mismatches:
                print(f"  {w.message}")
        else:
            print("No mismatch detected (types may not be statically comparable)")

    # --- Demo 3: Explicit type contracts ---
    print("\n--- Demo 3: Explicit type contracts on Step ---")

    # You can also set input/output types explicitly
    parse_step = Step(
        lambda text: text.split(","),
        name="csv_parse",
        input_type=str,
        output_type=list,
    )
    join_step = Step(
        lambda items: " | ".join(items),
        name="join",
        input_type=list,
        output_type=str,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explicit_pipeline = Pipeline(steps=[parse_step, join_step])
        if not caught:
            print("No warnings -- explicit types are compatible!")

    result = explicit_pipeline.run("alpha,beta,gamma,delta")
    print(f"Pipeline: {explicit_pipeline}")
    print(f"Result: {result.output}")

    # --- Demo 4: Inspect inferred types ---
    print("\n--- Demo 4: Inspecting inferred types ---")
    for s in [tokenize, count_tokens, format_counts]:
        in_t = getattr(s, "input_type", None)
        out_t = getattr(s, "output_type", None)
        in_name = getattr(in_t, "__name__", str(in_t)) if in_t else "?"
        out_name = getattr(out_t, "__name__", str(out_t)) if out_t else "?"
        print(f"  {s.name}: {in_name} -> {out_name}")

    print("\nDone! Type contracts catch pipeline wiring errors at construction time.")


if __name__ == "__main__":
    main()
