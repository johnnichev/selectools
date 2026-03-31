"""
Example 75: Stability Markers — @stable, @beta, @deprecated

The stability module provides three decorators for annotating the public API
stability of any class or function.

- @stable   — API is frozen; breaking changes require a major version bump
- @beta     — API may change in a minor release without a deprecation cycle
- @deprecated(since, replacement) — emits DeprecationWarning on use

All three are zero-overhead on the hot path. @deprecated wraps __init__
(for classes) or the function itself to emit the warning exactly once per call.

Run: python examples/75_stability_markers.py
"""

import warnings

from selectools import beta, deprecated, stable

# ── @stable: API contract is frozen ──────────────────────────────────────────


@stable
class PipelineConfig:
    """Configuration for a processing pipeline."""

    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps


@stable
def normalize(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower()


# ── @beta: API may change in a minor release ──────────────────────────────────


@beta
class ExperimentalRouter:
    """Dynamic routing strategy — API may change before stable release."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy

    def route(self, task: str) -> str:
        return f"[{self.strategy}] routing: {task}"


# ── @deprecated: emits DeprecationWarning on instantiation or call ────────────


@deprecated(since="0.19", replacement="PipelineConfig")
class OldPipelineConfig:
    """Deprecated — use PipelineConfig instead."""

    def __init__(self, name: str):
        self.name = name


@deprecated(since="0.19", replacement="normalize")
def clean_text(text: str) -> str:
    """Deprecated — use normalize() instead."""
    return text.strip().lower()


def main() -> None:
    # @stable and @beta: zero overhead, no warnings
    cfg = PipelineConfig(name="main", max_steps=5)
    print(f"PipelineConfig.__stability__ = {PipelineConfig.__stability__!r}")
    print(f"normalize.__stability__       = {normalize.__stability__!r}")

    router = ExperimentalRouter(strategy="dynamic")
    print(f"ExperimentalRouter.__stability__ = {ExperimentalRouter.__stability__!r}")
    print(f"  route result: {router.route('classify document')}")
    print()

    # @deprecated: DeprecationWarning is emitted on use
    print("Using deprecated APIs (warnings captured below):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        old_cfg = OldPipelineConfig(name="legacy")
        result = clean_text("  Hello World  ")

    for w in caught:
        print(f"  ⚠  {w.message}")

    print()
    print(
        f"OldPipelineConfig.__deprecated_since__       = {OldPipelineConfig.__deprecated_since__!r}"
    )
    print(
        f"OldPipelineConfig.__deprecated_replacement__ = {OldPipelineConfig.__deprecated_replacement__!r}"
    )
    print()

    # Programmatic introspection
    apis = [PipelineConfig, normalize, ExperimentalRouter, OldPipelineConfig, clean_text]
    print(f"{'API':<25} {'stability':<12}")
    print("-" * 38)
    for api in apis:
        marker = getattr(api, "__stability__", "unmarked")
        print(f"{api.__name__:<25} {marker:<12}")


if __name__ == "__main__":
    main()
