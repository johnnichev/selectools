"""
Architecture fitness tests.

Automated checks that verify structural invariants of the selectools codebase:
- No circular imports across public modules
- Provider protocol conformance
- Enum uniqueness and coverage
- Export consistency
- Model registry validity
"""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# 1. No circular imports
# ---------------------------------------------------------------------------

IMPORTABLE_MODULES = [
    "selectools",
    "selectools.agent",
    "selectools.providers.openai_provider",
    "selectools.providers.anthropic_provider",
    "selectools.providers.gemini_provider",
    "selectools.providers.ollama_provider",
    "selectools.providers.fallback",
    "selectools.providers.stubs",
    "selectools.tools",
    "selectools.guardrails",
    "selectools.rag",
    "selectools.embeddings",
    "selectools.trace",
    "selectools.types",
    "selectools.models",
    "selectools.observer",
]


@pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
def test_no_circular_imports(module_name: str) -> None:
    """Every public module must import without raising ImportError."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Provider protocol methods present on concrete providers
# ---------------------------------------------------------------------------

REQUIRED_METHODS = ["complete", "acomplete", "stream", "astream"]
REQUIRED_PROPERTIES = ["name", "supports_streaming", "supports_async"]


def _get_provider_classes():
    from selectools import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider

    return [
        ("OpenAIProvider", OpenAIProvider),
        ("AnthropicProvider", AnthropicProvider),
        ("GeminiProvider", GeminiProvider),
        ("OllamaProvider", OllamaProvider),
    ]


@pytest.mark.parametrize(
    "provider_name,provider_cls",
    _get_provider_classes(),
    ids=[name for name, _ in _get_provider_classes()],
)
@pytest.mark.parametrize("method", REQUIRED_METHODS)
def test_provider_has_required_method(provider_name: str, provider_cls: type, method: str) -> None:
    """Every concrete provider must implement all Provider protocol methods."""
    assert hasattr(provider_cls, method), f"{provider_name} is missing required method '{method}'"
    assert callable(getattr(provider_cls, method)), f"{provider_name}.{method} is not callable"


@pytest.mark.parametrize(
    "provider_name,provider_cls",
    _get_provider_classes(),
    ids=[name for name, _ in _get_provider_classes()],
)
@pytest.mark.parametrize("prop", REQUIRED_PROPERTIES)
def test_provider_has_required_property(provider_name: str, provider_cls: type, prop: str) -> None:
    """Every concrete provider must expose required properties/attributes."""
    assert hasattr(provider_cls, prop), f"{provider_name} is missing required attribute '{prop}'"


# ---------------------------------------------------------------------------
# 3. All StepType enum values appear in at least one test
# ---------------------------------------------------------------------------


def _collect_test_content() -> str:
    """Read all test files and return their concatenated content."""
    tests_dir = pathlib.Path(__file__).parent
    parts: list[str] = []
    for root, _dirs, files in os.walk(tests_dir):
        for fname in files:
            if fname.startswith("test_") and fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                # Skip this file itself to avoid self-referential matches
                if os.path.abspath(fpath) == os.path.abspath(__file__):
                    continue
                with open(fpath, encoding="utf-8") as f:
                    parts.append(f.read())
    return "\n".join(parts)


def test_all_step_types_covered_in_tests() -> None:
    """Every StepType enum member must appear in at least one test file."""
    from selectools.trace import StepType

    all_test_content = _collect_test_content()

    missing: list[str] = []
    for member in StepType:
        # Check for both the enum name (e.g. StepType.LLM_CALL) and the
        # string value (e.g. "llm_call")
        name_present = member.name in all_test_content
        value_present = member.value in all_test_content
        if not name_present and not value_present:
            missing.append(f"{member.name} ({member.value})")

    assert not missing, f"StepType members not referenced in any test file: {', '.join(missing)}"


# ---------------------------------------------------------------------------
# 4. All ModelType enum values used in the model registry
# ---------------------------------------------------------------------------


def test_all_model_types_used_in_registry() -> None:
    """Every ModelType member must appear in at least one ModelInfo in ALL_MODELS."""
    from selectools.models import ALL_MODELS, ModelType

    types_in_registry = {m.type for m in ALL_MODELS}

    missing = [member for member in ModelType if member not in types_in_registry]

    assert not missing, (
        f"ModelType members not used in ALL_MODELS: {', '.join(m.name for m in missing)}"
    )


# ---------------------------------------------------------------------------
# 5. Export consistency
# ---------------------------------------------------------------------------


def test_all_exports_importable() -> None:
    """Everything listed in selectools.__all__ must be importable from selectools."""
    import selectools

    missing: list[str] = []
    for name in selectools.__all__:
        if not hasattr(selectools, name):
            missing.append(name)

    assert not missing, f"Names in __all__ that are not importable: {', '.join(missing)}"


def _public_names() -> list[str]:
    import selectools

    return list(selectools.__all__)


@pytest.mark.parametrize("name", _public_names())
def test_export_resolves(name: str) -> None:
    """Every symbol in selectools.__all__ must resolve from the top level.

    The stability marker itself is asserted by the v1.0 marking gate below
    (``test_every_public_symbol_has_stability_marker``).
    """
    import selectools

    assert hasattr(selectools, name), f"selectools.{name} is in __all__ but does not resolve"


V1_WART_REMOVAL_ADDITIONS = [
    "Pipeline",
    "Step",
    "StepResult",
    "compose",
    "step",
    "parallel",
    "branch",
    "retry",
    "cache_step",
    "RouterProvider",
    "RouterConfig",
]


@pytest.mark.parametrize("name", V1_WART_REMOVAL_ADDITIONS)
def test_reconciled_exports_carry_stability_marker(name: str) -> None:
    """Names re-added to __all__ in the v1.0 wart-removal pass must be @beta."""
    import selectools

    obj = getattr(selectools, name)
    assert getattr(obj, "__stability__", None) == "beta", (
        f"selectools.{name} must carry the @beta stability marker"
    )


# ---------------------------------------------------------------------------
# 6. StepType enum has no duplicates
# ---------------------------------------------------------------------------


def test_step_type_no_duplicate_values() -> None:
    """All StepType enum values must be unique."""
    from selectools.trace import StepType

    values = [member.value for member in StepType]
    duplicates = [v for v in values if values.count(v) > 1]

    assert not duplicates, f"Duplicate StepType values: {set(duplicates)}"


def test_step_type_no_duplicate_names() -> None:
    """All StepType enum names must be unique (enforced by Python, but explicit)."""
    from selectools.trace import StepType

    names = [member.name for member in StepType]
    assert len(names) == len(set(names)), "Duplicate StepType names detected"


# ---------------------------------------------------------------------------
# 7. ModelType enum has no duplicates
# ---------------------------------------------------------------------------


def test_model_type_no_duplicate_values() -> None:
    """All ModelType enum values must be unique."""
    from selectools.models import ModelType

    values = [member.value for member in ModelType]
    duplicates = [v for v in values if values.count(v) > 1]

    assert not duplicates, f"Duplicate ModelType values: {set(duplicates)}"


def test_model_type_no_duplicate_names() -> None:
    """All ModelType enum names must be unique (enforced by Python, but explicit)."""
    from selectools.models import ModelType

    names = [member.name for member in ModelType]
    assert len(names) == len(set(names)), "Duplicate ModelType names detected"


# ---------------------------------------------------------------------------
# 8. All models have valid pricing
# ---------------------------------------------------------------------------


def test_all_models_have_valid_pricing() -> None:
    """Every model in ALL_MODELS must have non-negative costs and positive limits."""
    from selectools.models import ALL_MODELS

    errors: list[str] = []
    for model in ALL_MODELS:
        if model.prompt_cost < 0:
            errors.append(f"{model.id}: negative prompt_cost ({model.prompt_cost})")
        if model.completion_cost < 0:
            errors.append(f"{model.id}: negative completion_cost ({model.completion_cost})")
        if model.max_tokens <= 0:
            errors.append(f"{model.id}: max_tokens must be positive ({model.max_tokens})")
        if model.context_window <= 0:
            errors.append(f"{model.id}: context_window must be positive ({model.context_window})")

    assert not errors, f"Model registry pricing/limit violations:\n" + "\n".join(errors)


def test_all_models_have_required_fields() -> None:
    """Every model must have a non-empty id and provider."""
    from selectools.models import ALL_MODELS

    for model in ALL_MODELS:
        assert model.id, f"Model with empty id found (provider={model.provider})"
        assert model.provider, f"Model {model.id} has empty provider"


# ---------------------------------------------------------------------------
# 9. v1.0 stability marking gate
# ---------------------------------------------------------------------------
#
# Every public symbol — top-level ``selectools.__all__`` plus every public
# submodule's ``__all__`` — must carry a stability marker. Markers come from
# one of two places (both resolved by ``selectools.stability.get_stability``):
#
# 1. A ``__stability__`` attribute set by ``@stable``/``@beta``/``@deprecated``
#    on the symbol itself (inherited markers do NOT count: every public
#    subclass must be marked explicitly).
# 2. An entry in ``selectools.stability.STABILITY_REGISTRY`` for symbols that
#    cannot carry the attribute: module-level constants, typing aliases, and
#    ``@runtime_checkable`` Protocols (where a class attribute would become a
#    structural member on Python 3.9-3.11 and break ``isinstance()``).
#
# The ONLY exclusion is module references re-exported through ``__all__``
# (e.g. ``selectools.rag`` in the top-level ``__all__``): those declare their
# own module-level ``__stability__``, asserted separately below. There is no
# name-based exclusion list — a future unmarked public symbol fails this gate.

VALID_STABILITY_LEVELS = {"stable", "beta", "deprecated"}

PUBLIC_SUBMODULES = [
    "a2a",
    "agent",
    "cache",
    "embeddings",
    "evals",
    "exceptions",
    "guardrails",
    "knowledge_backends",
    "mcp",
    "models",
    "observe",
    "orchestration",
    "patterns",
    "pending",
    "pricing",
    "providers",
    "rag",
    "results",
    "serve",
    "sessions",
    "toolbox",
    "tools",
    "types",
    "unified_memory",
]

# Module-level stability promises (the ROADMAP v1.0 taxonomy): "stable" for
# core modules whose public surface is majority-stable, "beta" for everything
# still allowed to move in a minor release.
EXPECTED_MODULE_STABILITY = {
    "a2a": "beta",
    "agent": "stable",
    "cache": "stable",
    "embeddings": "beta",
    "evals": "beta",
    "exceptions": "stable",
    "guardrails": "stable",
    "knowledge_backends": "beta",
    "mcp": "beta",
    "models": "stable",
    "observe": "beta",
    "orchestration": "beta",
    "patterns": "beta",
    "pending": "beta",
    "pricing": "stable",
    "providers": "stable",
    "rag": "beta",
    "results": "beta",
    "serve": "beta",
    "sessions": "stable",
    "toolbox": "stable",
    "tools": "stable",
    "types": "stable",
    "unified_memory": "beta",
}


def _all_public_symbols() -> "list[tuple[str, str]]":
    cases: "list[tuple[str, str]]" = []
    for mod_name in ["selectools"] + [f"selectools.{s}" for s in PUBLIC_SUBMODULES]:
        mod = importlib.import_module(mod_name)
        for name in mod.__all__:
            cases.append((mod_name, name))
    return cases


@pytest.mark.parametrize(
    "mod_name,symbol",
    _all_public_symbols(),
    ids=[f"{m}.{n}" for m, n in _all_public_symbols()],
)
def test_every_public_symbol_has_stability_marker(mod_name: str, symbol: str) -> None:
    """v1.0 gate: every public symbol must carry a stability marker.

    Fails on any future addition to any public ``__all__`` that is neither
    decorated with ``@stable``/``@beta``/``@deprecated`` nor registered via
    ``selectools.stability.register_stability``.
    """
    import inspect

    from selectools.stability import get_stability

    mod = importlib.import_module(mod_name)
    obj = getattr(mod, symbol)
    if inspect.ismodule(obj):
        pytest.skip("module reference: covered by test_public_submodule_declares_module_stability")
    level = get_stability(obj, symbol)
    assert level in VALID_STABILITY_LEVELS, (
        f"{mod_name}.{symbol} has no stability marker. Decorate it with "
        f"@stable/@beta/@deprecated, or, if it cannot carry attributes "
        f"(constant, typing alias, runtime-checkable Protocol), call "
        f'register_stability("{symbol}", "<level>") at its definition site.'
    )


@pytest.mark.parametrize("submodule", PUBLIC_SUBMODULES)
def test_public_submodule_declares_module_stability(submodule: str) -> None:
    """Every public submodule must declare a module-level ``__stability__``."""
    mod = importlib.import_module(f"selectools.{submodule}")
    level = getattr(mod, "__stability__", None)
    assert level in VALID_STABILITY_LEVELS, (
        f"selectools.{submodule} must declare a module-level __stability__ (got {level!r})"
    )
    assert level == EXPECTED_MODULE_STABILITY[submodule], (
        f"selectools.{submodule}.__stability__ is {level!r} but the v1.0 "
        f"taxonomy expects {EXPECTED_MODULE_STABILITY[submodule]!r}; update "
        f"EXPECTED_MODULE_STABILITY only with a deliberate stability decision."
    )
