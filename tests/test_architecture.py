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
# There is no name-based exclusion list — a future unmarked public symbol
# fails this gate. Only two conventions narrow the surface:
#
# - Module references re-exported through ``__all__`` (e.g. ``selectools.rag``
#   in the top-level ``__all__``, or the toolbox category modules in
#   ``toolbox.__all__``) are asserted to carry a module-level
#   ``__stability__`` themselves, in-line in the symbol test.
# - ``_``-prefixed names are private by convention even when they appear in an
#   ``__all__`` for internal wiring (e.g. ``orchestration.state._Goto``) and
#   are not part of the public contract.
#
# ``PUBLIC_SUBMODULES`` is the full set of modules reachable by a ``pkgutil``
# walk that declare an ``__all__`` (``_``-prefixed modules are internal by
# convention). ``test_public_submodules_list_is_exhaustive`` regenerates the
# walk on every run, so a new module with an ``__all__`` cannot ship without
# joining this gate.

VALID_STABILITY_LEVELS = {"stable", "beta", "deprecated"}

# Genuinely internal modules that declare an ``__all__`` but are deliberately
# kept out of the public gate. Empty today — add a module here ONLY with an
# explanatory comment and a deliberate decision that its surface is internal.
INTERNAL_MODULES_WITH_ALL: "set[str]" = set()

PUBLIC_SUBMODULES = [
    "a2a",
    "a2a.client",
    "a2a.server",
    "a2a.types",
    "agent",
    "agent.config_groups",
    "analytics",
    "audit",
    "cache",
    "cache_redis",
    "cache_semantic",
    "cancellation",
    "checkpoint_postgres",
    "coherence",
    "compose",
    "embeddings",
    "embeddings.anthropic",
    "embeddings.cohere",
    "embeddings.gemini",
    "embeddings.openai",
    "embeddings.provider",
    "entity_memory",
    "env",
    "evals",
    "exceptions",
    "guardrails",
    "guardrails.base",
    "guardrails.format",
    "guardrails.injection",
    "guardrails.length",
    "guardrails.pii",
    "guardrails.pipeline",
    "guardrails.topic",
    "guardrails.toxicity",
    "knowledge",
    "knowledge_backends",
    "knowledge_graph",
    "knowledge_sanitizers",
    "knowledge_store_redis",
    "knowledge_store_supabase",
    "mcp",
    "memory",
    "models",
    "observe",
    "observe.langfuse",
    "observe.otel",
    "observe.trace_store",
    "observer",
    "orchestration",
    "orchestration.checkpoint",
    "orchestration.graph",
    "orchestration.node",
    "orchestration.state",
    "orchestration.supervisor",
    "parser",
    "patterns",
    "patterns.debate",
    "patterns.plan_and_execute",
    "patterns.reflective",
    "patterns.team_lead",
    "pending",
    "pipeline",
    "policy",
    "pricing",
    "prompt",
    "providers",
    "providers.anthropic_provider",
    "providers.azure_openai_provider",
    "providers.base",
    "providers.fallback",
    "providers.gemini_provider",
    "providers.litellm_provider",
    "providers.ollama_provider",
    "providers.openai_provider",
    "providers.router",
    "providers.stubs",
    "rag",
    "rag.bm25",
    "rag.chunking",
    "rag.hybrid",
    "rag.loaders",
    "rag.reranker",
    "rag.stores",
    "rag.stores.chroma",
    "rag.stores.faiss",
    "rag.stores.memory",
    "rag.stores.pgvector",
    "rag.stores.pinecone",
    "rag.stores.qdrant",
    "rag.stores.sqlite",
    "rag.tools",
    "rag.vector_store",
    "results",
    "scheduler",
    "security",
    "serve",
    "serve.api",
    "sessions",
    "stability",
    "structured",
    "templates",
    "token_estimation",
    "toolbox",
    "toolbox.browser_tools",
    "toolbox.calculator_tools",
    "toolbox.code_tools",
    "toolbox.data_tools",
    "toolbox.datetime_tools",
    "toolbox.db_tools",
    "toolbox.discord_tools",
    "toolbox.email_tools",
    "toolbox.file_tools",
    "toolbox.github_tools",
    "toolbox.image_tools",
    "toolbox.linear_tools",
    "toolbox.memory_tools",
    "toolbox.reasoning_tools",
    "toolbox.notion_tools",
    "toolbox.pdf_tools",
    "toolbox.s3_tools",
    "toolbox.search_tools",
    "toolbox.slack_tools",
    "toolbox.text_tools",
    "toolbox.web_tools",
    "tools",
    "tools.loader",
    "trace",
    "types",
    "unified_memory",
    "usage",
]

# Module-level stability promises (the ROADMAP v1.0 taxonomy): "stable" for
# core modules whose public surface is majority-stable, "beta" for everything
# still allowed to move in a minor release.
EXPECTED_MODULE_STABILITY = {
    "a2a": "beta",
    "a2a.client": "beta",
    "a2a.server": "beta",
    "a2a.types": "beta",
    "agent": "stable",
    "agent.config_groups": "stable",
    "analytics": "stable",
    "audit": "stable",
    "cache": "stable",
    "cache_redis": "stable",
    "cache_semantic": "stable",
    "cancellation": "stable",
    "checkpoint_postgres": "stable",
    "coherence": "beta",
    "compose": "beta",
    "embeddings": "beta",
    "embeddings.anthropic": "beta",
    "embeddings.cohere": "beta",
    "embeddings.gemini": "beta",
    "embeddings.openai": "beta",
    "embeddings.provider": "beta",
    "entity_memory": "stable",
    "env": "stable",
    "evals": "beta",
    "exceptions": "stable",
    "guardrails": "stable",
    "guardrails.base": "stable",
    "guardrails.format": "stable",
    "guardrails.injection": "beta",
    "guardrails.length": "stable",
    "guardrails.pii": "stable",
    "guardrails.pipeline": "stable",
    "guardrails.topic": "stable",
    "guardrails.toxicity": "stable",
    "knowledge": "beta",
    "knowledge_backends": "beta",
    "knowledge_graph": "beta",
    "knowledge_sanitizers": "beta",
    "knowledge_store_redis": "beta",
    "knowledge_store_supabase": "beta",
    "mcp": "beta",
    "memory": "stable",
    "models": "stable",
    "observe": "beta",
    "observe.langfuse": "beta",
    "observe.otel": "beta",
    "observe.trace_store": "beta",
    "observer": "stable",
    "orchestration": "beta",
    "orchestration.checkpoint": "beta",
    "orchestration.graph": "beta",
    "orchestration.node": "beta",
    "orchestration.state": "beta",
    "orchestration.supervisor": "beta",
    "parser": "stable",
    "patterns": "beta",
    "patterns.debate": "beta",
    "patterns.plan_and_execute": "beta",
    "patterns.reflective": "beta",
    "patterns.team_lead": "beta",
    "pending": "beta",
    "pipeline": "beta",
    "policy": "beta",
    "pricing": "stable",
    "prompt": "stable",
    "providers": "stable",
    "providers.anthropic_provider": "stable",
    "providers.azure_openai_provider": "stable",
    "providers.base": "stable",
    "providers.fallback": "stable",
    "providers.gemini_provider": "stable",
    "providers.litellm_provider": "beta",
    "providers.ollama_provider": "stable",
    "providers.openai_provider": "stable",
    "providers.router": "beta",
    "providers.stubs": "beta",
    "rag": "beta",
    "rag.bm25": "beta",
    "rag.chunking": "beta",
    "rag.hybrid": "beta",
    "rag.loaders": "beta",
    "rag.reranker": "beta",
    "rag.stores": "beta",
    "rag.stores.chroma": "beta",
    "rag.stores.faiss": "beta",
    "rag.stores.memory": "beta",
    "rag.stores.pgvector": "beta",
    "rag.stores.pinecone": "beta",
    "rag.stores.qdrant": "beta",
    "rag.stores.sqlite": "beta",
    "rag.tools": "beta",
    "rag.vector_store": "beta",
    "results": "beta",
    "scheduler": "beta",
    "security": "stable",
    "serve": "beta",
    "serve.api": "beta",
    "sessions": "stable",
    "stability": "stable",
    "structured": "stable",
    "templates": "beta",
    "token_estimation": "stable",
    "toolbox": "stable",
    "toolbox.browser_tools": "beta",
    "toolbox.calculator_tools": "stable",
    "toolbox.code_tools": "stable",
    "toolbox.data_tools": "stable",
    "toolbox.datetime_tools": "stable",
    "toolbox.db_tools": "stable",
    "toolbox.discord_tools": "beta",
    "toolbox.email_tools": "stable",
    "toolbox.file_tools": "stable",
    "toolbox.github_tools": "stable",
    "toolbox.image_tools": "beta",
    "toolbox.linear_tools": "stable",
    "toolbox.memory_tools": "stable",
    "toolbox.reasoning_tools": "beta",
    "toolbox.notion_tools": "stable",
    "toolbox.pdf_tools": "stable",
    "toolbox.s3_tools": "beta",
    "toolbox.search_tools": "stable",
    "toolbox.slack_tools": "stable",
    "toolbox.text_tools": "stable",
    "toolbox.web_tools": "stable",
    "tools": "stable",
    "tools.loader": "stable",
    "trace": "stable",
    "types": "stable",
    "unified_memory": "beta",
    "usage": "stable",
}


# Sentinel symbol for modules that cannot import in this environment because
# an OPTIONAL dependency is absent (CI's core matrix installs no extras).
# The gate still emits one parametrized case for such a module so its
# existence is visible in the test report; the case skips with the reason.
_OPTIONAL_DEP_SENTINEL = "<optional-dependency-missing>"


def _import_or_none(mod_name: str) -> "tuple[object | None, str | None]":
    """Import a module, returning (module, None) or (None, missing-dep reason).

    Only ImportError/ModuleNotFoundError are treated as "optional dependency
    absent" — any other exception is a real bug and propagates.
    """
    try:
        return importlib.import_module(mod_name), None
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        return None, str(exc)


def _all_public_symbols() -> "list[tuple[str, str]]":
    cases: "list[tuple[str, str]]" = []
    for mod_name in ["selectools"] + [f"selectools.{s}" for s in PUBLIC_SUBMODULES]:
        mod, missing = _import_or_none(mod_name)
        if mod is None:
            cases.append((mod_name, _OPTIONAL_DEP_SENTINEL))
            continue
        for name in mod.__all__:
            # ``_``-prefixed names are private by convention even when listed
            # in an ``__all__`` for internal wiring (orchestration.state
            # exports ``_Goto``/``_Update``, rag.vector_store exports two
            # ``_``-helpers for its store implementations).
            if name.startswith("_"):
                continue
            cases.append((mod_name, name))
    return cases


def _walked_modules_with_all() -> "list[str]":
    """Every selectools module reachable by a pkgutil walk that has an ``__all__``.

    ``_``-prefixed modules (and anything below them) are internal by
    convention and excluded.
    """
    import pkgutil

    import selectools

    walked: "list[str]" = []
    for info in pkgutil.walk_packages(selectools.__path__, prefix="selectools."):
        short = info.name[len("selectools.") :]
        if any(part.startswith("_") for part in short.split(".")):
            continue
        mod, _missing = _import_or_none(info.name)
        if mod is None:
            # Optional-dependency module not importable in this environment;
            # if it is gated it will be exercised in environments that
            # install the extras (and skipped-with-reason here).
            if short in PUBLIC_SUBMODULES:
                walked.append(short)
            continue
        if hasattr(mod, "__all__"):
            walked.append(short)
    return sorted(walked)


def test_public_submodules_list_is_exhaustive() -> None:
    """The gate enumerates EVERY module with an ``__all__`` — no holes.

    A module that declares an ``__all__`` is declaring a public surface; it
    must either be listed in ``PUBLIC_SUBMODULES`` (and thereby have all of
    its symbols marked) or be deliberately excluded via
    ``INTERNAL_MODULES_WITH_ALL`` with an explanatory comment.
    """
    walked = set(_walked_modules_with_all())
    gated = set(PUBLIC_SUBMODULES)
    missing = walked - gated - INTERNAL_MODULES_WITH_ALL
    assert not missing, (
        f"Modules with an __all__ missing from the stability gate: "
        f"{sorted(missing)}. Add them to PUBLIC_SUBMODULES (and "
        f"EXPECTED_MODULE_STABILITY), or — only for genuinely internal "
        f"surfaces — to INTERNAL_MODULES_WITH_ALL with a comment."
    )
    stale = gated - walked
    assert not stale, (
        f"PUBLIC_SUBMODULES entries that no longer exist or lost their __all__: {sorted(stale)}"
    )
    overlap = gated & INTERNAL_MODULES_WITH_ALL
    assert not overlap, f"Modules cannot be both public and internal: {sorted(overlap)}"
    assert set(EXPECTED_MODULE_STABILITY) == gated, (
        "EXPECTED_MODULE_STABILITY must pin a level for exactly the modules "
        "in PUBLIC_SUBMODULES; diff: "
        f"{sorted(set(EXPECTED_MODULE_STABILITY) ^ gated)}"
    )


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

    if symbol == _OPTIONAL_DEP_SENTINEL:
        pytest.skip(f"{mod_name} requires an optional dependency not installed here")
    mod, missing = _import_or_none(mod_name)
    if mod is None:
        pytest.skip(f"{mod_name} optional dependency missing: {missing}")
    try:
        obj = getattr(mod, symbol)
    except ImportError as exc:
        # Lazy __getattr__ exports (serve.AgentAPI, a2a.A2AServer) raise
        # ImportError when their optional backend is absent.
        pytest.skip(f"{mod_name}.{symbol} optional dependency missing: {exc}")
    if inspect.ismodule(obj):
        # A module re-exported through __all__ (selectools.rag, the toolbox
        # category modules, ...) must itself declare a module-level
        # __stability__ — asserted here so module refs cannot dodge the gate
        # by pointing at a test that does not enumerate them.
        ref_level = getattr(obj, "__stability__", None)
        assert ref_level in VALID_STABILITY_LEVELS, (
            f"{mod_name}.{symbol} re-exports module {obj.__name__}, which has "
            f"no module-level __stability__ (got {ref_level!r}). Set "
            f'__stability__ = "stable"|"beta" in {obj.__name__}.'
        )
        return
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
    mod, missing = _import_or_none(f"selectools.{submodule}")
    if mod is None:
        pytest.skip(f"selectools.{submodule} optional dependency missing: {missing}")
    level = getattr(mod, "__stability__", None)
    assert level in VALID_STABILITY_LEVELS, (
        f"selectools.{submodule} must declare a module-level __stability__ (got {level!r})"
    )
    assert level == EXPECTED_MODULE_STABILITY[submodule], (
        f"selectools.{submodule}.__stability__ is {level!r} but the v1.0 "
        f"taxonomy expects {EXPECTED_MODULE_STABILITY[submodule]!r}; update "
        f"EXPECTED_MODULE_STABILITY only with a deliberate stability decision."
    )
