"""
Agent templates and YAML configuration loading.

Create agents from YAML config files or use pre-built templates
for common use cases.

Usage::

    from selectools.templates import from_yaml, load_template

    # From YAML file
    agent = from_yaml("agent.yaml")

    # From built-in template
    agent = load_template("customer_support", provider=provider)
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..providers.base import Provider

_TEMPLATES: Dict[str, str] = {
    "customer_support": "selectools.templates.customer_support",
    "data_analyst": "selectools.templates.data_analyst",
    "research_assistant": "selectools.templates.research_assistant",
    "code_reviewer": "selectools.templates.code_reviewer",
    "rag_chatbot": "selectools.templates.rag_chatbot",
}


def from_yaml(path: str, provider: Optional["Provider"] = None) -> "Agent":
    """Create an Agent from a YAML configuration file.

    Args:
        path: Path to the YAML config file.
        provider: Optional provider instance. If not given, one is created
                  from the ``provider`` field in the YAML.

    Returns:
        A configured Agent instance.

    Example YAML::

        provider: openai
        model: gpt-4o
        tools:
          - selectools.toolbox.file_tools.read_file
          - ./my_custom_tool.py
        system_prompt: "You are a helpful assistant."
        retry:
          max_retries: 3
        budget:
          max_cost_usd: 0.50
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config loading. Install with: pip install pyyaml"
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a dict, got {type(raw).__name__}")

    return _build_agent_from_dict(raw, provider=provider, base_dir=config_path.parent)


def from_dict(config: Any, provider: Optional["Provider"] = None) -> "Agent":
    """Create an Agent from a configuration dictionary.

    Same format as YAML config but as a Python dict.
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config).__name__}")
    return _build_agent_from_dict(config, provider=provider)


def load_template(
    name: str,
    provider: "Provider",
    **overrides: Any,
) -> "Agent":
    """Load a pre-built agent template.

    Args:
        name: Template name. Available: customer_support, data_analyst,
              research_assistant, code_reviewer, rag_chatbot.
        provider: LLM provider instance.
        **overrides: Override any config field (e.g., model="gpt-4o").

    Returns:
        A configured Agent with template tools and system prompt.
    """
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise ValueError(f"Unknown template {name!r}. Available: {available}")

    mod = importlib.import_module(_TEMPLATES[name])
    build_fn = getattr(mod, "build", None)
    if build_fn is None:
        raise ValueError(f"Template {name!r} has no build() function")

    return build_fn(provider=provider, **overrides)  # type: ignore[no-any-return]


def list_templates() -> List[str]:
    """Return names of all available templates."""
    return sorted(_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


def _build_agent_from_dict(
    raw: Dict[str, Any],
    provider: Optional["Provider"] = None,
    base_dir: Optional[Path] = None,
) -> "Agent":
    """Build an Agent from a config dict."""
    from ..agent.config import AgentConfig
    from ..agent.config_groups import (
        BudgetConfig,
        CoherenceConfig,
        CompressConfig,
        RetryConfig,
        TraceConfig,
    )
    from ..agent.core import Agent

    # Resolve provider
    if provider is None:
        provider = _resolve_provider(raw.get("provider", "openai"))

    # Resolve tools
    tools = _resolve_tools(raw.get("tools", []), base_dir=base_dir)
    if not tools:
        from ..tools.decorators import tool

        @tool(description="No-op placeholder tool")
        def noop() -> str:
            return "ok"

        tools = [noop]

    # Build nested configs from YAML sections
    config_kwargs: Dict[str, Any] = {}

    # Direct fields
    for key in (
        "model",
        "temperature",
        "max_tokens",
        "max_iterations",
        "system_prompt",
        "verbose",
        "stream",
        "reasoning_strategy",
    ):
        if key in raw:
            config_kwargs[key] = raw[key]

    # Nested config groups
    if "retry" in raw and isinstance(raw["retry"], dict):
        config_kwargs["retry"] = RetryConfig(**raw["retry"])

    if "budget" in raw and isinstance(raw["budget"], dict):
        config_kwargs["budget"] = BudgetConfig(**raw["budget"])

    if "coherence" in raw and isinstance(raw["coherence"], dict):
        config_kwargs["coherence"] = CoherenceConfig(**raw["coherence"])

    if "compress" in raw and isinstance(raw["compress"], dict):
        config_kwargs["compress"] = CompressConfig(**raw["compress"])

    if "trace" in raw and isinstance(raw["trace"], dict):
        config_kwargs["trace"] = TraceConfig(**raw["trace"])

    config = AgentConfig(**config_kwargs)

    return Agent(provider=provider, tools=tools, config=config)


def _resolve_provider(name: str) -> "Provider":
    """Create a provider instance from a name string."""
    providers = {
        "openai": ("selectools.providers.openai_provider", "OpenAIProvider"),
        "anthropic": ("selectools.providers.anthropic_provider", "AnthropicProvider"),
        "gemini": ("selectools.providers.gemini_provider", "GeminiProvider"),
        "ollama": ("selectools.providers.ollama_provider", "OllamaProvider"),
        "local": ("selectools.providers.stubs", "LocalProvider"),
    }
    if name not in providers:
        available = ", ".join(sorted(providers.keys()))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}")

    mod_path, cls_name = providers[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls()  # type: ignore[no-any-return]


def _resolve_tools(tool_specs: List[Any], base_dir: Optional[Path] = None) -> list:
    """Resolve tool specifications to Tool objects.

    Specs can be:
    - Dotted import path: "selectools.toolbox.file_tools.read_file"
    - Relative file path: "./my_tool.py"
    - Already a Tool object
    """
    from ..tools.base import Tool

    tools = []
    for spec in tool_specs:
        if isinstance(spec, Tool):
            tools.append(spec)
        elif isinstance(spec, str):
            if spec.startswith("./") or spec.startswith("../") or spec.endswith(".py"):
                # File path
                from ..tools.loader import ToolLoader

                file_path = spec
                if base_dir and not os.path.isabs(spec):
                    resolved = (base_dir / spec).resolve()
                    if not str(resolved).startswith(str(base_dir.resolve())):
                        raise ValueError(f"Tool path escapes config directory: {spec!r}")
                    file_path = str(resolved)
                loaded = ToolLoader.from_file(file_path)
                tools.extend(loaded)
            else:
                # Dotted import path
                parts = spec.rsplit(".", 1)
                if len(parts) == 2:
                    mod = importlib.import_module(parts[0])
                    obj = getattr(mod, parts[1], None)
                    if obj is not None:
                        if isinstance(obj, Tool):
                            tools.append(obj)
                        elif callable(obj):
                            tools.append(obj)
    return tools


__all__ = [
    "from_yaml",
    "from_dict",
    "load_template",
    "list_templates",
]
