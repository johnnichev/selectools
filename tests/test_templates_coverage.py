"""Tests for templates/__init__.py — YAML loading, from_dict, list_templates, _resolve_provider."""

from __future__ import annotations

import os
import tempfile

import pytest

from selectools.providers.stubs import LocalProvider
from selectools.templates import from_dict, from_yaml, list_templates, load_template

# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_returns_sorted_list(self):
        templates = list_templates()
        assert templates == sorted(templates)

    def test_contains_all_expected(self):
        templates = list_templates()
        expected = [
            "code_reviewer",
            "customer_support",
            "data_analyst",
            "rag_chatbot",
            "research_assistant",
        ]
        assert templates == expected


# ---------------------------------------------------------------------------
# load_template edge cases
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    def test_unknown_template_error_message(self):
        with pytest.raises(ValueError, match="Unknown template") as exc_info:
            load_template("nonexistent", provider=LocalProvider())
        # Error message should list available templates
        assert "customer_support" in str(exc_info.value)

    def test_all_templates_load(self):
        """Every listed template should load without error."""
        for name in list_templates():
            agent = load_template(name, provider=LocalProvider())
            assert agent is not None
            assert agent.config.system_prompt is not None

    def test_template_override_model(self):
        agent = load_template(
            "customer_support",
            provider=LocalProvider(),
            model="custom-model",
        )
        assert agent.config.model == "custom-model"


# ---------------------------------------------------------------------------
# from_dict edge cases
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            from_dict("not a dict")

    def test_invalid_type_list_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            from_dict([1, 2, 3])

    def test_minimal_dict(self):
        agent = from_dict({}, provider=LocalProvider())
        assert agent is not None

    def test_all_direct_fields(self):
        agent = from_dict(
            {
                "model": "gpt-4o",
                "temperature": 0.5,
                "max_tokens": 1000,
                "max_iterations": 5,
                "system_prompt": "Test agent",
                "verbose": True,
                "stream": True,
            },
            provider=LocalProvider(),
        )
        assert agent.config.model == "gpt-4o"
        assert agent.config.temperature == 0.5
        assert agent.config.max_tokens == 1000
        assert agent.config.max_iterations == 5
        assert agent.config.system_prompt == "Test agent"

    def test_with_coherence_config(self):
        agent = from_dict(
            {"coherence": {"enabled": True}},
            provider=LocalProvider(),
        )
        assert agent is not None

    def test_with_trace_config(self):
        agent = from_dict(
            {"trace": {"tool_result_chars": 500}},
            provider=LocalProvider(),
        )
        assert agent is not None

    def test_with_all_nested_configs(self):
        agent = from_dict(
            {
                "model": "gpt-4o",
                "retry": {"max_retries": 3},
                "budget": {"max_cost_usd": 1.0},
                "compress": {"enabled": True},
            },
            provider=LocalProvider(),
        )
        assert agent.config.max_retries == 3
        assert agent.config.max_cost_usd == 1.0


# ---------------------------------------------------------------------------
# from_yaml edge cases
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            from_yaml("/tmp/nonexistent_config.yaml", provider=LocalProvider())

    def test_invalid_yaml_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- just\n- a\n- list\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="must be a dict"):
                    from_yaml(f.name, provider=LocalProvider())
            finally:
                os.unlink(f.name)

    def test_yaml_with_tools_dotted_path(self):
        yaml_content = """
model: gpt-4o
tools:
  - selectools.toolbox.datetime_tools.get_current_time
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                agent = from_yaml(f.name, provider=LocalProvider())
                assert len(agent.tools) >= 1
            finally:
                os.unlink(f.name)

    def test_yaml_without_provider_creates_default(self):
        """When no provider arg and YAML says 'local', should create LocalProvider."""
        yaml_content = """
provider: local
model: test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                agent = from_yaml(f.name)
                assert agent is not None
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------


class TestResolveProvider:
    def test_local_provider(self):
        from selectools.templates import _resolve_provider

        provider = _resolve_provider("local")
        from selectools.providers.stubs import LocalProvider

        assert isinstance(provider, LocalProvider)

    def test_unknown_provider_raises(self):
        from selectools.templates import _resolve_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            _resolve_provider("unknown_provider")

    def test_unknown_provider_lists_available(self):
        from selectools.templates import _resolve_provider

        with pytest.raises(ValueError) as exc_info:
            _resolve_provider("bad")
        assert "openai" in str(exc_info.value)
        assert "local" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_tool_object_passthrough(self):
        from selectools.templates import _resolve_tools
        from selectools.tools.decorators import tool

        @tool(description="test tool")
        def my_tool() -> str:
            return "ok"

        result = _resolve_tools([my_tool])
        assert len(result) == 1

    def test_dotted_path_resolves(self):
        from selectools.templates import _resolve_tools

        result = _resolve_tools(["selectools.toolbox.datetime_tools.get_current_time"])
        assert len(result) == 1

    def test_invalid_dotted_path_raises(self):
        from selectools.templates import _resolve_tools

        with pytest.raises(ModuleNotFoundError):
            _resolve_tools(["selectools.nonexistent.module.func"])

    def test_valid_module_missing_attr_ignored(self):
        from selectools.templates import _resolve_tools

        # Module exists but attr does not — getattr returns None, so no tool added
        result = _resolve_tools(["selectools.toolbox.datetime_tools.no_such_func"])
        assert len(result) == 0

    def test_path_traversal_blocked(self):
        from pathlib import Path

        from selectools.templates import _resolve_tools

        with pytest.raises(ValueError, match="escapes config directory"):
            _resolve_tools(
                ["../../etc/passwd"],
                base_dir=Path("/tmp/config"),
            )
