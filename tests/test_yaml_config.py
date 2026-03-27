"""Tests for YAML config loading and agent templates."""

from __future__ import annotations

import os
import tempfile

import pytest

from selectools.providers.stubs import LocalProvider
from selectools.templates import from_dict, from_yaml, list_templates, load_template


class TestFromDict:
    def test_basic_config(self):
        agent = from_dict(
            {"model": "gpt-4o-mini", "system_prompt": "You are helpful."},
            provider=LocalProvider(),
        )
        assert agent.config.model == "gpt-4o-mini"
        assert agent.config.system_prompt == "You are helpful."

    def test_with_retry_config(self):
        agent = from_dict(
            {"model": "test", "retry": {"max_retries": 5, "backoff_seconds": 3.0}},
            provider=LocalProvider(),
        )
        assert agent.config.max_retries == 5
        assert agent.config.retry_backoff_seconds == 3.0

    def test_with_budget_config(self):
        agent = from_dict(
            {"model": "test", "budget": {"max_total_tokens": 50000, "max_cost_usd": 1.0}},
            provider=LocalProvider(),
        )
        assert agent.config.max_total_tokens == 50000
        assert agent.config.max_cost_usd == 1.0

    def test_with_compress_config(self):
        agent = from_dict(
            {"model": "test", "compress": {"enabled": True, "threshold": 0.8}},
            provider=LocalProvider(),
        )
        assert agent.config.compress_context is True
        assert agent.config.compress_threshold == 0.8

    def test_empty_tools_gets_noop(self):
        agent = from_dict({"model": "test"}, provider=LocalProvider())
        assert len(agent.tools) >= 1

    def test_invalid_config_type_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            from_dict("not a dict", provider=LocalProvider())


class TestFromYaml:
    def test_basic_yaml(self):
        yaml_content = """
model: gpt-4o-mini
system_prompt: "You are a test agent."
max_iterations: 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                agent = from_yaml(f.name, provider=LocalProvider())
                assert agent.config.model == "gpt-4o-mini"
                assert agent.config.max_iterations == 3
            finally:
                os.unlink(f.name)

    def test_yaml_with_nested_config(self):
        yaml_content = """
model: gpt-4o
retry:
  max_retries: 5
  backoff_seconds: 2.0
budget:
  max_cost_usd: 0.50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                agent = from_yaml(f.name, provider=LocalProvider())
                assert agent.config.max_retries == 5
                assert agent.config.max_cost_usd == 0.50
            finally:
                os.unlink(f.name)

    def test_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            from_yaml("nonexistent.yaml", provider=LocalProvider())


class TestTemplates:
    def test_list_templates(self):
        templates = list_templates()
        assert "customer_support" in templates
        assert "data_analyst" in templates
        assert "research_assistant" in templates
        assert "code_reviewer" in templates
        assert "rag_chatbot" in templates
        assert len(templates) == 5

    def test_load_customer_support(self):
        agent = load_template("customer_support", provider=LocalProvider())
        assert agent.config.system_prompt is not None
        assert "support" in agent.config.system_prompt.lower()
        assert len(agent.tools) >= 2

    def test_load_data_analyst(self):
        agent = load_template("data_analyst", provider=LocalProvider())
        assert (
            "data" in agent.config.system_prompt.lower()
            or "analyst" in agent.config.system_prompt.lower()
        )

    def test_load_research_assistant(self):
        agent = load_template("research_assistant", provider=LocalProvider())
        assert "research" in agent.config.system_prompt.lower()

    def test_load_code_reviewer(self):
        agent = load_template("code_reviewer", provider=LocalProvider())
        assert "review" in agent.config.system_prompt.lower()

    def test_load_rag_chatbot(self):
        agent = load_template("rag_chatbot", provider=LocalProvider())
        assert "knowledge" in agent.config.system_prompt.lower()

    def test_template_with_model_override(self):
        agent = load_template("customer_support", provider=LocalProvider(), model="gpt-4o")
        assert agent.config.model == "gpt-4o"

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            load_template("nonexistent", provider=LocalProvider())


class TestImports:
    def test_templates_importable(self):
        from selectools.templates import from_dict, from_yaml, list_templates, load_template

        assert callable(from_yaml)
        assert callable(from_dict)
        assert callable(load_template)
        assert callable(list_templates)
