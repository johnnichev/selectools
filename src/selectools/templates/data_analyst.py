"""Data analyst agent template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import Provider

from ..agent.config import AgentConfig
from ..agent.core import Agent
from ..tools.decorators import tool


@tool(description="Execute a SQL query against the database")
def run_query(sql: str) -> str:
    """Execute SQL and return results. Supports SELECT, DESCRIBE, SHOW."""
    return f"Query executed: {sql[:100]}... | 42 rows returned"


@tool(description="Generate a data summary with statistics")
def summarize_data(dataset: str) -> str:
    """Compute summary statistics for a dataset."""
    return f"Summary of '{dataset}': mean=45.2, median=42.0, std=12.3, min=5, max=98, nulls=3%"


@tool(description="Create a chart or visualization description")
def create_chart(chart_type: str, data_description: str) -> str:
    """Describe a chart visualization. Types: bar, line, scatter, pie, histogram."""
    return f"Chart created: {chart_type} chart showing {data_description}"


SYSTEM_PROMPT = """You are a data analyst assistant.

Your responsibilities:
1. Answer data questions by querying the database with run_query
2. Provide statistical summaries using summarize_data
3. Suggest and describe visualizations with create_chart

Guidelines:
- Always start by understanding what data the user needs
- Write clean, readable SQL
- Explain your findings in plain language
- Suggest visualizations when they would clarify the data
- Mention caveats or limitations in the data"""


def build(provider: "Provider", **overrides: Any) -> Agent:
    """Build a data analyst agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o-mini"),
        "max_iterations": overrides.pop("max_iterations", 6),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[run_query, summarize_data, create_chart],
        config=AgentConfig(**config_kwargs),
    )
