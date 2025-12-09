"""
Data Analysis Agent Example

This example demonstrates how to build an AI-powered data analysis assistant using Selectools.

Features demonstrated:
- Data loading and inspection tools (CSV, JSON, dictionary)
- Statistical analysis (mean, median, correlation, distribution)
- Data filtering and aggregation
- Natural language interface for data exploration
- Conversation memory to maintain analysis context
- Error handling for data operations

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."

    # Run the example
    python examples/data_analysis_agent.py

Requirements:
    pip install selectools

Note: This example uses built-in Python libraries only. For production use,
      consider adding pandas, numpy, and matplotlib for more advanced analytics.
"""

import json
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

from selectools import (
    Agent,
    AgentConfig,
    ConversationMemory,
    Message,
    OpenAIProvider,
    Role,
    Tool,
    ToolParameter,
)

# ============================================================================
# Mock Data Storage (In production, replace with actual data sources)
# ============================================================================

# Simulated sales data
SALES_DATA = [
    {
        "date": "2024-01-15",
        "product": "Laptop",
        "quantity": 5,
        "revenue": 6499.95,
        "region": "North",
        "category": "Electronics",
    },
    {
        "date": "2024-01-16",
        "product": "Mouse",
        "quantity": 12,
        "revenue": 359.88,
        "region": "South",
        "category": "Electronics",
    },
    {
        "date": "2024-01-17",
        "product": "Keyboard",
        "quantity": 8,
        "revenue": 639.92,
        "region": "East",
        "category": "Electronics",
    },
    {
        "date": "2024-01-18",
        "product": "Monitor",
        "quantity": 3,
        "revenue": 1199.97,
        "region": "West",
        "category": "Electronics",
    },
    {
        "date": "2024-01-19",
        "product": "Desk Chair",
        "quantity": 7,
        "revenue": 1749.93,
        "region": "North",
        "category": "Furniture",
    },
    {
        "date": "2024-01-20",
        "product": "Laptop",
        "quantity": 4,
        "revenue": 5199.96,
        "region": "East",
        "category": "Electronics",
    },
    {
        "date": "2024-01-21",
        "product": "Desk",
        "quantity": 2,
        "revenue": 799.98,
        "region": "South",
        "category": "Furniture",
    },
    {
        "date": "2024-01-22",
        "product": "Mouse",
        "quantity": 15,
        "revenue": 449.85,
        "region": "West",
        "category": "Electronics",
    },
    {
        "date": "2024-01-23",
        "product": "Keyboard",
        "quantity": 10,
        "revenue": 799.90,
        "region": "North",
        "category": "Electronics",
    },
    {
        "date": "2024-01-24",
        "product": "Monitor",
        "quantity": 6,
        "revenue": 2399.94,
        "region": "East",
        "category": "Electronics",
    },
    {
        "date": "2024-01-25",
        "product": "Desk Chair",
        "quantity": 5,
        "revenue": 1249.95,
        "region": "South",
        "category": "Furniture",
    },
    {
        "date": "2024-01-26",
        "product": "Laptop",
        "quantity": 8,
        "revenue": 10399.92,
        "region": "West",
        "category": "Electronics",
    },
    {
        "date": "2024-01-27",
        "product": "Desk",
        "quantity": 3,
        "revenue": 1199.97,
        "region": "North",
        "category": "Furniture",
    },
    {
        "date": "2024-01-28",
        "product": "Mouse",
        "quantity": 20,
        "revenue": 599.80,
        "region": "East",
        "category": "Electronics",
    },
    {
        "date": "2024-01-29",
        "product": "Keyboard",
        "quantity": 12,
        "revenue": 959.88,
        "region": "South",
        "category": "Electronics",
    },
]

# Store loaded datasets (in-memory cache)
DATASETS: Dict[str, List[Dict[str, Any]]] = {"sales": SALES_DATA}


# ============================================================================
# Data Analysis Tools
# ============================================================================


def load_dataset(dataset_name: str) -> str:
    """
    Load a dataset into memory for analysis.

    Args:
        dataset_name: Name of the dataset to load (e.g., 'sales')

    Returns:
        Summary of the loaded dataset
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        return f"Dataset '{dataset_name}' not found. Available datasets: {available}"

    data = DATASETS[dataset_name]

    # Generate summary
    result = f"Dataset '{dataset_name}' loaded successfully!\n\n"
    result += f"Total rows: {len(data)}\n"

    if data:
        columns = list(data[0].keys())
        result += f"Columns: {', '.join(columns)}\n"
        result += f"\nFirst 3 rows:\n"

        for idx, row in enumerate(data[:3], 1):
            result += f"{idx}. {json.dumps(row, indent=2)}\n"

    return result


def get_column_stats(dataset_name: str, column_name: str) -> str:
    """
    Calculate statistics for a numeric column.

    Args:
        dataset_name: Name of the dataset
        column_name: Name of the column to analyze

    Returns:
        Statistical summary (mean, median, min, max, std)
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in DATASETS:
        return f"Dataset '{dataset_name}' not found."

    data = DATASETS[dataset_name]

    # Extract column values
    try:
        values = [row[column_name] for row in data if column_name in row]
    except KeyError:
        available_columns = ", ".join(data[0].keys()) if data else "none"
        return f"Column '{column_name}' not found. Available: {available_columns}"

    if not values:
        return f"Column '{column_name}' has no data."

    # Check if numeric
    if not all(isinstance(v, (int, float)) for v in values):
        return f"Column '{column_name}' is not numeric. Use 'count_values' for categorical data."

    # Calculate statistics
    result = f"Statistics for '{column_name}' in '{dataset_name}':\n\n"
    result += f"Count: {len(values)}\n"
    result += f"Mean: {statistics.mean(values):.2f}\n"
    result += f"Median: {statistics.median(values):.2f}\n"
    result += f"Min: {min(values):.2f}\n"
    result += f"Max: {max(values):.2f}\n"

    if len(values) > 1:
        result += f"Std Dev: {statistics.stdev(values):.2f}\n"

    result += f"Sum: {sum(values):.2f}"

    return result


def filter_data(dataset_name: str, column_name: str, operator: str, value: str) -> str:
    """
    Filter dataset by a condition and show results.

    Args:
        dataset_name: Name of the dataset to filter
        column_name: Column to filter on
        operator: Comparison operator (equals, greater_than, less_than, contains)
        value: Value to compare against

    Returns:
        Filtered data summary
    """
    dataset_name = dataset_name.lower().strip()
    operator = operator.lower().strip()

    if dataset_name not in DATASETS:
        return f"Dataset '{dataset_name}' not found."

    data = DATASETS[dataset_name]

    # Parse value type
    try:
        # Try to convert to number
        numeric_value = float(value)
        value = numeric_value
    except ValueError:
        # Keep as string
        pass

    # Apply filter
    filtered = []
    for row in data:
        if column_name not in row:
            continue

        row_value = row[column_name]

        match = False
        if operator == "equals":
            match = row_value == value
        elif operator == "greater_than":
            match = float(row_value) > float(value)
        elif operator == "less_than":
            match = float(row_value) < float(value)
        elif operator == "contains":
            match = str(value).lower() in str(row_value).lower()
        else:
            return f"Unknown operator '{operator}'. Use: equals, greater_than, less_than, contains"

        if match:
            filtered.append(row)

    # Generate result
    result = f"Filter: {column_name} {operator} {value}\n"
    result += f"Matched {len(filtered)} out of {len(data)} rows\n\n"

    if filtered:
        result += "Sample results (first 5):\n"
        for idx, row in enumerate(filtered[:5], 1):
            result += f"{idx}. {json.dumps(row)}\n"
    else:
        result += "No matching rows found."

    return result


def group_by(dataset_name: str, group_column: str, agg_column: str, operation: str) -> str:
    """
    Group data by a column and aggregate.

    Args:
        dataset_name: Name of the dataset
        group_column: Column to group by
        agg_column: Column to aggregate
        operation: Aggregation operation (sum, mean, count, min, max)

    Returns:
        Grouped and aggregated results
    """
    dataset_name = dataset_name.lower().strip()
    operation = operation.lower().strip()

    if dataset_name not in DATASETS:
        return f"Dataset '{dataset_name}' not found."

    data = DATASETS[dataset_name]

    # Group data
    groups: Dict[Any, List[Any]] = {}
    for row in data:
        if group_column not in row:
            continue

        key = row[group_column]
        if key not in groups:
            groups[key] = []

        if agg_column in row:
            groups[key].append(row[agg_column])

    # Aggregate
    results = {}
    for key, values in groups.items():
        if not values:
            results[key] = 0
            continue

        if operation == "sum":
            results[key] = sum(values)
        elif operation == "mean":
            results[key] = statistics.mean(values)
        elif operation == "count":
            results[key] = len(values)
        elif operation == "min":
            results[key] = min(values)
        elif operation == "max":
            results[key] = max(values)
        else:
            return f"Unknown operation '{operation}'. Use: sum, mean, count, min, max"

    # Format output
    result = f"Group by '{group_column}', aggregate '{agg_column}' using {operation}:\n\n"

    # Sort by value descending
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for key, value in sorted_results:
        if isinstance(value, float):
            result += f"{key}: {value:.2f}\n"
        else:
            result += f"{key}: {value}\n"

    return result


def count_values(dataset_name: str, column_name: str) -> str:
    """
    Count unique values in a column (for categorical data).

    Args:
        dataset_name: Name of the dataset
        column_name: Column to count values for

    Returns:
        Value counts sorted by frequency
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in DATASETS:
        return f"Dataset '{dataset_name}' not found."

    data = DATASETS[dataset_name]

    # Count values
    counts: Dict[Any, int] = {}
    for row in data:
        if column_name not in row:
            continue

        value = row[column_name]
        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return f"Column '{column_name}' not found or has no data."

    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    result = f"Value counts for '{column_name}' in '{dataset_name}':\n\n"
    result += f"Total unique values: {len(sorted_counts)}\n\n"

    for value, count in sorted_counts:
        percentage = (count / len(data)) * 100
        result += f"{value}: {count} ({percentage:.1f}%)\n"

    return result


def calculate_correlation(dataset_name: str, column1: str, column2: str) -> str:
    """
    Calculate correlation between two numeric columns.

    Args:
        dataset_name: Name of the dataset
        column1: First column name
        column2: Second column name

    Returns:
        Correlation coefficient and interpretation
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in DATASETS:
        return f"Dataset '{dataset_name}' not found."

    data = DATASETS[dataset_name]

    # Extract values
    pairs = []
    for row in data:
        if column1 in row and column2 in row:
            try:
                v1 = float(row[column1])
                v2 = float(row[column2])
                pairs.append((v1, v2))
            except (ValueError, TypeError):
                continue

    if len(pairs) < 2:
        return f"Not enough numeric data points to calculate correlation."

    # Calculate correlation
    x_values = [p[0] for p in pairs]
    y_values = [p[1] for p in pairs]

    correlation = statistics.correlation(x_values, y_values)

    # Interpret correlation
    if abs(correlation) > 0.7:
        strength = "strong"
    elif abs(correlation) > 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if correlation > 0 else "negative"

    result = f"Correlation between '{column1}' and '{column2}':\n\n"
    result += f"Coefficient: {correlation:.3f}\n"
    result += f"Interpretation: {strength.title()} {direction} correlation\n\n"

    if abs(correlation) > 0.7:
        result += f"There is a strong {direction} relationship between {column1} and {column2}."
    elif abs(correlation) > 0.4:
        result += f"There is a moderate {direction} relationship between {column1} and {column2}."
    else:
        result += f"The relationship between {column1} and {column2} is weak."

    return result


# ============================================================================
# Create Data Analysis Agent
# ============================================================================


def create_analysis_agent() -> Agent:
    """Create a data analysis agent with analytical tools."""

    tools = [
        Tool(
            name="load_dataset",
            description="Load a dataset into memory for analysis. Always start by loading the dataset before performing analysis.",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset to load (e.g., 'sales')",
                    required=True,
                )
            ],
            function=load_dataset,
        ),
        Tool(
            name="get_column_stats",
            description="Calculate statistics (mean, median, min, max, std) for a numeric column.",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset",
                    required=True,
                ),
                ToolParameter(
                    name="column_name",
                    type="string",
                    description="Name of the numeric column to analyze",
                    required=True,
                ),
            ],
            function=get_column_stats,
        ),
        Tool(
            name="filter_data",
            description="Filter the dataset by a condition and show matching rows.",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset",
                    required=True,
                ),
                ToolParameter(
                    name="column_name",
                    type="string",
                    description="Column to filter on",
                    required=True,
                ),
                ToolParameter(
                    name="operator",
                    type="string",
                    description="Comparison operator: equals, greater_than, less_than, contains",
                    required=True,
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="Value to compare against",
                    required=True,
                ),
            ],
            function=filter_data,
        ),
        Tool(
            name="group_by",
            description="Group data by a column and perform aggregation (sum, mean, count, min, max).",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset",
                    required=True,
                ),
                ToolParameter(
                    name="group_column",
                    type="string",
                    description="Column to group by",
                    required=True,
                ),
                ToolParameter(
                    name="agg_column",
                    type="string",
                    description="Column to aggregate",
                    required=True,
                ),
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Aggregation operation: sum, mean, count, min, max",
                    required=True,
                ),
            ],
            function=group_by,
        ),
        Tool(
            name="count_values",
            description="Count unique values in a categorical column and show distribution.",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset",
                    required=True,
                ),
                ToolParameter(
                    name="column_name",
                    type="string",
                    description="Column to count values for",
                    required=True,
                ),
            ],
            function=count_values,
        ),
        Tool(
            name="calculate_correlation",
            description="Calculate the correlation coefficient between two numeric columns.",
            parameters=[
                ToolParameter(
                    name="dataset_name",
                    type="string",
                    description="Name of the dataset",
                    required=True,
                ),
                ToolParameter(
                    name="column1",
                    type="string",
                    description="First column name",
                    required=True,
                ),
                ToolParameter(
                    name="column2",
                    type="string",
                    description="Second column name",
                    required=True,
                ),
            ],
            function=calculate_correlation,
        ),
    ]

    from selectools.models import OpenAI

    provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
    memory = ConversationMemory(max_messages=20)

    config = AgentConfig(
        max_iterations=5,
        verbose=True,
        temperature=0.3,  # Lower temperature for precise data analysis
        max_tokens=800,
    )

    system_prompt = """You are an expert data analyst assistant. You help users explore and understand their data through analysis.

Your responsibilities:
- Load datasets when users want to analyze data
- Perform statistical analysis and aggregations
- Answer questions about data patterns and trends
- Provide clear, actionable insights from data
- Guide users through exploratory data analysis

Guidelines:
- Always load the dataset first before analyzing it
- Show actual numbers and statistics, not just descriptions
- When users ask vague questions, suggest specific analyses
- Break complex analyses into steps
- Explain findings in clear, non-technical language when appropriate
- If a column doesn't exist, suggest similar column names

Remember: Your goal is to help users understand their data, not just run tools."""

    return Agent(
        tools=tools,
        provider=provider,
        memory=memory,
        config=config,
    )


# ============================================================================
# Example Analyses
# ============================================================================


def example_basic_stats():
    """Example: Get basic statistics for the sales dataset."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Statistics")
    print("=" * 70)

    agent = create_analysis_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="Analyze the sales dataset. What's the average revenue and total sales?",
            )
        ]
    )

    print(f"\n📊 Analyst: {response.content}\n")


def example_grouping():
    """Example: Group by analysis."""
    print("=" * 70)
    print("EXAMPLE 2: Sales by Region")
    print("=" * 70)

    agent = create_analysis_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="Which region has the highest sales? Show me revenue by region.",
            )
        ]
    )

    print(f"\n📊 Analyst: {response.content}\n")


def example_filtering():
    """Example: Filter and analyze specific data."""
    print("=" * 70)
    print("EXAMPLE 3: High-Value Orders")
    print("=" * 70)

    agent = create_analysis_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="Show me all electronics sales where revenue was greater than 1000",
            )
        ]
    )

    print(f"\n📊 Analyst: {response.content}\n")


def example_multi_turn_analysis():
    """Example: Multi-turn conversation with context."""
    print("=" * 70)
    print("EXAMPLE 4: Multi-Turn Analysis (With Memory)")
    print("=" * 70)

    agent = create_analysis_agent()

    # Initial question
    response = agent.run(
        [Message(role=Role.USER, content="Load the sales data and tell me what products we sell")]
    )
    print(f"\n📊 Analyst: {response.content}\n")

    # Follow-up 1 (using memory)
    response = agent.run(
        [Message(role=Role.USER, content="Which product generates the most revenue?")]
    )
    print(f"📊 Analyst: {response.content}\n")

    # Follow-up 2
    response = agent.run(
        [Message(role=Role.USER, content="Is there a correlation between quantity and revenue?")]
    )
    print(f"📊 Analyst: {response.content}\n")


def example_category_analysis():
    """Example: Analyze categorical data."""
    print("=" * 70)
    print("EXAMPLE 5: Category Distribution")
    print("=" * 70)

    agent = create_analysis_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="What's the distribution of sales by category? How many electronics vs furniture sales?",
            )
        ]
    )

    print(f"\n📊 Analyst: {response.content}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all data analysis examples."""
    print("\n📊 Data Analysis Agent Examples\n")

    example_basic_stats()
    print("\n")

    example_grouping()
    print("\n")

    example_filtering()
    print("\n")

    example_category_analysis()
    print("\n")

    example_multi_turn_analysis()

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\n💡 Tip: In production, integrate with pandas, numpy, and matplotlib")
    print("   for more advanced analytics and visualization capabilities.")


if __name__ == "__main__":
    main()
