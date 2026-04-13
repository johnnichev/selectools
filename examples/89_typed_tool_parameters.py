"""
Typed Tool Parameters — list[str], dict[str, str], list[int].

Since v0.22.0 (BUG-29), selectools emits proper JSON schema for typed
collections. OpenAI strict mode requires `items` / `additionalProperties`
in the schema — bare `list` / `dict` without type parameters are rejected.

Prerequisites: No API key needed (uses LocalProvider)
Run: python examples/89_typed_tool_parameters.py
"""

from selectools import Agent
from selectools.providers.stubs import LocalProvider
from selectools.tools import tool


@tool(description="Tag a document with labels")
def tag_document(doc_id: str, tags: list[str]) -> str:
    """Tags emits items: {type: string} in the schema."""
    return f"Tagged {doc_id} with {', '.join(tags)}"


@tool(description="Score items by category")
def score_items(category: str, scores: list[int]) -> str:
    """Scores emits items: {type: integer} in the schema."""
    return f"{category}: total={sum(scores)}, avg={sum(scores)/len(scores):.1f}"


@tool(description="Update key-value settings")
def update_settings(config: dict[str, str]) -> str:
    """Config emits additionalProperties: {type: string} in the schema."""
    return f"Updated {len(config)} settings: {config}"


def main() -> None:
    agent = Agent(
        tools=[tag_document, score_items, update_settings],
        provider=LocalProvider(),
    )

    # Inspect the generated schemas
    for t in agent.tools:
        schema = t.schema()
        print(f"\n{t.name}:")
        for pname, pschema in schema["parameters"]["properties"].items():
            print(f"  {pname}: {pschema}")

    # Show that list[str] produces {"type": "array", "items": {"type": "string"}}
    tag_schema = tag_document.schema()["parameters"]["properties"]["tags"]
    assert "items" in tag_schema, "list[str] must produce items in schema"
    assert tag_schema["items"]["type"] == "string"
    print("\n✓ Typed collection schemas are correct for OpenAI strict mode")


if __name__ == "__main__":
    main()
