"""Bridge between MCP tool definitions and selectools Tool objects."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..tools.base import Tool, ToolParameter

# JSON Schema type → Python type mapping
_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_params(input_schema: Dict[str, Any]) -> List[ToolParameter]:
    """Convert a JSON Schema inputSchema to a list of ToolParameter."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    params: List[ToolParameter] = []

    for name, prop in properties.items():
        # Handle type (can be string or list for nullable)
        raw_type = prop.get("type", "string")
        if isinstance(raw_type, list):
            # ["string", "null"] → use the non-null type
            raw_type = next((t for t in raw_type if t != "null"), "string")

        param_type = _TYPE_MAP.get(raw_type, str)
        description = prop.get("description", "")
        enum = prop.get("enum")

        # Include default in description if present
        if "default" in prop:
            default_str = repr(prop["default"])
            if description:
                description += f" (default: {default_str})"
            else:
                description = f"Default: {default_str}"

        param = ToolParameter(
            name=name,
            param_type=param_type,
            description=description,
            required=name in required,
            enum=enum,
        )
        # Store the full JSON Schema for this property for passthrough
        param._raw_schema = prop  # type: ignore[attr-defined]
        params.append(param)

    return params


def mcp_to_tool(
    mcp_tool: Any,
    call_fn: Any,  # async callable (name, arguments) -> str
    *,
    server_name: str = "",
    prefix: bool = False,
) -> Tool:
    """Convert an MCP tool definition to a selectools Tool.

    Args:
        mcp_tool: An MCP Tool object (from mcp.types.Tool).
        call_fn: Async callable that calls the MCP tool and returns text.
        server_name: Name of the MCP server (for prefixing).
        prefix: Whether to prefix the tool name with server_name.

    Returns:
        A selectools Tool object that proxies calls to the MCP server.
    """
    name = mcp_tool.name
    if prefix and server_name:
        name = f"{server_name}_{name}"

    description = mcp_tool.description or ""
    input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
    params = _schema_to_params(input_schema)

    # Check annotations for policy hints
    annotations: Dict[str, Any] = {}
    if hasattr(mcp_tool, "annotations") and mcp_tool.annotations:
        ann = mcp_tool.annotations
        if hasattr(ann, "readOnlyHint"):
            annotations["read_only"] = ann.readOnlyHint
        if hasattr(ann, "destructiveHint"):
            annotations["destructive"] = ann.destructiveHint
        if hasattr(ann, "idempotentHint"):
            annotations["idempotent"] = ann.idempotentHint

    # Create wrapper function that the Tool will call
    async def _mcp_call(**kwargs: Any) -> str:
        result = await call_fn(name=mcp_tool.name, arguments=kwargs)
        return str(result)

    tool = Tool(
        name=name,
        description=description,
        parameters=params,
        function=_mcp_call,
        _skip_validation=True,
    )
    # Mark as MCP-sourced for special handling
    tool._mcp_server = server_name  # type: ignore[attr-defined]
    tool._mcp_annotations = annotations  # type: ignore[attr-defined]
    tool._mcp_original_name = mcp_tool.name  # type: ignore[attr-defined]

    return tool


def tool_to_mcp_schema(tool: Tool) -> Dict[str, Any]:
    """Convert a selectools Tool to an MCP tool definition dict.

    Returns a dict matching the MCP Tool schema format.
    """
    schema = tool.schema()
    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "inputSchema": schema.get("parameters", {"type": "object", "properties": {}}),
    }
