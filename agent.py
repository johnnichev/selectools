"""
Convenience re-export module for the AI tool calling framework.

This allows imports like `from agent import Agent` while the core
implementation lives under `src/tool_calling/agent.py`.
"""

from src.tool_calling.agent import Agent, Message, Role, Tool, ToolParameter

__all__ = ["Agent", "Message", "Role", "Tool", "ToolParameter"]

