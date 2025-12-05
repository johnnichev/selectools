"""
Core message and role types for the tool-calling library.

These primitives are provider-agnostic and are reused across adapters,
the agent loop, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import base64


class Role(str, Enum):
    """Conversation role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


def _encode_image(image_path: str) -> str:
    """Load and base64-encode an image from disk."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


@dataclass
class Message:
    """
    Conversation message with optional inline image payload and tool metadata.

    The `image_base64` field is populated automatically when `image_path` is
    provided so adapters can forward vision content to providers without
    re-encoding.
    """

    role: Role
    content: str
    image_path: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None
    image_base64: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.image_path:
            self.image_base64 = _encode_image(self.image_path)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-JSON-safe representation for logging or debugging."""
        return {
            "role": self.role.value,
            "content": self.content,
            "image_base64": self.image_base64,
            "tool_name": self.tool_name,
            "tool_result": self.tool_result,
        }


@dataclass
class ToolCall:
    """Structured representation of a parsed tool call."""

    tool_name: str
    parameters: Dict[str, Any]


__all__ = ["Role", "Message", "ToolCall"]
