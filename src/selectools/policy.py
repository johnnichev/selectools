"""
Tool execution policy engine with allow / review / deny rules.

Evaluates declarative rules before every tool execution to control which
tools may run freely, which require approval, and which are blocked.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    DENY = "deny"


@dataclass
class PolicyResult:
    """Outcome of evaluating a tool call against the policy."""

    decision: PolicyDecision
    reason: str = ""
    matched_rule: str = ""


@dataclass
class ToolPolicy:
    """Declarative allow / review / deny rules for tool execution.

    Rules use glob patterns matched against tool names.
    Evaluation order: deny -> review -> allow -> default (review).

    Example::

        policy = ToolPolicy(
            allow=["search_*", "read_*", "get_*"],
            review=["send_*", "create_*", "update_*"],
            deny=["delete_*", "drop_*"],
        )
    """

    allow: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)
    deny_when: List[Dict[str, str]] = field(default_factory=list)

    def evaluate(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> PolicyResult:
        """Return the policy decision for a tool call.

        Evaluation order:
        1. ``deny_when`` argument-level conditions
        2. ``deny`` glob patterns
        3. ``review`` glob patterns
        4. ``allow`` glob patterns
        5. Default → review
        """
        if tool_args is not None:
            for rule in self.deny_when:
                if fnmatch.fnmatch(tool_name, rule.get("tool", "*")):
                    arg_name = rule.get("arg", "")
                    pattern = rule.get("pattern", "")
                    if arg_name in tool_args and fnmatch.fnmatch(str(tool_args[arg_name]), pattern):
                        return PolicyResult(
                            decision=PolicyDecision.DENY,
                            reason=f"Argument '{arg_name}' matches deny condition",
                            matched_rule=f"deny_when: {rule}",
                        )

        for pattern in self.deny:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=f"Tool matches deny pattern '{pattern}'",
                    matched_rule=f"deny:{pattern}",
                )

        for pattern in self.review:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyResult(
                    decision=PolicyDecision.REVIEW,
                    reason=f"Tool matches review pattern '{pattern}'",
                    matched_rule=f"review:{pattern}",
                )

        for pattern in self.allow:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyResult(
                    decision=PolicyDecision.ALLOW,
                    reason=f"Tool matches allow pattern '{pattern}'",
                    matched_rule=f"allow:{pattern}",
                )

        return PolicyResult(
            decision=PolicyDecision.REVIEW,
            reason="No matching rule; defaulting to review",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicy":
        return cls(
            allow=data.get("allow", []),
            review=data.get("review", []),
            deny=data.get("deny", []),
            deny_when=data.get("deny_when", []),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ToolPolicy":
        """Load a ``ToolPolicy`` from a YAML file.

        The YAML file should follow this structure::

            allow:
              - search_*
              - read_*
            review:
              - send_*
              - create_*
            deny:
              - delete_*
            deny_when:
              - tool: send_email
                arg: to
                pattern: "*@external.com"

        Requires ``pyyaml`` (``pip install pyyaml``).

        Args:
            path: Path to the YAML policy file.

        Raises:
            ImportError: If ``pyyaml`` is not installed.
            FileNotFoundError: If the file does not exist.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required to load policies from YAML files. "
                "Install it with: pip install pyyaml"
            ) from exc

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Policy file not found: {yaml_path}")

        with yaml_path.open() as fh:
            data: Dict[str, Any] = yaml.safe_load(fh) or {}

        return cls.from_dict(data)


__all__ = ["ToolPolicy", "PolicyDecision", "PolicyResult"]
