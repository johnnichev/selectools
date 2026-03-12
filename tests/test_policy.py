"""
Unit tests for ToolPolicy: evaluate(), from_dict(), from_yaml(), glob patterns,
deny_when conditions, and default behavior.

Previously only covered by E2E tests that were always skipped in CI.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import pytest

from selectools.policy import PolicyDecision, PolicyResult, ToolPolicy


class TestPolicyDecisionEnum:
    def test_values(self) -> None:
        assert PolicyDecision.ALLOW == "allow"
        assert PolicyDecision.REVIEW == "review"
        assert PolicyDecision.DENY == "deny"


class TestPolicyResult:
    def test_defaults(self) -> None:
        r = PolicyResult(decision=PolicyDecision.ALLOW)
        assert r.reason == ""
        assert r.matched_rule == ""


class TestEvaluateGlobPatterns:
    def test_allow_exact_match(self) -> None:
        policy = ToolPolicy(allow=["search"])
        result = policy.evaluate("search")
        assert result.decision == PolicyDecision.ALLOW
        assert "allow" in result.matched_rule

    def test_allow_wildcard(self) -> None:
        policy = ToolPolicy(allow=["get_*"])
        result = policy.evaluate("get_weather")
        assert result.decision == PolicyDecision.ALLOW

    def test_deny_exact_match(self) -> None:
        policy = ToolPolicy(deny=["delete_user"])
        result = policy.evaluate("delete_user")
        assert result.decision == PolicyDecision.DENY

    def test_deny_wildcard(self) -> None:
        policy = ToolPolicy(deny=["drop_*"])
        result = policy.evaluate("drop_table")
        assert result.decision == PolicyDecision.DENY

    def test_review_wildcard(self) -> None:
        policy = ToolPolicy(review=["send_*"])
        result = policy.evaluate("send_email")
        assert result.decision == PolicyDecision.REVIEW

    def test_no_match_defaults_to_review(self) -> None:
        policy = ToolPolicy(allow=["search"], deny=["delete"])
        result = policy.evaluate("unknown_tool")
        assert result.decision == PolicyDecision.REVIEW
        assert "default" in result.reason.lower()

    def test_empty_policy_defaults_to_review(self) -> None:
        policy = ToolPolicy()
        result = policy.evaluate("any_tool")
        assert result.decision == PolicyDecision.REVIEW


class TestEvaluationOrder:
    """Deny takes priority over review, which takes priority over allow."""

    def test_deny_beats_allow(self) -> None:
        policy = ToolPolicy(allow=["delete_*"], deny=["delete_*"])
        result = policy.evaluate("delete_user")
        assert result.decision == PolicyDecision.DENY

    def test_deny_beats_review(self) -> None:
        policy = ToolPolicy(review=["delete_*"], deny=["delete_*"])
        result = policy.evaluate("delete_user")
        assert result.decision == PolicyDecision.DENY

    def test_review_beats_allow(self) -> None:
        policy = ToolPolicy(allow=["send_*"], review=["send_*"])
        result = policy.evaluate("send_email")
        assert result.decision == PolicyDecision.REVIEW


class TestDenyWhen:
    def test_deny_when_arg_matches(self) -> None:
        policy = ToolPolicy(
            allow=["send_email"],
            deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*@external.com"}],
        )
        result = policy.evaluate("send_email", {"to": "hacker@external.com"})
        assert result.decision == PolicyDecision.DENY

    def test_deny_when_arg_no_match(self) -> None:
        policy = ToolPolicy(
            allow=["send_email"],
            deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*@external.com"}],
        )
        result = policy.evaluate("send_email", {"to": "friend@company.com"})
        assert result.decision == PolicyDecision.ALLOW

    def test_deny_when_tool_wildcard(self) -> None:
        policy = ToolPolicy(
            allow=["*"],
            deny_when=[{"tool": "*", "arg": "path", "pattern": "/etc/*"}],
        )
        result = policy.evaluate("read_file", {"path": "/etc/passwd"})
        assert result.decision == PolicyDecision.DENY

    def test_deny_when_no_args(self) -> None:
        policy = ToolPolicy(
            allow=["send_email"],
            deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*"}],
        )
        result = policy.evaluate("send_email")
        assert result.decision == PolicyDecision.ALLOW

    def test_deny_when_missing_arg(self) -> None:
        policy = ToolPolicy(
            allow=["send_email"],
            deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*"}],
        )
        result = policy.evaluate("send_email", {"subject": "Hello"})
        assert result.decision == PolicyDecision.ALLOW

    def test_deny_when_beats_everything(self) -> None:
        policy = ToolPolicy(
            allow=["send_email"],
            deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*"}],
        )
        result = policy.evaluate("send_email", {"to": "anyone"})
        assert result.decision == PolicyDecision.DENY


class TestFromDict:
    def test_basic(self) -> None:
        data: Dict[str, Any] = {
            "allow": ["search_*"],
            "review": ["send_*"],
            "deny": ["delete_*"],
        }
        policy = ToolPolicy.from_dict(data)
        assert policy.evaluate("search_web").decision == PolicyDecision.ALLOW
        assert policy.evaluate("send_email").decision == PolicyDecision.REVIEW
        assert policy.evaluate("delete_db").decision == PolicyDecision.DENY

    def test_empty_dict(self) -> None:
        policy = ToolPolicy.from_dict({})
        assert policy.allow == []
        assert policy.review == []
        assert policy.deny == []
        assert policy.deny_when == []

    def test_with_deny_when(self) -> None:
        data: Dict[str, Any] = {
            "allow": ["*"],
            "deny_when": [{"tool": "execute", "arg": "cmd", "pattern": "rm *"}],
        }
        policy = ToolPolicy.from_dict(data)
        result = policy.evaluate("execute", {"cmd": "rm -rf /"})
        assert result.decision == PolicyDecision.DENY


class TestFromYaml:
    def test_load_yaml(self) -> None:
        yaml_content = """
allow:
  - search_*
  - read_*
review:
  - send_*
deny:
  - delete_*
deny_when:
  - tool: send_email
    arg: to
    pattern: "*@evil.com"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            policy = ToolPolicy.from_yaml(path)
            assert policy.evaluate("search_web").decision == PolicyDecision.ALLOW
            assert policy.evaluate("send_email").decision == PolicyDecision.REVIEW
            assert policy.evaluate("delete_user").decision == PolicyDecision.DENY
            assert (
                policy.evaluate("send_email", {"to": "x@evil.com"}).decision == PolicyDecision.DENY
            )
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ToolPolicy.from_yaml("/nonexistent/path.yaml")

    def test_empty_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            policy = ToolPolicy.from_yaml(path)
            assert policy.allow == []
        finally:
            os.unlink(path)
