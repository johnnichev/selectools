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

    def test_whitespace_only_tool_name_is_denied(self) -> None:
        """Regression: whitespace-only tool names must be DENY, not bypass the empty guard.

        Previously, '   '.strip() == '' but 'not \"   \"' is False so the empty-name
        check was skipped, allowing whitespace names to match wildcard allow patterns.
        """
        policy = ToolPolicy(allow=["*"])
        result = policy.evaluate("   ")
        assert result.decision == PolicyDecision.DENY
        assert "empty" in result.reason.lower()

    def test_tabs_and_newlines_in_tool_name_are_denied(self) -> None:
        """Regression: tool names with only tabs/newlines must also be denied."""
        policy = ToolPolicy(allow=["*"])
        result = policy.evaluate("\t\n")
        assert result.decision == PolicyDecision.DENY


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

    def test_yaml_list_raises_value_error(self) -> None:
        """Regression: YAML file containing a list (not a dict) previously raised AttributeError.

        yaml.safe_load returns a list for ``- item`` YAML; calling .get() on a list
        raises AttributeError. The fix validates that the parsed value is a dict.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- search_*\n- read_*\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a mapping"):
                ToolPolicy.from_yaml(path)
        finally:
            os.unlink(path)

    def test_yaml_scalar_raises_value_error(self) -> None:
        """Regression: YAML file containing a bare scalar also raised AttributeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("just_a_string\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a mapping"):
                ToolPolicy.from_yaml(path)
        finally:
            os.unlink(path)

    def test_yaml_string_allow_raises_value_error(self) -> None:
        """Regression: YAML with 'allow: search_*' (string, not list) must raise ValueError.

        When allow is a bare string 'search_*', from_dict iterates characters including '*',
        which as a glob pattern matches ALL tool names — silently bypassing the entire policy.
        """
        yaml_content = "allow: search_*\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a list of strings"):
                ToolPolicy.from_yaml(path)
        finally:
            os.unlink(path)

    def test_yaml_deny_when_string_raises_value_error(self) -> None:
        """Regression: YAML with 'deny_when: search_*' (string) must raise ValueError.

        Previously, iterating a string in the deny_when loop caused AttributeError on .get().
        """
        yaml_content = "deny_when: search_*\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a list of mappings"):
                ToolPolicy.from_yaml(path)
        finally:
            os.unlink(path)

    def test_yaml_deny_when_list_of_strings_raises_value_error(self) -> None:
        """Regression: YAML deny_when entries must be mappings, not bare strings."""
        yaml_content = "deny_when:\n  - search_*\n  - read_*\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match=r"deny_when\[0\].*must be a mapping"):
                ToolPolicy.from_yaml(path)
        finally:
            os.unlink(path)


class TestFromDictValidation:
    def test_string_allow_raises_value_error(self) -> None:
        """Regression: from_dict with allow=str iterates characters, producing glob '*' which
        allows every tool.  A string value must raise ValueError, not silently misbehave.

        With allow='search_*', fnmatch iterates chars ['s','e','a','r','c','h','_','*'].
        The bare '*' pattern matches ALL tool names — the entire policy is bypassed.
        """
        with pytest.raises(ValueError, match="must be a list of strings"):
            ToolPolicy.from_dict({"allow": "search_*"})

    def test_int_deny_raises_value_error(self) -> None:
        """Regression: non-list deny causes TypeError in evaluate(); must raise at construction."""
        with pytest.raises(ValueError, match="must be a list of strings"):
            ToolPolicy.from_dict({"deny": 42})

    def test_none_allow_becomes_empty_list(self) -> None:
        """from_dict with allow=None must produce an empty allow list, not crash."""
        policy = ToolPolicy.from_dict({"allow": None})
        assert policy.allow == []

    def test_none_deny_when_becomes_empty_list(self) -> None:
        """from_dict with deny_when=None must produce an empty deny_when list."""
        policy = ToolPolicy.from_dict({"deny_when": None})
        assert policy.deny_when == []

    def test_string_deny_when_raises_value_error(self) -> None:
        """Regression: deny_when='bad' causes AttributeError on .get() in evaluate();
        must raise ValueError at construction time instead.
        """
        with pytest.raises(ValueError, match="must be a list of mappings"):
            ToolPolicy.from_dict({"deny_when": "search_*"})

    def test_deny_when_with_non_dict_entry_raises_value_error(self) -> None:
        """Regression: deny_when=[str] causes AttributeError on .get() in evaluate();
        must raise ValueError at construction time instead.
        """
        with pytest.raises(ValueError, match=r"deny_when\[0\].*must be a mapping"):
            ToolPolicy.from_dict({"deny_when": ["not_a_dict"]})

    def test_deny_when_with_int_entry_raises_value_error(self) -> None:
        """Regression: deny_when=[42] must raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match=r"deny_when\[0\].*must be a mapping"):
            ToolPolicy.from_dict({"deny_when": [42]})

    def test_int_element_in_allow_list_raises_value_error(self) -> None:
        """Regression: allow=[1, 2, 3] previously passed construction but crashed at evaluate()
        with TypeError from fnmatch.fnmatch(tool_name, 1).
        Must raise ValueError at construction time with a helpful message.
        """
        with pytest.raises(ValueError, match=r"allow\[0\].*must be a string"):
            ToolPolicy.from_dict({"allow": [1, 2, 3]})

    def test_int_element_in_deny_list_raises_value_error(self) -> None:
        """Regression: deny=[42] previously crashed with TypeError at evaluate time."""
        with pytest.raises(ValueError, match=r"deny\[0\].*must be a string"):
            ToolPolicy.from_dict({"deny": [42]})

    def test_int_element_in_review_list_raises_value_error(self) -> None:
        """Regression: review=[None] previously crashed with TypeError at evaluate time."""
        with pytest.raises(ValueError, match=r"review\[0\].*must be a string"):
            ToolPolicy.from_dict({"review": [None]})

    def test_deny_when_int_field_value_raises_value_error(self) -> None:
        """Regression: deny_when with non-string field values previously crashed at evaluate()
        with TypeError from fnmatch.fnmatch(..., 123).
        Must raise ValueError at construction time.
        """
        with pytest.raises(ValueError, match=r"deny_when\[0\]\['pattern'\].*must be a string"):
            ToolPolicy.from_dict({"deny_when": [{"tool": "search", "arg": "q", "pattern": 123}]})

    def test_deny_when_non_string_tool_field_raises_value_error(self) -> None:
        """Regression: deny_when entry with non-string 'tool' key must raise at construction."""
        with pytest.raises(ValueError, match=r"deny_when\[0\]\['tool'\].*must be a string"):
            ToolPolicy.from_dict({"deny_when": [{"tool": 42, "arg": "q", "pattern": "*"}]})

    def test_valid_from_dict_still_works(self) -> None:
        """Sanity check: valid from_dict must not be broken by the new validation."""
        policy = ToolPolicy.from_dict(
            {
                "allow": ["search_*", "get_*"],
                "review": ["send_*"],
                "deny": ["delete_*"],
                "deny_when": [{"tool": "send_email", "arg": "to", "pattern": "*@evil.com"}],
            }
        )
        assert policy.evaluate("search_web").decision == PolicyDecision.ALLOW
        assert policy.evaluate("send_sms").decision == PolicyDecision.REVIEW
        assert policy.evaluate("delete_user").decision == PolicyDecision.DENY
        assert policy.evaluate("send_email", {"to": "x@evil.com"}).decision == PolicyDecision.DENY
