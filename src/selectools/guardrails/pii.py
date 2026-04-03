"""
PIIGuardrail — detect and optionally redact personally identifiable information.

Uses regex patterns for common PII types: email, phone, SSN, credit card,
and IP address.  Custom patterns can be added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from selectools.stability import stable

from .base import Guardrail, GuardrailAction, GuardrailResult

_BUILTIN_PATTERNS: Dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{3}\s\d{2}\s\d{4}|\d{9})\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


@dataclass
class PIIMatch:
    """A single PII detection within text."""

    pii_type: str
    value: str
    start: int
    end: int


@stable
class PIIGuardrail(Guardrail):
    """Detect (and optionally redact) PII in content.

    Args:
        action: ``block``, ``rewrite`` (redact), or ``warn``. Default: ``rewrite``.
        detect: List of PII types to detect.  Default: all built-in types.
            Built-in types: ``email``, ``phone_us``, ``ssn``, ``credit_card``, ``ipv4``.
        redact_char: Character used for redaction.  Default: ``"*"``.
        custom_patterns: Additional ``{name: regex_pattern}`` pairs.
    """

    name = "pii"

    def __init__(
        self,
        *,
        action: GuardrailAction = GuardrailAction.REWRITE,
        detect: Optional[List[str]] = None,
        redact_char: str = "*",
        custom_patterns: Optional[Dict[str, str]] = None,
    ) -> None:
        self.action = action
        self._redact_char = redact_char

        patterns: Dict[str, re.Pattern[str]] = {}
        if detect is not None:
            for name in detect:
                if name in _BUILTIN_PATTERNS:
                    patterns[name] = _BUILTIN_PATTERNS[name]
        else:
            patterns = dict(_BUILTIN_PATTERNS)

        if custom_patterns:
            for name, pat in custom_patterns.items():
                try:
                    compiled = re.compile(pat, re.IGNORECASE)
                    # Safety check: test against a short string to catch catastrophic backtracking
                    compiled.search("a" * 100)  # nosec B105 — ReDoS smoke test
                    patterns[name] = compiled
                except re.error as exc:
                    raise ValueError(f"Invalid custom PII pattern '{name}': {exc}") from exc

        self._patterns = patterns

    def detect(self, content: str) -> List[PIIMatch]:
        """Return all PII matches found in *content*."""
        matches: List[PIIMatch] = []
        for pii_type, pattern in self._patterns.items():
            for m in pattern.finditer(content):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        matches.sort(key=lambda x: x.start)
        return matches

    def redact(self, content: str) -> str:
        """Return *content* with all detected PII replaced by redaction chars."""
        matches = self.detect(content)
        if not matches:
            return content
        parts: List[str] = []
        last_end = 0
        for m in matches:
            if m.start < last_end:
                # Skip overlapping matches — the region was already redacted
                continue
            parts.append(content[last_end : m.start])
            parts.append(f"[{m.pii_type.upper()}:{self._redact_char * min(len(m.value), 8)}]")
            last_end = m.end
        parts.append(content[last_end:])
        return "".join(parts)

    def check(self, content: str) -> GuardrailResult:
        matches = self.detect(content)
        if not matches:
            return GuardrailResult(passed=True, content=content, guardrail_name=self.name)

        types_found = sorted({m.pii_type for m in matches})
        reason = f"PII detected: {', '.join(types_found)} ({len(matches)} instance(s))"

        if self.action == GuardrailAction.REWRITE:
            redacted = self.redact(content)
            return GuardrailResult(
                passed=False,
                content=redacted,
                reason=reason,
                guardrail_name=self.name,
            )

        return GuardrailResult(
            passed=False,
            content=content,
            reason=reason,
            guardrail_name=self.name,
        )


__all__ = ["PIIGuardrail", "PIIMatch"]
