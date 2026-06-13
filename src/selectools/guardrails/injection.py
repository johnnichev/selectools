"""
PromptInjectionGuardrail — heuristic prompt-injection / jailbreak detection.

Scans content for high-signal injection phrasings ("ignore previous
instructions", "reveal your system prompt", role-delimiter spoofing like
``<system>`` / ``[INST]``, jailbreak markers like "developer mode" / "DAN").
Each pattern is deliberately high-precision so the default — block on a single
match — has a low false-positive rate.

This is the heuristic tier. A model-based detector (e.g. a small open-weight
classifier) is a heavier, optional future addition; the heuristic guard ships
now because it needs no model hosting and catches the common, templated attacks
that show up in practice.

Usage::

    from selectools.guardrails import PromptInjectionGuardrail, GuardrailsPipeline

    guard = PromptInjectionGuardrail()
    pipeline = GuardrailsPipeline(input_guardrails=[guard])

Tune sensitivity with ``min_matches`` (how many distinct patterns must fire) or
extend coverage with ``extra_patterns``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Sequence, Tuple

from selectools.stability import beta

from .base import Guardrail, GuardrailAction, GuardrailResult

# (label, regex) — each pattern targets a specific, templated attack so that a
# single match is high-confidence. Patterns are matched case-insensitively.
_DEFAULT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        "ignore-previous",
        r"ignore\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|prior|earlier|above|preceding)\s+"
        r"(?:instructions?|prompts?|messages?|directions?|rules?)",
    ),
    (
        "disregard-previous",
        r"disregard\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|prior|earlier|above|system)\s+"
        r"(?:instructions?|prompts?|rules?|directions?)",
    ),
    (
        "forget-instructions",
        r"forget\s+(?:everything|all|what|your|the)\s+"
        r"(?:(?:previous\s+|prior\s+)?(?:instructions?|rules?|context|prompts?)"
        r"|(?:you|i)\s+(?:were\s+|was\s+|been\s+|are\s+)?(?:told|instructed|said|taught))",
    ),
    (
        "reveal-system-prompt",
        r"(?:reveal|show|print|repeat|display|tell\s+me|give\s+me)\s+"
        r"(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+|initial\s+|original\s+)?"
        r"(?:prompt|instructions|rules|directives?)",
    ),
    (
        "override-safety",
        r"(?:override|bypass|disable|turn\s+off|ignore)\s+(?:your\s+|the\s+|all\s+)?"
        r"(?:safety|security|guidelines?|guardrails?|filters?|restrictions?|programming)",
    ),
    (
        "no-restrictions",
        r"(?:you\s+have\s+no|without\s+any|with\s+no|free\s+from)\s+"
        r"(?:restrictions?|limitations?|rules?|filters?|guidelines?)",
    ),
    (
        "developer-mode",
        r"\bdeveloper\s+mode\b",
    ),
    (
        "jailbreak",
        r"\bjailbr(?:eak|oken)\b",
    ),
    (
        "dan-mode",
        r"\b(?:do\s+anything\s+now|DAN\s+mode)\b",
    ),
    (
        "you-are-now-unrestricted",
        r"you\s+are\s+now\s+(?:a\s+|an\s+|in\s+)?"
        r"(?:DAN|developer\s+mode|free|unrestricted|jailbroken|unfiltered|uncensored)",
    ),
    (
        "act-as-unrestricted",
        r"(?:act|behave|respond)\s+as\s+(?:if\s+you\s+(?:are|were)\s+|an?\s+)?"
        r"(?:DAN|jailbroken|unrestricted|unfiltered|uncensored)",
    ),
    (
        "pretend-no-rules",
        r"pretend\s+(?:that\s+)?you\s+(?:are\s+(?:not\s+)?|have\s+no\b)",
    ),
    (
        "role-delimiter-system",
        r"<\s*/?\s*system\s*>|(?:^|\n)\s*(?:#{2,}\s*)?system\s*[:>]",
    ),
    (
        "inst-delimiter",
        r"\[/?INST\]|<\|im_start\|>|<\|im_end\|>",
    ),
    (
        "new-instructions",
        r"(?:here\s+are\s+|these\s+are\s+)?your\s+new\s+(?:instructions?|rules?|directives?)",
    ),
)


@beta
class PromptInjectionGuardrail(Guardrail):
    """Block content that matches known prompt-injection / jailbreak patterns.

    Args:
        min_matches: Number of distinct patterns that must fire to trigger a
            failure. ``1`` (default) blocks on any single high-signal match;
            raise it to require corroborating signals.
        extra_patterns: Additional ``(label, regex)`` pairs to match (case-
            insensitive), merged with the built-in set.
        patterns: Replace the built-in pattern set entirely with these
            ``(label, regex)`` pairs. Mutually exclusive with ``extra_patterns``.
        action: ``block`` (default), ``rewrite`` (returns content unchanged —
            injection isn't safely rewritable, so this degrades to ``warn``
            semantics), or ``warn``.

    The heuristic tier. For higher recall on novel phrasings, pair it with a
    model-based classifier (a future optional integration).
    """

    name = "prompt_injection"

    def __init__(
        self,
        *,
        min_matches: int = 1,
        extra_patterns: Optional[Sequence[Tuple[str, str]]] = None,
        patterns: Optional[Sequence[Tuple[str, str]]] = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> None:
        if min_matches < 1:
            raise ValueError(f"min_matches must be >= 1, got {min_matches!r}")
        if patterns is not None and extra_patterns is not None:
            raise ValueError("pass either patterns or extra_patterns, not both")
        raw: Sequence[Tuple[str, str]]
        if patterns is not None:
            raw = patterns
        elif extra_patterns is not None:
            raw = (*_DEFAULT_PATTERNS, *extra_patterns)
        else:
            raw = _DEFAULT_PATTERNS
        self.min_matches = min_matches
        self.action = action
        self._patterns: List[Tuple[str, Pattern[str]]] = [
            (label, re.compile(rx, re.IGNORECASE)) for label, rx in raw
        ]

    def detected(self, content: str) -> List[str]:
        """Return the labels of every pattern that matched *content*."""
        return [label for label, rx in self._patterns if rx.search(content)]

    def check(self, content: str) -> GuardrailResult:
        labels = self.detected(content)
        if len(labels) >= self.min_matches:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=(
                    f"Possible prompt injection: matched {len(labels)} pattern(s) "
                    f"({', '.join(labels)})"
                ),
                guardrail_name=self.name,
            )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


__stability__ = "beta"

__all__ = ["PromptInjectionGuardrail"]
