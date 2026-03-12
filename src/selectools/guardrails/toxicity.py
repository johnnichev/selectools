"""
ToxicityGuardrail — keyword-based toxicity detection.

Uses a configurable keyword blocklist.  Content is scored by the
ratio of toxic words found.  A ``threshold`` controls sensitivity
(0.0 = block on any match, 1.0 = never block).
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from .base import Guardrail, GuardrailAction, GuardrailResult

_DEFAULT_BLOCKLIST: Set[str] = {
    "kill",
    "murder",
    "attack",
    "hate",
    "racist",
    "sexist",
    "slur",
    "terrorist",
    "bomb",
    "assault",
    "harass",
    "abuse",
    "threat",
    "violence",
    "extremist",
    "supremacist",
}


class ToxicityGuardrail(Guardrail):
    """Block content that exceeds a toxicity threshold.

    Toxicity is measured as the fraction of unique blocklist words found
    in the content.  For production use, consider replacing the keyword
    list with a dedicated moderation model.

    Args:
        threshold: Fraction of blocklist words found that triggers a
            failure.  ``0.0`` means *any* match fails.  Default: ``0.0``.
        blocklist: Custom set of toxic keywords.  Defaults to a built-in
            list of ~16 high-signal terms.
        action: ``block``, ``rewrite``, or ``warn``.  Default: ``block``.
    """

    name = "toxicity"

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        blocklist: Optional[Set[str]] = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> None:
        self.threshold = threshold
        self._blocklist = blocklist or _DEFAULT_BLOCKLIST
        self.action = action
        self._word_re = re.compile(r"\b\w+\b")

    def score(self, content: str) -> float:
        """Return a toxicity score between 0.0 and 1.0."""
        words = {w.lower() for w in self._word_re.findall(content)}
        if not words:
            return 0.0
        hits = words & self._blocklist
        return len(hits) / len(self._blocklist) if self._blocklist else 0.0

    def matched_words(self, content: str) -> List[str]:
        """Return the blocklist words found in *content*."""
        words = {w.lower() for w in self._word_re.findall(content)}
        return sorted(words & self._blocklist)

    def check(self, content: str) -> GuardrailResult:
        score = self.score(content)
        if score > self.threshold:
            matched = self.matched_words(content)
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Toxicity score {score:.2f} exceeds threshold {self.threshold:.2f} "
                f"(matched: {', '.join(matched)})",
                guardrail_name=self.name,
            )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


__all__ = ["ToxicityGuardrail"]
