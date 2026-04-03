"""
Tool output screening — detect prompt injection in tool results.

When a tool returns untrusted content (web pages, emails, user-uploaded
files), that content is fed back to the LLM as context.  An attacker can
embed instructions in the content that hijack the agent's behaviour.

This module provides pattern-based screening that runs *after* tool
execution and *before* the result is appended to conversation history.
Suspicious outputs are replaced with a safe placeholder.

Usage:
    Mark individual tools with ``screen_output=True`` in the ``@tool``
    decorator, or enable screening globally via
    ``AgentConfig(screen_tool_output=True)``.

    Custom patterns can be supplied via ``AgentConfig(output_screening_patterns=[...])``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

_DEFAULT_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*:\s*you", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE),
    re.compile(r"</s>", re.IGNORECASE),
    re.compile(r"IMPORTANT:\s*override", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    # OpenAI/Llama special tokens used to inject synthetic turns
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
    # Common jailbreak / role-override patterns
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(instructions|rules|guidelines|constraints)", re.IGNORECASE),
    re.compile(r"from\s+now\s+on[,\s]", re.IGNORECASE),
    re.compile(r"act\s+as\s+(DAN|an?\s+AI\s+without\s+restrictions?)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]


_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Common homoglyphs: visually similar characters mapped to their ASCII equivalents.
# Covers Cyrillic, Greek, and other scripts commonly used for evasion.
_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0410": "A",  # Cyrillic А
    "\u0412": "B",  # Cyrillic В
    "\u0421": "C",  # Cyrillic С
    "\u0415": "E",  # Cyrillic Е
    "\u041d": "H",  # Cyrillic Н
    "\u041a": "K",  # Cyrillic К
    "\u041c": "M",  # Cyrillic М
    "\u041e": "O",  # Cyrillic О
    "\u0420": "P",  # Cyrillic Р
    "\u0422": "T",  # Cyrillic Т
    "\u0425": "X",  # Cyrillic Х
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0443": "y",  # Cyrillic у
    "\u0445": "x",  # Cyrillic х
    "\u0456": "i",  # Cyrillic і
    "\u0391": "A",  # Greek Α
    "\u0392": "B",  # Greek Β
    "\u0395": "E",  # Greek Ε
    "\u0397": "H",  # Greek Η
    "\u0399": "I",  # Greek Ι
    "\u039a": "K",  # Greek Κ
    "\u039c": "M",  # Greek Μ
    "\u039d": "N",  # Greek Ν
    "\u039f": "O",  # Greek Ο
    "\u03a1": "P",  # Greek Ρ
    "\u03a4": "T",  # Greek Τ
    "\u03a5": "Y",  # Greek Υ
    "\u03a7": "X",  # Greek Χ
    "\u03b1": "a",  # Greek α (visual approximation)
    "\u03bf": "o",  # Greek ο
    "\u0131": "i",  # Latin dotless ı
    "\uff41": "a",  # Fullwidth ａ
    "\uff45": "e",  # Fullwidth ｅ
    "\uff49": "i",  # Fullwidth ｉ
    "\uff4f": "o",  # Fullwidth ｏ
    "\uff55": "u",  # Fullwidth ｕ
}

_HOMOGLYPH_TRANS = str.maketrans(_HOMOGLYPH_MAP)


def _normalize_for_screening(text: str) -> str:
    """Normalize Unicode to prevent homoglyph bypass of security patterns."""
    text = unicodedata.normalize("NFKD", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.translate(_HOMOGLYPH_TRANS)
    return text


@dataclass
class ScreeningResult:
    """Result of screening a tool output.

    Attributes:
        safe: Whether the content passed screening.
        content: The (possibly replaced) content.
        matched_patterns: List of pattern descriptions that matched.
    """

    safe: bool
    content: str
    matched_patterns: List[str]


def screen_output(
    content: str,
    *,
    extra_patterns: Optional[List[str]] = None,
) -> ScreeningResult:
    """Screen tool output for prompt injection patterns.

    Args:
        content: Raw tool output string.
        extra_patterns: Additional regex patterns (strings) to check.

    Returns:
        A :class:`ScreeningResult` indicating whether the content is safe.
    """
    if content is None:
        content = ""

    patterns = list(_DEFAULT_INJECTION_PATTERNS)
    if extra_patterns:
        for pat in extra_patterns:
            try:
                compiled = re.compile(pat, re.IGNORECASE)
                compiled.search("a" * 100)  # nosec B105 — ReDoS smoke test
                patterns.append(compiled)
            except re.error as exc:
                raise ValueError(f"Invalid extra screening pattern: {exc}") from exc

    normalized = _normalize_for_screening(content)

    matched: List[str] = []
    for pattern in patterns:
        if pattern.search(normalized):
            matched.append(pattern.pattern)

    if matched:
        safe_content = (
            "[Tool output blocked: potential prompt injection detected. "
            f"{len(matched)} suspicious pattern(s) found.]"
        )
        return ScreeningResult(
            safe=False,
            content=safe_content,
            matched_patterns=matched,
        )

    return ScreeningResult(safe=True, content=content, matched_patterns=[])


__all__ = ["ScreeningResult", "screen_output"]
