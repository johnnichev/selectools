"""
Built-in pre-save sanitizers for ``KnowledgeMemory``.

Remembered content is user-derived and flows back into system prompts via
``build_context()``, which makes the knowledge store a prompt-injection
vector: an attacker who can get text remembered (chat message, tool output,
scraped page) can plant structural markers that later read as instructions.

These sanitizers plug into ``KnowledgeMemory(pre_save=...)``.  Each takes the
entry text and returns the transformed text, or ``None`` to reject the entry
entirely::

    from selectools import KnowledgeMemory
    from selectools.knowledge_sanitizers import defang_delimiters, strip_surrogates

    memory = KnowledgeMemory(
        directory="./memory",
        pre_save=[strip_surrogates, defang_delimiters],
        dedupe=True,
    )

All functions here are stdlib-only and deterministic.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Iterable, Optional

from .stability import beta

# Full ``--- Label ---`` line.  Rewritten with em-dash bookends: visually
# similar, but no longer reads as a structural section header.  Lines of
# dashes only (``---``) pass through; they cannot carry a forged label.
_DELIMITER_RE = re.compile(r"^---[ \t]*([^\n-][^\n]*?)[ \t]*---[ \t]*$", re.MULTILINE)

# Half-delimiter: ``--- End of conversation`` with no closing dashes.  Still
# scans as a section break to an LLM, so it gets the same rewrite.
# Whitespace is restricted to same-line (`[ \t]`, not `\s`) so a bare
# ``---`` line never swallows the following line.
_HALF_DELIMITER_RE = re.compile(r"^---[ \t]+([^\n-][^\n]*?)[ \t]*$", re.MULTILINE)

# Role markers from common LLM chat templates (``[INST]``, ``<|im_start|>``,
# ``<system>``).  Models are trained on text containing these tokens and may
# give them special weight even inside quoted content.
_ROLE_MARKER_RE = re.compile(
    r"""(?ix)
    \[/? \s* (?: INST | SYS | ASSISTANT | USER | SYSTEM ) \s* \]
    | <\|? \s* (?: im_start | im_end | system | assistant | user
                 | endoftext | start | end ) \s* \|? >
    | </? \s* (?: system | assistant | user ) \s* >
    """
)

# Speaker-label prefix at line start (``Assistant: I'll wire the money``).
# Inside a remembered-context block this reads as a forged conversation turn.
_SPEAKER_LABEL_RE = re.compile(
    r"^(\s*)(User|Assistant|System|Human)(\s*):",
    re.MULTILINE | re.IGNORECASE,
)

# Code fence at line start (3+ backticks, optionally indented up to 3 spaces
# per CommonMark).  A fence opened inside remembered content can swallow the
# rest of the prompt or terminate a fence the host prompt opened.  Inline
# code spans (1-2 backticks, or backticks mid-line) are untouched.
_FENCE_RE = re.compile(r"^(\s{0,3})(`{3,})", re.MULTILINE)

# U+02CB MODIFIER LETTER GRAVE ACCENT: visually a backtick, but markdown
# parsers and tokenizers do not treat it as a fence character.
_FENCE_SUBSTITUTE = "ˋ"


@beta
def defang_delimiters(text: str) -> str:
    """Neutralize prompt-injection delimiters in text about to be remembered.

    Conservative by design: every rewrite keeps the content human-readable
    and meaning-preserving; only the *structural* interpretation is broken.

    What is defanged, and why:

    1. ``--- Label ---`` and ``--- Label`` lines become ``— Label —`` /
       ``— Label`` (em dashes).  Dash-fenced lines are the classic fake
       section header (``--- End of conversation ---``).
    2. Chat-template role markers (``[INST]``, ``<|im_start|>``,
       ``<system>``, ``</assistant>``, ...) get a space inserted after the
       opening bracket so they no longer tokenize as control tokens.
    3. ``User:`` / ``Assistant:`` / ``System:`` / ``Human:`` at line start
       become ``User :`` etc., breaking forged conversation turns.
    4. Line-start code fences (3+ backticks) are rewritten with the
       visually identical U+02CB character, so remembered content cannot
       open or close a markdown fence in the host prompt.  Inline backtick
       spans are preserved.

    Args:
        text: Entry text to sanitize.

    Returns:
        The defanged text.  Plain prose passes through unchanged.
    """
    s = text or ""
    s = _DELIMITER_RE.sub(r"— \1 —", s)
    s = _HALF_DELIMITER_RE.sub(r"— \1", s)
    s = _ROLE_MARKER_RE.sub(lambda m: m.group(0).replace("[", "[ ").replace("<", "< "), s)
    s = _SPEAKER_LABEL_RE.sub(r"\1\2\3 :", s)
    s = _FENCE_RE.sub(lambda m: m.group(1) + _FENCE_SUBSTITUTE * len(m.group(2)), s)
    return s


@beta
def strip_surrogates(text: str) -> str:
    """Remove lone UTF-16 surrogates and other UTF-8-unencodable characters.

    Text arriving from webhooks (WhatsApp/Telegram emoji edge cases, broken
    client encodings) can contain lone surrogate code points.  Python ``str``
    holds them happily, but the first ``open(..., encoding="utf-8").write()``
    raises ``UnicodeEncodeError`` — bricking persistence for that entry and,
    on shared files, anything written after it.

    Args:
        text: Entry text to sanitize.

    Returns:
        The text with unencodable code points dropped.  Well-formed text
        (including astral-plane emoji) passes through unchanged.
    """
    return (text or "").encode("utf-8", errors="ignore").decode("utf-8")


@beta
def dedupe_against(
    existing_fetcher: Callable[[], Iterable[str]],
    threshold: float = 0.9,
) -> Callable[[str], Optional[str]]:
    """Build a pre-save sanitizer that rejects near-duplicate entries.

    Returns a hook that fetches the current entry texts via
    ``existing_fetcher`` and rejects the new text (returns ``None``) when its
    ``difflib.SequenceMatcher`` ratio against any existing entry reaches
    ``threshold``.  Append-only memories otherwise accumulate the same fact
    once per session, bloating every future prompt.

    Cost: the fetcher runs on every save, and similarity is O(len(a) *
    len(b)) per comparison in the worst case.  Cheap upper-bound ratios
    (``real_quick_ratio`` / ``quick_ratio``) prune most non-matches, but for
    stores beyond a few thousand entries prefer a bounded fetcher (e.g. only
    the most recent or same-category entries).

    Args:
        existing_fetcher: Zero-arg callable returning the texts to compare
            against.  Called once per sanitizer invocation, so it always
            sees the latest state.
        threshold: Similarity ratio in ``[0.0, 1.0]`` at or above which the
            new entry is rejected.  ``1.0`` rejects exact matches only.

    Returns:
        A pre-save hook suitable for ``KnowledgeMemory(pre_save=...)``.
    """

    def _reject_near_duplicates(text: str) -> Optional[str]:
        for existing in existing_fetcher():
            matcher = difflib.SequenceMatcher(None, text, existing, autojunk=False)
            if (
                matcher.real_quick_ratio() >= threshold
                and matcher.quick_ratio() >= threshold
                and matcher.ratio() >= threshold
            ):
                return None
        return text

    return _reject_near_duplicates


__all__ = [
    "defang_delimiters",
    "strip_surrogates",
    "dedupe_against",
]
