"""Answer extraction + normalization.

Two responsibilities:

1. Pull the final boxed answer out of a model response (containing thinking
   tokens, multiple boxed blocks, etc.).
2. Normalize a raw answer string into a canonical form so two answers can be
   compared without sympy.

Why not delegate everything to ``judger.Judger``? The judger does heavy
LaTeX/sympy work that's slow when called millions of times during voting.
This module is **pure string/regex**, fast, and deterministic — perfect for
self-consistency vote tallying. ``evaluate.py`` still uses the real judger
for the final correctness call.
"""

from __future__ import annotations

import re
import unicodedata


# ─── Boxed-answer extraction ────────────────────────────────────────────────

_BOXED_OPEN = "\\boxed{"


def _strip_thinking_tags(text: str) -> str:
    """Drop everything before the last ``</think>`` if present."""
    end = text.rfind("</think>")
    return text[end + len("</think>") :] if end >= 0 else text


def find_all_boxed(text: str) -> list[str]:
    """Return every ``\\boxed{...}`` content in order, balanced braces aware."""
    out: list[str] = []
    i = 0
    while True:
        idx = text.find(_BOXED_OPEN, i)
        if idx < 0:
            break
        start = idx + len(_BOXED_OPEN)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[start : j - 1])
        i = j
    return out


def extract_final_boxed(response: str) -> str | None:
    """Return the *last* boxed answer in the response (after ``</think>``).

    Returns ``None`` if no boxed answer is found anywhere.
    """
    boxes = find_all_boxed(_strip_thinking_tags(response))
    if boxes:
        return boxes[-1]
    boxes = find_all_boxed(response)
    return boxes[-1] if boxes else None


def extract_all_final_boxed(response: str) -> list[str]:
    """Return every boxed block after ``</think>`` (multi-part friendly).

    Falls back to all boxed in the whole text if nothing after ``</think>``.
    """
    after = find_all_boxed(_strip_thinking_tags(response))
    return after if after else find_all_boxed(response)


# ─── MCQ letter extraction ──────────────────────────────────────────────────

_LETTER_IN_BOX = re.compile(r"^([A-Za-z])$")
_LETTER_LOOSE = re.compile(r"\b([A-Z])\b")


def extract_letter(response: str) -> str:
    """Extract a single capital letter answer from an MCQ response.

    Order:
    1. Last ``\\boxed{X}`` where X is one letter (after thinking tag).
    2. Fallback: last standalone ``[A-Z]`` word in the post-thinking text.
    Returns empty string if nothing found.
    """
    box = extract_final_boxed(response)
    if box is not None:
        m = _LETTER_IN_BOX.match(box.strip())
        if m:
            return m.group(1).upper()
    tail = _strip_thinking_tags(response).upper()
    matches = _LETTER_LOOSE.findall(tail)
    return matches[-1] if matches else ""


# ─── Numerical normalization ────────────────────────────────────────────────

_FRAC_RX = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SLASH_FRAC_RX = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
_SCI_RX = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[eE]\s*(-?\d+)\s*$")
_THOUSANDS_RX = re.compile(r"(?<=\d),(?=\d{3}(?:[^\d]|$))")
_TRAILING_ZEROS_RX = re.compile(r"^(-?\d+\.\d*?)0+$")
_TRAILING_DOT_RX = re.compile(r"^(-?\d+)\.$")


def normalize_answer(s: str) -> str:
    """Canonicalize an answer string for equality comparison.

    Pure string/regex — never calls sympy. Handles:

    - whitespace, surrounding ``$``, surrounding ``\\(``/``\\)``
    - unicode minus ``−`` → ``-``
    - ``-512`` ≡ ``-512.0`` ≡ ``-512.00`` (trim trailing zeros after decimal)
    - ``1/2`` ≡ ``0.5`` (parse simple int fractions to decimal)
    - ``\\frac{1}{2}`` and ``\\dfrac{1}{2}`` likewise
    - ``1,000`` ≡ ``1000`` (remove thousands separators)
    - ``2.5e3`` ≡ ``2500`` (parse scientific)
    - Trailing ``%`` stripped (judger does the same)
    - ``\\pi``, symbolic forms left alone (don't approximate)
    """
    if s is None:
        return ""

    # Unicode normalize then translate funky minus signs.
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("−", "-")

    # Strip $...$ / \(...\) / \[...\] wrappers.
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()

    # Trailing percent — judger strips %, follow suit.
    # Check \% before % so the trailing backslash doesn't get left behind.
    if s.endswith("\\%"):
        s = s[:-2].strip()
    elif s.endswith("%"):
        s = s[:-1].strip()

    # Drop \left and \right (purely cosmetic LaTeX).
    s = s.replace("\\left", "").replace("\\right", "")

    # \dfrac and \tfrac → \frac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # \frac{a}{b} → a/b for further numeric parsing.
    s = _FRAC_RX.sub(r"(\1)/(\2)", s)

    # Strip wrapping outer parens introduced by frac substitution if numeric.
    s_stripped = s.strip()

    # Try integer fraction → decimal.
    if "\\pi" not in s_stripped and "\\" not in s_stripped:
        m = _SLASH_FRAC_RX.match(s_stripped.replace("(", "").replace(")", ""))
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den != 0:
                val = num / den
                return _format_float(val)

    # Strip thousands separators in all-digit chunks: 1,000 -> 1000
    s = _THOUSANDS_RX.sub("", s)

    # Scientific notation -> decimal.
    m = _SCI_RX.match(s.strip())
    if m:
        val = float(m.group(1)) * (10 ** int(m.group(2)))
        return _format_float(val)

    # Trim trailing zeros after decimal point.
    s = s.strip()
    m = _TRAILING_ZEROS_RX.match(s)
    if m:
        s = m.group(1)
    m = _TRAILING_DOT_RX.match(s)
    if m:
        s = m.group(1)

    # Final whitespace clean.
    s = re.sub(r"\s+", "", s)
    return s


def _format_float(val: float) -> str:
    """Render float as the shortest unambiguous decimal string."""
    if val == int(val):
        return str(int(val))
    s = repr(val)
    # repr may give 0.5 or 0.6666666666666666; both are acceptable.
    return s
