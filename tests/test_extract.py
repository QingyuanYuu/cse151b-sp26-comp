"""Tests for extract.py.

Coverage targets: 30+ tests across boxed extraction, MCQ letter, and
numerical normalization edge cases (per Phase 2 spec requirement).
"""

from __future__ import annotations

import pytest

from cse151b_comp.extract import (
    extract_all_final_boxed,
    extract_final_boxed,
    extract_letter,
    find_all_boxed,
    normalize_answer,
)


# ─── find_all_boxed ─────────────────────────────────────────────────────────


def test_find_all_boxed_single() -> None:
    assert find_all_boxed("answer is \\boxed{42}") == ["42"]


def test_find_all_boxed_multiple() -> None:
    assert find_all_boxed("\\boxed{1} then \\boxed{2}") == ["1", "2"]


def test_find_all_boxed_nested_braces() -> None:
    assert find_all_boxed("\\boxed{\\frac{1}{2}}") == ["\\frac{1}{2}"]


def test_find_all_boxed_empty_when_absent() -> None:
    assert find_all_boxed("no box here") == []


def test_find_all_boxed_unmatched_brace_dropped() -> None:
    # Truncated response — opening brace but no close.
    assert find_all_boxed("starts \\boxed{42 and never closes") == []


# ─── extract_final_boxed (with thinking tag) ────────────────────────────────


def test_extract_final_boxed_uses_post_thinking() -> None:
    text = "<think>\\boxed{99}</think> final \\boxed{42}"
    assert extract_final_boxed(text) == "42"


def test_extract_final_boxed_falls_back_to_full_text() -> None:
    text = "<think>\\boxed{42}</think>"  # nothing after </think>
    assert extract_final_boxed(text) == "42"


def test_extract_final_boxed_returns_none_when_absent() -> None:
    assert extract_final_boxed("no boxes anywhere") is None


def test_extract_all_final_boxed_multi_part() -> None:
    text = "<think>...</think> \\boxed{41} and \\boxed{35} and \\boxed{16}"
    assert extract_all_final_boxed(text) == ["41", "35", "16"]


# ─── extract_letter (MCQ) ───────────────────────────────────────────────────


def test_extract_letter_simple() -> None:
    assert extract_letter("answer \\boxed{C}") == "C"


def test_extract_letter_lowercase_normalized() -> None:
    assert extract_letter("answer \\boxed{c}") == "C"


def test_extract_letter_after_thinking() -> None:
    # boxed inside <think> should be ignored
    assert extract_letter("<think> guess \\boxed{B} </think> \\boxed{D}") == "D"


def test_extract_letter_fallback_when_no_box() -> None:
    # Truncated response without boxed — fall back to last standalone letter
    assert extract_letter("the right one is C in my view") == "C"


def test_extract_letter_returns_empty_when_nothing() -> None:
    assert extract_letter("12345") == ""


def test_extract_letter_ignores_paren_form_in_box() -> None:
    # \boxed{(C)} is NOT a single letter → falls back to scanning text
    assert extract_letter("\\boxed{(C)} so the answer is C") == "C"


# ─── normalize_answer: integers and floats ──────────────────────────────────


def test_normalize_int_unchanged() -> None:
    assert normalize_answer("42") == "42"


def test_normalize_negative_int() -> None:
    assert normalize_answer("-512") == "-512"


def test_normalize_unicode_minus() -> None:
    assert normalize_answer("−512") == "-512"


def test_normalize_int_with_trailing_zero_decimal() -> None:
    assert normalize_answer("-512.0") == "-512"
    assert normalize_answer("-512.00") == "-512"


def test_normalize_float_keeps_significant_digits() -> None:
    assert normalize_answer("3.14") == "3.14"


def test_normalize_float_trims_trailing_zeros() -> None:
    assert normalize_answer("3.1400") == "3.14"


def test_normalize_handles_dollar_wrappers() -> None:
    assert normalize_answer("$-512$") == "-512"


def test_normalize_handles_paren_latex_wrappers() -> None:
    assert normalize_answer("\\(-512\\)") == "-512"


def test_normalize_strips_whitespace() -> None:
    assert normalize_answer("   42   ") == "42"


# ─── normalize_answer: fractions ────────────────────────────────────────────


def test_normalize_simple_slash_fraction() -> None:
    assert normalize_answer("1/2") == "0.5"


def test_normalize_negative_fraction() -> None:
    assert normalize_answer("-3/4") == "-0.75"


def test_normalize_latex_frac() -> None:
    assert normalize_answer("\\frac{1}{2}") == "0.5"


def test_normalize_latex_dfrac() -> None:
    assert normalize_answer("\\dfrac{3}{4}") == "0.75"


def test_normalize_latex_tfrac() -> None:
    assert normalize_answer("\\tfrac{1}{4}") == "0.25"


def test_normalize_zero_denominator_left_alone() -> None:
    # Don't crash — just leave it (judger will reject)
    assert normalize_answer("1/0") in {"1/0", ""}


# ─── normalize_answer: thousands and scientific ─────────────────────────────


def test_normalize_thousands_separator() -> None:
    assert normalize_answer("1,000") == "1000"


def test_normalize_thousands_million() -> None:
    assert normalize_answer("1,000,000") == "1000000"


def test_normalize_scientific_lower() -> None:
    assert normalize_answer("2.5e3") == "2500.0" or normalize_answer("2.5e3") == "2500"


def test_normalize_scientific_upper() -> None:
    assert normalize_answer("2.5E3") == "2500.0" or normalize_answer("2.5E3") == "2500"


def test_normalize_scientific_negative_exp() -> None:
    assert normalize_answer("5e-2") == "0.05"


# ─── normalize_answer: percent ──────────────────────────────────────────────


def test_normalize_strips_percent() -> None:
    assert normalize_answer("50%") == "50"


def test_normalize_strips_latex_percent() -> None:
    assert normalize_answer("50\\%") == "50"


# ─── normalize_answer: symbolic answers untouched ───────────────────────────


def test_normalize_pi_kept() -> None:
    # \pi must NOT be approximated
    out = normalize_answer("\\pi")
    assert "\\pi" in out


def test_normalize_pi_in_expression() -> None:
    out = normalize_answer("2\\pi")
    assert "\\pi" in out


def test_normalize_left_right_stripped() -> None:
    assert normalize_answer("\\left(3\\right)") == "(3)"


# ─── normalize_answer: edge cases ───────────────────────────────────────────


def test_normalize_none_returns_empty() -> None:
    assert normalize_answer(None) == ""  # type: ignore[arg-type]


def test_normalize_empty_string() -> None:
    assert normalize_answer("") == ""


def test_normalize_only_whitespace() -> None:
    assert normalize_answer("   ") == ""


def test_normalize_internal_whitespace_collapsed() -> None:
    assert normalize_answer("4 2") == "42"


def test_normalize_idempotent() -> None:
    once = normalize_answer("$-512.00$")
    twice = normalize_answer(once)
    assert once == twice
