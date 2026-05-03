"""Unit tests for prompts.py.

These guard against regressions in the prompt rules — every Phase 1 fix
checked here has a documented reason in prompts.py.
"""
from __future__ import annotations

from cse151b_comp.prompts import SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, build_prompt


# ─── MCQ prompt ─────────────────────────────────────────────────────────────


def test_mcq_returns_mcq_system() -> None:
    sys_p, _ = build_prompt("Q?", ["a", "b", "c"])
    assert sys_p is SYSTEM_PROMPT_MCQ


def test_mcq_user_includes_letter_labels() -> None:
    _, user_p = build_prompt("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user_p
    assert "B. 4" in user_p
    assert "C. 5" in user_p


def test_mcq_user_strips_option_whitespace() -> None:
    _, user_p = build_prompt("Q", [" 3 ", "  4"])
    assert "A. 3" in user_p
    assert "B. 4" in user_p


def test_mcq_supports_more_than_4_options() -> None:
    options = [str(i) for i in range(10)]
    _, user_p = build_prompt("Q", options)
    for letter, val in zip("ABCDEFGHIJ", options):
        assert f"{letter}. {val}" in user_p


def test_mcq_system_has_positive_letter_example() -> None:
    assert "\\boxed{C}" in SYSTEM_PROMPT_MCQ


def test_mcq_system_forbids_parenthesised_letter() -> None:
    assert "\\boxed{(C)}" in SYSTEM_PROMPT_MCQ
    assert "Do NOT" in SYSTEM_PROMPT_MCQ


def test_mcq_system_forbids_dotted_letter() -> None:
    assert "\\boxed{C.}" in SYSTEM_PROMPT_MCQ


def test_mcq_system_has_token_budget_rule() -> None:
    assert "running out" in SYSTEM_PROMPT_MCQ
    assert "best-guess" in SYSTEM_PROMPT_MCQ


# ─── Free-form / math prompt ────────────────────────────────────────────────


def test_freeform_returns_math_system() -> None:
    sys_p, user_p = build_prompt("Compute the integral.", None)
    assert sys_p is SYSTEM_PROMPT_MATH
    assert user_p == "Compute the integral."


def test_freeform_with_empty_options_treated_as_freeform() -> None:
    sys_p, _ = build_prompt("Q", [])
    assert sys_p is SYSTEM_PROMPT_MATH


def test_math_system_explains_multi_part_kbox_style() -> None:
    assert "Multiple boxed blocks" in SYSTEM_PROMPT_MATH or "K sub-answers" in SYSTEM_PROMPT_MATH
    assert "\\boxed{41}" in SYSTEM_PROMPT_MATH
    assert "\\boxed{35}" in SYSTEM_PROMPT_MATH


def test_math_system_warns_against_labels_between_boxes() -> None:
    """Crucial: judger only takes contiguous boxes. (a)/(b) labels between
    them break parsing. Phase-1 test we caught while wiring up score_response."""
    assert "BREAKS the parser" in SYSTEM_PROMPT_MATH
    assert "(a)" in SYSTEM_PROMPT_MATH or "labels like" in SYSTEM_PROMPT_MATH


def test_math_system_explains_multi_part_csv_style() -> None:
    assert "\\boxed{41, 35, 16}" in SYSTEM_PROMPT_MATH


def test_math_system_warns_against_mixed_styles() -> None:
    assert "Do NOT mix" in SYSTEM_PROMPT_MATH or "do not mix" in SYSTEM_PROMPT_MATH.lower()


def test_math_system_handles_ANS_placeholder() -> None:
    assert "[ANS]" in SYSTEM_PROMPT_MATH


def test_math_system_requires_last_line_is_boxed() -> None:
    assert "last line" in SYSTEM_PROMPT_MATH


def test_math_system_anti_rounding_rule_present() -> None:
    assert "Do not round" in SYSTEM_PROMPT_MATH
    assert "6 significant figures" in SYSTEM_PROMPT_MATH


def test_math_system_anti_rounding_examples() -> None:
    # The 5-question smoke test bug: rounded 143.224 -> 143
    assert "143.224229" in SYSTEM_PROMPT_MATH
    assert "143" in SYSTEM_PROMPT_MATH


def test_math_system_no_units_rule() -> None:
    assert "no units" in SYSTEM_PROMPT_MATH


def test_math_system_no_trailing_punctuation_rule() -> None:
    assert "no trailing punctuation" in SYSTEM_PROMPT_MATH


def test_math_system_token_budget_rule() -> None:
    assert "running out" in SYSTEM_PROMPT_MATH
    assert "\\boxed" in SYSTEM_PROMPT_MATH


def test_math_system_no_x_equals_rule() -> None:
    # Judger strips "x = " but leaving it consumes the boxed value
    assert "x = " in SYSTEM_PROMPT_MATH


# ─── Sanity: prompts are actually different ─────────────────────────────────


def test_mcq_and_math_prompts_differ() -> None:
    assert SYSTEM_PROMPT_MATH != SYSTEM_PROMPT_MCQ


def test_mcq_does_not_have_freeform_rules() -> None:
    # MCQ should NOT instruct about multi-part [ANS]
    assert "[ANS]" not in SYSTEM_PROMPT_MCQ
