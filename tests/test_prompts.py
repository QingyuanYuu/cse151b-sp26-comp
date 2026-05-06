"""Tests for v6 prompts (per-type routing).

These guard against:

- regressions in the rules retained in v6 (positive assertions),
- accidental re-introduction of rules diagnosed as harmful in v4
  (negative assertions),
- correctness of question-type routing in :func:`detect_question_type`.
"""
from __future__ import annotations

from cse151b_comp.prompts import (
    RUNB_SYSTEM_PROMPT_FREE,
    RUNB_SYSTEM_PROMPT_MCQ,
    SYSTEM_PROMPT_FREE_MULTI,
    SYSTEM_PROMPT_FREE_SINGLE,
    SYSTEM_PROMPT_MATH,
    SYSTEM_PROMPT_MCQ,
    build_prompt,
    build_prompt_runb,
    detect_question_type,
)


# ─── Question-type detection ────────────────────────────────────────────────


def test_detect_mc_when_options_present() -> None:
    assert detect_question_type("Q?", ["a", "b"]) == "mc"


def test_detect_mc_overrides_ans_placeholder() -> None:
    # If options exist, treat as MCQ even if [ANS] appears.
    assert detect_question_type("Pick: [ANS]", ["a", "b"]) == "mc"


def test_detect_free_single_no_options_no_multi_markers() -> None:
    assert detect_question_type("Solve: 2+2 = [ANS]", None) == "free_single"


def test_detect_free_multi_two_ans_placeholders() -> None:
    assert detect_question_type("a) [ANS] b) [ANS]", None) == "free_multi"


def test_detect_free_multi_letter_markers() -> None:
    q = "(a) compute X. (b) compute Y. (c) compute Z."
    assert detect_question_type(q, None) == "free_multi"


def test_detect_free_single_only_one_letter_marker() -> None:
    # "compute (a + b)" has one (a) marker but should be free_single
    assert detect_question_type("compute (a+b)^2 = [ANS]", None) == "free_single"


def test_detect_handles_empty_options_list() -> None:
    assert detect_question_type("Q?", []) == "free_single"


# ─── build_prompt routes correctly ──────────────────────────────────────────


def test_build_prompt_mc_returns_mcq_system() -> None:
    sys_p, _ = build_prompt("What is 2+2?", ["3", "4"])
    assert sys_p is SYSTEM_PROMPT_MCQ


def test_build_prompt_free_single_returns_single_system() -> None:
    sys_p, _ = build_prompt("Compute 1+1.", None)
    assert sys_p is SYSTEM_PROMPT_FREE_SINGLE


def test_build_prompt_free_multi_returns_multi_system() -> None:
    sys_p, _ = build_prompt("a) [ANS] b) [ANS]", None)
    assert sys_p is SYSTEM_PROMPT_FREE_MULTI


def test_build_prompt_returns_2_tuple_for_back_compat() -> None:
    out = build_prompt("Q?", None)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_mcq_user_includes_letter_labels() -> None:
    _, user_p = build_prompt("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user_p
    assert "B. 4" in user_p
    assert "C. 5" in user_p


def test_mcq_user_strips_option_whitespace() -> None:
    _, user_p = build_prompt("Q", [" 3 ", "  4"])
    assert "A. 3" in user_p


def test_mcq_user_includes_letter_only_reminder() -> None:
    _, user_p = build_prompt("What is 2+2?", ["3", "4"])
    assert "letter only" in user_p.lower()


def test_free_multi_user_includes_kbox_example() -> None:
    _, user_p = build_prompt("a) [ANS] b) [ANS]", None)
    assert "\\boxed{ans1}" in user_p


def test_mcq_supports_more_than_4_options() -> None:
    options = [str(i) for i in range(10)]
    _, user_p = build_prompt("Q", options)
    for letter, val in zip("ABCDEFGHIJ", options):
        assert f"{letter}. {val}" in user_p


# ─── MCQ system rules retained ──────────────────────────────────────────────


def test_mcq_system_has_letter_only_rule() -> None:
    assert "LETTER only" in SYSTEM_PROMPT_MCQ


def test_mcq_system_forbids_paren_period_text_macros() -> None:
    # Anti-(C), anti-period, anti-LaTeX-text-macros all in one rule.
    assert "Do NOT include parentheses" in SYSTEM_PROMPT_MCQ
    assert "\\text" in SYSTEM_PROMPT_MCQ
    assert "\\textbf" in SYSTEM_PROMPT_MCQ


def test_mcq_does_not_have_token_budget_rule() -> None:
    # v4→v5 ablation removed token-budget rescue; v6 keeps it removed.
    assert "running out" not in SYSTEM_PROMPT_MCQ
    assert "best-guess" not in SYSTEM_PROMPT_MCQ


def test_mcq_does_not_have_freeform_rules() -> None:
    assert "[ANS]" not in SYSTEM_PROMPT_MCQ


# ─── Free-single rules retained / dropped ──────────────────────────────────


def test_free_single_has_symbolic_examples() -> None:
    # The whole point of v6: explicit symbolic forms are GOOD answers.
    assert "\\frac{1}{2}" in SYSTEM_PROMPT_FREE_SINGLE
    assert "\\pi" in SYSTEM_PROMPT_FREE_SINGLE


def test_free_single_has_equation_form_example() -> None:
    # Counters the v4 'no x =' rule which stripped equation prefixes.
    assert "D = 800 - 50d" in SYSTEM_PROMPT_FREE_SINGLE


def test_free_single_has_text_answer_example() -> None:
    # Counters Bug A (True/False → 1/0) and Bug B (letter → digit).
    assert "Yes" in SYSTEM_PROMPT_FREE_SINGLE


def test_free_single_does_not_force_plain_numbers() -> None:
    # v4→v5: "Use plain numbers" removed because it caused True/False → 1/0.
    assert "Use plain numbers" not in SYSTEM_PROMPT_FREE_SINGLE
    assert "plain numbers" not in SYSTEM_PROMPT_FREE_SINGLE


def test_free_single_does_not_strip_x_equals() -> None:
    # v4→v5: "no 'x = '" rule removed because it stripped equation prefixes.
    assert "x = " not in SYSTEM_PROMPT_FREE_SINGLE
    assert "x =" not in SYSTEM_PROMPT_FREE_SINGLE.replace("x = 800", "")


def test_free_single_does_not_have_anti_rounding_rule() -> None:
    assert "Do not round" not in SYSTEM_PROMPT_FREE_SINGLE
    assert "6 significant figures" not in SYSTEM_PROMPT_FREE_SINGLE


def test_free_single_does_not_have_token_budget_rule() -> None:
    assert "running out" not in SYSTEM_PROMPT_FREE_SINGLE


# ─── Free-multi rules retained / dropped ───────────────────────────────────


def test_free_multi_has_kbox_example() -> None:
    assert "\\boxed{41}" in SYSTEM_PROMPT_FREE_MULTI
    assert "\\boxed{35}" in SYSTEM_PROMPT_FREE_MULTI
    assert "\\boxed{16}" in SYSTEM_PROMPT_FREE_MULTI


def test_free_multi_warns_against_labels_between_boxes() -> None:
    assert "(a)" in SYSTEM_PROMPT_FREE_MULTI
    assert "breaks" in SYSTEM_PROMPT_FREE_MULTI.lower()


def test_free_multi_forbids_combining_in_single_box() -> None:
    assert "DO NOT combine multiple values inside a single box" in SYSTEM_PROMPT_FREE_MULTI


def test_free_multi_does_not_have_anti_rounding_rule() -> None:
    assert "Do not round" not in SYSTEM_PROMPT_FREE_MULTI


def test_free_multi_does_not_have_token_budget_rule() -> None:
    assert "running out" not in SYSTEM_PROMPT_FREE_MULTI


# ─── Sanity ─────────────────────────────────────────────────────────────────


def test_three_prompts_all_distinct() -> None:
    assert SYSTEM_PROMPT_MCQ != SYSTEM_PROMPT_FREE_SINGLE
    assert SYSTEM_PROMPT_FREE_SINGLE != SYSTEM_PROMPT_FREE_MULTI
    assert SYSTEM_PROMPT_MCQ != SYSTEM_PROMPT_FREE_MULTI


def test_system_prompt_math_back_compat_alias() -> None:
    # Existing notebook + scripts import SYSTEM_PROMPT_MATH; alias must work.
    assert SYSTEM_PROMPT_MATH is SYSTEM_PROMPT_FREE_MULTI


# ─── Run B prompt rules ────────────────────────────────────────────────────


def test_runb_mcq_has_letter_only_rule() -> None:
    assert "ONLY the letter" in RUNB_SYSTEM_PROMPT_MCQ


def test_runb_mcq_forbids_paren_period_variants() -> None:
    assert "\\boxed{(C)}" in RUNB_SYSTEM_PROMPT_MCQ
    assert "\\boxed{C.}" in RUNB_SYSTEM_PROMPT_MCQ


def test_runb_mcq_no_anti_rounding_or_token_rescue() -> None:
    # These were Phase 1 rules diagnosed harmful on private.
    assert "Do not round" not in RUNB_SYSTEM_PROMPT_MCQ
    assert "running out" not in RUNB_SYSTEM_PROMPT_MCQ
    assert "best-guess" not in RUNB_SYSTEM_PROMPT_MCQ


def test_runb_free_uses_single_box_comma_format() -> None:
    # The whole point of Run B's free-form prompt: single box, comma-sep.
    assert "ONE \\boxed{}" in RUNB_SYSTEM_PROMPT_FREE
    assert "\\boxed{3, 7, 12}" in RUNB_SYSTEM_PROMPT_FREE


def test_runb_free_forbids_quad_and_multibox() -> None:
    # The judger contiguity bug: \\quad / multi-box truncates to last box.
    assert "\\quad" in RUNB_SYSTEM_PROMPT_FREE
    assert "\\qquad" in RUNB_SYSTEM_PROMPT_FREE
    assert "Do NOT use multiple \\boxed{} blocks" in RUNB_SYSTEM_PROMPT_FREE


def test_runb_free_has_symbolic_preference() -> None:
    # Targets private gold distribution: -7\sqrt{149}/149 etc.
    assert "irrational" in RUNB_SYSTEM_PROMPT_FREE.lower()
    assert "\\sqrt" in RUNB_SYSTEM_PROMPT_FREE
    assert "do not convert" in RUNB_SYSTEM_PROMPT_FREE.lower()


def test_runb_free_does_not_use_ambiguous_e_or_log() -> None:
    # Bare "e" and "log" trigger over-symbolic-ification on questions
    # where e is a variable name or where log has multiple LaTeX forms.
    # Use \ln and e^x instead — both are unambiguous.
    assert ", e," not in RUNB_SYSTEM_PROMPT_FREE
    assert ", log," not in RUNB_SYSTEM_PROMPT_FREE


def test_runb_free_no_anti_rounding_or_token_rescue() -> None:
    assert "Do not round" not in RUNB_SYSTEM_PROMPT_FREE
    assert "6 significant figures" not in RUNB_SYSTEM_PROMPT_FREE
    assert "running out" not in RUNB_SYSTEM_PROMPT_FREE
    assert "best-guess" not in RUNB_SYSTEM_PROMPT_FREE


def test_runb_two_prompts_distinct() -> None:
    assert RUNB_SYSTEM_PROMPT_MCQ != RUNB_SYSTEM_PROMPT_FREE


def test_build_prompt_runb_routes_mcq_by_options() -> None:
    sys_p, _ = build_prompt_runb("Q?", ["a", "b"])
    assert sys_p is RUNB_SYSTEM_PROMPT_MCQ


def test_build_prompt_runb_routes_freeform_when_no_options() -> None:
    sys_p, _ = build_prompt_runb("Compute 1+1.", None)
    assert sys_p is RUNB_SYSTEM_PROMPT_FREE


def test_build_prompt_runb_freeform_used_for_multipart_too() -> None:
    # Run B intentionally collapses single + multi into one prompt.
    sys_p, _ = build_prompt_runb("(a) X (b) Y (c) Z", None)
    assert sys_p is RUNB_SYSTEM_PROMPT_FREE


def test_build_prompt_runb_mcq_user_includes_labels() -> None:
    _, user = build_prompt_runb("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user
    assert "B. 4" in user
    assert "C. 5" in user


def test_runb_free_prompt_under_token_budget() -> None:
    # Length sanity: stay well under v6's 349-token Phase 1 prompt.
    # Rough char/token ratio is ~4. 600 chars ≈ 150 tokens.
    assert len(RUNB_SYSTEM_PROMPT_FREE) < 600
