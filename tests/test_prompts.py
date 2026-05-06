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
    RUNC_SYSTEM_PROMPT_FREE,
    RUNC_SYSTEM_PROMPT_MCQ,
    RUND_SYSTEM_PROMPT_FREE,
    RUND_SYSTEM_PROMPT_MCQ,
    RUNE_SYSTEM_PROMPT_FREE_BASE,
    RUNE_SYSTEM_PROMPT_MCQ,
    SYSTEM_PROMPT_FREE_MULTI,
    SYSTEM_PROMPT_FREE_SINGLE,
    SYSTEM_PROMPT_MATH,
    SYSTEM_PROMPT_MCQ,
    build_prompt,
    build_prompt_runb,
    build_prompt_runc,
    build_prompt_rund,
    build_prompt_rune,
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


# ─── Run C prompt rules ────────────────────────────────────────────────────


def test_runc_inherits_runb_anti_pattern_rules() -> None:
    # Run C must keep all of Run B's working anti-pattern rules.
    assert "\\quad" in RUNC_SYSTEM_PROMPT_FREE
    assert "\\qquad" in RUNC_SYSTEM_PROMPT_FREE
    assert "Do NOT use multiple \\boxed{} blocks" in RUNC_SYSTEM_PROMPT_FREE
    assert "irrational" in RUNC_SYSTEM_PROMPT_FREE.lower()
    assert "\\sqrt" in RUNC_SYSTEM_PROMPT_FREE
    assert "\\boxed{(C)}" in RUNC_SYSTEM_PROMPT_MCQ
    assert "\\boxed{C.}" in RUNC_SYSTEM_PROMPT_MCQ


def test_runc_mcq_has_end_with_box_rule() -> None:
    # Run C addition: structural end-with-box rule.
    assert "must end with" in RUNC_SYSTEM_PROMPT_MCQ.lower()
    assert "\\boxed{X}" in RUNC_SYSTEM_PROMPT_MCQ


def test_runc_free_has_end_with_box_rule() -> None:
    # Run C addition: structural end-with-box rule (free-form variant).
    assert "End your response" in RUNC_SYSTEM_PROMPT_FREE


def test_runc_free_has_text_bool_examples() -> None:
    # Run C addition: counter-balance the symbolic-preference rule for
    # non-numeric answers.
    assert "\\boxed{Yes}" in RUNC_SYSTEM_PROMPT_FREE
    assert "\\boxed{Tuesday}" in RUNC_SYSTEM_PROMPT_FREE
    assert "\\boxed{True}" in RUNC_SYSTEM_PROMPT_FREE


def test_runc_does_not_use_v2_token_rescue_phrasing() -> None:
    # v2 Phase 1's token-rescue phrasing produced literal \\boxed{...}
    # placeholders. Run C avoids the "if running out" trigger.
    for prompt in (RUNC_SYSTEM_PROMPT_MCQ, RUNC_SYSTEM_PROMPT_FREE):
        assert "running out" not in prompt
        assert "best-guess" not in prompt
        assert "best guess" not in prompt
        assert "if you are unable" not in prompt.lower()


def test_runc_does_not_have_anti_rounding_or_token_rescue() -> None:
    # Inherited from Run B: these Phase 1 rules regressed on private.
    for prompt in (RUNC_SYSTEM_PROMPT_MCQ, RUNC_SYSTEM_PROMPT_FREE):
        assert "Do not round" not in prompt
        assert "6 significant figures" not in prompt


def test_runc_does_not_use_ambiguous_e_or_log() -> None:
    # Same guard as Run B against bare "e" and "log".
    assert ", e," not in RUNC_SYSTEM_PROMPT_FREE
    assert ", log," not in RUNC_SYSTEM_PROMPT_FREE


def test_runc_two_prompts_distinct() -> None:
    assert RUNC_SYSTEM_PROMPT_MCQ != RUNC_SYSTEM_PROMPT_FREE


def test_runc_distinct_from_runb() -> None:
    # Verify Run C actually differs from Run B (no accidental no-op).
    assert RUNC_SYSTEM_PROMPT_MCQ != RUNB_SYSTEM_PROMPT_MCQ
    assert RUNC_SYSTEM_PROMPT_FREE != RUNB_SYSTEM_PROMPT_FREE


def test_build_prompt_runc_routes_mcq_by_options() -> None:
    sys_p, _ = build_prompt_runc("Q?", ["a", "b"])
    assert sys_p is RUNC_SYSTEM_PROMPT_MCQ


def test_build_prompt_runc_routes_freeform_when_no_options() -> None:
    sys_p, _ = build_prompt_runc("Compute 1+1.", None)
    assert sys_p is RUNC_SYSTEM_PROMPT_FREE


def test_build_prompt_runc_freeform_used_for_multipart_too() -> None:
    sys_p, _ = build_prompt_runc("(a) X (b) Y (c) Z", None)
    assert sys_p is RUNC_SYSTEM_PROMPT_FREE


def test_build_prompt_runc_mcq_user_includes_labels() -> None:
    _, user = build_prompt_runc("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user
    assert "B. 4" in user
    assert "C. 5" in user


def test_runc_prompts_under_token_budget() -> None:
    # Length sanity: Run C is allowed to grow ~25% over Run B for the
    # end-with-box rule + text examples, but stay well under Phase 1's
    # 349-token regression zone.
    # MCQ: ~110 tokens / ~440 chars
    # FREE: ~175 tokens / ~700 chars
    assert len(RUNC_SYSTEM_PROMPT_MCQ) < 500
    assert len(RUNC_SYSTEM_PROMPT_FREE) < 750


# ─── Run D prompt rules ────────────────────────────────────────────────────


def test_rund_inherits_runc_anti_pattern_rules() -> None:
    # Run D must keep all of Run C's working anti-pattern rules.
    assert "\\quad" in RUND_SYSTEM_PROMPT_FREE
    assert "\\qquad" in RUND_SYSTEM_PROMPT_FREE
    assert "Do NOT use multiple \\boxed{} blocks" in RUND_SYSTEM_PROMPT_FREE
    assert "irrational" in RUND_SYSTEM_PROMPT_FREE.lower()
    assert "\\sqrt" in RUND_SYSTEM_PROMPT_FREE
    assert "\\boxed{(C)}" in RUND_SYSTEM_PROMPT_MCQ
    assert "\\boxed{C.}" in RUND_SYSTEM_PROMPT_MCQ


def test_rund_keeps_end_with_box_rule() -> None:
    assert "must end with" in RUND_SYSTEM_PROMPT_MCQ.lower()
    assert "End your response" in RUND_SYSTEM_PROMPT_FREE


def test_rund_mcq_has_one_worked_example() -> None:
    # MCQ adds one worked example demonstrating letter-only output.
    assert "Example:" in RUND_SYSTEM_PROMPT_MCQ
    assert "Q:" in RUND_SYSTEM_PROMPT_MCQ
    assert "A:" in RUND_SYSTEM_PROMPT_MCQ
    # The example resolves to \boxed{C} (matching the rule's letter).
    assert "\\boxed{C}" in RUND_SYSTEM_PROMPT_MCQ


def test_rund_free_has_three_worked_examples() -> None:
    # Three Q/A demonstrations: symbolic, multi-part, bool.
    # The "A:" pattern after "Q:" identifies each demonstration.
    qa_count = RUND_SYSTEM_PROMPT_FREE.count("Q:")
    assert qa_count == 3, f"Expected 3 Q→A examples, got {qa_count}"


def test_rund_free_examples_target_failure_modes() -> None:
    # Symbolic example: \\boxed{9\\pi} (counter decimal-conversion).
    assert "\\boxed{9\\pi}" in RUND_SYSTEM_PROMPT_FREE
    # Multi-part comma example: \\boxed{4, -7} (counter multi-box).
    assert "\\boxed{4, -7}" in RUND_SYSTEM_PROMPT_FREE
    # Text example: \\boxed{Tuesday} (counter symbolic-only bias for
    # natural-form answers; weekday picked instead of bool to avoid
    # any Yes/No directional bias on private gold distribution).
    assert "\\boxed{Tuesday}" in RUND_SYSTEM_PROMPT_FREE


def test_rund_free_does_not_have_yes_no_example() -> None:
    # Avoid biasing the model toward Yes or No on bool questions —
    # private gold's Yes/No distribution is unknown, so a worked
    # example with a directional answer is asymmetric risk. The inline
    # rule still lists Yes/True/Tuesday so bool answers remain valid.
    assert "\\boxed{Yes}" not in RUND_SYSTEM_PROMPT_FREE
    assert "\\boxed{No}" not in RUND_SYSTEM_PROMPT_FREE
    assert "\\boxed{True}" not in RUND_SYSTEM_PROMPT_FREE
    assert "\\boxed{False}" not in RUND_SYSTEM_PROMPT_FREE


def test_rund_uses_qa_format_to_resist_echo() -> None:
    # The id=5 echo bug in Run C came from inline boxed values without
    # surrounding question context. Q→A frame requires the model to also
    # fabricate a question text to plagiarise, much less likely.
    # Both prompts must wrap any boxed example in a Q/A pair.
    for prompt in (RUND_SYSTEM_PROMPT_MCQ, RUND_SYSTEM_PROMPT_FREE):
        # Every \\boxed{...} block in the prompt should follow either a
        # rule explanation ("like \\boxed{...}", "e.g. \\boxed{...}") or
        # appear after an "A:" line (worked example answer).
        assert "Q:" in prompt
        assert "A:" in prompt


def test_rund_does_not_have_anti_rounding_or_token_rescue() -> None:
    for prompt in (RUND_SYSTEM_PROMPT_MCQ, RUND_SYSTEM_PROMPT_FREE):
        assert "Do not round" not in prompt
        assert "running out" not in prompt
        assert "best-guess" not in prompt
        assert "best guess" not in prompt


def test_rund_does_not_use_ambiguous_e_or_log() -> None:
    assert ", e," not in RUND_SYSTEM_PROMPT_FREE
    assert ", log," not in RUND_SYSTEM_PROMPT_FREE


def test_rund_distinct_from_runc() -> None:
    assert RUND_SYSTEM_PROMPT_MCQ != RUNC_SYSTEM_PROMPT_MCQ
    assert RUND_SYSTEM_PROMPT_FREE != RUNC_SYSTEM_PROMPT_FREE


def test_rund_two_prompts_distinct() -> None:
    assert RUND_SYSTEM_PROMPT_MCQ != RUND_SYSTEM_PROMPT_FREE


def test_build_prompt_rund_routes_mcq_by_options() -> None:
    sys_p, _ = build_prompt_rund("Q?", ["a", "b"])
    assert sys_p is RUND_SYSTEM_PROMPT_MCQ


def test_build_prompt_rund_routes_freeform_when_no_options() -> None:
    sys_p, _ = build_prompt_rund("Compute 1+1.", None)
    assert sys_p is RUND_SYSTEM_PROMPT_FREE


def test_build_prompt_rund_freeform_used_for_multipart_too() -> None:
    sys_p, _ = build_prompt_rund("(a) X (b) Y (c) Z", None)
    assert sys_p is RUND_SYSTEM_PROMPT_FREE


def test_build_prompt_rund_mcq_user_includes_labels() -> None:
    _, user = build_prompt_rund("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user
    assert "B. 4" in user
    assert "C. 5" in user


def test_rund_prompts_under_token_budget() -> None:
    # Run D adds few-shot examples on top of Run C. Length budget:
    # MCQ ~130 tokens / ~520 chars (Run C was 92t/370c).
    # FREE ~270 tokens / ~1080 chars (Run C was 167t/668c).
    # Hard ceiling: 349-token Phase 1 prompt regressed -8.1pp on private,
    # so cap free at 1300 chars (~325 tokens) with margin.
    assert len(RUND_SYSTEM_PROMPT_MCQ) < 600
    assert len(RUND_SYSTEM_PROMPT_FREE) < 1300


# ─── Run E prompt rules (ceiling probe) ────────────────────────────────────


def test_rune_inherits_runb_anti_pattern_rules() -> None:
    assert "\\quad" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\qquad" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "Do NOT use multiple \\boxed{} blocks" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "irrational" in RUNE_SYSTEM_PROMPT_FREE_BASE.lower()
    assert "\\boxed{(C)}" in RUNE_SYSTEM_PROMPT_MCQ
    assert "\\boxed{C.}" in RUNE_SYSTEM_PROMPT_MCQ


def test_rune_keeps_end_with_box_rule() -> None:
    assert "must end with" in RUNE_SYSTEM_PROMPT_MCQ.lower()
    assert "End your response" in RUNE_SYSTEM_PROMPT_FREE_BASE


def test_rune_mcq_has_elimination_strategy() -> None:
    # Run E adds explicit MCQ elimination instruction (10-opt rescue).
    assert "eliminate" in RUNE_SYSTEM_PROMPT_MCQ.lower()
    assert "8+" in RUNE_SYSTEM_PROMPT_MCQ or "many options" in RUNE_SYSTEM_PROMPT_MCQ.lower()


def test_rune_free_has_concise_hint() -> None:
    # Run E adds anti-truncation conciseness instruction.
    assert "concise" in RUNE_SYSTEM_PROMPT_FREE_BASE.lower()
    assert "do not restate" in RUNE_SYSTEM_PROMPT_FREE_BASE.lower()


def test_rune_free_has_five_examples() -> None:
    # Run E expands Run D's 3 examples to 5 (adds R^2 + derivative).
    qa_count = RUNE_SYSTEM_PROMPT_FREE_BASE.count("Q:")
    assert qa_count == 5, f"Expected 5 Q→A examples, got {qa_count}"


def test_rune_free_examples_target_failure_modes() -> None:
    # Original 3 from Run D.
    assert "\\boxed{9\\pi}" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{4, -7}" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{Tuesday}" in RUNE_SYSTEM_PROMPT_FREE_BASE
    # Two new examples for Run E: statistics R^2, calculus derivative.
    assert "R^2" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{0.8}" in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{3x^2 + 2}" in RUNE_SYSTEM_PROMPT_FREE_BASE


def test_rune_free_does_not_have_yes_no_example() -> None:
    # Same audit as Run D: no directional bool worked example.
    assert "\\boxed{Yes}" not in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{No}" not in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{True}" not in RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "\\boxed{False}" not in RUNE_SYSTEM_PROMPT_FREE_BASE


def test_rune_does_not_have_anti_rounding_or_token_rescue() -> None:
    for prompt in (RUNE_SYSTEM_PROMPT_MCQ, RUNE_SYSTEM_PROMPT_FREE_BASE):
        assert "Do not round" not in prompt
        assert "running out" not in prompt
        assert "best-guess" not in prompt
        assert "best guess" not in prompt


def test_rune_topic_routing_stats() -> None:
    # Stats keywords route to the stats suffix.
    sys_p, _ = build_prompt_rune(
        "An ANOVA test yields F = 4.5. State the conclusion.", None
    )
    assert "Statistics tip" in sys_p


def test_rune_topic_routing_calculus() -> None:
    sys_p, _ = build_prompt_rune("Compute the derivative of x^2.", None)
    assert "Calculus tip" in sys_p


def test_rune_topic_routing_linalg() -> None:
    sys_p, _ = build_prompt_rune(
        "Find the eigenvalues of the 2x2 matrix [[1,2],[3,4]].", None
    )
    assert "Linear algebra tip" in sys_p


def test_rune_topic_routing_probability() -> None:
    sys_p, _ = build_prompt_rune(
        "What is the probability of rolling a 6?", None
    )
    assert "Probability tip" in sys_p


def test_rune_topic_routing_default_no_suffix() -> None:
    # No keyword match → no suffix.
    sys_p, _ = build_prompt_rune("Compute 2 + 2.", None)
    assert sys_p == RUNE_SYSTEM_PROMPT_FREE_BASE
    assert "tip:" not in sys_p


def test_rune_distinct_from_rund() -> None:
    assert RUNE_SYSTEM_PROMPT_MCQ != RUND_SYSTEM_PROMPT_MCQ
    assert RUNE_SYSTEM_PROMPT_FREE_BASE != RUND_SYSTEM_PROMPT_FREE


def test_build_prompt_rune_routes_mcq_by_options() -> None:
    sys_p, _ = build_prompt_rune("Q?", ["a", "b"])
    assert sys_p is RUNE_SYSTEM_PROMPT_MCQ


def test_build_prompt_rune_routes_freeform_when_no_options() -> None:
    sys_p, _ = build_prompt_rune("Compute 1+1.", None)
    # Default topic suffix is empty, so equality with base.
    assert sys_p == RUNE_SYSTEM_PROMPT_FREE_BASE


def test_build_prompt_rune_mcq_user_includes_labels() -> None:
    _, user = build_prompt_rune("What is 2+2?", ["3", "4", "5"])
    assert "A. 3" in user
    assert "B. 4" in user
    assert "C. 5" in user


def test_rune_prompts_under_token_budget() -> None:
    # Run E intentionally pushes the upper end of the empirical sweet
    # spot. MCQ ~150 tokens (600 chars). Free base ~270 tokens (1080
    # chars). With a topic suffix, free can reach ~320 tokens (1280
    # chars). Hard ceiling: stay below 1400 chars on free base
    # (= ~350 tokens, just under Phase 1's 349-token regression line).
    assert len(RUNE_SYSTEM_PROMPT_MCQ) < 800
    assert len(RUNE_SYSTEM_PROMPT_FREE_BASE) < 1400
