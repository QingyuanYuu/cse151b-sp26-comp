from cse151b_comp.prompts import SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, build_prompt


def test_mcq_prompt_includes_letter_format() -> None:
    sys_p, user_p = build_prompt("What is 1+1?", ["1", "2", "3"])
    assert sys_p == SYSTEM_PROMPT_MCQ
    assert "\\boxed{C}" in sys_p
    assert "A. 1" in user_p
    assert "B. 2" in user_p
    assert "C. 3" in user_p


def test_freeform_prompt_skips_options_block() -> None:
    sys_p, user_p = build_prompt("Compute the integral.", None)
    assert sys_p == SYSTEM_PROMPT_MATH
    assert "\\boxed{}" in sys_p
    assert user_p == "Compute the integral."


def test_freeform_prompt_handles_empty_options_list() -> None:
    sys_p, _ = build_prompt("Compute the integral.", [])
    assert sys_p == SYSTEM_PROMPT_MATH
