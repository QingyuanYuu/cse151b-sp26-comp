"""Tests for cse151b_comp.budget.allocate_max_tokens."""

from __future__ import annotations

from cse151b_comp.budget import _CEILING, _FLOOR, allocate_max_tokens


# ─── MCQ scaling ────────────────────────────────────────────────────────────


def test_mcq_4_options_clamped_to_floor() -> None:
    # 8000 + 4*800 = 11200 — below floor (12000), clamps up.
    assert allocate_max_tokens("Q?", ["a", "b", "c", "d"]) == _FLOOR


def test_mcq_5_options_clamped_to_floor() -> None:
    # 8000 + 5*800 = 12000 — at floor.
    assert allocate_max_tokens("Q?", ["a", "b", "c", "d", "e"]) == _FLOOR


def test_mcq_2_options_clamped_to_floor() -> None:
    # 8000 + 2*800 = 9600 — below floor, clamps to 12000.
    assert allocate_max_tokens("Q?", ["a", "b"]) == _FLOOR


def test_mcq_8_options_above_floor() -> None:
    # 8000 + 8*800 = 14400.
    assert allocate_max_tokens("Q?", [str(i) for i in range(8)]) == 14400


def test_mcq_10_options_caps_at_16k() -> None:
    # 8000 + 10*800 = 16000 — at MCQ ceiling.
    assert allocate_max_tokens("Q?", [str(i) for i in range(10)]) == 16000


def test_mcq_15_options_still_caps_at_16k() -> None:
    assert allocate_max_tokens("Q?", [str(i) for i in range(15)]) == 16000


# ─── Multi-part scaling ─────────────────────────────────────────────────────


def test_multi_part_two_letter_markers_clamped_to_floor() -> None:
    # 6000 + 2*2200 = 10400 — below floor (12000), clamps up.
    q = "(a) compute X. (b) compute Y."
    assert allocate_max_tokens(q, None) == _FLOOR


def test_multi_part_three_letter_markers_above_floor() -> None:
    # 6000 + 3*2200 = 12600 — above floor.
    q = "(a) X (b) Y (c) Z"
    assert allocate_max_tokens(q, None) == 12600


def test_multi_part_five_letter_markers() -> None:
    # 6000 + 5*2200 = 17000.
    q = "(a) X (b) Y (c) Z (d) W (e) V"
    assert allocate_max_tokens(q, None) == 17000


def test_multi_part_eight_ans_caps_at_ceiling() -> None:
    # 6000 + 8*2200 = 23600 — caps at 22000.
    q = "[ANS] [ANS] [ANS] [ANS] [ANS] [ANS] [ANS] [ANS]"
    assert allocate_max_tokens(q, None) == _CEILING


def test_multi_part_uses_max_of_ans_and_letters() -> None:
    # 4 [ANS] vs 2 letters → uses 4.
    q = "[ANS] [ANS] [ANS] [ANS] (a) X (b) Y"
    # 6000 + 4*2200 = 14800
    assert allocate_max_tokens(q, None) == 14800


# ─── Free-form single ───────────────────────────────────────────────────────


def test_free_single_at_floor() -> None:
    # free_single sits at the floor (12k) by design.
    assert allocate_max_tokens("Compute 1+1.", None) == _FLOOR


def test_free_single_with_one_ans_placeholder() -> None:
    # One [ANS] is still single-answer; budget stays at floor.
    assert allocate_max_tokens("Solve: 2+2 = [ANS]", None) == _FLOOR


def test_free_single_with_one_letter_marker() -> None:
    # "compute (a+b)^2" has one (a) marker — still single answer.
    assert allocate_max_tokens("compute (a+b)^2", None) == _FLOOR


# ─── Floor / ceiling invariants ─────────────────────────────────────────────


def test_all_outputs_within_bounds() -> None:
    cases = [
        ("Q", None),
        ("Q [ANS]", None),
        ("Q [ANS] [ANS]", None),
        ("(a) X (b) Y (c) Z", None),
        ("Q", []),
        ("Q", ["a", "b"]),
        ("Q", [str(i) for i in range(20)]),
    ]
    for q, opts in cases:
        v = allocate_max_tokens(q, opts)
        assert _FLOOR <= v <= _CEILING, f"{(q, opts)!r} → {v}"


def test_empty_options_treated_as_free_form() -> None:
    # Empty list is falsy — should route to free-form, not MCQ.
    assert allocate_max_tokens("Compute 2+2.", []) == _FLOOR


def test_floor_matches_phase0_baseline() -> None:
    # Phase 0 used max_tokens=12288. Floor must be at least 12000 so
    # no question type gets less budget than the proven baseline.
    assert _FLOOR >= 12000
