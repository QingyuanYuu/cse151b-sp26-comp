"""Tests for budget.allocate_max_tokens.

Calibration sanity checks against the floor/ceiling rules in
:mod:`cse151b_comp.budget`'s docstring.
"""

from __future__ import annotations

from cse151b_comp.budget import allocate_max_tokens, count_parts

# ─── count_parts ───────────────────────────────────────────────────────────


def test_count_parts_via_ANS_placeholders():
    q = "Find x = [ANS] and y = [ANS]."
    assert count_parts(q) == 2


def test_count_parts_via_part_letters():
    q = "(a) Find x. (b) Find y. (c) Find z."
    assert count_parts(q) == 3


def test_count_parts_takes_max_of_signals():
    # Both signals agree at 2.
    q = "[ANS] [ANS] (a) (b)"
    assert count_parts(q) == 2


def test_count_parts_returns_at_least_one():
    assert count_parts("Compute the integral.") == 1
    assert count_parts("") == 1


def test_count_parts_ignores_repeated_same_letter():
    # 5 (a)s are still one distinct part marker.
    q = "consider (a + b)^2 where (a) is large"
    # \(a\) appears via part-letter regex once; result is max(0_ANS, 1_letter, 1) = 1
    assert count_parts(q) == 1


# ─── allocate_max_tokens — MCQ ────────────────────────────────────────────


def test_mcq_4_options_clamps_to_floor():
    # 8000 + 4*800 = 11.2k < 12288 floor → clamped up.
    assert allocate_max_tokens("Pick one.", options=["A", "B", "C", "D"]) == 12288


def test_mcq_5_options_clamps_to_floor():
    # 8000 + 5*800 = 12k < 12288 floor.
    assert allocate_max_tokens("Pick one.", options=list("ABCDE")) == 12288


def test_mcq_10_options_above_floor():
    # 8000 + 10*800 = 16000.
    assert allocate_max_tokens("Pick one.", options=list("ABCDEFGHIJ")) == 16000


def test_mcq_15_options_clamps_to_ceiling():
    # 8000 + 15*800 = 20k. Just below ceiling 20480.
    assert allocate_max_tokens("Pick one.", options=list("A" * 15)) == 20000


def test_mcq_huge_options_clamps_to_ceiling():
    # 8000 + 30*800 = 32000 → ceiling.
    assert allocate_max_tokens("Pick one.", options=["X"] * 30) == 20480


# ─── allocate_max_tokens — free-form multi ────────────────────────────────


def test_multi_2parts_clamps_to_floor():
    # 6000 + 2*2200 = 10400 < 12288 floor.
    q = "(a) Find x. (b) Find y."
    assert allocate_max_tokens(q, options=None) == 12288


def test_multi_3parts_above_floor():
    # 6000 + 3*2200 = 12600.
    q = "(a) (b) (c)"
    assert allocate_max_tokens(q, options=None) == 12600


def test_multi_5parts_above_floor():
    # 6000 + 5*2200 = 17000.
    q = "[ANS] [ANS] [ANS] [ANS] [ANS]"
    assert allocate_max_tokens(q, options=None) == 17000


def test_multi_8parts_clamps_to_ceiling():
    # 6000 + 8*2200 = 23600 > 20480 ceiling.
    q = "[ANS]" * 8
    assert allocate_max_tokens(q, options=None) == 20480


# ─── allocate_max_tokens — free-form single ───────────────────────────────


def test_single_free_returns_floor():
    assert allocate_max_tokens("Compute the integral.", options=None) == 12288


def test_single_free_with_one_part_marker_returns_floor():
    q = "consider x^2 where (a + 1) is the leading coefficient"
    assert allocate_max_tokens(q, options=None) == 12288


# ─── Floor / ceiling overrides ────────────────────────────────────────────


def test_custom_floor_lowers_minimum():
    # Caller can override floor to allow tighter budgets.
    assert allocate_max_tokens("Compute X.", options=None, floor=8000) == 8000


def test_custom_ceiling_caps_max():
    # Caller can lower ceiling to keep KV cache headroom.
    assert allocate_max_tokens("[ANS]" * 8, options=None, ceiling=18000) == 18000


def test_no_question_text_with_options_returns_mcq_budget():
    """Even a degenerate empty question, options-bearing routes to MCQ math."""
    assert allocate_max_tokens("", options=["A", "B", "C", "D"]) == 12288
