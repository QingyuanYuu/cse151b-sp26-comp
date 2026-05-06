"""Tests for cse151b_comp.self_consistency.question_type.

Specifically guards against the v6_sc_k8 0.448 regression: the previous
gold-based heuristic returned ``free_single`` for every multi-part
free-form question on the private set (no gold), causing the
``vote_free_multi`` branch to never fire on private. The fix reuses
prompts.detect_question_type, which works from the question text only.
"""

from __future__ import annotations

from cse151b_comp.self_consistency import question_type


def test_mc_when_options_present() -> None:
    assert question_type({"question": "Q?", "options": ["a", "b"]}) == "mc"


def test_mc_when_options_present_even_without_gold() -> None:
    # Private-set scenario: no gold, but options imply MC.
    assert question_type({"question": "Q?", "options": ["a", "b", "c"]}) == "mc"


def test_free_multi_detected_from_question_when_no_gold() -> None:
    # The regression case: private rows have no gold; must still detect multi.
    item = {"question": "(a) compute X. (b) compute Y. (c) compute Z.", "options": None}
    assert question_type(item) == "free_multi"


def test_free_multi_detected_from_ans_placeholders() -> None:
    item = {"question": "Solve: [ANS] [ANS]", "options": None}
    assert question_type(item) == "free_multi"


def test_free_single_when_no_multi_markers_and_no_gold() -> None:
    item = {"question": "Compute 2+2.", "options": None}
    assert question_type(item) == "free_single"


def test_free_single_with_one_letter_marker() -> None:
    # The (a) in (a+b)^2 should NOT trigger free_multi.
    item = {"question": "Expand (a+b)^2", "options": None}
    assert question_type(item) == "free_single"


def test_options_missing_field() -> None:
    # Robust to missing 'options' key entirely.
    assert question_type({"question": "Compute 1+1."}) == "free_single"


def test_question_missing_returns_free_single() -> None:
    # Defensive: even with no question text, do not crash.
    assert question_type({}) == "free_single"


def test_does_not_use_gold_field() -> None:
    # Even if a stale gold list is present, routing must follow the question.
    item = {
        "question": "Compute 1+1.",
        "options": None,
        "answer": ["1", "2", "3", "4"],  # would have triggered free_multi under old logic
    }
    assert question_type(item) == "free_single"
