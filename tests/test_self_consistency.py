"""Tests for ``self_consistency.question_type`` routing.

Critical regression test: on the private set (no gold), ``question_type`` was
previously routing **every** free-form question to ``free_single`` because
its routing key was ``isinstance(gold, list) and len(gold) > 1``. The bug
caused the K=8 SC vote on multi-part questions to extract+vote on the LAST
boxed only, then write the full response to CSV — where the judger saw the
``\\quad``-separated boxes, only kept the last contiguous group, and silently
dropped all earlier sub-answers.

After the fix, ``question_type`` mirrors ``prompts.detect_question_type``
(routes off question text + options, never gold).
"""

from __future__ import annotations

from cse151b_comp.self_consistency import question_type


def test_routes_mcq_via_options_present():
    item = {"question": "What is 2+2?", "options": ["3", "4", "5"]}
    assert question_type(item) == "mc"


def test_routes_free_multi_via_ANS_placeholders_no_gold():
    """Private-set scenario: no gold, but question text has [ANS] placeholders.

    Pre-fix: returned 'free_single' (wrong → vote on last-box only).
    Post-fix: returns 'free_multi' (vote on tuple, matches prompt routing).
    """
    item = {
        "question": "Compute (a) [ANS] and (b) [ANS] given the system.",
        "options": None,
    }
    assert question_type(item) == "free_multi"


def test_routes_free_multi_via_part_letter_markers_no_gold():
    item = {
        "question": "(a) Find x. (b) Find y. (c) Find z.",
        "options": None,
    }
    assert question_type(item) == "free_multi"


def test_routes_free_single_when_no_multi_marker():
    item = {"question": "Compute the integral of x^2 from 0 to 1.", "options": None}
    assert question_type(item) == "free_single"


def test_does_not_use_gold_for_routing():
    """Even if gold IS present and is a list of length > 1, routing should
    fall back to question-text inspection (so train and inference agree)."""
    item_unambiguous = {
        "question": "Compute the single integral.",
        "options": None,
        "answer": [1, 2, 3],  # gold is a list, but question is single-answer
    }
    # Pre-fix this would return "free_multi"; post-fix sees question text
    # has no [ANS] / (a)(b) markers, so it's free_single.
    assert question_type(item_unambiguous) == "free_single"


def test_options_overrides_question_markers():
    """MCQ with parenthetical letters in question text should still route
    to 'mc', not 'free_multi'. (Edge case: options-bearing questions
    sometimes mention (a)/(b) inside the prompt.)"""
    item = {
        "question": "Which of the following is true? (a) and (b) below:",
        "options": ["A", "B", "C", "D"],
    }
    assert question_type(item) == "mc"
