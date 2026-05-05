"""Tests for prepare_numina_data.py.

Real ``datasets.load_dataset`` and the HF download path are not exercised
here — they require network and ~5 GB of cache. The pure logic
(filter_quality, split_at_last_boxed, format_sft_row, decontaminate) is
tested directly with in-memory inputs.
"""

from __future__ import annotations

import pytest

from cse151b_comp.prepare_numina_data import (
    decontaminate,
    filter_quality,
    format_sft_row,
    is_valid_row,
    split_at_last_boxed,
)

# ─── is_valid_row / filter_quality ──────────────────────────────────────────


def test_is_valid_row_requires_boxed_in_solution():
    assert not is_valid_row("Compute 1+1.", "The answer is 2.")
    assert is_valid_row("Compute 1+1.", "The answer is \\boxed{2}.")


def test_is_valid_row_rejects_trivial_problem():
    assert not is_valid_row("ok", "\\boxed{42}")
    assert not is_valid_row("", "\\boxed{42}")


def test_is_valid_row_rejects_multipart_ans():
    problem = "Find values for [ANS] and [ANS] given the system below."
    assert not is_valid_row(problem, "\\boxed{42}")


def test_filter_quality_drops_invalid_rows():
    rows = [
        {"problem": "Compute the sum.", "solution": "It is \\boxed{6}.", "source": "math"},
        {"problem": "ok", "solution": "\\boxed{1}"},
        {"problem": "Compute the integral.", "solution": "no boxed here"},
        {"problem": "Compute X.", "solution": "\\boxed{42}"},
    ]
    kept = filter_quality(rows)
    assert len(kept) == 2
    assert {r["problem"] for r in kept} == {"Compute the sum.", "Compute X."}


# ─── split_at_last_boxed ────────────────────────────────────────────────────


def test_split_at_last_boxed_returns_prefix_and_box():
    sol = "First we compute. Then \\boxed{42} is the answer."
    prefix, box = split_at_last_boxed(sol)
    assert prefix == "First we compute. Then"
    assert box == "\\boxed{42}"


def test_split_at_last_boxed_uses_LAST_box_when_multiple():
    sol = "We try \\boxed{wrong} first, but actually \\boxed{42}."
    prefix, box = split_at_last_boxed(sol)
    assert "\\boxed{wrong}" in prefix
    assert box == "\\boxed{42}"


def test_split_at_last_boxed_handles_nested_braces():
    sol = "Therefore \\boxed{\\frac{1}{2}}."
    prefix, box = split_at_last_boxed(sol)
    assert box == "\\boxed{\\frac{1}{2}}"


def test_split_at_last_boxed_returns_none_when_no_box():
    assert split_at_last_boxed("no boxed answer here") is None


def test_split_at_last_boxed_returns_none_for_unbalanced_braces():
    assert split_at_last_boxed("malformed \\boxed{unfinished") is None


# ─── format_sft_row ─────────────────────────────────────────────────────────


def test_format_sft_row_wraps_reasoning_in_think_tags():
    row = {
        "problem": "Compute 2+2.",
        "solution": "We add: 2+2 = 4. Therefore \\boxed{4}.",
        "source": "math",
    }
    sft = format_sft_row(row, idx=7)
    assert sft is not None
    assert sft["id"] == "numina:7"
    assert sft["question_type"] == "free_single"
    assert "<think>" in sft["target_response"]
    assert "</think>" in sft["target_response"]
    assert sft["target_response"].rstrip().endswith("\\boxed{4}")
    # Reasoning text appears inside the think block, not after
    think_start = sft["target_response"].index("<think>")
    think_end = sft["target_response"].index("</think>")
    assert "We add: 2+2 = 4" in sft["target_response"][think_start:think_end]


def test_format_sft_row_routes_to_freesingle_user_prompt():
    row = {"problem": "Compute X.", "solution": "X = 1, so \\boxed{1}.", "source": "math"}
    sft = format_sft_row(row, idx=0)
    assert "Compute X." in sft["user_prompt"]
    assert "\\boxed{}" in sft["user_prompt"]


def test_format_sft_row_returns_none_when_split_fails():
    row = {"problem": "Compute X.", "solution": "no box at all", "source": "math"}
    assert format_sft_row(row, idx=0) is None


# ─── decontaminate ──────────────────────────────────────────────────────────


def test_decontaminate_drops_exact_matches():
    pytest.importorskip("sklearn")
    numina = [
        {"problem": "Find the integral of x squared from 0 to 1.", "solution": "\\boxed{1/3}"},
        {"problem": "Compute the determinant of the 2x2 matrix.", "solution": "\\boxed{0}"},
        {"problem": "Solve for x in 2x + 3 = 7.", "solution": "\\boxed{2}"},
    ]
    contam = [
        "Find the integral of x squared from 0 to 1.",  # exact match for row 0
    ]
    kept, sims = decontaminate(numina, contam, tfidf_threshold=0.5)
    assert len(kept) == 2  # row 0 dropped
    problems = {r["problem"] for r in kept}
    assert "Find the integral of x squared from 0 to 1." not in problems


def test_decontaminate_keeps_unrelated_problems():
    pytest.importorskip("sklearn")
    numina = [
        {"problem": "Find the integral of sin(x) dx.", "solution": "\\boxed{-cos(x)}"},
        {"problem": "Solve the linear system Ax = b.", "solution": "\\boxed{x=A^{-1}b}"},
    ]
    contam = ["What is the boiling point of water in Celsius?"]
    kept, sims = decontaminate(numina, contam, tfidf_threshold=0.85)
    assert len(kept) == 2  # nothing should match a chemistry question
    assert all(s < 0.85 for s in sims)


def test_decontaminate_no_contam_returns_all():
    numina = [{"problem": "Q1.", "solution": "\\boxed{1}"}]
    kept, sims = decontaminate(numina, contam_texts=[], tfidf_threshold=0.85)
    assert kept == numina
    assert sims == [0.0]


def test_decontaminate_no_numina_returns_empty():
    kept, sims = decontaminate([], contam_texts=["something"], tfidf_threshold=0.85)
    assert kept == []
