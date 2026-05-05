"""Tests for evaluate.py — score_response + evaluate_rows + summarize."""

from __future__ import annotations

import sys
import pathlib

# Make judger importable from repo root.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cse151b_comp.evaluate import evaluate_rows, score_response, summarize
from judger import Judger


def test_score_mcq_correct() -> None:
    j = Judger(strict_extract=False)
    response = "<think>guessing</think> answer is \\boxed{C}"
    assert score_response(response, "C", ["a", "b", "c", "d"], j) is True


def test_score_mcq_wrong() -> None:
    j = Judger(strict_extract=False)
    response = "answer is \\boxed{A}"
    assert score_response(response, "C", ["a", "b", "c", "d"], j) is False


def test_score_freeform_correct_decimal_vs_fraction() -> None:
    j = Judger(strict_extract=False)
    # gold "1/2" should match boxed 0.5
    response = "Final answer: \\boxed{0.5}"
    assert score_response(response, ["1/2"], None, j) is True


def test_score_freeform_multi_part_csv_style() -> None:
    """Comma-separated style inside one \\boxed{} works."""
    j = Judger(strict_extract=False)
    response = "Final: \\boxed{41, 35, 16}"
    assert score_response(response, ["41", "35", "16"], None, j) is True


def test_score_freeform_multi_part_consecutive_boxes() -> None:
    """Multiple \\boxed{} blocks separated by ONLY whitespace work."""
    j = Judger(strict_extract=False)
    response = "Final answers: \\boxed{41} \\boxed{35} \\boxed{16}"
    assert score_response(response, ["41", "35", "16"], None, j) is True


def test_score_freeform_multi_part_nonconsecutive_FAILS() -> None:
    """Words/parens between boxes break contiguity → judger only sees the last."""
    j = Judger(strict_extract=False)
    response = "(a) \\boxed{41} (b) \\boxed{35} (c) \\boxed{16}"
    # Only \boxed{16} is captured; gold has 3 entries → mismatch.
    assert score_response(response, ["41", "35", "16"], None, j) is False


def test_evaluate_rows_skips_private() -> None:
    rows = [{"id": 1, "response": "anything"}]  # no answer field
    out = evaluate_rows(rows)
    assert "correct" not in out[0]


def test_evaluate_rows_scores_public() -> None:
    rows = [
        {"id": 0, "question": "Q", "answer": "C", "options": ["a", "b", "c"], "response": "\\boxed{C}"},
        {"id": 1, "question": "Q", "answer": ["1"], "response": "\\boxed{1}"},
    ]
    out = evaluate_rows(rows)
    assert all("correct" in r for r in out)
    assert out[0]["correct"] is True
    assert out[1]["correct"] is True


def test_summarize_empty_returns_zero() -> None:
    assert summarize([])["n"] == 0


def test_summarize_basic() -> None:
    rows = [
        {"is_mcq": True, "correct": True, "response": "\\boxed{C}"},
        {"is_mcq": True, "correct": False, "response": "no box"},
        {"is_mcq": False, "correct": True, "response": "\\boxed{42}"},
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["mcq_n"] == 2
    assert s["free_n"] == 1
    assert abs(s["overall_acc"] - 2 / 3) < 1e-9
    assert abs(s["mcq_acc"] - 0.5) < 1e-9
    assert s["free_acc"] == 1.0
    assert abs(s["no_box_rate"] - 1 / 3) < 1e-9
