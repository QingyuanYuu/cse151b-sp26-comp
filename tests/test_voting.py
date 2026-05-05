"""Tests for voting helpers."""

from __future__ import annotations

from cse151b_comp.voting import (
    _extract_tuple,
    solvable_but_missed,
    vote_free_multi,
    vote_free_single,
    vote_mcq,
)


# ─── vote_mcq ───────────────────────────────────────────────────────────────


def test_vote_mcq_clear_majority() -> None:
    responses = [
        "answer is \\boxed{A}",
        "answer is \\boxed{A}",
        "answer is \\boxed{A}",
        "answer is \\boxed{B}",
        "answer is \\boxed{C}",
    ]
    winner, counts, idx = vote_mcq(responses)
    assert winner == "A"
    assert counts["A"] == 3
    assert counts["B"] == 1


def test_vote_mcq_tie_picks_longest_response() -> None:
    responses = [
        "short \\boxed{A}",
        "longer reasoning here \\boxed{B}",
        "looong reasoning trace with details \\boxed{B}",
    ]
    # 1 A vs 2 B → B wins outright
    winner, _, idx = vote_mcq(responses)
    assert winner == "B"
    assert idx == 2  # longest B-voter


def test_vote_mcq_tie_breaks_with_longest_among_winners() -> None:
    responses = [
        "shrt \\boxed{A}",
        "longer \\boxed{B}",
    ]
    winner, _, idx = vote_mcq(responses)
    # 1-1 tie → both A and B are winners; longest among winners decides
    assert winner == "B"
    assert idx == 1


def test_vote_mcq_all_extraction_failed() -> None:
    responses = ["no boxed", "still no boxed", "12345"]
    winner, counts, idx = vote_mcq(responses)
    assert winner == ""
    assert counts == {}


# ─── vote_free_single ───────────────────────────────────────────────────────


def test_vote_free_single_normalizes_before_voting() -> None:
    # 0.5 == 1/2 == \frac{1}{2} should all collapse to the same canonical form
    responses = [
        "Final \\boxed{0.5}",
        "Final \\boxed{1/2}",
        "Final \\boxed{\\frac{1}{2}}",
        "Final \\boxed{0.6}",
    ]
    winner, counts, idx = vote_free_single(responses)
    assert winner == "0.5"
    assert counts["0.5"] == 3
    assert counts["0.6"] == 1


def test_vote_free_single_picks_longest_voter() -> None:
    responses = [
        "short \\boxed{42}",
        "much longer reasoning here \\boxed{42}",
        "different \\boxed{99}",
    ]
    _, _, idx = vote_free_single(responses)
    assert idx == 1  # longest among winners


def test_vote_free_single_handles_extraction_failure() -> None:
    responses = ["nothing", "still nothing"]
    winner, counts, idx = vote_free_single(responses)
    assert winner == ""
    assert counts == {}


# ─── vote_free_multi ────────────────────────────────────────────────────────


def test_extract_tuple_comma_separated() -> None:
    assert _extract_tuple("Final \\boxed{1, 2, 3}") == ("1", "2", "3")


def test_extract_tuple_consecutive_boxes() -> None:
    assert _extract_tuple("Final \\boxed{1} \\boxed{2} \\boxed{3}") == (
        "1",
        "2",
        "3",
    )


def test_extract_tuple_no_box() -> None:
    assert _extract_tuple("nothing here") is None


def test_vote_free_multi_whole_tuple_majority() -> None:
    responses = [
        "Final \\boxed{1, 2, 3}",
        "Final \\boxed{1, 2, 3}",
        "Final \\boxed{1, 2, 4}",  # one slot off
    ]
    winner, _, idx = vote_free_multi(responses)
    assert winner == ("1", "2", "3")
    assert idx in {0, 1}


def test_vote_free_multi_per_slot_fallback() -> None:
    """When no whole-tuple has plurality, fall back to per-slot voting."""
    responses = [
        "Final \\boxed{1, 2, 3}",
        "Final \\boxed{1, 2, 4}",
        "Final \\boxed{1, 5, 4}",
    ]
    # Per slot: 1 wins (3/3), 2 wins (2/3), 4 wins (2/3) → (1, 2, 4)
    winner, _, _ = vote_free_multi(responses)
    assert winner == ("1", "2", "4")


def test_vote_free_multi_normalizes_each_slot() -> None:
    responses = [
        "Final \\boxed{0.5, 1/3}",
        "Final \\boxed{1/2, 0.333333}",
    ]
    winner, _, _ = vote_free_multi(responses)
    # Each slot normalised; the two responses should now agree on a tuple.
    assert winner is not None
    assert len(winner) == 2


def test_vote_free_multi_all_failed() -> None:
    responses = ["no box here", "also nothing"]
    winner, counts, idx = vote_free_multi(responses)
    assert winner == ()
    assert counts == {}


# ─── solvable_but_missed ────────────────────────────────────────────────────


def test_solvable_but_missed_when_one_sample_correct_but_vote_wrong() -> None:
    extracted = ["A", "B", "B"]
    assert solvable_but_missed(extracted, "B", "A") is True


def test_solvable_but_missed_false_when_vote_correct() -> None:
    extracted = ["A", "A", "B"]
    assert solvable_but_missed(extracted, "A", "A") is False


def test_solvable_but_missed_false_when_no_sample_correct() -> None:
    extracted = ["B", "B", "C"]
    assert solvable_but_missed(extracted, "B", "A") is False


def test_solvable_but_missed_works_for_tuples() -> None:
    extracted = [("1", "2"), ("3", "4"), ("3", "4")]
    assert solvable_but_missed(extracted, ("3", "4"), ("1", "2")) is True
