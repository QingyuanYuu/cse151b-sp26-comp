"""Voting helpers for self-consistency.

Three voting strategies, one per question type:

- ``vote_mcq`` — majority over extracted letters, tied resolved by longest
  valid reasoning trace.
- ``vote_free_single`` — extracts the final boxed value from each sample,
  normalizes it via :func:`extract.normalize_answer`, takes the mode of
  the normalized form, and returns both the canonical winner and the
  longest original response that voted for it.
- ``vote_free_multi`` — extracts every sample's full tuple (either a
  comma-split single boxed or a list of K boxed blocks), normalizes
  each element, and votes on the **whole tuple** because gold equality
  is tuple-level (all-or-nothing). If no tuple has plurality it falls
  back to per-slot voting and reports the chosen path.

All voters return a triple ``(winning_canonical, vote_counts, winning_idx)``,
where ``winning_idx`` indexes into the per-sample lists for use as the
``winning_response`` shown to the grader.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from cse151b_comp.extract import (
    extract_all_final_boxed,
    extract_letter,
    normalize_answer,
)


# ─── Helper: pick the longest response among indices ────────────────────────


def _longest_idx(indices: Iterable[int], responses: list[str]) -> int:
    """Among ``indices``, return the one whose response is longest."""
    return max(indices, key=lambda i: len(responses[i]))


# ─── MCQ ────────────────────────────────────────────────────────────────────


def vote_mcq(responses: list[str]) -> tuple[str, dict[str, int], int]:
    """Vote a single-letter MCQ answer across K samples.

    Returns ``(winning_letter, vote_counts, winning_idx)``.
    Empty letters (extraction failure) are not voted on, but if all
    samples failed, returns ``("", {}, 0)``.
    """
    letters = [extract_letter(r) for r in responses]
    counts = Counter(l for l in letters if l)
    if not counts:
        return "", {}, 0
    top_count = max(counts.values())
    winners = {l for l, c in counts.items() if c == top_count}
    voters = [i for i, l in enumerate(letters) if l in winners]
    winning_idx = _longest_idx(voters, responses)
    return letters[winning_idx], dict(counts), winning_idx


# ─── Free-form single ───────────────────────────────────────────────────────


def vote_free_single(
    responses: list[str],
) -> tuple[str, dict[str, int], int]:
    """Vote a single-value free-form answer across K samples.

    Returns ``(winning_normalized, vote_counts, winning_idx)``.
    """
    extracted = []
    for r in responses:
        boxes = extract_all_final_boxed(r)
        extracted.append(boxes[-1] if boxes else "")
    normed = [normalize_answer(e) for e in extracted]

    counts = Counter(n for n in normed if n)
    if not counts:
        return "", {}, 0
    top_count = max(counts.values())
    winners = {n for n, c in counts.items() if c == top_count}
    voters = [i for i, n in enumerate(normed) if n in winners]
    winning_idx = _longest_idx(voters, responses)
    return normed[winning_idx], dict(counts), winning_idx


# ─── Free-form multi ────────────────────────────────────────────────────────


def _extract_tuple(response: str) -> tuple[str, ...] | None:
    """Pull out the K-tuple of normalized answers from a single response.

    Two source shapes are accepted:

    1. ``\\boxed{a, b, c}`` — split by top-level comma.
    2. ``\\boxed{a} \\boxed{b} \\boxed{c}`` (contiguous group) — already a
       list from :func:`extract_all_final_boxed`.

    Returns ``None`` if the response yields no boxed content.
    """
    boxes = extract_all_final_boxed(response)
    if not boxes:
        return None
    if len(boxes) == 1 and "," in boxes[0]:
        # Split top-level commas (no nested-brace handling — judger does the
        # same simple split, so we mirror its behaviour).
        parts = [p.strip() for p in boxes[0].split(",")]
    else:
        parts = boxes
    return tuple(normalize_answer(p) for p in parts)


def vote_free_multi(
    responses: list[str],
) -> tuple[tuple[str, ...] | tuple[()], dict[str, int], int]:
    """Vote a multi-answer free-form response across K samples.

    Whole-tuple equality is the primary scheme (because the grader is
    all-or-nothing). When the winning tuple is *no plurality* (every tuple
    appears exactly once), fall back to per-slot voting; the per-slot
    fallback assumes all candidate tuples have the same K.
    """
    tuples = [_extract_tuple(r) for r in responses]
    valid = [t for t in tuples if t is not None]
    if not valid:
        return (), {}, 0

    counts: Counter[tuple[str, ...]] = Counter(valid)
    top_count = max(counts.values())

    if top_count > 1:
        # Plurality wins.
        winners = {t for t, c in counts.items() if c == top_count}
        voters = [i for i, t in enumerate(tuples) if t in winners]
        winning_idx = _longest_idx(voters, responses)
        winning_tuple = tuples[winning_idx]
    else:
        # No plurality — fall back to per-slot voting.
        # Use the modal K (most-common length) so we don't average across
        # different shapes.
        lengths = Counter(len(t) for t in valid)
        modal_k = max(lengths, key=lambda k: lengths[k])
        same_shape = [t for t in valid if len(t) == modal_k]
        winning_tuple = tuple(
            Counter(t[j] for t in same_shape).most_common(1)[0][0]
            for j in range(modal_k)
        )
        # Winning response: the one whose tuple matches the constructed
        # winner (if any), else the longest sample of the modal shape.
        match_idx = next(
            (i for i, t in enumerate(tuples) if t == winning_tuple), -1
        )
        if match_idx >= 0:
            winning_idx = match_idx
        else:
            shape_voters = [i for i, t in enumerate(tuples)
                            if t is not None and len(t) == modal_k]
            winning_idx = _longest_idx(shape_voters, responses)

    # Reduce keys to JSON-friendly strings for the on-disk schema.
    counts_str: dict[str, int] = {repr(k): v for k, v in counts.items()}
    return winning_tuple, counts_str, winning_idx


# ─── Solvable-but-missed metric ─────────────────────────────────────────────


def solvable_but_missed(
    sample_extracted: list[str | tuple[str, ...]],
    winning_extracted: str | tuple[str, ...],
    gold_normalized: str | tuple[str, ...],
) -> bool:
    """Return True if at least one sample matched the gold but the vote
    selected a different answer.

    Caller must pre-normalize both sides.
    """
    if winning_extracted == gold_normalized:
        return False
    return any(s == gold_normalized for s in sample_extracted)
