"""Per-question ``max_tokens`` allocation for vLLM inference.

Why this exists
---------------

The Phase 0 baseline had a 17.4 % no-``\\boxed{}`` rate on private (164/943
responses), and 57 % of those were due to the uniform ``max_tokens=12288``
budget being hit mid-thinking. The truncation rate is **not uniform**
across question types:

- MCQ with ``len(options) >= 8``: model expands every option, runs out of
  reasoning room (28 % no-box, the worst bucket on Phase 0).
- Multi-part free-form with ``K >= 4`` sub-answers: each part needs its
  own derivation; cumulative budget consumption is roughly linear in K.
- Free-form single: short, plenty of headroom at 12k.

A flat budget either over-spends on the easy single-answer cases or
starves the hard multi-part cases. This module gives each question
exactly what it needs based on its surface features (no gold required,
so it works on the private set).

Calibration
-----------

The coefficients below were derived from
``reports/baseline_public_v1.json``'s truncation distribution. The bands
are intentionally conservative — we only ever **add** budget relative to
the 12k baseline, never reduce, so a miscalibration cannot degrade the
no-truncation cases. ``_FLOOR = 12000`` matches Phase 0's effective
``max_tokens=12288`` (the v5_sanity 0.583 baseline rounded down to a
clean 12k), so no question type ever receives less budget than the
proven baseline.

Hard ceiling: 22000 tokens per sample. Combined with vLLM
``max_model_len=24576`` and ~2k of prompt+system overhead, this fits the
RTX 4090 24 GB INT4 quantized footprint with K=8 self-consistency at
``max_num_seqs=6``.
"""

from __future__ import annotations

import re


_PART_RX = re.compile(r"\([a-e]\)")

# 12000 = Phase 0's effective max_tokens floor (12288 rounded). Never go
# below this — see module docstring + reports/public_private_gap_analysis.md.
_FLOOR = 12000
_CEILING = 22000


def _count_parts(question: str) -> int:
    """Estimate the number of sub-answers a question expects.

    Two signals, take the max:

    - ``[ANS]`` placeholder count.
    - distinct ``(a)``…``(e)`` alphabetical part markers.

    Returns 0 if neither is present (treat as single-answer).
    """
    n_ans = question.count("[ANS]")
    n_letters = len({m.lower() for m in _PART_RX.findall(question)})
    return max(n_ans, n_letters)


def allocate_max_tokens(question: str, options: list[str] | None) -> int:
    """Return per-question ``max_tokens`` in ``[12000, 22000]``.

    Routing:

    - ``options`` truthy → MCQ. Budget scales with ``len(options)``
      because the Qwen3-Thinking model tends to enumerate every option
      in its reasoning trace. Capped at 18k (10-option case).
    - ``[ANS]`` count or ``(a)/(b)/(c)`` markers >= 2 → multi-part.
      Budget scales linearly with K, capped at 22k.
    - else → free-form single. Flat at the floor (12k) — matches
      Phase 0's effective baseline.

    All branches floor at ``_FLOOR = 12000`` so the worst-case budget
    matches the Phase 0 ``max_tokens=12288`` baseline exactly.

    Run C calibration (vs Run B): MCQ formula raised from
    ``8000 + 800·n cap 16k`` to ``8000 + 1000·n cap 18k``. The 10-option
    bucket holds 89 % of all private MCQ (267 / 300) and was Run B's
    worst no-box class (29 / 267 = 10.9 % truncated). +2 k headroom is
    the cheapest available rescue for that bucket.
    """
    if options:
        # MCQ: 4 opt → 12k (floor), 5 opt → 13k, 8 opt → 16k, 10 opt → 18k
        n_opts = len(options)
        return max(_FLOOR, min(8000 + n_opts * 1000, 18000))

    n_parts = _count_parts(question)
    if n_parts >= 2:
        # K=2 → 12k (floor), K=3 → 12.6k, K=5 → 17k, K=8 → 22k (capped)
        return max(_FLOOR, min(6000 + n_parts * 2200, _CEILING))

    return _FLOOR  # free-form single — floored at 12k


# ─── v2 budget (Run G): raise floor to 16k, ceilings +2k ──────────────────
#
# Run C private regression analysis isolated free_single's 12k allocation
# as a likely contributor: Run B used flat 16k and didn't lose free_single
# accuracy; Run C reduced free_single to 12k and the median response was
# 384 chars shorter (less reasoning depth on hard questions).
#
# v2 raises the floor to 16k so every question gets at least Run B's
# proven baseline budget. Ceilings also lift +2k:
#
# - MCQ ceiling 18k → 20k (10-opt bucket gets +2k headroom).
# - multi-part ceiling 22k → 24k (K=8+ questions get +2k).
#
# Required: --max-model-len ≥ 26624 to fit 24k output + ~2k input.

_FLOOR_V2 = 16000
_CEILING_V2_MCQ = 20000
_CEILING_V2_MULTI = 24000


def allocate_max_tokens_v2(question: str, options: list[str] | None) -> int:
    """v2: aggressive budget. Floor 16k, MCQ cap 20k, multi cap 24k.

    Used by Run G (Run F prompt + v2 budget). Targeted fix for Run C's
    private regression where free_single's 12k starved hard reasoning.
    """
    if options:
        # MCQ: 4 opt → 16k (floor), 8 opt → 17.6k, 10 opt → 20k (cap)
        n_opts = len(options)
        return max(_FLOOR_V2, min(8000 + n_opts * 1200, _CEILING_V2_MCQ))

    n_parts = _count_parts(question)
    if n_parts >= 2:
        # K=2 → 16k (floor), K=5 → 18.5k, K=8 → 24k (cap)
        return max(_FLOOR_V2, min(6000 + n_parts * 2500, _CEILING_V2_MULTI))

    return _FLOOR_V2  # free_single — floored at 16k (Run B baseline)
