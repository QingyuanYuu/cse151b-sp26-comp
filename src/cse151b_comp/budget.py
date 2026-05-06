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
no-truncation cases.

Hard ceiling: 22000 tokens per sample. Combined with vLLM
``max_model_len=24576`` and ~2k of prompt+system overhead, this fits the
RTX 4090 24 GB INT4 quantized footprint with K=8 self-consistency at
``max_num_seqs=6``.
"""

from __future__ import annotations

import re


_PART_RX = re.compile(r"\([a-e]\)")

_FLOOR = 10000
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
    """Return per-question ``max_tokens`` in ``[10000, 22000]``.

    Routing:

    - ``options`` truthy → MCQ. Budget scales with ``len(options)``
      because the Qwen3-Thinking model tends to enumerate every option
      in its reasoning trace. Capped at 16k (10-option case).
    - ``[ANS]`` count or ``(a)/(b)/(c)`` markers >= 2 → multi-part.
      Budget scales linearly with K, capped at 22k.
    - else → free-form single. Flat 12k (Phase 0's effective ceiling
      after raising from 12288 to 16384 only mattered for outliers).
    """
    if options:
        # MCQ: 4 opt → 11.2k, 5 opt → 12k, 8 opt → 14.4k, 10 opt → 16k
        n_opts = len(options)
        return max(_FLOOR, min(8000 + n_opts * 800, 16000))

    n_parts = _count_parts(question)
    if n_parts >= 2:
        # K=2 → 10.4k, K=3 → 12.6k, K=5 → 17k, K=8 → 22k (capped)
        return max(_FLOOR, min(6000 + n_parts * 2200, _CEILING))

    return 12000  # free-form single
