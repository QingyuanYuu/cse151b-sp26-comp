"""Per-question max_tokens budget allocator for vLLM SamplingParams.

Why this exists
---------------
Phase 0 starter prompts on private leaderboard had a 17.4 % no-box rate
(``reports/baseline_public_v1.md``); ~57 % of those were caused by hitting
``max_tokens`` mid-thinking and never reaching the final ``\\boxed{}``.
Worst-hit bucket: MCQ at 28 % no-box rate, because 10-option questions
walk through every option in thinking before reaching the final letter.
Multi-part free-form is also truncation-heavy on the LAST sub-answer.

A uniform ``max_tokens=12288`` (notebook default) or ``16384`` (current
launcher) is wasteful for short single-answer questions and not enough
for the failure-mode classes. vLLM's ``llm.generate(prompts,
sampling_params=list[SamplingParams])`` accepts a per-prompt budget; this
module computes that budget from cheap question-text features (no gold,
no model needed).

Calibration source
------------------
Numbers below are read off ``reports/baseline_public_v1.md``'s truncation
distribution (where the no-box came from, by question type and length).
Re-tune if a fresh run shows specific buckets still truncating.

Floor / ceiling discipline
--------------------------
- **Floor = 12288**: no question gets LESS than the previous uniform
  default. The point is to RECOVER truncation cases, not introduce new
  ones in the simple-question buckets.
- **Ceiling = 20480**: leaves ~4k headroom under
  ``max_model_len=24576`` for the system+user prompt + chat template
  overhead. Going closer to max_model_len risks the prompt itself getting
  squeezed.
"""

from __future__ import annotations

import re

# Distinct part markers like (a), (b), (c) — case-insensitive single-letter.
_PART_RX = re.compile(r"\(([a-e])\)", re.IGNORECASE)


def count_parts(question: str) -> int:
    """Estimate how many sub-answers this question expects.

    Returns ``max(n_ANS_placeholders, n_distinct_part_letters, 1)``.

    The ``[ANS]`` placeholder is the more reliable signal; ``(a)/(b)/(c)``
    fallback handles questions that use part-letter formatting instead.
    Returns at least 1 (single-answer baseline).
    """
    n_ans = question.count("[ANS]")
    distinct_part_markers = {m.lower() for m in _PART_RX.findall(question)}
    return max(n_ans, len(distinct_part_markers), 1)


def allocate_max_tokens(
    question: str,
    options: list[str] | None,
    ceiling: int = 20480,
    floor: int = 12288,
) -> int:
    """Return ``max_tokens`` budget for one question.

    Rules:

    - **MCQ**: ``8000 + 800 * len(options)``. 4 options → 11.2k clamps to
      floor; 10 options → 16k.
    - **Multi-part free-form** (≥ 2 parts detected): ``6000 + 2200 *
      n_parts``. K=2 → 10.4k clamps to floor; K=5 → 17k; K=8+ → ceiling.
    - **Single free-form**: floor.

    All results are clamped to ``[floor, ceiling]``. ``floor=12288``
    matches the previous uniform default, so this allocator only ever
    *raises* budget, never lowers it.
    """
    if options:
        budget = 8000 + 800 * len(options)
    else:
        n_parts = count_parts(question)
        if n_parts >= 2:
            budget = 6000 + 2200 * n_parts
        else:
            # Single free-form: keep at floor (was 12288 uniform default).
            budget = floor

    return max(min(budget, ceiling), floor)
