"""System and user prompt templates for the CSE 151B math competition.

Phase 1 fixes (vs the starter prompts):

- MCQ: explicit negative examples for `\\boxed{(C)}`, `\\boxed{C.}`, etc., which the
  judger silently rejects.
- Multi-part free-form: explicit instruction to use either K consecutive
  `\\boxed{}` blocks OR one comma-separated `\\boxed{a, b, ..., k}`. Prevents
  mixed styles that confuse the extractor.
- `[ANS]` placeholder questions (66% of dataset): tell model the final line must
  be the boxed answer.
- Anti-rounding: smoke test on 5 questions revealed model truncating
  `143.224229...` to `143` and `2.32624...` to `2.33`, both fail at 1e-8
  tolerance.
- Token budget self-rescue: 17% of baseline_v1 responses hit MAX_TOKENS without
  emitting `\\boxed{}`. Tell model to bail out with a best guess if running low.
"""

from __future__ import annotations

# ─── Common boilerplate appended to every system prompt ──────────────────────
_TOKEN_BUDGET_RULE = (
    "If you are running out of reasoning space, IMMEDIATELY output your "
    "best-guess answer inside \\boxed{...} before stopping. Never end "
    "without a final boxed answer."
)

_NUMERIC_PRECISION_RULE = (
    "Do not round numerical answers. Report at least 6 significant figures or "
    "the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33."
)

_FORMAT_RULES_FREEFORM = (
    "Output rules for the FINAL answer:\n"
    "- Use plain numbers (write 0.5, not '1/2 of pi'), no units, no 'x = ', "
    "no trailing punctuation.\n"
    "- " + _NUMERIC_PRECISION_RULE + "\n"
    "- " + _TOKEN_BUDGET_RULE
)

_FORMAT_RULES_MCQ = (
    "Output rules for the FINAL answer:\n"
    "- Output ONLY the letter inside \\boxed{}. Example: \\boxed{C}.\n"
    "- Do NOT write \\boxed{(C)}, \\boxed{C.}, \\boxed{C)}, or \\boxed{C: ...}. "
    "Letter only, no punctuation, no parentheses.\n"
    "- " + _TOKEN_BUDGET_RULE
)


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step.\n\n"
    "If the problem has K sub-answers (e.g. parts (a) (b) (c), or multiple "
    "[ANS] placeholders), use ONE of these two styles for the FINAL line:\n"
    "  (1) PREFERRED — single boxed, comma-separated:\n"
    "      \\boxed{41, 35, 16}\n"
    "  (2) Multiple boxed blocks separated ONLY by whitespace/commas:\n"
    "      \\boxed{41} \\boxed{35} \\boxed{16}\n"
    "      DO NOT put labels like '(a)', '(b)', words, or sentences "
    "BETWEEN boxed blocks — that BREAKS the parser.\n"
    "      Example of what NOT to do: '(a) \\boxed{41} (b) \\boxed{35}' — "
    "the parser only sees the last box.\n"
    "Do NOT mix the two styles in one response.\n\n"
    "If the question contains [ANS] placeholders, replace each with your "
    "boxed answer in order. The very last line of your response must be a "
    "\\boxed{...}.\n\n"
    + _FORMAT_RULES_FREEFORM
)


SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single "
    "best answer.\n\n"
    + _FORMAT_RULES_MCQ
)


def build_prompt(question: str, options: list[str] | None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question.

    `options` truthy ⇒ MCQ. `options=[]` is treated as free-form (defensive).
    """
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question
